import os
import pickle
import time
import uuid
import torch
import torch.distributed as dist
from sparsevllm.utils.log import logger
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from sparsevllm.config import (
    Config,
    _resolve_decode_cuda_graph_capture_sizes,
    _resolve_decode_cuda_graph_context_sizes,
)
from sparsevllm.engine.sequence import Sequence
from sparsevllm.models.qwen2 import Qwen2ForCausalLM
from sparsevllm.models.llama import LlamaForCausalLM
from sparsevllm.layers.sampler import Sampler
from sparsevllm.utils.context import set_context, get_context, reset_context
from sparsevllm.utils.loader import load_model, sync_deltakv_config_from_checkpoint

from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparsevllm.engine.sparse_controller import SparseController
import sparsevllm.platforms as platforms
from sparsevllm.utils.profiler import profiler

try:
    from sparsevllm.models.qwen3 import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None


TP_SHM_NAME_PREFIX = "sparsevllm_"
PREFIX_CACHE_CONTROL_RPC_METHODS = {
    "prefix_cache_inspect",
    "prefix_cache_match",
    "prefix_cache_delete_subtree",
    "prefix_cache_set_eviction_priority",
}
TP_RPC_STATUS_SYNC_METHODS = PREFIX_CACHE_CONTROL_RPC_METHODS | {"run"}


def make_tp_shm_name() -> str:
    return f"{TP_SHM_NAME_PREFIX}{os.getpid()}_{uuid.uuid4().hex}"

class ModelRunner:
    """
    负责模型执行的类。每个 GPU Rank 进程都拥有一个 ModelRunner 实例。
    主要职责：权重加载、显存分配 (KV Cache)、槽位管理 (Rank-Local)、前向计算。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event], tp_shm_name: str | None = None):
        self.config = config
        # Inference-only engine: disable autograd graph construction globally in this process.
        # (This is process-local; must be set inside every spawned TP worker.)
        torch.set_grad_enabled(False)
        profiler.set_rank(rank)
        profiler.set_enabled(config.enable_profiler and rank == 0)
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.tp_shm_name = tp_shm_name
        self.platform = platforms.current_platform
        self.platform.validate_inference()
        self.platform.init_backend()
        self.device = self.platform.get_device(rank)

        # 初始化分布式环境并绑定对应的设备
        self.platform.set_device(self.device)
        if not dist.is_initialized():
            master_port = int(os.getenv("SPARSEVLLM_MASTER_PORT", "2333"))
            dist.init_process_group(
                self.platform.get_distributed_backend(),
                f"tcp://localhost:{master_port}",
                world_size=self.world_size,
                rank=rank,
            )
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)
        
        # 加载对应的模型分片 (Shards)
        if hf_config.model_type == "qwen2":
            self.model = Qwen2ForCausalLM(hf_config)
        elif hf_config.model_type == "qwen3":
            if Qwen3ForCausalLM is None:
                raise ImportError(
                    "Qwen3ForCausalLM is unavailable in this Transformers installation. "
                    "Use a Transformers version with Qwen3 support for Qwen3 models."
                )
            self.model = Qwen3ForCausalLM(hf_config)
        elif hf_config.model_type == "llama":
            self.model = LlamaForCausalLM(hf_config)
        else:
            raise NotImplementedError(f"Unsupported Sparse-vLLM model_type={hf_config.model_type!r}.")
        load_model(self.model, config.model, rank=rank, world_size=self.world_size)
        
        self.sampler = Sampler()

        # DeltaKV cache allocation depends on latent dimension / compressor architecture.
        # Sync those fields from the compressor checkpoint before creating CacheManager.
        sync_deltakv_config_from_checkpoint(config)
        
        # 初始化 CacheManager (负责 KV Cache + 物理槽位)
        self.cache_manager = CacheManager.create(config, rank, self.world_size)

        # 初始化稀疏控制器
        self.sparse_controller = SparseController(config, self.cache_manager)
        # 注入模型
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.sparse_controller = self.sparse_controller
            self.sparse_controller.set_modules(self.model.model.layers)
            if hasattr(self.cache_manager, "set_model_layers"):
                self.cache_manager.set_model_layers(self.model.model.layers)

        # 加载 DeltaKV 压缩器
        self.load_deltakv_compressors()

        decode_static_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
            self.config.decode_cuda_graph_capture_sizes,
            self.config.max_decoding_seqs,
        )
        decode_static_context_sizes = _resolve_decode_cuda_graph_context_sizes(
            self.config.decode_cuda_graph_context_sizes,
            self.config.max_model_len,
        )
        self.cuda_graph_pool = torch.cuda.graph_pool_handle() if self.config.decode_cuda_graph else None
        self.decode_cuda_graph_runner = DecodeCudaGraphRunner(
            cache_manager=self.cache_manager,
            sparse_controller=self.sparse_controller,
            run_model=self.run_model,
            is_long_text_batch=self._is_long_text_batch,
            method=self.config.vllm_sparse_method,
            rank=self.rank,
            capture_sizes=decode_static_capture_sizes,
            context_sizes=decode_static_context_sizes,
            graph_pool=self.cuda_graph_pool,
        )
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # TP 场景下的多进程指令同步
        if self.world_size > 1:
            if not self.tp_shm_name:
                raise ValueError("tp_shm_name is required when tensor_parallel_size > 1.")
            if rank == 0:
                # Rank 0 创建共享内存用于发送方法调用指令
                self.shm = SharedMemory(name=self.tp_shm_name, create=True, size=2**20)
                dist.barrier(device_ids=self.platform.barrier_device_ids(rank))
            else:
                # 其他 Rank 监听共享内存中的指令
                dist.barrier(device_ids=self.platform.barrier_device_ids(rank))
                self.shm = SharedMemory(name=self.tp_shm_name)
                self.loop()

    def exit(self):
        """释放资源并注销分布式进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier(device_ids=self.platform.barrier_device_ids(self.rank))
            if self.rank == 0:
                self.shm.unlink()
        self.platform.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """子进程的主循环：监听共享内存，解析并执行来自 Rank 0 的方法指令"""
        while True:
            method_name, args = self.read_shm()
            try:
                self.call(method_name, *args)
            except Exception as exc:
                if method_name in PREFIX_CACHE_CONTROL_RPC_METHODS:
                    logger.error("TP worker prefix-cache control RPC failed: {}: {}", type(exc).__name__, exc)
                else:
                    raise
            if method_name == "exit":
                break

    def read_shm(self):
        """反序列化共享内存中的方法名和参数"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """序列化方法名 and 参数并写入共享内存"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise RuntimeError(
                f"Shared memory command is too large: {n + 4} > {len(self.shm.buf)}"
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()
        timeout_s = float(os.getenv("SPARSEVLLM_TP_RPC_ACK_TIMEOUT_S", "30"))
        deadline = time.monotonic() + timeout_s
        for event in self.event:
            while event.is_set():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for TP worker to read shared-memory RPC "
                        f"{method_name!r} after {timeout_s:.1f}s."
                    )
                time.sleep(0.0001)

    def call(self, method_name, *args):
        """RPC 风格的调用：如果是 Rank 0 则先广播指令，然后所有进程执行本地逻辑"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        # Ensure *all* runner-side ops (including sparse post-processing like DeltaKV eviction)
        # run without autograd bookkeeping to avoid large activation graphs / OOM.
        if method_name in TP_RPC_STATUS_SYNC_METHODS:
            local_error: BaseException | None = None
            result = None
            try:
                with torch.inference_mode():
                    result = method(*args)
            except BaseException as exc:
                local_error = exc
            self._sync_tp_rpc_status(method_name, local_error)
            if local_error is not None:
                raise local_error
            return result
        with torch.inference_mode():
            return method(*args)

    def _sync_tp_rpc_status(
        self,
        method_name: str,
        local_error: BaseException | None,
    ) -> None:
        if self.world_size <= 1 or not dist.is_initialized():
            return
        failed = torch.tensor(
            [1 if local_error is not None else 0],
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed.item()) != 0 and local_error is None:
            raise RuntimeError(f"At least one TP worker failed during {method_name}.")

    def _sync_prefix_cache_control_rpc_status(
        self,
        method_name: str,
        local_error: BaseException | None,
    ) -> None:
        self._sync_tp_rpc_status(method_name, local_error)

    def load_deltakv_compressors(self):
        """加载 DeltaKV 压缩器权重"""
        method = str(self.config.vllm_sparse_method or "")
        if not method.startswith('deltakv') or self.config.deltakv_path is None:
            return
        
        logger.info(f"Loading DeltaKV compressors from {self.config.deltakv_path}")
        from sparsevllm.utils.loader import load_deltakv_compressors_to_cache_manager

        load_deltakv_compressors_to_cache_manager(self.cache_manager, self.config.deltakv_path)

    def free_slots(self, seq_id: int):
        """通知 CacheManager 释放该序列占用的物理显存位子"""
        with profiler.record("model_free_slots"):
            if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                before = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots seq_id={} before={}", seq_id, before)
            self.cache_manager.free_seq(seq_id)
            if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                after = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots seq_id={} after={}", seq_id, after)

    def free_slots_batch(self, seq_ids: list[int]):
        """Release cache slots for a batch of finished/preempted sequences."""
        with profiler.record("model_free_slots_batch"):
            seq_ids = [int(seq_id) for seq_id in seq_ids]
            if not seq_ids:
                return
            if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                before = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots_batch seq_ids={} before={}", seq_ids, before)
            for seq_id in seq_ids:
                self.cache_manager.free_seq(seq_id)
            if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                after = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots_batch seq_ids={} after={}", seq_ids, after)

    def set_tokenizer_metadata(
        self,
        delimiter_token_ids: list[int],
        non_execution_token_ids: list[int] | None = None,
    ):
        setter = getattr(self.sparse_controller, "set_tokenizer_metadata", None)
        if setter is not None:
            setter(
                delimiter_token_ids=[int(x) for x in delimiter_token_ids],
                non_execution_token_ids=(
                    None
                    if non_execution_token_ids is None
                    else [int(x) for x in non_execution_token_ids]
                ),
            )

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        include_subtree: bool = False,
    ) -> dict[str, object]:
        return self.cache_manager.prefix_cache_inspect(
            [int(token_id) for token_id in token_ids],
            include_subtree=bool(include_subtree),
        )

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        return self.cache_manager.prefix_cache_match(
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        return self.cache_manager.prefix_cache_delete_subtree(
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        priority: int,
    ) -> dict[str, object]:
        return self.cache_manager.prefix_cache_set_eviction_priority(
            [int(token_id) for token_id in token_ids],
            priority=int(priority),
        )

    def _long_text_threshold(self, is_prefill: bool) -> int:
        if self.config.vllm_sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            base = self.config.num_sink_tokens + self.config.num_recent_tokens
        else:
            base = (
                self.config.num_sink_tokens
                + self.config.num_recent_tokens
                + self.config.decode_keep_tokens
            )
        return base + (self.config.chunk_prefill_size if is_prefill else 0)

    def _is_long_text_batch(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        # `is_long_text` is a batch-level flag used to gate sparse logic. We compute it
        # dynamically from the *current* sequence lengths so short prompts can become
        # long during decode.
        if not seqs:
            return False
        if is_prefill and self.cache_manager.is_full_prefill_step(seqs):
            return True
        threshold = self._long_text_threshold(is_prefill)
        if is_prefill:
            flags = [int(seq.num_prompt_tokens) > int(threshold) for seq in seqs]
        else:
            flags = [int(seq.num_tokens) > int(threshold) for seq in seqs]
        is_long = bool(flags[0])
        if any(bool(flag) != is_long for flag in flags):
            raise ValueError("Mixed long/short batch detected; scheduler should enforce separation.")
        return is_long

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        """准备前向上下文并设置 Context"""
        input_ids, positions, cu_seqlens_q = self.cache_manager.prepare_step(seqs, is_prefill)
        set_context(
            is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cache_manager=self.cache_manager,
            is_long_text=self._is_long_text_batch(seqs, is_prefill),
            seqs=seqs,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样超参数"""
        temperatures = [seq.temperature for seq in seqs]
        top_ps = [seq.top_p for seq in seqs]
        top_ks = [seq.top_k for seq in seqs]
        pin_memory = self.platform.supports_pin_memory()
        return (
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=pin_memory).to(
                device=self.device,
                non_blocking=pin_memory,
            ),
            torch.tensor(top_ps, dtype=torch.float32, pin_memory=pin_memory).to(
                device=self.device,
                non_blocking=pin_memory,
            ),
            torch.tensor(top_ks, dtype=torch.int64, pin_memory=pin_memory).to(
                device=self.device,
                non_blocking=pin_memory,
            ),
        )

    def _auto_capture_greedy_sampling(self, seqs: list[Sequence]) -> bool:
        if self.config.decode_cuda_graph_capture_sampling:
            return True
        if self.config.tensor_parallel_size != 1:
            return False
        if self.config.enable_prefix_caching:
            return False
        if str(self.config.vllm_sparse_method or "") not in {"", "omnikv"}:
            return False
        return all(seq.temperature <= 1e-10 for seq in seqs)

    def set_decode_cuda_graph_max_context_len_override(self, max_context_len: int | None):
        self.decode_cuda_graph_runner.set_max_context_len_override(max_context_len)

    def set_omnikv_decode_graph_max_context_len_override(self, max_context_len: int | None):
        self.set_decode_cuda_graph_max_context_len_override(max_context_len)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """物理执行逻辑：统一使用 Eager 模式"""
        _stage = 'prefill' if is_prefill else 'decode'
        with profiler.record(f"model_run_model_{_stage}"):
            return self.model.compute_logits(self.model(input_ids, positions))

    def run_logits_for_compare(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor | None:
        """Debug logits-alignment path: execute one step and return rank-0 logits without sampling."""
        try:
            if is_prefill:
                ctx = get_context()
                input_ids, positions = self.prepare_step(seqs, is_prefill)
                with profiler.record("model_sparse_prepare"):
                    ctx.sparse_controller = self.sparse_controller
                    self.sparse_controller.prepare_forward(seqs, is_prefill)
                logits = self.run_model(input_ids, positions, is_prefill)
            else:
                logits = self.decode_cuda_graph_runner.run_eager_static(seqs)
            with profiler.record("model_sparse_post"):
                self.sparse_controller.post_forward(seqs, is_prefill)
            return logits if self.rank == 0 else None
        finally:
            reset_context()

    def _post_sparse_forward(self, seqs: list[Sequence], is_prefill: bool) -> None:
        with profiler.record("model_sparse_post"):
            with profiler.record("sparse_post_forward"):
                self.sparse_controller.post_forward(seqs, is_prefill)
            with profiler.record("cache_on_forward_end"):
                self.cache_manager.on_forward_end(seqs, is_prefill)

    def _collect_logprobs(
        self,
        logits: torch.Tensor,
        token_ids: list[int],
        seqs: list[Sequence],
    ) -> tuple[list[float | None], list[dict[int, float] | None]] | tuple[None, None]:
        if not any(seq.logprobs is not None for seq in seqs):
            return None, None

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_tensor = torch.tensor(token_ids, device=log_probs.device, dtype=torch.long)
        sampled = log_probs.gather(1, token_tensor.unsqueeze(1)).squeeze(1)
        sampled_logprobs: list[float | None] = sampled.detach().cpu().tolist()

        max_top_logprobs = max(int(seq.logprobs or 0) for seq in seqs)
        top_logprobs: list[dict[int, float] | None]
        if max_top_logprobs <= 0:
            top_logprobs = [None] * len(seqs)
        else:
            top_values, top_indices = torch.topk(
                log_probs,
                k=min(max_top_logprobs, log_probs.shape[-1]),
                dim=-1,
            )
            top_logprobs = []
            for row, seq in enumerate(seqs):
                requested = int(seq.logprobs or 0)
                if requested <= 0:
                    top_logprobs.append(None)
                    continue
                values = top_values[row, :requested].detach().cpu().tolist()
                indices = top_indices[row, :requested].detach().cpu().tolist()
                top_logprobs.append({int(token_id): float(value) for token_id, value in zip(indices, values)})
        return sampled_logprobs, top_logprobs

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> tuple[list[int], tuple[list[float | None], list[dict[int, float] | None]] | None]:
        """单步执行主逻辑"""
        name = "model_run_prefill" if is_prefill else "model_run_decode"
        with profiler.record(name):
            if not is_prefill:
                try:
                    if self.config.decode_cuda_graph:
                        logits, graph_token_ids = self.decode_cuda_graph_runner.run(
                            seqs,
                            capture_sampling=self._auto_capture_greedy_sampling(seqs),
                        )
                    else:
                        logits = self.decode_cuda_graph_runner.run_eager_static(seqs)
                        graph_token_ids = None
                    if self.rank != 0:
                        self._post_sparse_forward(seqs, is_prefill)
                        return None, None
                    self._post_sparse_forward(seqs, is_prefill)
                    with profiler.record("model_sampler"):
                        if graph_token_ids is not None:
                            token_ids = graph_token_ids.tolist()
                        else:
                            all_greedy = all(seq.temperature <= 1e-10 for seq in seqs)
                            temperatures = None
                            top_ps = None
                            top_ks = None
                            if not all_greedy:
                                temperatures, top_ps, top_ks = self.prepare_sample(seqs)
                            token_ids = self.sampler(
                                logits,
                                temperatures,
                                top_ps,
                                top_ks,
                                all_greedy=all_greedy,
                            ).tolist()
                    logprob_outputs = self._collect_logprobs(logits, token_ids, seqs)
                    return token_ids, logprob_outputs
                finally:
                    reset_context()

            # 1. 准备前向上下文
            ctx = get_context()
            input_ids, positions = self.prepare_step(seqs, is_prefill)
            
            # 2. 准备稀疏化状态
            with profiler.record("model_sparse_prepare"):
                ctx.sparse_controller = self.sparse_controller
                self.sparse_controller.prepare_forward(seqs, is_prefill)
            
            all_greedy = all(seq.temperature <= 1e-10 for seq in seqs) if self.rank == 0 else False
            temperatures = None
            top_ps = None
            top_ks = None
            if self.rank == 0 and not all_greedy:
                temperatures, top_ps, top_ks = self.prepare_sample(seqs)
            
            # 3. 前向计算
            logits = self.run_model(input_ids, positions, is_prefill)
            
            # 4. Token 采样 (仅 Rank 0)
            with profiler.record("model_sampler"):
                token_ids = self.sampler(logits, temperatures, top_ps, top_ks, all_greedy=all_greedy).tolist() if self.rank == 0 else None
            logprob_outputs = self._collect_logprobs(logits, token_ids, seqs) if self.rank == 0 else None

            # 5. 后置稀疏处理 (如 SnapKV 驱逐)
            self._post_sparse_forward(seqs, is_prefill)

            reset_context()
            return token_ids, logprob_outputs
