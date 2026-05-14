import os
import pickle
import torch
import torch.distributed as dist
from sparsevllm.utils.log import logger
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.models.qwen2 import Qwen2ForCausalLM
from sparsevllm.models.qwen3 import Qwen3ForCausalLM
from sparsevllm.models.llava_onevision import LlavaOnevisionForCausalLM
from sparsevllm.models.deepseek_v2 import DeepSeekV2ForCausalLM
from sparsevllm.layers.sampler import Sampler
from sparsevllm.utils.context import set_context, get_context, reset_context
from sparsevllm.utils.loader import load_model, sync_deltakv_config_from_checkpoint

from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.utils.profiler import profiler

class ModelRunner:
    """
    负责模型执行的类。每个 GPU Rank 进程都拥有一个 ModelRunner 实例。
    主要职责：权重加载、显存分配 (KV Cache)、槽位管理 (Rank-Local)、前向计算。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
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

        # 初始化分布式环境并绑定对应的 GPU 卡
        if not dist.is_initialized():
            master_port = int(os.getenv("SPARSEVLLM_MASTER_PORT", "2333"))
            dist.init_process_group(
                "nccl",
                f"tcp://localhost:{master_port}",
                world_size=self.world_size,
                rank=rank,
            )
        torch.cuda.set_device(rank)
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 加载对应的模型分片 (Shards)
        if hf_config.model_type == "llava_onevision":
            self.model = LlavaOnevisionForCausalLM(hf_config)
        elif hf_config.model_type in ("qwen2", "llama"):
            self.model = Qwen2ForCausalLM(hf_config)
        elif hf_config.model_type == "deepseek_v2":
            self.model = DeepSeekV2ForCausalLM(
                hf_config,
                dsa_topk=config.dsa_topk,
                use_flash_mla=config.dsa_use_flash_mla,
            )
        elif hf_config.model_type == "deepseek_v32":
            raise NotImplementedError(
                "DeepSeek-V3.2 sparsevllm support is disabled. "
                "Use DeepSeek-V2 or another backend for now."
            )
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model, rank=rank, world_size=self.world_size)
        
        self.sampler = Sampler()

        # DeltaKV cache allocation depends on latent dimension / compressor architecture.
        # Sync those fields from the compressor checkpoint before creating CacheManager.
        sync_deltakv_config_from_checkpoint(config)
        
        # 初始化 CacheManager (负责 KV Cache + 物理槽位)
        self.cache_manager = CacheManager.create(config, rank, self.world_size)

        # 初始化稀疏控制器
        self.sparse_controller = SparseController(config, self.cache_manager)
        # 注入模型 (Qwen-style text backbones only; DeepSeek MLA does not use SparseController callbacks)
        text_backbone = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            text_backbone = self.model.model
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model"):
            if hasattr(self.model.language_model.model, "layers"):
                text_backbone = self.model.language_model.model
        if text_backbone is not None:
            text_backbone.sparse_controller = self.sparse_controller
            self.sparse_controller.set_modules(text_backbone.layers)

        # 加载 DeltaKV 压缩器
        self.load_deltakv_compressors()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # TP 场景下的多进程指令同步
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 创建共享内存用于发送方法调用指令
                self.shm = SharedMemory(name="sparsevllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 其他 Rank 监听共享内存中的指令
                dist.barrier()
                self.shm = SharedMemory(name="sparsevllm")
                self.loop()

    def exit(self):
        """释放资源并注销分布式进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """子进程的主循环：监听共享内存，解析并执行来自 Rank 0 的方法指令"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
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
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """RPC 风格的调用：如果是 Rank 0 则先广播指令，然后所有进程执行本地逻辑"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        # Ensure *all* runner-side ops (including sparse post-processing like DeltaKV eviction)
        # run without autograd bookkeeping to avoid large activation graphs / OOM.
        with torch.inference_mode():
            return method(*args)

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

    def _long_text_threshold(self, is_prefill: bool) -> int:
        if self.config.vllm_sparse_method == "deltakv-snapkv":
            base = (
                self.config.num_sink_tokens
                + self.config.num_recent_tokens
                + self.config.snapkv_window_size
            )
        elif self.config.vllm_sparse_method == "deltakv-standalone":
            base = self.config.num_sink_tokens + self.config.num_recent_tokens
        elif self.config.vllm_sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            base = self.config.num_sink_tokens + self.config.num_recent_tokens
        else:
            base = (
                self.config.num_sink_tokens
                + self.config.num_recent_tokens
                + self.config.num_top_tokens
            )
        return base + (self.config.chunk_prefill_size if is_prefill else 0)

    def _is_long_text_batch(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        # `is_long_text` is a batch-level flag used to gate sparse logic. We compute it
        # dynamically from the *current* sequence lengths so short prompts can become
        # long during decode.
        threshold = self._long_text_threshold(is_prefill)
        if not seqs:
            return False
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
        inputs_embeds = None
        if is_prefill and any(seq.prompt_embeds is not None for seq in seqs):
            if not all(seq.prompt_embeds is not None for seq in seqs):
                raise ValueError("Cannot mix prompt-embedding and token-id-only sequences in one prefill batch.")
            embed_chunks = []
            for seq in seqs:
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + seq.current_chunk_size
                embed_chunks.append(seq.prompt_embeds[start_idx:end_idx])
            inputs_embeds = torch.cat(embed_chunks, dim=0).to(device=input_ids.device, non_blocking=True)
            if inputs_embeds.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    "Prepared prompt embeddings do not match flattened input token count: "
                    f"{inputs_embeds.shape[0]} != {input_ids.shape[0]}"
                )
        set_context(
            is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cache_manager=self.cache_manager,
            is_long_text=self._is_long_text_batch(seqs, is_prefill),
        )
        return input_ids, positions, inputs_embeds

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样超参数"""
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        inputs_embeds: torch.Tensor | None = None,
    ):
        """物理执行逻辑：统一使用 Eager 模式"""
        _stage = 'prefill' if is_prefill else 'decode'
        with profiler.record(f"model_run_model_{_stage}"):
            return self.model.compute_logits(self.model(input_ids, positions, inputs_embeds=inputs_embeds))

    def _move_multimodal_inputs_to_cuda(self, inputs: dict) -> dict:
        moved = {}
        dtype = self.config.hf_config.torch_dtype
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(device="cuda", dtype=dtype, non_blocking=True)
                else:
                    moved[key] = value.to(device="cuda", non_blocking=True)
            else:
                moved[key] = value
        return moved

    def prepare_multimodal_prompts(self, inputs: dict):
        if self.world_size != 1:
            raise ValueError("Sparse-vLLM multimodal prompt embeddings currently require tensor_parallel_size=1.")
        if not hasattr(self.model, "prepare_prompt_embeds"):
            raise ValueError(f"Model {type(self.model).__name__} does not support multimodal prompt embeddings.")
        inputs = self._move_multimodal_inputs_to_cuda(dict(inputs))
        return self.model.prepare_prompt_embeds(inputs)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> tuple[list[int], torch.Tensor | None]:
        """单步执行主逻辑"""
        name = "model_run_prefill" if is_prefill else "model_run_decode"
        with profiler.record(name):
            # 1. 准备前向上下文
            ctx = get_context()
            input_ids, positions, inputs_embeds = self.prepare_step(seqs, is_prefill)
            
            # 2. 准备稀疏化状态
            with profiler.record("model_sparse_prepare"):
                ctx.sparse_controller = self.sparse_controller
                self.sparse_controller.prepare_forward(seqs, is_prefill)
            
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            
            # 3. 前向计算
            logits = self.run_model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)
            
            # 4. Token 采样 (仅 Rank 0)
            with profiler.record("model_sampler"):
                token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

            # 5. 后置稀疏处理 (如 SnapKV 驱逐)
            with profiler.record("model_sparse_post"):
                self.sparse_controller.post_forward(seqs, is_prefill)

            reset_context()
            return token_ids, None # attn_score 不再作为返回值
