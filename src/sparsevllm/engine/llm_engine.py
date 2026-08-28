import atexit
import gc
import os
import pickle
from dataclasses import fields
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter
import threading
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig, Qwen2Tokenizer
import torch
import torch.multiprocessing as mp
from sparsevllm.utils.code_revision import code_revision_info
from sparsevllm.utils.log import logger
import sys

from sparsevllm.configs.cuda_graph import (
    build_decode_cuda_graph_startup_family_plan,
)

from sparsevllm.config import Config
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.model_runner import ModelRunner, make_tp_shm_name, select_master_port
from sparsevllm.engine.input_processor import tokenize_text_prompt
from sparsevllm.multimodal.inputs import (
    MultiModalInputProcessor,
    MultiModalPrompt,
    is_multimodal_prompt,
)
from sparsevllm.engine.prefix_cache import PrefixCacheRoutingSnapshot
from sparsevllm.engine.chain_cache import (
    ChainCacheIndex,
    ChainRoutingSnapshot,
    ChainModeError,
    ChainNotFoundError,
    ChainOwnerMismatchError,
    ChainPrefixMismatchError,
    RequestAdmission,
    stable_token_digest,
)
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.utils.profiler import profiler

def _deltakv_graph_warmup_profile(config: Config) -> str:
    graph_warmup = bool(getattr(config, "decode_graph", False))
    method = normalize_sparse_method(getattr(config, "sparse_method", "") or "")
    if not graph_warmup:
        return "decode_1seq"
    if method == "deltakv":
        warmup_policy = os.getenv("SPARSEVLLM_DELTAKV_GRAPH_WARMUP", "graph").strip().lower()
        if warmup_policy in ("eager", "minimal", "current", "prefill", "prefill_only"):
            return "prefill_only"
        if warmup_policy in ("decode_1seq", "decode-1seq", "decode"):
            return "decode_1seq"
        if warmup_policy in ("big_prefill_only", "big-prefill-only", "prefill_graph_batch"):
            return "big_prefill_only"
        if warmup_policy in ("graph", "full"):
            return "graph"
        raise ValueError(
            "SPARSEVLLM_DELTAKV_GRAPH_WARMUP must be one of "
            "'prefill_only', 'decode_1seq', 'big_prefill_only', or 'graph', "
            f"got {warmup_policy!r}."
        )
    return "graph"


def _use_graph_scaled_warmup(config: Config) -> bool:
    return _deltakv_graph_warmup_profile(config) == "graph"


def _moe_workspace_warmup_token_counts(config: Config) -> tuple[int, ...]:
    if config.model_spec.num_experts_field is None:
        return ()

    max_batched_tokens = int(config.max_num_batched_tokens)
    mlp_chunk_size = int(config.mlp_chunk_size)
    if max_batched_tokens <= 0 or mlp_chunk_size <= 0:
        raise ValueError(
            "MoE workspace warmup requires positive max_num_batched_tokens and "
            f"mlp_chunk_size, got {max_batched_tokens} and {mlp_chunk_size}."
        )

    max_moe_tokens = min(max_batched_tokens, mlp_chunk_size)
    decode_tokens = min(
        max_moe_tokens,
        max(1, int(config.max_decoding_seqs)),
    )
    return tuple(dict.fromkeys((decode_tokens, max_moe_tokens)))


class _ThroughputIntervalLogger:
    def __init__(self, interval_s: float):
        self._interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prefill_tokens = 0
        self._decode_tokens = 0
        self._running_seqs = 0
        self._prefill_seqs = 0
        self._decode_seqs = 0
        self._prefill_chunked_seqs = 0
        self._prefill_full_seqs = 0
        self._prefill_raw_offload_seqs = 0
        self._decode_long_seqs = 0
        self._decode_short_seqs = 0
        self._last_batch = "idle"
        self._last_report_t = perf_counter()

    def start(self):
        if self._interval_s <= 0:
            return
        if self._thread is not None:
            return
        with self._lock:
            self._last_report_t = perf_counter()
        self._thread = threading.Thread(target=self._run, name="svllm-throughput-logger", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=self._interval_s + 1.0)

    def record_step(self, num_tokens: int):
        if num_tokens == 0:
            return
        with self._lock:
            if num_tokens > 0:
                self._prefill_tokens += int(num_tokens)
            else:
                self._decode_tokens += int(-num_tokens)

    def record_state(
        self,
        running_seqs: int,
        prefill_seqs: int,
        decode_seqs: int,
        prefill_chunked_seqs: int,
        prefill_full_seqs: int,
        prefill_raw_offload_seqs: int,
        decode_long_seqs: int,
        decode_short_seqs: int,
        last_batch: str,
    ):
        with self._lock:
            self._running_seqs = int(running_seqs)
            self._prefill_seqs = int(prefill_seqs)
            self._decode_seqs = int(decode_seqs)
            self._prefill_chunked_seqs = int(prefill_chunked_seqs)
            self._prefill_full_seqs = int(prefill_full_seqs)
            self._prefill_raw_offload_seqs = int(prefill_raw_offload_seqs)
            self._decode_long_seqs = int(decode_long_seqs)
            self._decode_short_seqs = int(decode_short_seqs)
            self._last_batch = str(last_batch)

    def _run(self):
        while not self._stop.wait(self._interval_s):
            now = perf_counter()
            with self._lock:
                prefill_tokens = self._prefill_tokens
                decode_tokens = self._decode_tokens
                running_seqs = self._running_seqs
                prefill_seqs = self._prefill_seqs
                decode_seqs = self._decode_seqs
                prefill_chunked_seqs = self._prefill_chunked_seqs
                prefill_full_seqs = self._prefill_full_seqs
                prefill_raw_offload_seqs = self._prefill_raw_offload_seqs
                decode_long_seqs = self._decode_long_seqs
                decode_short_seqs = self._decode_short_seqs
                last_batch = self._last_batch
                self._prefill_tokens = 0
                self._decode_tokens = 0
                last_t = self._last_report_t
                self._last_report_t = now

            dt = max(now - last_t, 1e-9)
            prefill_tp = prefill_tokens / dt
            decode_tp = decode_tokens / dt
            logger.info(
                "Avg TP (last {dt:.1f}s): prefill_tp={prefill_tp:.0f} tok/s, decode_tp={decode_tp:.0f} tok/s "
                "| seq(run/prf/dc)={running_seqs}/{prefill_seqs}/{decode_seqs} "
                "| prf(chunked/full/raw_offload)={prefill_chunked_seqs}/{prefill_full_seqs}/{prefill_raw_offload_seqs} "
                "dc(L/S)={decode_long_seqs}/{decode_short_seqs} "
                "| last_batch={last_batch} "
                "(prefill_tokens={prefill_tokens}, decode_tokens={decode_tokens})",
                dt=dt,
                prefill_tokens=prefill_tokens,
                prefill_tp=prefill_tp,
                decode_tokens=decode_tokens,
                decode_tp=decode_tp,
                running_seqs=running_seqs,
                prefill_seqs=prefill_seqs,
                decode_seqs=decode_seqs,
                prefill_chunked_seqs=prefill_chunked_seqs,
                prefill_full_seqs=prefill_full_seqs,
                prefill_raw_offload_seqs=prefill_raw_offload_seqs,
                decode_long_seqs=decode_long_seqs,
                decode_short_seqs=decode_short_seqs,
                last_batch=last_batch,
            )

class LLMEngine:
    """
    Sparse-vLLM 推理引擎的核心入口类。
    负责协调 Tokenizer、调度器 (Scheduler) 和模型执行器 (ModelRunner)。
    管理多进程张量并行 (Tensor Parallelism) 的生命周期。
    """

    def __init__(self, model, **kwargs):
        # 1. 初始化配置
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        ignored_keys = sorted(set(kwargs) - config_fields)
        if ignored_keys:
            raise ValueError(
                f"Unknown Sparse-vLLM config keys: {ignored_keys}. "
                "Runtime parameter aliases and unknown keys are not accepted."
            )
        config = Config(model, **config_kwargs)
        self.config = config
        
        # 初始化 Profiler
        profiler.set_enabled(config.enable_profiler)
        
        # 2. 启动 world worker 进程；TP/EP/DP 语义由 ParallelContext 管理。
        master_port = select_master_port()
        logger.info("Using distributed master port: {}", master_port)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        tp_shm_name = make_tp_shm_name() if config.world_size > 1 else None
        for i in range(1, config.world_size):
            event = (ctx.Event(), ctx.Event())
            # 为每一个非零 Rank 启动一个独立的 ModelRunner 进程
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event, tp_shm_name, master_port),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 3. 初始化主进程的 ModelRunner (Rank 0)
        # 注意：必须先初始化 ModelRunner 以便在本地 GPU 分配 KV Cache 账本
        self.model_runner = ModelRunner(config, 0, self.events, tp_shm_name, master_port)
        
        # 加载分词器
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.multimodal_processor = (
            MultiModalInputProcessor(config.model)
            if config.enable_multimodal
            and callable(getattr(self.model_runner.model, "encode_multimodal", None))
            else None
        )
        generation_config = GenerationConfig.from_pretrained(config.model)
        eos_values = generation_config.eos_token_id
        if eos_values is None:
            eos_values = []
        elif isinstance(eos_values, int):
            eos_values = [eos_values]
        else:
            eos_values = list(eos_values)
        if self.tokenizer.eos_token_id is not None:
            eos_values.append(int(self.tokenizer.eos_token_id))
        config.eos_token_ids = tuple(dict.fromkeys(int(token_id) for token_id in eos_values))
        config.eos = config.eos_token_ids[0] if config.eos_token_ids else -1
        self.model_runner.call(
            "set_tokenizer_metadata",
            self._build_delimiter_token_ids(self.tokenizer),
            self._build_non_execution_token_ids(self.tokenizer),
        )
        
        # 4. 初始化调度器
        # 关键设计：将 Rank 0 的 CacheManager 传给 Scheduler。
        # Scheduler 通过它来感知全局显存的余量，从而做出调度和抢占决策。
        self.scheduler = Scheduler(
            config,
            self.model_runner.runtime_state,
            prefix_cache_hit_refresher=(
                self._refresh_prefix_cache_hit
                if config.enable_prefix_caching
                else None
            ),
        )
        
        self._exited = False
        self._throughput_logger = _ThroughputIntervalLogger(config.throughput_log_interval_s)
        self.last_step_token_outputs: list[tuple[int, list[int]]] = []
        self.last_step_prompt_cache_hits: list[tuple[int, int]] = []
        self.last_step_logprob_outputs: list[
            tuple[int, list[float | None], list[dict[int, float] | None]]
        ] = []
        self._active_chain_sequences: dict[int, Sequence] = {}
        # 注册退出钩子，确保程序崩溃或结束时能正确释放多进程资源
        self._atexit_callback = self.exit
        atexit.register(self._atexit_callback)

        # 5. 预热模型
        self._warmup()
        if os.getenv("SPARSEVLLM_PROFILER_RESET_AFTER_WARMUP", "0") == "1":
            profiler.reset()
        self._throughput_logger.start()

    @staticmethod
    def _build_delimiter_token_ids(tokenizer) -> list[int]:
        # Match SkipKV's official newline-oriented split set. Plain "." or "?"
        # would trigger steering far more often than the paper implementation.
        delimiter_texts = [
            "\n",
            ".\n",
            ")\n",
            "\n\n",
            ".\n\n",
            ")\n\n",
            "?\n\n",
        ]
        token_ids: set[int] = set()
        for text in delimiter_texts:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                ids = []
            if ids:
                token_ids.add(int(ids[-1]))
        return sorted(token_ids)

    @staticmethod
    def _build_non_execution_token_ids(tokenizer) -> list[int]:
        marker_texts = [
            "Alternatively",
            "Wait",
            "again",
        ]
        token_ids: set[int] = set()
        for text in marker_texts:
            candidates = {text, " " + text, text.lower(), " " + text.lower()}
            for candidate in candidates:
                try:
                    ids = tokenizer.encode(candidate, add_special_tokens=False)
                except Exception:
                    ids = []
                if ids:
                    token_ids.add(int(ids[-1]))
        return sorted(token_ids)

    def _warmup(self):
        """预热模型，确保所有算子和显存都已就绪"""
        logger.info("Warming up the engine...")
        
        warmup_profile = _deltakv_graph_warmup_profile(self.config)
        graph_sized_batch = warmup_profile in ("graph", "big_prefill_only")
        decode_warmup = warmup_profile in ("graph", "decode_1seq")
        num_seqs = int(self.config.max_decoding_seqs) if graph_sized_batch else 1
        startup_capture = bool(
            getattr(self.config, "decode_graph_startup_capture", False)
        )
        
        # Startup precapture owns decode warmup when enabled. Keep this first
        # pass prefill-only so it cannot create unplanned short/long graph keys.
        sampling_params = SamplingParams(
            max_tokens=2 if decode_warmup and not startup_capture else 1,
            temperature=0.0,
            ignore_eos=decode_warmup and not startup_capture,
        )
        max_prompt_len = max(1, int(self.config.max_model_len) - int(sampling_params.max_tokens))
        warmup_len = min(int(self.config.engine_prefill_chunk_size), max_prompt_len)
        warmup_len_override = os.getenv("SPARSEVLLM_DELTAKV_GRAPH_WARMUP_PROMPT_LEN", "").strip().lower()
        if warmup_len_override:
            if warmup_len_override in {"max", "full", "max_model_len", "max-model-len"}:
                warmup_len = max_prompt_len
            else:
                try:
                    warmup_len = int(warmup_len_override)
                except ValueError as exc:
                    raise ValueError(
                        "SPARSEVLLM_DELTAKV_GRAPH_WARMUP_PROMPT_LEN must be a positive integer or 'max', "
                        f"got {warmup_len_override!r}."
                    ) from exc
                if warmup_len <= 0:
                    raise ValueError(
                        "SPARSEVLLM_DELTAKV_GRAPH_WARMUP_PROMPT_LEN must be positive, "
                        f"got {warmup_len}."
                    )
        free_slots = int(self.model_runner.runtime_state.prompt_admission_free_slots())
        capacity = free_slots - (num_seqs - 1) - num_seqs * int(sampling_params.max_tokens)
        if capacity <= 0:
            raise RuntimeError(
                "Insufficient runtime capacity for warmup: "
                f"free_slots={free_slots} "
                f"num_seqs={num_seqs} max_tokens={sampling_params.max_tokens}."
            )
        max_warmup_len = min(max_prompt_len, capacity)
        if warmup_len > max_warmup_len:
            logger.warning(
                "Warmup prompt length ({}) exceeds runtime capacity ({}); clamping.",
                warmup_len,
                max_warmup_len,
            )
            warmup_len = max_warmup_len
        startup_plan = (
            build_decode_cuda_graph_startup_family_plan(self.config)
            if startup_capture
            else []
        )
        capture_groups: dict[tuple[int, bool], list[int]] = {}
        for batch_size, context_capacity, is_long_text in startup_plan:
            capture_groups.setdefault((batch_size, is_long_text), []).append(
                context_capacity
            )

        num_warmup_rounds = 2 if warmup_profile == "graph" else 1
        vocab_size = int(self.config.hf_config.vocab_size)
        num_dummy_prompts = (
            num_seqs * num_warmup_rounds
            + sum(batch_size for batch_size, _ in capture_groups)
        )
        if num_dummy_prompts > vocab_size:
            raise ValueError(
                "Warmup requires one distinct leading token per dummy prompt: "
                f"need={num_dummy_prompts} vocab_size={vocab_size}."
            )
        logger.info(
            f"Warmup profile: {warmup_profile} "
            f"(num_seqs={num_seqs}, max_tokens={sampling_params.max_tokens}, "
            f"ignore_eos={sampling_params.ignore_eos})."
        )

        def run_warmup(
            params: SamplingParams,
            prompt_offset: int,
            *,
            batch_size: int = num_seqs,
            first_prompt_len: int = warmup_len,
        ) -> int:
            for request_idx in range(batch_size):
                # Distinct leading tokens prevent prefix-cache reuse within or
                # across warmup rounds.
                prompt_len = first_prompt_len if request_idx == 0 else 1
                dummy_prompt = [prompt_offset + request_idx] + [0] * (prompt_len - 1)
                self.add_request(dummy_prompt, params)
            while not self.is_finished():
                self.step()
            return prompt_offset + batch_size

        def prepare_capture_batch(
            params: SamplingParams,
            prompt_offset: int,
            *,
            batch_size: int,
            prompt_len: int,
        ) -> tuple[list[Sequence], int]:
            seq_ids = []
            for request_idx in range(batch_size):
                dummy_prompt = [prompt_offset + request_idx] + [0] * (prompt_len - 1)
                seq_ids.append(self.add_request(dummy_prompt, params))

            parked: list[Sequence] = []
            while self.scheduler.waiting:
                self.step()
                while self.scheduler.decoding:
                    parked.append(self.scheduler.decoding.popleft())
            while self.scheduler.decoding:
                parked.append(self.scheduler.decoding.popleft())
            if len(parked) != batch_size:
                raise RuntimeError(
                    "Startup decode CUDA Graph prefill did not park the requested "
                    f"batch: expected={batch_size}, actual={len(parked)}."
                )
            if {int(seq.seq_id) for seq in parked} != set(seq_ids):
                raise RuntimeError("Startup decode CUDA Graph prefill parked unexpected sequences.")
            return parked, prompt_offset + batch_size

        prompt_offset = run_warmup(sampling_params, prompt_offset=0)

        if startup_plan:
            self.model_runner.call("begin_decode_cuda_graph_capture")
            short_graphs = sum(not is_long for _, _, is_long in startup_plan)
            long_graphs = len(startup_plan) - short_graphs
            logger.info(
                "Startup decode CUDA Graph capture: {} coarse graphs "
                "(limit={}, short={}, long={}, plan={}).",
                len(startup_plan),
                self.config.decode_graph_max_cached_graphs,
                short_graphs,
                long_graphs,
                startup_plan,
            )
            capture_params = SamplingParams(
                max_tokens=2,
                temperature=0.0,
                ignore_eos=True,
            )
            threshold = self.scheduler._long_text_threshold(is_prefill=False)
            for (batch_size, is_long_text), context_capacities in capture_groups.items():
                prompt_len = int(threshold) if is_long_text else 1
                parked, prompt_offset = prepare_capture_batch(
                    capture_params,
                    prompt_offset,
                    batch_size=batch_size,
                    prompt_len=prompt_len,
                )
                try:
                    observed_long = self.scheduler._is_long_text(
                        parked[0],
                        is_prefill=False,
                    )
                    if bool(observed_long) != bool(is_long_text):
                        raise RuntimeError(
                            "Startup decode CUDA Graph family prefill crossed the "
                            "wrong long-text boundary: "
                            f"expected={is_long_text}, observed={observed_long}, "
                            f"threshold={threshold}, num_tokens={parked[0].num_tokens}."
                        )
                    for context_capacity in context_capacities:
                        self.model_runner.call(
                            "set_decode_cuda_graph_max_context_len_override",
                            context_capacity,
                        )
                        self.model_runner.call(
                            "capture_decode_cuda_graph_warmup",
                            parked,
                        )
                finally:
                    self.model_runner.call(
                        "set_decode_cuda_graph_max_context_len_override",
                        None,
                    )
                    self.scheduler.decoding.extend(parked)
                    for seq in parked:
                        self.abort_request(int(seq.seq_id))
            self.model_runner.call(
                "set_decode_cuda_graph_reuse_larger_context_graphs",
                True,
            )
            graph_runner = self.model_runner.decode_graph_runner
            captured = {
                (
                    int(key.batch_size),
                    int(
                        state.capture_context_capacity
                        if key.shape_policy == "batch_only"
                        else key.context_capacity
                    ),
                    bool(key.is_long_text),
                )
                for key, state in graph_runner._graphs.items()
                if state.graph is not None
                and key.method == str(self.config.sparse_method or "")
                and not key.capture_sampling
            }
            missing = sorted(set(startup_plan) - captured)
            if missing:
                raise RuntimeError(
                    "Startup decode CUDA Graph capture did not materialize its plan: "
                    f"missing={missing}."
                )
            self.model_runner.call("collect_decode_cuda_graph_metadata")
            self.model_runner.call("exchange_decode_cuda_graph_metadata")
            self.model_runner.call("register_decode_cuda_graph_buffers")
            self.model_runner.call("seal_decode_cuda_graph_startup_plan")
            logger.info(
                "Startup decode CUDA Graph capture finished: cached={} "
                "capture_count={} replay_count={}.",
                len(captured),
                graph_runner.capture_count,
                graph_runner.replay_count,
            )

        if warmup_profile == "graph":
            # CUDA Graph capture establishes its private allocator pool. Warm
            # prefill once more against the final allocator layout.
            logger.info(f"Post-capture prefill warmup (num_seqs={num_seqs}).")
            prompt_offset = run_warmup(
                SamplingParams(max_tokens=1, temperature=0.0),
                prompt_offset=prompt_offset,
            )

        self._warmup_moe_workspaces()
        self._after_warmup_debug_cleanup()
        self.model_runner.call("log_operator_implementations")
        logger.info("Warmup finished.")

    def _warmup_moe_workspaces(self) -> None:
        token_counts = _moe_workspace_warmup_token_counts(self.config)
        if not token_counts:
            return
        logger.info(
            "Post-allocation MoE workspace warmup token counts: {}. "
            "An OOM here is fatal so gpu_memory_utilization can be tuned before serving.",
            token_counts,
        )
        for num_tokens in token_counts:
            self.model_runner.call("warmup_moe_workspace", num_tokens)

    def _after_warmup_debug_cleanup(self):
        self.model_runner.call("reset_after_warmup")

    @staticmethod
    def _cleanup_model_runner_shared_memory(model_runner):
        shm = getattr(model_runner, "shm", None)
        if shm is None:
            return
        try:
            shm.close()
        except Exception as exc:
            logger.warning("Failed to close ModelRunner shared memory during shutdown: {}", repr(exc))
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to unlink ModelRunner shared memory during shutdown: {}", repr(exc))

    def exit(self):
        """优雅地退出所有子进程并清理共享内存"""
        atexit_callback = getattr(self, "_atexit_callback", None)
        if atexit_callback is not None:
            atexit.unregister(atexit_callback)
            del self._atexit_callback
        if self._exited:
            return
        self._exited = True

        profiler.print_stats()
        if hasattr(self, "_throughput_logger"):
            self._throughput_logger.stop()
        runner_exit_completed, runner_platform = self._shutdown_runtime()
        if runner_exit_completed:
            # Collect only after _shutdown_runtime() returns. Its worker-thread
            # closure temporarily owns ModelRunner, so collecting inside that
            # frame can leave cyclic model/cache objects alive until exit().
            gc.collect()
            if runner_platform is not None:
                runner_platform.empty_cache()

    def _shutdown_runtime(self):
        """Stop the runner/workers and drop engine-owned runtime references."""
        runner_exit_completed = True
        runner_platform = None
        if hasattr(self, "model_runner"):
            model_runner = self.model_runner
            runner_platform = getattr(model_runner, "platform", None)
            timeout_s = float(os.getenv("SPARSEVLLM_ENGINE_EXIT_TIMEOUT_S", "10"))
            errors: list[BaseException] = []

            def call_model_runner_exit():
                try:
                    model_runner.call("exit")
                except BaseException as exc:  # pragma: no cover - surfaced by warning below.
                    errors.append(exc)

            exit_thread = threading.Thread(
                target=call_model_runner_exit,
                name="sparsevllm-engine-exit",
                daemon=True,
            )
            exit_thread.start()
            exit_thread.join(timeout=max(0.0, timeout_s))
            if exit_thread.is_alive():
                runner_exit_completed = False
                logger.warning(
                    "Timed out waiting {:.1f}s for ModelRunner exit RPC; terminating workers.",
                    timeout_s,
                )
                self._cleanup_model_runner_shared_memory(model_runner)
            elif errors:
                logger.warning("ModelRunner exit RPC failed during shutdown: {}", repr(errors[0]))
                self._cleanup_model_runner_shared_memory(model_runner)
            errors.clear()
            del self.model_runner
        if hasattr(self, "scheduler"):
            del self.scheduler
        if hasattr(self, "ps"):
            join_timeout_s = float(os.getenv("SPARSEVLLM_WORKER_JOIN_TIMEOUT_S", "5"))
            for p in self.ps:
                # The exit RPC has already asked each worker to leave its loop.
                # Give it time to release distributed/Event resources before using
                # terminate(), which can leave multiprocessing semaphores registered.
                p.join(timeout=max(0.0, join_timeout_s))
                if p.is_alive():
                    logger.warning(
                        "Worker process pid={} did not stop after the exit RPC; terminating.",
                        p.pid,
                    )
                    p.terminate()
                    p.join(timeout=max(0.0, join_timeout_s))
                if p.is_alive():
                    logger.warning(
                        "Worker process pid={} did not stop after terminate; killing.",
                        p.pid,
                    )
                    p.kill()
                    p.join(timeout=max(0.0, join_timeout_s))
                close = getattr(p, "close", None)
                if callable(close) and not p.is_alive():
                    close()
        if hasattr(self, "events"):
            self.events.clear()
        return runner_exit_completed, runner_platform

    def _tokenize_prompt(self, prompt: str | list[int]) -> list[int]:
        tokenizer = self.tokenizer if isinstance(prompt, str) else None
        return tokenize_text_prompt(tokenizer, prompt)

    def admit_request(
        self,
        prompt: str | list[int] | MultiModalPrompt | dict,
        sampling_params: SamplingParams,
        chain_id: str | None = None,
        chain_append_only: bool = False,
    ) -> RequestAdmission:
        """Validate and synchronously admit one request.

        In chain mode the returned seq_id is the resident sequence identity and
        remains stable across turns. The caller's request identity is separate.
        """
        multimodal = None
        if is_multimodal_prompt(prompt):
            if self.multimodal_processor is None:
                raise NotImplementedError(
                    "Multimodal input is disabled or unsupported by this model."
                )
            if chain_id or chain_append_only:
                raise ChainModeError("Multimodal requests do not support chain mode.")
            multimodal = self.multimodal_processor.process(prompt)
            prompt = multimodal.token_ids
        mode = str(
            getattr(self.config, "resolved_prefix_cache_mode", "disabled")
        )
        normalized_chain_id = str(chain_id or "").strip()
        existing = None
        if mode == "chain" and normalized_chain_id:
            coordinator = (
                self.model_runner.runtime_state.chain_cache_coordinator
            )
            if coordinator is None:
                raise RuntimeError(
                    "Config resolved chain prefix caching but the runtime has no "
                    "ChainCacheCoordinator."
                )
            existing = coordinator.index.records.get(normalized_chain_id)
            if existing is None:
                coordinator.index.lookup(normalized_chain_id)
        if chain_append_only:
            if existing is None:
                raise ChainNotFoundError(
                    "chain_append_only requires an existing chain.",
                    chain_id=normalized_chain_id or None,
                )
            suffix_token_ids = (
                [
                    int(token_id)
                    for token_id in self.tokenizer.encode(
                        prompt,
                        add_special_tokens=False,
                    )
                ]
                if isinstance(prompt, str)
                else [int(token_id) for token_id in prompt]
            )
            if not suffix_token_ids:
                raise ChainPrefixMismatchError(
                    "A chain append must contain at least one suffix token.",
                    chain_id=normalized_chain_id,
                )
            prompt = [
                int(token_id) for token_id in existing.token_ids
            ] + suffix_token_ids
        else:
            prompt = self._tokenize_prompt(prompt)
        prompt_len = len(prompt)
        max_tokens = sampling_params.max_tokens
        if prompt_len + max_tokens > self.config.max_model_len:
            raise ValueError(
                "Prompt length + max_tokens exceeds max_model_len: "
                f"{prompt_len} + {max_tokens} > {self.config.max_model_len}. "
                "Reduce prompt/decoding length or increase max_model_len if the model supports it."
            )
        logger.debug(f'add prompt with {len(prompt)} tokens.')
        seq = Sequence(prompt, sampling_params)
        if multimodal is not None:
            seq.multimodal_digest = multimodal.digest
            seq.multimodal_full_prefill = (
                getattr(self.config.hf_config, "use_bidirectional_attention", None)
                == "vision"
            )
            payload = pickle.dumps(multimodal.tensors, protocol=pickle.HIGHEST_PROTOCOL)
            payload_shm = SharedMemory(create=True, size=len(payload))
            try:
                payload_shm.buf[: len(payload)] = payload
                seq.multimodal_position_delta = int(
                    self.model_runner.call(
                        "register_multimodal_shared",
                        int(seq.seq_id),
                        list(prompt),
                        payload_shm.name,
                        len(payload),
                    )
                )
            except Exception as register_error:
                try:
                    self.model_runner.call("free_multimodal", int(seq.seq_id))
                except Exception as cleanup_error:
                    logger.error(
                        "Failed to roll back multimodal seq_id={} after registration "
                        "error {}: {}",
                        seq.seq_id,
                        type(register_error).__name__,
                        cleanup_error,
                    )
                raise
            finally:
                payload_shm.close()
                payload_shm.unlink()
        if mode != "chain":
            if normalized_chain_id:
                raise ChainModeError(
                    "chain_id requires enable_prefix_caching=True with "
                    "prefix_cache_mode='chain'.",
                    chain_id=normalized_chain_id,
                )
            try:
                self.scheduler.add(seq)
            except Exception:
                if multimodal is not None:
                    self.model_runner.call("free_multimodal", int(seq.seq_id))
                raise
            return RequestAdmission(
                seq_id=int(seq.seq_id),
                chain_id=None,
                chain_status="disabled",
                reused_tokens=0,
                prefilled_tokens=prompt_len,
                prompt_token_ids=list(prompt),
            )

        coordinator = self.model_runner.runtime_state.chain_cache_coordinator
        if coordinator is None:
            raise RuntimeError(
                "Config resolved chain prefix caching but the runtime has no "
                "ChainCacheCoordinator."
            )
        created = not normalized_chain_id
        if created:
            normalized_chain_id = ChainCacheIndex.new_chain_id()
        if existing is not None:
            seq.seq_id = int(existing.seq_id)
        recreated = False
        try:
            plan = self.model_runner.runtime_state.chain_admission_plan(
                normalized_chain_id,
                int(seq.seq_id),
                prompt,
                int(sampling_params.max_tokens),
            )
        except ChainPrefixMismatchError as exc:
            if existing is None or chain_append_only:
                raise
            replaced_chain_id = normalized_chain_id
            replaced_seq_id = int(existing.seq_id)
            self.model_runner.call(
                "chain_invalidate",
                replaced_chain_id,
                replaced_seq_id,
            )
            normalized_chain_id = ChainCacheIndex.new_chain_id()
            logger.warning(
                "Recreating chain after strict token-prefix mismatch: "
                "old_chain_id={} new_chain_id={} input_tokens={} reason={}",
                replaced_chain_id,
                normalized_chain_id,
                prompt_len,
                str(exc),
            )
            plan = self.model_runner.runtime_state.chain_admission_plan(
                normalized_chain_id,
                int(seq.seq_id),
                prompt,
                int(sampling_params.max_tokens),
            )
            recreated = True
        if plan.status == "resumed" and prompt_len <= int(plan.reused_tokens):
            raise ChainPrefixMismatchError(
                "A resumed chain request must include at least one suffix token "
                "beyond the processed boundary.",
                chain_id=normalized_chain_id,
            )
        self.model_runner.call(
            "chain_validate_admission_plan",
            plan,
            prompt_len,
            stable_token_digest(prompt, count=int(plan.reused_tokens)),
            int(sampling_params.max_tokens),
        )
        self.model_runner.call("chain_apply_admission", plan)
        chain_status = "recreated" if recreated else str(plan.status)
        seq.chain_id = normalized_chain_id
        seq.chain_status = chain_status
        seq.chain_reused_tokens = int(plan.reused_tokens)
        seq.num_prefilled_tokens = int(plan.reused_tokens)
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(plan.reused_tokens)
        seq.prefix_cache_method = str(self.config.sparse_method or "")
        try:
            self.scheduler.add(seq)
        except Exception:
            self.model_runner.call(
                "chain_invalidate",
                normalized_chain_id,
                int(seq.seq_id),
            )
            raise
        self._active_chain_sequences[int(seq.seq_id)] = seq
        return RequestAdmission(
            seq_id=int(seq.seq_id),
            chain_id=normalized_chain_id,
            chain_status=chain_status,
            reused_tokens=int(plan.reused_tokens),
            prefilled_tokens=prompt_len - int(plan.reused_tokens),
            prompt_token_ids=list(prompt),
        )

    def add_request(
        self,
        prompt: str | list[int] | MultiModalPrompt | dict,
        sampling_params: SamplingParams,
    ):
        """Backward-compatible request API returning only seq_id."""
        return self.admit_request(prompt, sampling_params).seq_id

    def _refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self.model_runner.call("refresh_prefix_cache_hit", seq)

    def abort_request(self, seq_id: int, disposition: str = "invalidate"):
        """Abort a queued or running request and release any owned KV slots."""
        disposition = str(disposition)
        if disposition != "invalidate":
            raise ValueError(
                "abort_request only supports 'invalidate'; interrupted chain "
                "state cannot be retained safely, got "
                f"{disposition!r}."
            )
        chain_seq = self._active_chain_sequences.get(int(seq_id))
        multimodal = any(
            seq.seq_id == seq_id and seq.multimodal_digest is not None
            for queue in (
                getattr(self.scheduler, "waiting", ()),
                getattr(self.scheduler, "decoding", ()),
            )
            for seq in queue
        )
        should_free = self.scheduler.abort(seq_id)
        if chain_seq is not None:
            self._active_chain_sequences.pop(int(seq_id), None)
            self.model_runner.call(
                "chain_invalidate",
                str(chain_seq.chain_id),
                int(chain_seq.seq_id),
            )
            return
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is not None:
            chain_id = coordinator.index.seq_id_to_chain_id.get(
                int(seq_id)
            )
            if chain_id is not None:
                self.model_runner.call(
                    "chain_invalidate",
                    str(chain_id),
                    int(seq_id),
                )
                return
        if should_free:
            self.model_runner.call("free_slots", seq_id)
        elif multimodal:
            self.model_runner.call("free_multimodal", seq_id)

    def chain_cache_routing_match(self, chain_id: str) -> dict[str, object]:
        return self.model_runner.runtime_state.chain_routing_match(
            str(chain_id)
        )

    def invalidate_chain(self, chain_id: str) -> None:
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is None:
            raise ChainModeError(
                "Chain prefix cache is not enabled.",
                chain_id=str(chain_id),
            )
        record = coordinator.index.lookup(str(chain_id))
        self._active_chain_sequences.pop(int(record.seq_id), None)
        self.model_runner.call(
            "chain_invalidate",
            str(chain_id),
            int(record.seq_id),
        )

    def discard_chain(
        self,
        chain_id: str,
        *,
        expected_seq_id: int,
    ) -> bool:
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is None:
            raise ChainModeError(
                "Chain prefix cache is not enabled.",
                chain_id=str(chain_id),
            )
        record = coordinator.index.records.get(str(chain_id))
        if record is None:
            return False
        if int(record.seq_id) != int(expected_seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, "
                f"expected_seq_id={int(expected_seq_id)}.",
                chain_id=str(chain_id),
            )
        seq_id = int(record.seq_id)
        self.scheduler.abort(seq_id)
        self._active_chain_sequences.pop(seq_id, None)
        self.model_runner.call(
            "chain_invalidate",
            str(chain_id),
            seq_id,
        )
        return True

    def chain_cache_routing_snapshot(self) -> ChainRoutingSnapshot:
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is None:
            return ChainRoutingSnapshot(enabled=False)
        return coordinator.index.routing_snapshot()

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        include_subtree: bool = False,
    ) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_inspect",
            [int(token_id) for token_id in token_ids],
            bool(include_subtree),
        )

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_match",
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_delete_subtree",
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        priority: int,
    ) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_set_eviction_priority",
            [int(token_id) for token_id in token_ids],
            int(priority),
        )

    def debug_sparse_state_summaries(self) -> list[dict[str, object]]:
        summaries = self.model_runner.call("debug_sparse_state_summaries")
        if not isinstance(summaries, list) or len(summaries) != self.config.world_size:
            raise RuntimeError(
                "Sparse-state summary did not return one record per world rank: "
                f"expected={self.config.world_size}, got={summaries!r}."
            )
        return summaries

    def operator_runtime_stats(self) -> list[dict[str, object]]:
        stats = self.model_runner.call("operator_runtime_stats")
        if not isinstance(stats, list) or len(stats) != self.config.world_size:
            raise RuntimeError(
                "Operator runtime stats did not return one record per world rank: "
                f"expected={self.config.world_size}, got={stats!r}."
            )
        return stats

    def debug_last_logits(self) -> torch.Tensor:
        logits = self.model_runner.call("debug_last_logits_cpu")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"Rank 0 did not return debug logits: {logits!r}.")
        return logits

    def debug_hidden_states(self) -> dict[int, torch.Tensor]:
        snapshots = self.model_runner.call("debug_hidden_states_cpu")
        if not isinstance(snapshots, dict) or not all(
            isinstance(layer_idx, int) and isinstance(tensor, torch.Tensor)
            for layer_idx, tensor in snapshots.items()
        ):
            raise RuntimeError(
                f"Rank 0 did not return hidden-state snapshots: {snapshots!r}."
            )
        return snapshots

    def debug_moe_states(self) -> dict[int, dict[str, torch.Tensor]]:
        snapshots = self.model_runner.call("debug_moe_states_cpu")
        if not isinstance(snapshots, dict):
            raise RuntimeError(f"Rank 0 did not return MoE snapshots: {snapshots!r}.")
        return snapshots

    def worker_info(
        self,
        served_model_name: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, object]:
        config = self.config
        benchmark_config_keys = (
            "gpu_memory_utilization",
            "num_kvcache_slots",
            "max_num_batched_tokens",
            "prefill_schedule_policy",
            "engine_prefill_chunk_size",
            "long_prefill_offload_threshold",
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "full_attention_layers",
            "obs_layer_ids",
            "snapkv_window_size",
            "snapkv_num_full_layers",
            "sparse_prefill_score_mode",
            "h2o_decode_budget",
            "h2o_prefill_budget",
            "h2o_recent_ratio",
            "h2o_prefill_score_window",
            "pool_kernel_size",
            "sparse_attn_score_dtype",
            "pyramid_layer_ratios",
            "pyramidkv_start_layer",
            "pyramidkv_start_ratio",
            "pyramidkv_least_layer",
            "pyramidkv_least_ratio",
            "quest_chunk_size",
            "quest_token_budget",
            "quest_skip_layers",
            "deltakv_checkpoint_path",
            "deltakv_center_ratio",
            "deltakv_latent_dim",
            "deltakv_latent_quant_bits",
            "deltakv_latent_quant_group_size",
            "decode_graph",
            "decode_graph_shape_policy",
            "decode_graph_capture_sampling",
            "decode_graph_capture_sizes",
            "decode_graph_context_sizes",
            "decode_graph_context_policy",
            "decode_graph_max_cached_graphs",
            "decode_graph_startup_capture",
            "decode_graph_startup_capture_limit",
            "enable_prefix_caching",
            "prefix_cache_mode",
            "resolved_prefix_cache_mode",
            "chain_cache_max_tombstones",
            "prefix_cache_block_size",
            "prefix_cache_requested_max_blocks",
            "prefix_cache_max_blocks",
            "enable_prefix_cache_offload",
            "prefix_cache_host_size_gb",
            "recurrent_state_max_bytes",
            "prefix_cache_max_recurrent_bytes",
            "recurrent_state_pool_bytes",
            "recurrent_state_bytes_per_row",
            "recurrent_state_row_capacity",
            "prefix_recurrent_bytes_per_block",
            "prefix_recurrent_capacity_bytes",
            "prefix_kv_bytes_per_block",
            "prefix_kv_block_capacity",
            "kv_allocatable_bytes",
        )

        def jsonable(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [jsonable(item) for item in value]
            if isinstance(value, dict):
                return {
                    str(key): jsonable(item)
                    for key, item in value.items()
                }
            raise TypeError(
                "Worker benchmark metadata is not JSON serializable: "
                f"type={type(value).__name__} value={value!r}."
            )

        return {
            "served_model_name": served_model_name or str(config.model),
            "model": str(config.model),
            "model_type": str(getattr(config.hf_config, "model_type", "")),
            "vocab_size": int(
                getattr(config.hf_config, "vocab_size", 0) or 0
            ),
            "sparse_method": str(getattr(config, "sparse_method", "") or ""),
            "world_size": int(getattr(config, "world_size", 1)),
            "tensor_parallel_size": int(getattr(config, "tensor_parallel_size", 1)),
            "expert_parallel_size": int(getattr(config, "expert_parallel_size", 1)),
            "data_parallel_size": int(getattr(config, "data_parallel_size", 1)),
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs_in_batch": int(getattr(config, "max_num_seqs_in_batch", 0) or 0),
            "max_decoding_seqs": int(getattr(config, "max_decoding_seqs", 0) or 0),
            "max_num_seqs_in_gpu": int(getattr(config, "max_num_seqs_in_gpu", 0) or 0),
            "prefix_cache_enabled": bool(getattr(config, "enable_prefix_caching", False)),
            "prefix_cache_mode": str(
                getattr(config, "resolved_prefix_cache_mode", "disabled")
            ),
            "prefix_cache_block_size": getattr(config, "prefix_cache_block_size", None),
            "code_revision": code_revision_info(),
            "benchmark_config": {
                key: jsonable(getattr(config, key))
                for key in benchmark_config_keys
                if hasattr(config, key)
            },
            "tags": sorted(str(tag) for tag in (tags or []) if str(tag)),
        }

    def worker_routing_load(self) -> dict[str, object]:
        scheduler = self.scheduler
        waiting = len(scheduler.waiting)
        decoding = len(scheduler.decoding)
        return {
            "waiting_requests": int(waiting),
            "decoding_requests": int(decoding),
            "active_requests": int(waiting + decoding),
            "total_preemptions": int(getattr(scheduler, "total_preemptions", 0)),
            "total_recompute_replays": int(
                getattr(scheduler, "total_recompute_replays", 0)
            ),
            "max_num_seqs_in_batch": int(getattr(scheduler, "max_num_seqs_in_batch", 0)),
            "max_decoding_seqs": int(getattr(scheduler, "max_decoding_seqs", 0)),
            "max_num_seqs_in_gpu": int(getattr(scheduler.config, "max_num_seqs_in_gpu", 0)),
        }

    def worker_load(self) -> dict[str, object]:
        result = self.worker_routing_load()
        cache_stats = self.model_runner.runtime_state.free_slot_stats()
        result["cache"] = {
            str(key): int(value)
            for key, value in cache_stats.items()
            if isinstance(value, int)
        }
        return result

    def prefix_cache_routing_snapshot(self) -> PrefixCacheRoutingSnapshot:
        runtime_state = self.model_runner.runtime_state
        owner = (
            runtime_state.prefix_cache_coordinator
            if runtime_state.prefix_cache_coordinator is not None
            else runtime_state.cache_manager
        )
        method = str(self.config.sparse_method or "")
        prefix_cache = getattr(owner, "prefix_cache", None)
        if prefix_cache is not None:
            return prefix_cache.routing_snapshot(method)

        supported = hasattr(owner, "prefix_cache")
        return PrefixCacheRoutingSnapshot(
            supported=supported,
            enabled=False,
            method=method,
            reason=(
                "prefix cache is not enabled for this runtime."
                if supported
                else "prefix cache is not supported by this cache manager."
            ),
        )

    def _release_preempted_sequences(self, preempted_seqs: list[Sequence]) -> None:
        preempted_seq_ids = [int(seq.seq_id) for seq in preempted_seqs]
        if not preempted_seq_ids:
            return
        # Preemption is transient: retain the logical request and chain
        # identity, release only runtime KV/recurrent state, and let
        # scheduler-driven recompute rebuild it later.
        self.model_runner.call("free_slots_batch", preempted_seq_ids)

    def step(self):
        """
        执行单个推理步进（一个 Batch）。
        包含：调度、抢占处理、模型前向计算、状态更新、资源回收。
        """
        with profiler.record("step"):
            self.last_step_token_outputs = []
            self.last_step_prompt_cache_hits = []
            self.last_step_logprob_outputs = []
            # 1. 调度：决定哪些序列进入本次 Batch
            with profiler.record("schedule"):
                seqs, is_prefill, preempted_seqs = self.scheduler.schedule()
            if is_prefill:
                self.last_step_prompt_cache_hits = [
                    (int(seq.seq_id), int(seq.prefix_cache_hit_len))
                    for seq in seqs
                ]
            prefill_batch_mode = (
                self.scheduler.prefill_execution_mode_for_batch(seqs)
                if seqs and is_prefill
                else None
            )
            
            # 2. 显式处理抢占 (Eviction)：
            # 如果有序列被调度器踢出，立即广播指令让所有 Rank 释放其占用的物理槽位
            with profiler.record("preempt_free"):
                self._release_preempted_sequences(preempted_seqs)
                
            if not seqs:
                # No progress can be made; avoid infinite busy-looping in callers.
                if preempted_seqs or self.is_finished():
                    prefill_seqs = len(self.scheduler.waiting)
                    decode_seqs = len(self.scheduler.decoding)
                    prefill_modes = self.scheduler.prefill_execution_mode_counts()
                    decode_threshold = self.scheduler._long_text_threshold(is_prefill=False)
                    decode_long = sum(
                        1 for s in self.scheduler.decoding if int(s.num_tokens) > int(decode_threshold)
                    )
                    self._throughput_logger.record_state(
                        prefill_seqs + decode_seqs,
                        prefill_seqs,
                        decode_seqs,
                        prefill_modes["chunked"],
                        prefill_modes["full"],
                        prefill_modes["raw_offload"],
                        decode_long,
                        decode_seqs - decode_long,
                        "idle",
                    )
                    return [], 0
                # Most commonly: a prompt is larger than KV cache capacity (for methods that keep all tokens),
                # or scheduling constraints prevent any chunk from being placed.
                raise RuntimeError(
                    "Scheduler returned no runnable sequences and no preemptions; "
                    "this would hang the generation loop. "
                    f"method={self.config.sparse_method} free_slots={self.model_runner.runtime_state.num_free_slots} "
                    f"waiting={len(self.scheduler.waiting)} decoding={len(self.scheduler.decoding)}"
                )
                
            # 3. 跨进程广播并执行推理：
            # Rank 0 会驱动所有 Rank 进程同步运行本地的 ModelRunner.run
            with profiler.record("model_run_call"):
                try:
                    token_ids, logprob_outputs = self.model_runner.call(
                        "run", seqs, is_prefill
                    )
                except Exception:
                    for seq in seqs:
                        chain_seq = self._active_chain_sequences.pop(
                            int(seq.seq_id), None
                        )
                        if chain_seq is None:
                            continue
                        try:
                            self.model_runner.call(
                                "chain_invalidate",
                                str(chain_seq.chain_id),
                                int(chain_seq.seq_id),
                            )
                            # Remove scheduler ownership after RuntimeState has
                            # reclaimed the resident payload. A later serving
                            # cleanup must not free the same sparse rows twice.
                            self.scheduler.abort(int(chain_seq.seq_id))
                        except Exception:
                            logger.exception(
                                "Failed to invalidate chain {} after model failure.",
                                chain_seq.chain_id,
                            )
                    raise
            token_logprobs, top_logprobs = (
                logprob_outputs if logprob_outputs is not None else (None, None)
            )

            token_outputs: list[tuple[int, list[int]]] = []
            logprob_step_outputs: list[
                tuple[int, list[float | None], list[dict[int, float] | None]]
            ] = []
            step_token_logprobs = token_logprobs or [None] * len(seqs)
            step_top_logprobs = top_logprobs or [None] * len(seqs)
            for seq, token_id, token_logprob, top_logprob in zip(
                seqs,
                token_ids,
                step_token_logprobs,
                step_top_logprobs,
            ):
                if (
                    seq.should_publish_sample
                    and (not is_prefill or seq.is_last_chunk_prefill)
                ):
                    token_outputs.append((seq.seq_id, [int(token_id)]))
                    logprob_step_outputs.append((seq.seq_id, [token_logprob], [top_logprob]))
            
            # 4. 逻辑后处理：更新序列的 Token 列表和状态机
            with profiler.record("postprocess"):
                self.scheduler.postprocess(
                    seqs,
                    token_ids,
                    is_prefill,
                    token_logprobs=token_logprobs,
                    top_logprobs=top_logprobs,
                )
            self.last_step_token_outputs = token_outputs
            self.last_step_logprob_outputs = logprob_step_outputs
            
            # 5. 完成序列的资源回收：
            # 遍历序列，如果已达到 EOS 或最大长度，则通知所有进程释放物理槽位
            with profiler.record("finished_free"):
                finished_outputs = []
                finished_seq_ids = []
                for seq in seqs:
                    if seq.is_finished:
                        chain_seq = self._active_chain_sequences.pop(
                            int(seq.seq_id), None
                        )
                        if chain_seq is None:
                            finished_seq_ids.append(int(seq.seq_id))
                        else:
                            processed_token_count = max(
                                0, int(chain_seq.num_tokens) - 1
                            )
                            self.model_runner.call(
                                "chain_finish",
                                str(chain_seq.chain_id),
                                int(chain_seq.seq_id),
                                stable_token_digest(
                                    chain_seq.token_ids,
                                    count=processed_token_count,
                                ),
                                processed_token_count,
                            )
                            coordinator = (
                                self.model_runner.runtime_state
                                .chain_cache_coordinator
                            )
                            if coordinator is None:
                                raise RuntimeError(
                                    "Finished a chain request without a chain "
                                    "cache coordinator."
                                )
                            coordinator.remember_processed_tokens(
                                chain_id=str(chain_seq.chain_id),
                                seq_id=int(chain_seq.seq_id),
                                token_ids=list(chain_seq.token_ids),
                                processed_token_count=processed_token_count,
                            )
                        finished_outputs.append(
                            (
                                seq.seq_id,
                                seq.completion_token_ids,
                                seq.completion_token_logprobs,
                                seq.completion_top_logprobs,
                            )
                        )
                if finished_seq_ids:
                    self.model_runner.call("finish_slots_batch", finished_seq_ids)
        
        # 计算吞吐量统计数据 (正数表示 Prefill，负数表示 Decode)
        num_tokens = sum(seq.current_chunk_size for seq in seqs) if is_prefill else -len(seqs)
        self._throughput_logger.record_step(num_tokens)
        prefill_seqs = len(self.scheduler.waiting)
        decode_seqs = len(self.scheduler.decoding)
        prefill_modes = self.scheduler.prefill_execution_mode_counts()
        decode_threshold = self.scheduler._long_text_threshold(is_prefill=False)
        decode_long = sum(1 for s in self.scheduler.decoding if int(s.num_tokens) > int(decode_threshold))
        if is_prefill:
            if prefill_batch_mode is None:
                raise RuntimeError("Missing execution mode for a scheduled prefill batch.")
            last_batch = f"pf-{prefill_batch_mode}"
        else:
            batch_is_long = bool(int(seqs[0].num_tokens) > int(decode_threshold))
            last_batch = f"dc-{'L' if batch_is_long else 'S'}"
        self._throughput_logger.record_state(
            prefill_seqs + decode_seqs,
            prefill_seqs,
            decode_seqs,
            prefill_modes["chunked"],
            prefill_modes["full"],
            prefill_modes["raw_offload"],
            decode_long,
            decode_seqs - decode_long,
            last_batch,
        )
        return finished_outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已处理完毕"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[MultiModalPrompt] | list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        高层 API：批量输入 Prompt，阻塞直至全部生成完成。
        返回包含生成的 text 和 token_ids 的字典列表。
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 提交所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 主推理循环
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # 更新吞吐量统计
            if use_tqdm:
                dt = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / dt
                else:
                    decode_throughput = -num_tokens / dt
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集已完成的输出
            for seq_id, token_ids, _token_logprobs, _top_logprobs in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 按照请求提交顺序排序并解码
        results = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = [{"text": self.tokenizer.decode(tids, skip_special_tokens=True), "token_ids": tids} for tids in results]
        
        if use_tqdm:
            pbar.close()
        return results
