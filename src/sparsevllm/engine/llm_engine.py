import atexit
import gc
import os
from dataclasses import fields
from time import perf_counter
import threading
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig, Qwen2Tokenizer
import torch
import torch.multiprocessing as mp
from sparsevllm.utils.log import logger
import sys

from deltakv.configs.runtime_params import normalize_runtime_params

from sparsevllm.config import Config
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.model_runner import ModelRunner, make_tp_shm_name
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.utils.profiler import profiler

def _deltakv_graph_warmup_profile(config: Config) -> str:
    graph_warmup = bool(getattr(config, "decode_cuda_graph", False))
    method = normalize_sparse_method(getattr(config, "vllm_sparse_method", "") or "")
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
        self._prefill_long_seqs = 0
        self._prefill_short_seqs = 0
        self._decode_long_seqs = 0
        self._decode_short_seqs = 0
        self._last_batch = "idle"  # "pf-L", "pf-S", "dc-L", "dc-S", "idle"
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
        prefill_long_seqs: int,
        prefill_short_seqs: int,
        decode_long_seqs: int,
        decode_short_seqs: int,
        last_batch: str,
    ):
        with self._lock:
            self._running_seqs = int(running_seqs)
            self._prefill_seqs = int(prefill_seqs)
            self._decode_seqs = int(decode_seqs)
            self._prefill_long_seqs = int(prefill_long_seqs)
            self._prefill_short_seqs = int(prefill_short_seqs)
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
                prefill_long_seqs = self._prefill_long_seqs
                prefill_short_seqs = self._prefill_short_seqs
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
                "| prf(L/S)={prefill_long_seqs}/{prefill_short_seqs} dc(L/S)={decode_long_seqs}/{decode_short_seqs} "
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
                prefill_long_seqs=prefill_long_seqs,
                prefill_short_seqs=prefill_short_seqs,
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
        normalized_params = normalize_runtime_params(kwargs, backend="sparsevllm")
        for warning in normalized_params.warnings:
            logger.info(f"Runtime parameter normalization: {warning}")

        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {
            k: v for k, v in normalized_params.infer_config.items() if k in config_fields
        }
        ignored_keys = sorted(set(normalized_params.infer_config) - config_fields)
        if ignored_keys:
            if normalized_params.infer_config.get("allow_unknown_config_keys", False):
                logger.warning(f"Ignoring unknown Sparse-vLLM config keys: {ignored_keys}")
            else:
                raise ValueError(
                    f"Unknown Sparse-vLLM config keys: {ignored_keys}. "
                    "Refusing to ignore possible experiment parameter typos. "
                    "Set allow_unknown_config_keys=True only for explicitly validated compatibility runs."
                )
        config = Config(model, **config_kwargs)
        self.config = config
        
        # 初始化 Profiler
        profiler.set_enabled(config.enable_profiler)
        
        # 2. 启动 world worker 进程；TP/EP/DP 语义由 ParallelContext 管理。
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        tp_shm_name = make_tp_shm_name() if config.world_size > 1 else None
        for i in range(1, config.world_size):
            event = (ctx.Event(), ctx.Event())
            # 为每一个非零 Rank 启动一个独立的 ModelRunner 进程
            process = ctx.Process(target=ModelRunner, args=(config, i, event, tp_shm_name))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 3. 初始化主进程的 ModelRunner (Rank 0)
        # 注意：必须先初始化 ModelRunner 以便在本地 GPU 分配 KV Cache 账本
        self.model_runner = ModelRunner(config, 0, self.events, tp_shm_name)
        
        # 加载分词器
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
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
        self.last_step_logprob_outputs: list[
            tuple[int, list[float | None], list[dict[int, float] | None]]
        ] = []
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
        
        # 预热只需触发算子编译，使用固定短长度即可
        warmup_len = self.config.num_sink_tokens + self.config.decode_keep_tokens\
                     + self.config.num_recent_tokens + self.config.chunk_prefill_size + 1024
        warmup_profile = _deltakv_graph_warmup_profile(self.config)
        graph_sized_batch = warmup_profile in ("graph", "big_prefill_only")
        decode_warmup = warmup_profile in ("graph", "decode_1seq")
        num_seqs = int(self.config.max_decoding_seqs) if graph_sized_batch else 1
        
        # 预热 1 个 Token 的生成（包含 Prefill 和 Decode）
        sampling_params = SamplingParams(
            max_tokens=2 if decode_warmup else 1,
            temperature=0.0,
            ignore_eos=decode_warmup,
        )
        max_prompt_len = max(1, int(self.config.max_model_len) - int(sampling_params.max_tokens))
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
        if warmup_len > max_prompt_len:
            logger.warning(
                f"Warmup prompt length ({warmup_len}) exceeds max_model_len - max_tokens "
                f"({max_prompt_len}). Clamping warmup_len to {max_prompt_len}."
            )
            warmup_len = max_prompt_len
        dummy_prompt = [0] * warmup_len
        logger.info(
            f"Warmup profile: {warmup_profile} "
            f"(num_seqs={num_seqs}, max_tokens={sampling_params.max_tokens}, "
            f"ignore_eos={sampling_params.ignore_eos})."
        )
        
        for _ in range(num_seqs):
            self.add_request(dummy_prompt, sampling_params)

        while not self.is_finished():
            self.step()
        self._after_warmup_debug_cleanup()
        logger.info("Warmup finished.")

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

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将一个新的推理请求加入系统"""
        if isinstance(prompt, str):
            # Match HF manual_generate: add BOS for raw prompts, but do not
            # duplicate it when a chat template already starts with BOS.
            add_special_tokens = True
            if self.tokenizer.bos_token is None or prompt.startswith(self.tokenizer.bos_token):
                add_special_tokens = False
            prompt = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
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
        self.scheduler.add(seq)
        return seq.seq_id

    def _refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self.model_runner.call("refresh_prefix_cache_hit", seq)

    def abort_request(self, seq_id: int):
        """Abort a queued or running request and release any owned KV slots."""
        should_free = self.scheduler.abort(seq_id)
        if should_free:
            self.model_runner.call("free_slots", seq_id)

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
        return {
            "served_model_name": served_model_name or str(config.model),
            "model": str(config.model),
            "model_type": str(getattr(config.hf_config, "model_type", "")),
            "sparse_method": str(getattr(config, "vllm_sparse_method", "") or ""),
            "world_size": int(getattr(config, "world_size", 1)),
            "tensor_parallel_size": int(getattr(config, "tensor_parallel_size", 1)),
            "expert_parallel_size": int(getattr(config, "expert_parallel_size", 1)),
            "data_parallel_size": int(getattr(config, "data_parallel_size", 1)),
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs_in_batch": int(getattr(config, "max_num_seqs_in_batch", 0) or 0),
            "max_decoding_seqs": int(getattr(config, "max_decoding_seqs", 0) or 0),
            "prefix_cache_enabled": bool(getattr(config, "enable_prefix_caching", False)),
            "prefix_cache_block_size": getattr(config, "prefix_cache_block_size", None),
            "tags": sorted(str(tag) for tag in (tags or []) if str(tag)),
        }

    def worker_load(self) -> dict[str, object]:
        scheduler = self.scheduler
        waiting = len(scheduler.waiting)
        decoding = len(scheduler.decoding)
        cache_stats = self.model_runner.runtime_state.free_slot_stats()
        return {
            "waiting_requests": int(waiting),
            "decoding_requests": int(decoding),
            "active_requests": int(waiting + decoding),
            "total_preemptions": int(getattr(scheduler, "total_preemptions", 0)),
            "max_num_seqs_in_batch": int(getattr(scheduler, "max_num_seqs_in_batch", 0)),
            "max_decoding_seqs": int(getattr(scheduler, "max_decoding_seqs", 0)),
            "cache": {str(key): int(value) for key, value in cache_stats.items() if isinstance(value, int)},
        }

    def step(self):
        """
        执行单个推理步进（一个 Batch）。
        包含：调度、抢占处理、模型前向计算、状态更新、资源回收。
        """
        with profiler.record("step"):
            self.last_step_token_outputs = []
            self.last_step_logprob_outputs = []
            # 1. 调度：决定哪些序列进入本次 Batch
            with profiler.record("schedule"):
                seqs, is_prefill, preempted_seqs = self.scheduler.schedule()
            
            # 2. 显式处理抢占 (Eviction)：
            # 如果有序列被调度器踢出，立即广播指令让所有 Rank 释放其占用的物理槽位
            with profiler.record("preempt_free"):
                preempted_seq_ids = [int(seq.seq_id) for seq in preempted_seqs]
                if preempted_seq_ids:
                    self.model_runner.call("free_slots_batch", preempted_seq_ids)
                
            if not seqs:
                # No progress can be made; avoid infinite busy-looping in callers.
                if preempted_seqs or self.is_finished():
                    prefill_seqs = len(self.scheduler.waiting)
                    decode_seqs = len(self.scheduler.decoding)
                    prefill_threshold = self.scheduler._long_text_threshold(is_prefill=True)
                    decode_threshold = self.scheduler._long_text_threshold(is_prefill=False)
                    prefill_long = sum(
                        1 for s in self.scheduler.waiting if int(s.num_prompt_tokens) > int(prefill_threshold)
                    )
                    decode_long = sum(
                        1 for s in self.scheduler.decoding if int(s.num_tokens) > int(decode_threshold)
                    )
                    self._throughput_logger.record_state(
                        prefill_seqs + decode_seqs,
                        prefill_seqs,
                        decode_seqs,
                        prefill_long,
                        prefill_seqs - prefill_long,
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
                    f"method={self.config.vllm_sparse_method} free_slots={self.model_runner.runtime_state.num_free_slots} "
                    f"waiting={len(self.scheduler.waiting)} decoding={len(self.scheduler.decoding)}"
                )
                
            # 3. 跨进程广播并执行推理：
            # Rank 0 会驱动所有 Rank 进程同步运行本地的 ModelRunner.run
            with profiler.record("model_run_call"):
                token_ids, logprob_outputs = self.model_runner.call("run", seqs, is_prefill)
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
                if not is_prefill or seq.is_last_chunk_prefill:
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
                        finished_seq_ids.append(int(seq.seq_id))
                        finished_outputs.append(
                            (
                                seq.seq_id,
                                seq.completion_token_ids,
                                seq.completion_token_logprobs,
                                seq.completion_top_logprobs,
                            )
                        )
                if finished_seq_ids:
                    self.model_runner.call("free_slots_batch", finished_seq_ids)
        
        # 计算吞吐量统计数据 (正数表示 Prefill，负数表示 Decode)
        num_tokens = sum(seq.current_chunk_size for seq in seqs) if is_prefill else -len(seqs)
        self._throughput_logger.record_step(num_tokens)
        prefill_seqs = len(self.scheduler.waiting)
        decode_seqs = len(self.scheduler.decoding)
        prefill_threshold = self.scheduler._long_text_threshold(is_prefill=True)
        decode_threshold = self.scheduler._long_text_threshold(is_prefill=False)
        prefill_long = sum(1 for s in self.scheduler.waiting if int(s.num_prompt_tokens) > int(prefill_threshold))
        decode_long = sum(1 for s in self.scheduler.decoding if int(s.num_tokens) > int(decode_threshold))
        if is_prefill:
            batch_is_long = bool(int(seqs[0].num_prompt_tokens) > int(prefill_threshold))
            stage = "pf"
        else:
            batch_is_long = bool(int(seqs[0].num_tokens) > int(decode_threshold))
            stage = "dc"
        last_batch = f"{stage}-{'L' if batch_is_long else 'S'}"
        self._throughput_logger.record_state(
            prefill_seqs + decode_seqs,
            prefill_seqs,
            decode_seqs,
            prefill_long,
            prefill_seqs - prefill_long,
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
        prompts: list[str] | list[list[int]],
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
