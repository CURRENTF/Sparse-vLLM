import atexit
import os
import socket
from dataclasses import fields
from time import perf_counter
import threading
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Qwen2Tokenizer
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


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _deltakv_graph_warmup_profile(config: Config) -> str:
    graph_warmup = bool(getattr(config, "decode_cuda_graph", False))
    method = normalize_sparse_method(getattr(config, "vllm_sparse_method", "") or "")
    if not graph_warmup:
        return "prefill_only"
    if method == "deltakv":
        warmup_policy = os.getenv("SPARSEVLLM_DELTAKV_GRAPH_WARMUP", "prefill_only").strip().lower()
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

        config_fields = {field.name for field in fields(Config)}
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
        
        # 2. 启动并行 ModelRunner 环境。EP v1 的多进程 world 不能再等同于 TP world。
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        parallel_world_size = int(config.parallel_world_size)
        if parallel_world_size > 1 and config.distributed_master_port is None:
            env_master_port = os.getenv("SPARSEVLLM_MASTER_PORT")
            config.distributed_master_port = int(env_master_port) if env_master_port else _find_free_tcp_port()
        tp_shm_name = make_tp_shm_name() if parallel_world_size > 1 else None
        for i in range(1, parallel_world_size):
            event = ctx.Event()
            # 为每一个非零 parallel rank 启动一个独立的 ModelRunner 进程
            process = ctx.Process(target=ModelRunner, args=(config, i, event, tp_shm_name))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 3. 初始化主进程的 ModelRunner (Rank 0)
        # 注意：必须先初始化 ModelRunner 以便在本地 GPU 分配 KV Cache 账本
        self.model_runner = ModelRunner(config, 0, self.events, tp_shm_name)
        
        # 加载分词器
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.model_runner.call(
            "set_tokenizer_metadata",
            self._build_delimiter_token_ids(self.tokenizer),
            self._build_non_execution_token_ids(self.tokenizer),
        )
        
        # 4. 初始化调度器
        # 关键设计：将 Rank 0 的 CacheManager 传给 Scheduler。
        # Scheduler 通过它来感知全局显存的余量，从而做出调度和抢占决策。
        self.scheduler = Scheduler(config, self.model_runner.cache_manager)
        
        self._exited = False
        self._throughput_logger = _ThroughputIntervalLogger(config.throughput_log_interval_s)
        self.last_step_token_outputs: list[tuple[int, list[int]]] = []
        self.last_step_logprob_outputs: list[
            tuple[int, list[float | None], list[dict[int, float] | None]]
        ] = []
        self._dp_seq_owner: dict[int, int] = {}
        # 注册退出钩子，确保程序崩溃或结束时能正确释放多进程资源
        atexit.register(self.exit)

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
        model_runner = getattr(self, "model_runner", None)
        cache_manager = getattr(model_runner, "cache_manager", None)
        reset_after_warmup = getattr(cache_manager, "reset_after_warmup", None)
        if callable(reset_after_warmup):
            reset_after_warmup()
        else:
            reset_prefix_cache = getattr(cache_manager, "reset_prefix_cache", None)
            if callable(reset_prefix_cache):
                reset_prefix_cache()

        runner = getattr(model_runner, "decode_cuda_graph_runner", None)
        if runner is not None and os.getenv("SPARSEVLLM_DELTAKV_CLEAR_GRAPHS_AFTER_WARMUP", "0") == "1":
            runner.clear_captured_graphs()
            logger.info("Cleared decode CUDA graphs after warmup.")

        sparse_controller = getattr(model_runner, "sparse_controller", None)
        if (
            sparse_controller is not None
            and os.getenv("SPARSEVLLM_DELTAKV_CLEAR_ATTN_SCORE_BUFFERS_AFTER_WARMUP", "0") == "1"
        ):
            sparse_controller.clear_decode_attn_score_buffers()
            logger.info("Cleared decode attention score buffers after warmup.")

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
        if self._exited:
            return
        self._exited = True

        profiler.print_stats()
        if hasattr(self, "_throughput_logger"):
            self._throughput_logger.stop()
        if hasattr(self, "model_runner"):
            model_runner = self.model_runner
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
                logger.warning(
                    "Timed out waiting {:.1f}s for ModelRunner exit RPC; terminating workers.",
                    timeout_s,
                )
                self._cleanup_model_runner_shared_memory(model_runner)
            elif errors:
                logger.warning("ModelRunner exit RPC failed during shutdown: {}", repr(errors[0]))
                self._cleanup_model_runner_shared_memory(model_runner)
            del self.model_runner
        if hasattr(self, "ps"):
            join_timeout_s = float(os.getenv("SPARSEVLLM_WORKER_JOIN_TIMEOUT_S", "5"))
            for p in self.ps:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=max(0.0, join_timeout_s))
                if p.is_alive():
                    logger.warning(
                        "Worker process pid={} did not stop after terminate; killing.",
                        p.pid,
                    )
                    p.kill()
                    p.join(timeout=max(0.0, join_timeout_s))

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

    def abort_request(self, seq_id: int):
        """Abort a queued or running request and release any owned KV slots."""
        seq_id = int(seq_id)
        should_free = self.scheduler.abort(seq_id)
        if should_free:
            self._free_seq_slots([seq_id])
        else:
            self._dp_seq_owner.pop(seq_id, None)

    def _use_overlapped_dp_ep(self) -> bool:
        return bool(getattr(self.config, "expert_parallel_overlap_data_parallel", False))

    def _owner_rank_for_seq(self, seq: Sequence) -> int:
        seq_id = int(seq.seq_id)
        owner = self._dp_seq_owner.get(seq_id)
        if owner is not None:
            return int(owner)
        dp_size = int(getattr(self.config, "data_parallel_size", 1) or 1)
        if dp_size <= 1:
            owner = 0
        else:
            owner_counts = [0 for _ in range(dp_size)]
            live_seq_ids = {int(s.seq_id) for s in self.scheduler.waiting}
            live_seq_ids.update(int(s.seq_id) for s in self.scheduler.decoding)
            live_seq_ids.add(seq_id)
            for live_seq_id, live_owner in self._dp_seq_owner.items():
                if int(live_seq_id) in live_seq_ids and 0 <= int(live_owner) < dp_size:
                    owner_counts[int(live_owner)] += 1
            owner = min(range(dp_size), key=lambda rank: (owner_counts[rank], rank))
        self._dp_seq_owner[seq_id] = int(owner)
        return int(owner)

    def _build_rank_batches(self, seqs: list[Sequence]) -> list[list[Sequence]]:
        world_size = int(self.config.parallel_world_size)
        rank_batches: list[list[Sequence]] = [[] for _ in range(world_size)]
        for seq in seqs:
            owner = self._owner_rank_for_seq(seq)
            if owner < 0 or owner >= world_size:
                raise RuntimeError(
                    f"Invalid overlapped DP owner rank {owner} for seq_id={seq.seq_id}; "
                    f"world_size={world_size}."
                )
            rank_batches[owner].append(seq)
        return rank_batches

    def _free_seq_slots(self, seq_ids: list[int]) -> None:
        seq_ids = [int(seq_id) for seq_id in seq_ids]
        if not seq_ids:
            return
        if not self._use_overlapped_dp_ep():
            self.model_runner.call("free_slots_batch", seq_ids)
            for seq_id in seq_ids:
                self._dp_seq_owner.pop(seq_id, None)
            return

        world_size = int(self.config.parallel_world_size)
        rank_seq_ids: list[list[int]] = [[] for _ in range(world_size)]
        for seq_id in seq_ids:
            owner = self._dp_seq_owner.get(seq_id)
            if owner is None:
                raise RuntimeError(f"Missing overlapped DP owner rank for seq_id={seq_id} during free.")
            owner = int(owner)
            if owner < 0 or owner >= world_size:
                raise RuntimeError(
                    f"Invalid overlapped DP owner rank {owner} for seq_id={seq_id}; "
                    f"world_size={world_size}."
                )
            rank_seq_ids[owner].append(seq_id)
        self.model_runner.call("free_rank_slots_batch", rank_seq_ids)
        for seq_id in seq_ids:
            self._dp_seq_owner.pop(seq_id, None)

    def _run_overlapped_dp_step(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> tuple[list[int], tuple[list[float | None], list[dict[int, float] | None]]]:
        rank_batches = self._build_rank_batches(seqs)
        gathered = self.model_runner.call("run_rank_batches", rank_batches, is_prefill)
        if not isinstance(gathered, list) or len(gathered) != int(self.config.parallel_world_size):
            raise RuntimeError(
                "Overlapped DP+EP run did not gather one payload per rank: "
                f"got {type(gathered).__name__} len={len(gathered) if isinstance(gathered, list) else 'n/a'}."
            )

        token_by_seq: dict[int, int] = {}
        token_logprob_by_seq: dict[int, float | None] = {}
        top_logprob_by_seq: dict[int, dict[int, float] | None] = {}
        for payload in gathered:
            if payload is None:
                raise RuntimeError("Overlapped DP+EP gathered an empty rank payload.")
            seq_ids = [int(seq_id) for seq_id in payload["seq_ids"]]
            token_ids = [int(token_id) for token_id in payload["token_ids"]]
            token_logprobs = list(payload["token_logprobs"])
            top_logprobs = list(payload["top_logprobs"])
            if not (
                len(seq_ids)
                == len(token_ids)
                == len(token_logprobs)
                == len(top_logprobs)
            ):
                raise RuntimeError(
                    "Overlapped DP+EP rank payload length mismatch: "
                    f"rank={payload.get('rank')} seq_ids={len(seq_ids)} "
                    f"token_ids={len(token_ids)} token_logprobs={len(token_logprobs)} "
                    f"top_logprobs={len(top_logprobs)}."
                )
            for seq_id, token_id, token_logprob, top_logprob in zip(
                seq_ids,
                token_ids,
                token_logprobs,
                top_logprobs,
            ):
                if seq_id in token_by_seq:
                    raise RuntimeError(f"Duplicate overlapped DP token output for seq_id={seq_id}.")
                token_by_seq[seq_id] = token_id
                token_logprob_by_seq[seq_id] = token_logprob
                top_logprob_by_seq[seq_id] = top_logprob

        missing = [int(seq.seq_id) for seq in seqs if int(seq.seq_id) not in token_by_seq]
        if missing:
            raise RuntimeError(f"Missing overlapped DP token outputs for seq_ids={missing[:16]}.")
        token_ids = [int(token_by_seq[int(seq.seq_id)]) for seq in seqs]
        token_logprobs = [token_logprob_by_seq[int(seq.seq_id)] for seq in seqs]
        top_logprobs = [top_logprob_by_seq[int(seq.seq_id)] for seq in seqs]
        return token_ids, (token_logprobs, top_logprobs)

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
                    self._free_seq_slots(preempted_seq_ids)
                
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
                    f"method={self.config.vllm_sparse_method} free_slots={self.model_runner.cache_manager.num_free_slots} "
                    f"waiting={len(self.scheduler.waiting)} decoding={len(self.scheduler.decoding)}"
                )
                
            # 3. 跨进程执行推理。普通 TP/EP correctness 路径广播同一 batch；
            # overlapped DP+EP 路径按 seq owner rank 分发本地 batch 后收集采样结果。
            with profiler.record("model_run_call"):
                if self._use_overlapped_dp_ep():
                    token_ids, logprob_outputs = self._run_overlapped_dp_step(seqs, is_prefill)
                else:
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
                    self._free_seq_slots(finished_seq_ids)
        
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
