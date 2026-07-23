import asyncio
import os
import queue
import threading
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer
from sparsevllm.entrypoints.openai.sampling import _find_stop_index
from sparsevllm.entrypoints.openai.sampling import _safe_stream_text_len
from sparsevllm.llm import LLM
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.utils.log import logger


@dataclass
class RequestHandle:
    output_queue: asyncio.Queue
    cancelled: threading.Event
    seq_id: int | None = None


@dataclass
class _QueuedRequest:
    prompt: str | list[int]
    sampling_params: SamplingParams
    index: int
    stop: list[str]
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue
    cancelled: threading.Event
    handle: RequestHandle


@dataclass
class _ActiveRequest:
    index: int
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue
    prompt_token_ids: list[int]
    max_tokens: int
    stop: list[str]
    completion_token_ids: list[int]
    completion_token_logprobs: list[float | None]
    completion_top_logprobs: list[dict[int, float] | None]
    detokenizer: IncrementalDetokenizer
    emitted_text_len: int = 0
    pending_token_ids: list[int] = field(default_factory=list)
    pending_token_logprobs: list[float | None] = field(default_factory=list)
    pending_top_logprobs: list[dict[int, float] | None] = field(default_factory=list)


@dataclass
class _ControlRequest:
    operation: str
    kwargs: dict[str, Any]
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue


_WAKEUP = object()


class AsyncEngineDispatcher:
    def __init__(self, engine: LLM):
        self.engine = engine
        self._pending: queue.Queue[_QueuedRequest | object | None] = queue.Queue()
        self._aborts: queue.Queue[int] = queue.Queue()
        self._controls: queue.Queue[_ControlRequest] = queue.Queue()
        self._closing = threading.Event()
        self._failed_message: str | None = None
        self._fatal_callback: Callable[[str], None] | None = None
        self._state_lock = threading.Lock()
        self._routing_snapshot_lock = threading.Lock()
        self._worker_load_snapshot: dict[str, Any] | None = None
        self._prefix_cache_routing_snapshot: Any | None = None
        self._refresh_routing_snapshots()
        self._thread = threading.Thread(target=self._run, name="sparsevllm-openai-dispatcher", daemon=True)
        self._thread.start()

    @property
    def failure_message(self) -> str | None:
        with self._state_lock:
            return self._failed_message

    @property
    def is_ready(self) -> bool:
        with self._state_lock:
            return self._terminal_message_locked() is None

    def set_fatal_callback(self, callback: Callable[[str], None]) -> None:
        with self._state_lock:
            self._fatal_callback = callback
            failure_message = self._failed_message
        if failure_message is not None:
            callback(failure_message)

    def _mark_failed(self, message: str) -> Callable[[str], None] | None:
        with self._state_lock:
            self._failed_message = message
            return self._fatal_callback

    def _terminal_message_locked(self) -> str | None:
        if self._failed_message is not None:
            return self._failed_message
        if self._closing.is_set():
            return "Sparse-vLLM server is shutting down."
        if not self._thread.is_alive():
            return "Sparse-vLLM dispatcher thread is not running."
        return None

    async def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        index: int,
        stop: list[str] | None = None,
    ) -> RequestHandle:
        output_queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        handle = RequestHandle(output_queue=output_queue, cancelled=cancelled)
        queued = _QueuedRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            index=index,
            stop=list(stop or []),
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
            cancelled=cancelled,
            handle=handle,
        )
        with self._state_lock:
            terminal_message = self._terminal_message_locked()
            if terminal_message is None:
                self._pending.put(queued)
        if terminal_message is not None:
            output_queue.put_nowait(
                {"type": "error", "message": terminal_message}
            )
        return handle

    async def control(self, operation: str, **kwargs: Any) -> Any:
        output_queue: asyncio.Queue = asyncio.Queue()
        queued = _ControlRequest(
            operation=operation,
            kwargs=dict(kwargs),
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
        )
        with self._state_lock:
            terminal_message = self._terminal_message_locked()
            if terminal_message is None:
                self._controls.put(queued)
                self._pending.put(_WAKEUP)
        if terminal_message is not None:
            raise RuntimeError(terminal_message)
        result = await output_queue.get()
        if result["type"] == "error":
            raise RuntimeError(result["message"])
        return result["value"]

    def worker_load_snapshot(self) -> dict[str, Any]:
        self._raise_if_unavailable()
        with self._routing_snapshot_lock:
            snapshot = self._worker_load_snapshot
        if snapshot is None:
            raise RuntimeError(
                "Engine does not expose worker_load for routing snapshots."
            )
        return {**snapshot, "snapshot": True}

    def prefix_cache_routing_match(
        self,
        token_ids: list[int],
    ) -> dict[str, Any]:
        self._raise_if_unavailable()
        with self._routing_snapshot_lock:
            snapshot = self._prefix_cache_routing_snapshot
        if snapshot is None:
            raise RuntimeError(
                "Engine does not expose a prefix-cache routing snapshot."
            )
        result = snapshot.match([int(token_id) for token_id in token_ids])
        if not isinstance(result, dict):
            raise RuntimeError(
                "Prefix-cache routing snapshot returned a non-object result: "
                f"{type(result).__name__}."
            )
        return result

    def _raise_if_unavailable(self) -> None:
        with self._state_lock:
            terminal_message = self._terminal_message_locked()
        if terminal_message is not None:
            raise RuntimeError(terminal_message)

    def _refresh_routing_snapshots(self) -> None:
        worker_load_fn = getattr(self.engine, "worker_load", None)
        prefix_snapshot_fn = getattr(
            self.engine,
            "prefix_cache_routing_snapshot",
            None,
        )
        worker_load = (
            worker_load_fn()
            if callable(worker_load_fn)
            else None
        )
        prefix_snapshot = (
            prefix_snapshot_fn()
            if callable(prefix_snapshot_fn)
            else None
        )
        if worker_load is not None and not isinstance(worker_load, dict):
            raise RuntimeError(
                "worker_load must return an object for routing snapshots, "
                f"got {type(worker_load).__name__}."
            )
        with self._routing_snapshot_lock:
            if worker_load is not None:
                self._worker_load_snapshot = dict(worker_load)
            if prefix_snapshot is not None:
                self._prefix_cache_routing_snapshot = prefix_snapshot

    def cancel(self, handle: RequestHandle):
        handle.cancelled.set()
        if handle.seq_id is not None:
            self._aborts.put(handle.seq_id)

    def close(self):
        with self._state_lock:
            if not self._closing.is_set():
                self._closing.set()
                self._pending.put(None)
        timeout_s = float(os.getenv("SPARSEVLLM_OPENAI_SHUTDOWN_TIMEOUT_S", "5"))
        self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            logger.warning(
                "OpenAI dispatcher did not stop within {:.1f}s; forcing engine shutdown.",
                timeout_s,
            )
        self.engine.exit()

    def _put(self, request: _ActiveRequest | _QueuedRequest, item: dict[str, Any]):
        request.loop.call_soon_threadsafe(request.output_queue.put_nowait, item)

    def _put_control(self, request: _ControlRequest, item: dict[str, Any]):
        request.loop.call_soon_threadsafe(request.output_queue.put_nowait, item)

    def _run(self):
        active: dict[int, _ActiveRequest] = {}
        stopping = False
        fatal_callback: Callable[[str], None] | None = None
        try:
            while not stopping:
                controls_changed = self._drain_controls()
                self._drain_aborts(active)
                if controls_changed:
                    self._refresh_routing_snapshots()
                if not active:
                    item = self._pending.get()
                    if item is None:
                        break
                    if item is _WAKEUP:
                        continue
                    self._admit(item, active)

                while True:
                    try:
                        item = self._pending.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        stopping = True
                        continue
                    if item is _WAKEUP:
                        continue
                    self._admit(item, active)

                self._drain_controls()
                self._drain_aborts(active)
                if not active:
                    self._refresh_routing_snapshots()
                    continue

                self._refresh_routing_snapshots()
                finished_outputs, _num_tokens = self.engine.step()
                self._publish_token_deltas(active)
                self._publish_finished(active, finished_outputs)
                self._refresh_routing_snapshots()
        except Exception as exc:
            failed_message = f"{type(exc).__name__}: {exc}"
            fatal_callback = self._mark_failed(failed_message)
            logger.exception("OpenAI dispatcher stopped after a fatal error: {}", failed_message)
        finally:
            terminal_message = self.failure_message or "Sparse-vLLM server is shutting down."
            try:
                self._fail_active_requests(active, terminal_message)
                self._drain_queued_requests(terminal_message)
            finally:
                if fatal_callback is not None:
                    try:
                        fatal_callback(terminal_message)
                    except Exception:
                        logger.exception("OpenAI dispatcher fatal callback failed")

    def _drain_controls(self) -> bool:
        changed = False
        while True:
            try:
                item = self._controls.get_nowait()
            except queue.Empty:
                return changed
            if self._closing.is_set():
                self._put_control(item, {"type": "error", "message": "Sparse-vLLM server is shutting down."})
                continue
            if self._failed_message is not None:
                self._put_control(item, {"type": "error", "message": self._failed_message})
                continue
            try:
                method = getattr(self.engine, item.operation)
                value = method(**item.kwargs)
                changed = True
            except Exception as exc:
                self._put_control(item, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                continue
            self._put_control(item, {"type": "result", "value": value})

    def _admit(self, item: _QueuedRequest, active: dict[int, _ActiveRequest]):
        if item.cancelled.is_set():
            return
        if self._closing.is_set():
            self._put(item, {"type": "error", "message": "Sparse-vLLM server is shutting down."})
            return
        if self._failed_message is not None:
            self._put(item, {"type": "error", "message": self._failed_message})
            return
        try:
            detokenizer = IncrementalDetokenizer(self.engine.tokenizer)
            seq_id = self.engine.add_request(item.prompt, item.sampling_params)
            item.handle.seq_id = seq_id
            if item.cancelled.is_set():
                self.engine.abort_request(seq_id)
                return
            prompt_token_ids = (
                list(item.prompt)
                if isinstance(item.prompt, list)
                else self.engine.tokenizer.encode(item.prompt)
            )
            active[seq_id] = _ActiveRequest(
                index=item.index,
                loop=item.loop,
                output_queue=item.output_queue,
                prompt_token_ids=prompt_token_ids,
                max_tokens=item.sampling_params.max_tokens,
                stop=item.stop,
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=detokenizer,
            )
        except Exception as exc:
            self._put(item, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _publish_token_deltas(self, active: dict[int, _ActiveRequest]):
        logprob_outputs = {
            seq_id: (token_logprobs, top_logprobs)
            for seq_id, token_logprobs, top_logprobs in getattr(
                self.engine,
                "last_step_logprob_outputs",
                [],
            )
        }
        for seq_id, token_ids in self.engine.last_step_token_outputs:
            request = active.get(seq_id)
            if request is None:
                continue
            token_logprobs, top_logprobs = logprob_outputs.get(
                seq_id,
                ([None] * len(token_ids), [None] * len(token_ids)),
            )
            request.completion_token_ids.extend(token_ids)
            request.completion_token_logprobs.extend(token_logprobs)
            request.completion_top_logprobs.extend(top_logprobs)
            request.pending_token_ids.extend(token_ids)
            request.pending_token_logprobs.extend(token_logprobs)
            request.pending_top_logprobs.extend(top_logprobs)
            decoded = request.detokenizer.push(token_ids)
            raw_text_delta = decoded.raw_text
            full_text = request.detokenizer.text
            stop_index = _find_stop_index(full_text, request.stop)
            visible_text = full_text if stop_index is None else full_text[:stop_index]
            emit_len = (
                len(visible_text)
                if stop_index is not None
                else _safe_stream_text_len(visible_text, request.stop)
            )
            text = visible_text[request.emitted_text_len:emit_len]
            request.emitted_text_len = emit_len
            if text or (stop_index is None and raw_text_delta):
                self._publish_pending_token_event(request, text, raw_text_delta)
            if stop_index is not None:
                final = request.detokenizer.finish(request.completion_token_ids)
                self.engine.abort_request(seq_id)
                active.pop(seq_id, None)
                self._put(
                    request,
                    {
                        "type": "final",
                        "index": request.index,
                        "text": visible_text,
                        "raw_text": final.raw_text,
                        "text_delta": visible_text[request.emitted_text_len:],
                        "finish_reason": "stop",
                        "prompt_tokens": len(request.prompt_token_ids),
                        "completion_tokens": len(request.completion_token_ids),
                        "token_ids": request.completion_token_ids,
                        "token_logprobs": request.completion_token_logprobs,
                        "top_logprobs": request.completion_top_logprobs,
                    },
                )

    def _publish_pending_token_event(
        self,
        request: _ActiveRequest,
        text: str,
        raw_text_delta: str,
    ):
        self._put(
            request,
            {
                "type": "token",
                "index": request.index,
                "text": text,
                "raw_text_delta": raw_text_delta,
                "token_ids": list(request.pending_token_ids),
                "token_logprobs": list(request.pending_token_logprobs),
                "top_logprobs": list(request.pending_top_logprobs),
            },
        )
        request.pending_token_ids.clear()
        request.pending_token_logprobs.clear()
        request.pending_top_logprobs.clear()

    def _drain_aborts(self, active: dict[int, _ActiveRequest]):
        while True:
            try:
                seq_id = self._aborts.get_nowait()
            except queue.Empty:
                return
            if seq_id in active:
                self.engine.abort_request(seq_id)
                active.pop(seq_id)

    def _fail_active_requests(self, active: dict[int, _ActiveRequest], message: str):
        for seq_id, request in list(active.items()):
            active.pop(seq_id, None)
            try:
                self._put(request, {"type": "error", "message": message})
            except Exception:
                logger.exception("Failed to notify OpenAI request {} during dispatcher shutdown", seq_id)
            try:
                self.engine.abort_request(seq_id)
            except Exception:
                logger.exception("Failed to abort OpenAI request {} during dispatcher shutdown", seq_id)

    def _drain_queued_requests(self, message: str):
        while True:
            try:
                item = self._pending.get_nowait()
            except queue.Empty:
                break
            if item is not None and item is not _WAKEUP:
                try:
                    self._put(item, {"type": "error", "message": message})
                except Exception:
                    logger.exception("Failed to notify queued OpenAI request during dispatcher shutdown")
        while True:
            try:
                item = self._controls.get_nowait()
            except queue.Empty:
                return
            try:
                self._put_control(item, {"type": "error", "message": message})
            except Exception:
                logger.exception("Failed to notify queued OpenAI control request during dispatcher shutdown")

    def _publish_finished(
        self,
        active: dict[int, _ActiveRequest],
        finished_outputs: list[
            tuple[
                int,
                list[int],
                list[float | None],
                list[dict[int, float] | None],
            ]
        ],
    ):
        for seq_id, completion_token_ids, token_logprobs, top_logprobs in finished_outputs:
            request = active.get(seq_id)
            if request is None:
                continue
            observed = len(request.completion_token_ids)
            final = request.detokenizer.finish(completion_token_ids)
            request.pending_token_ids.extend(completion_token_ids[observed:])
            request.pending_token_logprobs.extend(token_logprobs[observed:])
            request.pending_top_logprobs.extend(top_logprobs[observed:])
            active.pop(seq_id, None)
            request.completion_token_ids = list(completion_token_ids)
            request.completion_token_logprobs = list(token_logprobs)
            request.completion_top_logprobs = list(top_logprobs)
            finish_reason = "length" if len(completion_token_ids) >= request.max_tokens else "stop"
            text = final.text
            stop_index = _find_stop_index(text, request.stop)
            if stop_index is not None:
                text = text[:stop_index]
                finish_reason = "stop"
            text_delta = text[request.emitted_text_len:]
            has_pending_logprobs = any(
                value is not None for value in request.pending_token_logprobs
            ) or any(value is not None for value in request.pending_top_logprobs)
            if request.pending_token_ids and (
                text_delta or final.raw_text_delta or has_pending_logprobs
            ):
                self._publish_pending_token_event(request, text_delta, final.raw_text_delta)
                request.emitted_text_len = len(text)
            self._put(
                request,
                {
                    "type": "final",
                    "index": request.index,
                    "text": text,
                    "raw_text": final.raw_text,
                    "text_delta": text[request.emitted_text_len:],
                    "finish_reason": finish_reason,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "completion_tokens": len(completion_token_ids),
                    "token_ids": completion_token_ids,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                },
            )
