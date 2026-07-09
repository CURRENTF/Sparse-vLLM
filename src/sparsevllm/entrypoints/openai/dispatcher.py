import asyncio
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any

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
    emitted_text_len: int = 0


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
        self._thread = threading.Thread(target=self._run, name="sparsevllm-openai-dispatcher", daemon=True)
        self._thread.start()

    async def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        index: int,
        stop: list[str] | None = None,
    ) -> RequestHandle:
        if self._failed_message is not None:
            output_queue: asyncio.Queue = asyncio.Queue()
            output_queue.put_nowait({"type": "error", "message": self._failed_message})
            return RequestHandle(output_queue=output_queue, cancelled=threading.Event())
        if self._closing.is_set():
            output_queue = asyncio.Queue()
            output_queue.put_nowait({"type": "error", "message": "Sparse-vLLM server is shutting down."})
            return RequestHandle(output_queue=output_queue, cancelled=threading.Event())
        output_queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        handle = RequestHandle(output_queue=output_queue, cancelled=cancelled)
        self._pending.put(
            _QueuedRequest(
                prompt=prompt,
                sampling_params=sampling_params,
                index=index,
                stop=list(stop or []),
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                cancelled=cancelled,
                handle=handle,
            )
        )
        return handle

    async def control(self, operation: str, **kwargs: Any) -> Any:
        if self._failed_message is not None:
            raise RuntimeError(self._failed_message)
        if self._closing.is_set():
            raise RuntimeError("Sparse-vLLM server is shutting down.")
        output_queue: asyncio.Queue = asyncio.Queue()
        self._controls.put(
            _ControlRequest(
                operation=operation,
                kwargs=dict(kwargs),
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
            )
        )
        self._pending.put(_WAKEUP)
        result = await output_queue.get()
        if result["type"] == "error":
            raise RuntimeError(result["message"])
        return result["value"]

    def cancel(self, handle: RequestHandle):
        handle.cancelled.set()
        if handle.seq_id is not None:
            self._aborts.put(handle.seq_id)

    def close(self):
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
        while not stopping:
            self._drain_controls()
            self._drain_aborts(active)
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
                continue

            try:
                finished_outputs, _num_tokens = self.engine.step()
                self._publish_token_deltas(active)
                self._publish_finished(active, finished_outputs)
            except Exception as exc:
                self._failed_message = f"{type(exc).__name__}: {exc}"
                for request in list(active.values()):
                    self._put(request, {"type": "error", "message": self._failed_message})
                self._abort_all(active)
                self._drain_pending_after_failure()
                break

        self._abort_all(active)
        self._drain_pending_after_failure("Sparse-vLLM server is shutting down.")

    def _drain_controls(self):
        while True:
            try:
                item = self._controls.get_nowait()
            except queue.Empty:
                return
            if self._closing.is_set():
                self._put_control(item, {"type": "error", "message": "Sparse-vLLM server is shutting down."})
                continue
            if self._failed_message is not None:
                self._put_control(item, {"type": "error", "message": self._failed_message})
                continue
            try:
                method = getattr(self.engine, item.operation)
                value = method(**item.kwargs)
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
            full_text = self.engine.tokenizer.decode(request.completion_token_ids, skip_special_tokens=True)
            stop_index = _find_stop_index(full_text, request.stop)
            visible_text = full_text if stop_index is None else full_text[:stop_index]
            emit_len = (
                len(visible_text)
                if stop_index is not None
                else _safe_stream_text_len(visible_text, request.stop)
            )
            text = visible_text[request.emitted_text_len:emit_len]
            request.emitted_text_len = emit_len
            if text:
                self._put(
                    request,
                    {
                        "type": "token",
                        "index": request.index,
                        "text": text,
                        "token_ids": token_ids,
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                    },
                )
            if stop_index is not None:
                active.pop(seq_id, None)
                self.engine.abort_request(seq_id)
                self._put(
                    request,
                    {
                        "type": "final",
                        "index": request.index,
                        "text": visible_text,
                        "text_delta": visible_text[request.emitted_text_len:],
                        "finish_reason": "stop",
                        "prompt_tokens": len(request.prompt_token_ids),
                        "completion_tokens": len(request.completion_token_ids),
                        "token_ids": request.completion_token_ids,
                        "token_logprobs": request.completion_token_logprobs,
                        "top_logprobs": request.completion_top_logprobs,
                    },
                )

    def _drain_aborts(self, active: dict[int, _ActiveRequest]):
        while True:
            try:
                seq_id = self._aborts.get_nowait()
            except queue.Empty:
                return
            if seq_id in active:
                active.pop(seq_id)
                self.engine.abort_request(seq_id)

    def _abort_all(self, active: dict[int, _ActiveRequest]):
        for seq_id in list(active):
            active.pop(seq_id)
            self.engine.abort_request(seq_id)

    def _drain_pending_after_failure(self, message: str | None = None):
        error = message or self._failed_message
        if error is None:
            return
        while True:
            try:
                item = self._pending.get_nowait()
            except queue.Empty:
                return
            if item is not None and item is not _WAKEUP:
                self._put(item, {"type": "error", "message": error})

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
            request = active.pop(seq_id, None)
            if request is None:
                continue
            request.completion_token_ids = list(completion_token_ids)
            request.completion_token_logprobs = list(token_logprobs)
            request.completion_top_logprobs = list(top_logprobs)
            finish_reason = "length" if len(completion_token_ids) >= request.max_tokens else "stop"
            text = self.engine.tokenizer.decode(completion_token_ids, skip_special_tokens=True)
            stop_index = _find_stop_index(text, request.stop)
            if stop_index is not None:
                text = text[:stop_index]
                finish_reason = "stop"
            self._put(
                request,
                {
                    "type": "final",
                    "index": request.index,
                    "text": text,
                    "text_delta": text[request.emitted_text_len:],
                    "finish_reason": finish_reason,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "completion_tokens": len(completion_token_ids),
                    "token_ids": completion_token_ids,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                },
            )

