import argparse
import asyncio
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from sparsevllm.config import Config
from sparsevllm.llm import LLM
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.utils.log import logger


UNSUPPORTED_SERVING_METHOD_PREFIXES = ("deltakv",)
SEMANTIC_ENGINE_ARGS = {
    "sparse_method",
    "deltakv_checkpoint_path",
    "decode_keep_tokens",
    "prefill_keep_tokens",
    "sink_keep_tokens",
    "recent_keep_tokens",
    "full_attention_layers",
    "engine_prefill_chunk_size",
    "deltakv_neighbor_count",
    "observation_layers",
}


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    logprobs: int | None = Field(default=None, ge=0)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0)


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


class AsyncEngineDispatcher:
    def __init__(self, engine: LLM):
        self.engine = engine
        self._pending: queue.Queue[_QueuedRequest | None] = queue.Queue()
        self._aborts: queue.Queue[int] = queue.Queue()
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

    def _run(self):
        active: dict[int, _ActiveRequest] = {}
        stopping = False
        while not stopping:
            self._drain_aborts(active)
            if not active:
                item = self._pending.get()
                if item is None:
                    break
                self._admit(item, active)

            while True:
                try:
                    item = self._pending.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    continue
                self._admit(item, active)

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
                active.clear()
                self._drain_pending_after_failure()
                break

        self._abort_all(active)
        self._drain_pending_after_failure("Sparse-vLLM server is shutting down.")

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
            if item is not None:
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


def create_app(
    model: str,
    engine_kwargs: dict[str, Any] | None = None,
    *,
    served_model_name: str | None = None,
    engine: LLM | None = None,
) -> FastAPI:
    served_model_name = served_model_name or model
    engine_kwargs = dict(engine_kwargs or {})
    if engine is None:
        engine_kwargs.setdefault("throughput_log_interval_s", 0.0)
    _validate_serving_method(engine_kwargs, engine)
    engine = engine or LLM(model, **engine_kwargs)
    dispatcher = AsyncEngineDispatcher(engine)

    app = FastAPI(title="Sparse-vLLM OpenAI-compatible API")
    app.state.dispatcher = dispatcher
    app.state.served_model_name = served_model_name

    @app.on_event("shutdown")
    def _shutdown():
        dispatcher.close()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "sparsevllm",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        _validate_request(request, served_model_name)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        started = time.perf_counter()
        prompts = _normalize_prompts(request.prompt)
        logger.info(
            "request_start id={} model={} stream={} prompts={} max_tokens={} temperature={} top_p={} top_k={}",
            request_id,
            request.model,
            request.stream,
            len(prompts),
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
        )
        sampling_params = _sampling_params_from_request(request)
        stop = _normalize_stop(request.stop)

        handles = [
            await dispatcher.submit(prompt, sampling_params, index, stop)
            for index, prompt in enumerate(prompts)
        ]

        if request.stream:
            return StreamingResponse(
                _completion_stream(dispatcher, request_id, created, request.model, handles, started, engine.tokenizer),
                media_type="text/event-stream",
            )

        try:
            response = await _completion_response(request_id, created, request.model, handles, engine.tokenizer)
        except asyncio.CancelledError:
            for handle in handles:
                dispatcher.cancel(handle)
            logger.info(
                "request_cancel id={} model={} stream=false elapsed_s={:.3f}",
                request_id,
                request.model,
                time.perf_counter() - started,
            )
            raise
        except Exception:
            for handle in handles:
                dispatcher.cancel(handle)
            raise
        usage = response["usage"]
        elapsed_s = time.perf_counter() - started
        logger.info(
            "request_finish id={} model={} stream=false prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
            request_id,
            request.model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            elapsed_s,
            _tokens_per_second(usage["completion_tokens"], elapsed_s),
            _tokens_per_second(usage["total_tokens"], elapsed_s),
        )
        return JSONResponse(response)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        _validate_chat_request(request, served_model_name)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        started = time.perf_counter()
        logger.info(
            "request_start id={} model={} endpoint=chat stream={} messages={} max_tokens={} temperature={} top_p={} top_k={}",
            request_id,
            request.model,
            request.stream,
            len(request.messages),
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
        )
        sampling_params = _sampling_params_from_request(request)
        stop = _normalize_stop(request.stop)
        prompt = _chat_prompt(engine.tokenizer, request.messages)
        handle = await dispatcher.submit(prompt, sampling_params, 0, stop)
        handles = [handle]

        if request.stream:
            return StreamingResponse(
                _chat_completion_stream(dispatcher, request_id, created, request.model, handles, started, engine.tokenizer),
                media_type="text/event-stream",
            )

        try:
            response = await _chat_completion_response(request_id, created, request.model, handles, engine.tokenizer)
        except asyncio.CancelledError:
            dispatcher.cancel(handle)
            logger.info(
                "request_cancel id={} model={} stream=false elapsed_s={:.3f}",
                request_id,
                request.model,
                time.perf_counter() - started,
            )
            raise
        except Exception:
            dispatcher.cancel(handle)
            raise
        usage = response["usage"]
        elapsed_s = time.perf_counter() - started
        logger.info(
            "request_finish id={} model={} stream=false prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
            request_id,
            request.model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            elapsed_s,
            _tokens_per_second(usage["completion_tokens"], elapsed_s),
            _tokens_per_second(usage["total_tokens"], elapsed_s),
        )
        return JSONResponse(response)

    return app


def _validate_serving_method(engine_kwargs: dict[str, Any], engine: LLM | None = None):
    method = (
        getattr(getattr(engine, "config", None), "vllm_sparse_method", "")
        if engine is not None
        else engine_kwargs.get("sparse_method", engine_kwargs.get("vllm_sparse_method", ""))
    )
    method = normalize_sparse_method(method)
    if any(method.startswith(prefix) for prefix in UNSUPPORTED_SERVING_METHOD_PREFIXES):
        raise ValueError(
            f"vllm_sparse_method={method!r} is not supported by the OpenAI API server yet. "
            "Run this method through the offline experiment entrypoints until serving support is validated."
        )


def _validate_request(request: CompletionRequest, served_model_name: str):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Sparse-vLLM completions currently supports n=1 only.")
    if request.stop and request.logprobs is not None:
        raise HTTPException(status_code=400, detail="stop with logprobs is not supported yet.")


def _validate_chat_request(request: ChatCompletionRequest, served_model_name: str):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Sparse-vLLM chat completions currently supports n=1 only.")
    if request.top_logprobs is not None and not request.logprobs:
        raise HTTPException(status_code=400, detail="top_logprobs requires logprobs=true.")
    if request.stop and request.logprobs:
        raise HTTPException(status_code=400, detail="stop with logprobs is not supported yet.")


def _sampling_params_from_request(request: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    logprobs = request.logprobs
    if isinstance(request, ChatCompletionRequest):
        logprobs = (
            request.top_logprobs
            if request.top_logprobs is not None
            else 0
        ) if request.logprobs else None
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
        logprobs=logprobs,
    )


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [item for item in stop if item]


def _chat_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    chat = [{"role": message.role, "content": message.content} for message in messages]
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    rendered = []
    for message in messages:
        rendered.append(f"{message.role}: {message.content}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _normalize_prompts(prompt: str | list[int] | list[str] | list[list[int]]) -> list[str | list[int]]:
    if isinstance(prompt, str):
        return [prompt]
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty.")
    if all(isinstance(item, int) for item in prompt):
        return [prompt]
    if all(isinstance(item, str) for item in prompt):
        return list(prompt)
    if all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in prompt):
        return list(prompt)
    raise HTTPException(status_code=400, detail="prompt must be a string, token id list, or homogeneous prompt list.")


def _find_stop_index(text: str, stop: list[str]) -> int | None:
    matches = [text.find(item) for item in stop if item and text.find(item) >= 0]
    return min(matches) if matches else None


def _safe_stream_text_len(text: str, stop: list[str]) -> int:
    if not stop:
        return len(text)
    max_overlap = 0
    for item in stop:
        max_prefix = min(len(item) - 1, len(text))
        for overlap in range(max_prefix, 0, -1):
            if text.endswith(item[:overlap]):
                max_overlap = max(max_overlap, overlap)
                break
    return len(text) - max_overlap


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=True)


def _completion_logprobs(
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float | None],
    top_logprobs: list[dict[int, float] | None],
) -> dict[str, Any] | None:
    if not token_logprobs or all(value is None for value in token_logprobs):
        return None
    tokens = [_token_text(tokenizer, token_id) for token_id in token_ids]
    text_offsets = []
    offset = 0
    for token in tokens:
        text_offsets.append(offset)
        offset += len(token)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": [
            None
            if item is None
            else {_token_text(tokenizer, token_id): value for token_id, value in item.items()}
            for item in top_logprobs
        ],
        "text_offset": text_offsets,
    }


def _chat_logprobs(
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float | None],
    top_logprobs: list[dict[int, float] | None],
) -> dict[str, Any] | None:
    if not token_logprobs or all(value is None for value in token_logprobs):
        return None
    content = []
    for token_id, logprob, top_items in zip(token_ids, token_logprobs, top_logprobs):
        token = _token_text(tokenizer, token_id)
        top = []
        if top_items is not None:
            top = [
                {"token": _token_text(tokenizer, top_token_id), "logprob": value, "bytes": None}
                for top_token_id, value in top_items.items()
            ]
        content.append({"token": token, "logprob": logprob, "bytes": None, "top_logprobs": top})
    return {"content": content}


async def _completion_response(
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    choices = []
    prompt_tokens = 0
    completion_tokens = 0
    for handle in handles:
        final = await _wait_final(handle.output_queue)
        choices.append(
            {
                "text": final["text"],
                "index": final["index"],
                "logprobs": _completion_logprobs(
                    tokenizer,
                    final.get("token_ids", []),
                    final.get("token_logprobs", []),
                    final.get("top_logprobs", []),
                )
                if tokenizer is not None
                else None,
                "finish_reason": final["finish_reason"],
            }
        )
        prompt_tokens += final["prompt_tokens"]
        completion_tokens += final["completion_tokens"]

    choices.sort(key=lambda choice: choice["index"])
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _chat_completion_response(
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if len(handles) != 1:
        raise HTTPException(status_code=500, detail="chat completions expects exactly one request handle.")
    final = await _wait_final(handles[0].output_queue)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": final["index"],
                "message": {"role": "assistant", "content": final["text"]},
                "logprobs": _chat_logprobs(
                    tokenizer,
                    final.get("token_ids", []),
                    final.get("token_logprobs", []),
                    final.get("top_logprobs", []),
                )
                if tokenizer is not None
                else None,
                "finish_reason": final["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": final["prompt_tokens"],
            "completion_tokens": final["completion_tokens"],
            "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
        },
    }


async def _completion_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    started: float | None = None,
    tokenizer: Any | None = None,
):
    pending = {index: handle for index, handle in enumerate(handles)}
    prompt_tokens = 0
    completion_tokens = 0
    try:
        while pending:
            tasks = {
                asyncio.create_task(handle.output_queue.get()): index
                for index, handle in pending.items()
            }
            done, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending_tasks:
                task.cancel()
            for task in done:
                item = task.result()
                if item["type"] == "error":
                    yield _sse({"object": "error", "message": item["message"]})
                    pending.pop(tasks[task], None)
                    continue
                if item["type"] == "token":
                    completion_tokens += len(item["token_ids"])
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "text": item["text"],
                                    "index": item["index"],
                                    "logprobs": _completion_logprobs(
                                        tokenizer,
                                        item.get("token_ids", []),
                                        item.get("token_logprobs", []),
                                        item.get("top_logprobs", []),
                                    )
                                    if tokenizer is not None
                                    else None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                elif item["type"] == "final":
                    prompt_tokens += item["prompt_tokens"]
                    completion_tokens = max(completion_tokens, item["completion_tokens"])
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "text": item.get("text_delta", ""),
                                    "index": item["index"],
                                    "logprobs": None,
                                    "finish_reason": item["finish_reason"],
                                }
                            ],
                        }
                    )
                    pending.pop(tasks[task], None)
        yield "data: [DONE]\n\n"
        if started is not None:
            elapsed_s = time.perf_counter() - started
            total_tokens = prompt_tokens + completion_tokens
            logger.info(
                "request_finish id={} model={} stream=true prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
                request_id,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                elapsed_s,
                _tokens_per_second(completion_tokens, elapsed_s),
                _tokens_per_second(total_tokens, elapsed_s),
            )
    except asyncio.CancelledError:
        for handle in pending.values():
            dispatcher.cancel(handle)
        logger.info(
            "request_cancel id={} model={} stream=true completion_tokens={} elapsed_s={:.3f}",
            request_id,
            model,
            completion_tokens,
            time.perf_counter() - started if started is not None else 0.0,
        )
        raise


async def _chat_completion_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    started: float | None = None,
    tokenizer: Any | None = None,
):
    pending = {index: handle for index, handle in enumerate(handles)}
    prompt_tokens = 0
    completion_tokens = 0
    first_chunk = False
    try:
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )
        while pending:
            tasks = {
                asyncio.create_task(handle.output_queue.get()): index
                for index, handle in pending.items()
            }
            done, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending_tasks:
                task.cancel()
            for task in done:
                item = task.result()
                if item["type"] == "error":
                    yield _sse({"object": "error", "message": item["message"]})
                    pending.pop(tasks[task], None)
                    continue
                if item["type"] == "token":
                    completion_tokens += len(item["token_ids"])
                    delta: dict[str, Any] = {"content": item["text"]}
                    if first_chunk:
                        delta["role"] = "assistant"
                        first_chunk = False
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": item["index"],
                                    "delta": delta,
                                    "logprobs": _chat_logprobs(
                                        tokenizer,
                                        item.get("token_ids", []),
                                        item.get("token_logprobs", []),
                                        item.get("top_logprobs", []),
                                    )
                                    if tokenizer is not None
                                    else None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                elif item["type"] == "final":
                    prompt_tokens += item["prompt_tokens"]
                    completion_tokens = max(completion_tokens, item["completion_tokens"])
                    text_delta = item.get("text_delta", "")
                    if text_delta:
                        delta = {"content": text_delta}
                        if first_chunk:
                            delta["role"] = "assistant"
                            first_chunk = False
                        yield _sse(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": item["index"],
                                        "delta": delta,
                                        "logprobs": None,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": item["index"],
                                    "delta": {},
                                    "logprobs": None,
                                    "finish_reason": item["finish_reason"],
                                }
                            ],
                        }
                    )
                    pending.pop(tasks[task], None)
        yield "data: [DONE]\n\n"
        if started is not None:
            elapsed_s = time.perf_counter() - started
            total_tokens = prompt_tokens + completion_tokens
            logger.info(
                "request_finish id={} model={} stream=true prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
                request_id,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                elapsed_s,
                _tokens_per_second(completion_tokens, elapsed_s),
                _tokens_per_second(total_tokens, elapsed_s),
            )
    except asyncio.CancelledError:
        for handle in pending.values():
            dispatcher.cancel(handle)
        logger.info(
            "request_cancel id={} model={} stream=true completion_tokens={} elapsed_s={:.3f}",
            request_id,
            model,
            completion_tokens,
            time.perf_counter() - started if started is not None else 0.0,
        )
        raise


async def _wait_final(queue_item: asyncio.Queue) -> dict[str, Any]:
    while True:
        item = await queue_item.get()
        if item["type"] == "error":
            raise HTTPException(status_code=500, detail=item["message"])
        if item["type"] == "final":
            return item


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _tokens_per_second(tokens: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return tokens / elapsed_s


def _parse_engine_kwargs(raw_args: list[str]) -> dict[str, Any]:
    config_fields = Config.__dataclass_fields__
    allowed_fields = set(config_fields) | SEMANTIC_ENGINE_ARGS
    kwargs: dict[str, Any] = {}
    idx = 0
    while idx < len(raw_args):
        key = raw_args[idx]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected engine argument {key!r}; expected --name value.")
        name = key[2:].replace("-", "_")
        if name not in allowed_fields:
            raise ValueError(f"Unknown Sparse-vLLM engine argument {key!r}.")
        if idx + 1 >= len(raw_args) or raw_args[idx + 1].startswith("--"):
            if name not in config_fields or not isinstance(config_fields[name].default, bool):
                raise ValueError(f"Missing value for Sparse-vLLM engine argument {key!r}.")
            value: Any = True
            idx += 1
        else:
            value = _coerce_cli_value(raw_args[idx + 1])
            idx += 2
        kwargs[name] = value
    return kwargs


def _coerce_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Sparse-vLLM with an OpenAI-compatible completions API.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model path to load.")
    parser.add_argument("--served-model-name", default=None, help="Model name accepted by /v1/completions.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main():
    parser = build_arg_parser()
    args, raw_engine_args = parser.parse_known_args()
    engine_kwargs = _parse_engine_kwargs(raw_engine_args)
    app = create_app(args.model, engine_kwargs, served_model_name=args.served_model_name)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
