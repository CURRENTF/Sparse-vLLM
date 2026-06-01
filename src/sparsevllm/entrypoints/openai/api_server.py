import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from sparsevllm.config import Config
from sparsevllm.llm import LLM
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.sampling_params import SamplingParams


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
    logprobs: int | None = None


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
    completion_token_ids: list[int]
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
        self._thread.join()
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
                completion_token_ids=[],
            )
        except Exception as exc:
            self._put(item, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _publish_token_deltas(self, active: dict[int, _ActiveRequest]):
        for seq_id, token_ids in self.engine.last_step_token_outputs:
            request = active.get(seq_id)
            if request is None:
                continue
            request.completion_token_ids.extend(token_ids)
            full_text = self.engine.tokenizer.decode(request.completion_token_ids, skip_special_tokens=True)
            text = full_text[request.emitted_text_len:] if len(full_text) >= request.emitted_text_len else ""
            request.emitted_text_len = len(full_text)
            self._put(
                request,
                {
                    "type": "token",
                    "index": request.index,
                    "text": text,
                    "token_ids": token_ids,
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
        finished_outputs: list[tuple[int, list[int]]],
    ):
        for seq_id, completion_token_ids in finished_outputs:
            request = active.pop(seq_id, None)
            if request is None:
                continue
            request.completion_token_ids = list(completion_token_ids)
            finish_reason = "length" if len(completion_token_ids) >= request.max_tokens else "stop"
            text = self.engine.tokenizer.decode(completion_token_ids, skip_special_tokens=True)
            self._put(
                request,
                {
                    "type": "final",
                    "index": request.index,
                    "text": text,
                    "finish_reason": finish_reason,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "completion_tokens": len(completion_token_ids),
                    "token_ids": completion_token_ids,
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
    _validate_serving_method(engine_kwargs or {}, engine)
    engine = engine or LLM(model, **(engine_kwargs or {}))
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
        prompts = _normalize_prompts(request.prompt)
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )

        handles = [
            await dispatcher.submit(prompt, sampling_params, index)
            for index, prompt in enumerate(prompts)
        ]

        if request.stream:
            return StreamingResponse(
                _completion_stream(dispatcher, request_id, created, request.model, handles),
                media_type="text/event-stream",
            )

        try:
            response = await _completion_response(request_id, created, request.model, handles)
        except asyncio.CancelledError:
            for handle in handles:
                dispatcher.cancel(handle)
            raise
        except Exception:
            for handle in handles:
                dispatcher.cancel(handle)
            raise
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
    if request.stop:
        raise HTTPException(status_code=400, detail="stop strings are not supported yet.")
    if request.logprobs is not None:
        raise HTTPException(status_code=400, detail="logprobs is not supported yet.")


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


async def _completion_response(
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
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
                "logprobs": None,
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


async def _completion_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
):
    pending = {index: handle for index, handle in enumerate(handles)}
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
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "text": item["text"],
                                    "index": item["index"],
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                elif item["type"] == "final":
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "text": "",
                                    "index": item["index"],
                                    "logprobs": None,
                                    "finish_reason": item["finish_reason"],
                                }
                            ],
                        }
                    )
                    pending.pop(tasks[task], None)
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        for handle in pending.values():
            dispatcher.cancel(handle)
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
