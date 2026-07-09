import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from sparsevllm.entrypoints.openai.dispatcher import AsyncEngineDispatcher
from sparsevllm.entrypoints.openai.dispatcher import RequestHandle
from sparsevllm.entrypoints.openai.protocol.completion import CompletionRequest
from sparsevllm.entrypoints.openai.sampling import _normalize_prompts
from sparsevllm.entrypoints.openai.sampling import _normalize_stop
from sparsevllm.entrypoints.openai.sampling import _sampling_params_from_request
from sparsevllm.entrypoints.openai.serving.base import _completion_logprobs
from sparsevllm.entrypoints.openai.serving.base import _model_dump_json
from sparsevllm.entrypoints.openai.serving.base import _sse
from sparsevllm.entrypoints.openai.serving.base import _tokens_per_second
from sparsevllm.entrypoints.openai.serving.base import _wait_final
from sparsevllm.entrypoints.openai.serving.base import _write_request_log
from sparsevllm.utils.log import logger


async def serve_completion(
    request: CompletionRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    served_model_name: str,
    request_log_path: Path | None,
):
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
    if request.stream:
        _write_request_log(
            request_log_path,
            {
                "status": "stream_started",
                "endpoint": "/v1/completions",
                "request_id": request_id,
                "request": _model_dump_json(request),
            },
        )

    handles = [
        await dispatcher.submit(prompt, sampling_params, index, stop)
        for index, prompt in enumerate(prompts)
    ]

    if request.stream:
        return StreamingResponse(
            _completion_stream(dispatcher, request_id, created, request.model, handles, started, tokenizer),
            media_type="text/event-stream",
        )

    try:
        response = await _completion_response(request_id, created, request.model, handles, tokenizer)
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
    _write_request_log(
        request_log_path,
        {
            "status": "success",
            "endpoint": "/v1/completions",
            "request_id": request_id,
            "elapsed_s": elapsed_s,
            "request": _model_dump_json(request),
            "response": response,
        },
    )
    return JSONResponse(response)


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
                    if not item["text"]:
                        continue
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
