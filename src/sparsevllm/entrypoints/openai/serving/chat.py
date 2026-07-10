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
from sparsevllm.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparsevllm.entrypoints.openai.render import _chat_request_prompt
from sparsevllm.entrypoints.openai.render import resolve_chat_template_kwargs
from sparsevllm.entrypoints.openai.render import resolve_chat_tools
from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningParseError
from sparsevllm.entrypoints.openai.responses.tools import ToolCallParseError
from sparsevllm.entrypoints.openai.sampling import _field_was_set
from sparsevllm.entrypoints.openai.sampling import _normalize_stop
from sparsevllm.entrypoints.openai.sampling import _sampling_params_from_request
from sparsevllm.entrypoints.openai.serving.base import _chat_logprobs
from sparsevllm.entrypoints.openai.serving.base import _model_dump_json
from sparsevllm.entrypoints.openai.serving.base import _sse
from sparsevllm.entrypoints.openai.serving.base import _tokens_per_second
from sparsevllm.entrypoints.openai.serving.base import _wait_final
from sparsevllm.entrypoints.openai.serving.base import _write_request_log
from sparsevllm.entrypoints.openai.serving.chat_parsing import ParsedChatOutput
from sparsevllm.entrypoints.openai.serving.chat_parsing import parse_chat_output
from sparsevllm.utils.log import logger


async def serve_chat_completion(
    request: ChatCompletionRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    served_model_name: str,
    request_log_path: Path | None,
    reasoning_parser_name: str | None = None,
):
    _validate_chat_request(
        request,
        served_model_name,
        tokenizer,
        reasoning_parser_name=reasoning_parser_name,
    )
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
    try:
        prompt = _chat_request_prompt(tokenizer, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.stream:
        _write_request_log(
            request_log_path,
            {
                "status": "stream_started",
                "endpoint": "/v1/chat/completions",
                "request_id": request_id,
                "request": _model_dump_json(request),
            },
        )
    handle = await dispatcher.submit(prompt, sampling_params, 0, stop)
    handles = [handle]

    if request.stream:
        return StreamingResponse(
            _chat_completion_stream(
                dispatcher,
                request_id,
                created,
                request.model,
                handles,
                started,
                tokenizer,
                _stream_include_usage(request.stream_options),
            ),
            media_type="text/event-stream",
        )

    try:
        response = await _chat_completion_response(
            request_id,
            created,
            request.model,
            handles,
            tokenizer,
            reasoning_parser_name=reasoning_parser_name,
            parse_tools=bool(resolve_chat_tools(request)),
        )
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
    _write_request_log(
        request_log_path,
        {
            "status": "success",
            "endpoint": "/v1/chat/completions",
            "request_id": request_id,
            "elapsed_s": elapsed_s,
            "request": _model_dump_json(request),
            "response": response,
        },
    )
    return JSONResponse(response)


def _validate_chat_request(
    request: ChatCompletionRequest,
    served_model_name: str,
    tokenizer: Any | None = None,
    *,
    reasoning_parser_name: str | None = None,
):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Sparse-vLLM chat completions currently supports n=1 only.")
    if (
        request.max_completion_tokens is not None
        and _field_was_set(request, "max_tokens")
        and request.max_tokens != request.max_completion_tokens
    ):
        raise HTTPException(
            status_code=400,
            detail="max_tokens and max_completion_tokens disagree; set only one value.",
        )
    if request.top_logprobs is not None and not request.logprobs:
        raise HTTPException(status_code=400, detail="top_logprobs requires logprobs=true.")
    if request.stop and request.logprobs:
        raise HTTPException(status_code=400, detail="stop with logprobs is not supported yet.")
    try:
        chat_template_kwargs = resolve_chat_template_kwargs(request)
        tools = resolve_chat_tools(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if chat_template_kwargs and tokenizer is not None and not getattr(tokenizer, "chat_template", None):
        raise HTTPException(status_code=400, detail="chat_template_kwargs requires a tokenizer chat_template.")
    if request.logprobs and (reasoning_parser_name is not None or tools):
        raise HTTPException(
            status_code=400,
            detail="Chat logprobs cannot be aligned with parsed reasoning or tool calls.",
        )


def _stream_include_usage(stream_options: dict[str, Any] | None) -> bool:
    if stream_options is None:
        return False
    return bool(stream_options.get("include_usage"))


async def _chat_completion_response(
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    tokenizer: Any | None = None,
    *,
    reasoning_parser_name: str | None = None,
    parse_tools: bool = False,
) -> dict[str, Any]:
    if len(handles) != 1:
        raise HTTPException(status_code=500, detail="chat completions expects exactly one request handle.")
    final = await _wait_final(handles[0].output_queue)
    parser_input = final.get("raw_text", final["text"]) if reasoning_parser_name else final["text"]
    try:
        parsed = parse_chat_output(
            parser_input,
            final["finish_reason"],
            reasoning_parser_name=reasoning_parser_name,
            parse_tools=parse_tools,
        )
    except (ReasoningParseError, ToolCallParseError) as exc:
        raise HTTPException(status_code=500, detail=f"Chat Completions parse failed: {exc}") from exc
    message = _chat_message(parsed)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": final["index"],
                "message": message,
                "logprobs": _chat_logprobs(
                    tokenizer,
                    final.get("token_ids", []),
                    final.get("token_logprobs", []),
                    final.get("top_logprobs", []),
                )
                if tokenizer is not None
                else None,
                "finish_reason": "tool_calls" if parsed.tool_calls else final["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": final["prompt_tokens"],
            "completion_tokens": final["completion_tokens"],
            "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
        },
    }


def _chat_message(parsed: ParsedChatOutput) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None if parsed.tool_calls else parsed.content,
    }
    if parsed.reasoning_content is not None:
        message["reasoning_content"] = parsed.reasoning_content
    if parsed.tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in parsed.tool_calls
        ]
    return message


async def _chat_completion_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    started: float | None = None,
    tokenizer: Any | None = None,
    include_usage: bool = False,
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
                    logprobs = (
                        _chat_logprobs(
                            tokenizer,
                            item.get("token_ids", []),
                            item.get("token_logprobs", []),
                            item.get("top_logprobs", []),
                        )
                        if tokenizer is not None
                        else None
                    )
                    if not item["text"] and logprobs is None:
                        continue
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
                                    "logprobs": logprobs,
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
        if include_usage:
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
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
