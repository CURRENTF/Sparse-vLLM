import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from sparsevllm.entrypoints.openai.dispatcher import AsyncEngineDispatcher
from sparsevllm.entrypoints.openai.dispatcher import RequestHandle
from sparsevllm.entrypoints.openai.protocol.responses import ResponseRequest
from sparsevllm.entrypoints.openai.render import _response_prompt
from sparsevllm.entrypoints.openai.render import resolve_response_chat_template_kwargs
from sparsevllm.entrypoints.openai.responses.reasoning import ParsedReasoning
from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningParseError
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_parser
from sparsevllm.entrypoints.openai.responses.tools import ToolCallParseError
from sparsevllm.entrypoints.openai.responses.tools import normalize_tools
from sparsevllm.entrypoints.openai.responses.tools import parse_tool_calls
from sparsevllm.entrypoints.openai.sampling import _sampling_params_from_response_request
from sparsevllm.entrypoints.openai.serving.base import _model_dump_json
from sparsevllm.entrypoints.openai.serving.base import _tokens_per_second
from sparsevllm.entrypoints.openai.serving.base import _wait_final
from sparsevllm.entrypoints.openai.serving.base import _write_request_log
from sparsevllm.utils.log import logger


async def serve_response(
    request: ResponseRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    served_model_name: str,
    request_log_path: Path | None,
    reasoning_parser_name: str | None,
):
    _validate_response_request(request, served_model_name)
    if request.stream:
        raise HTTPException(status_code=400, detail="Responses streaming is not implemented yet.")

    request_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    started = time.perf_counter()
    logger.info(
        "request_start id={} model={} endpoint=responses stream=false max_output_tokens={} temperature={} top_p={} top_k={}",
        request_id,
        request.model,
        request.max_output_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
    )
    try:
        prompt = _response_prompt(tokenizer, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    handle = await dispatcher.submit(prompt, _sampling_params_from_response_request(request), 0, [])
    try:
        response = await _response_response(
            request_id,
            created_at,
            request.model,
            handle,
            reasoning_parser_name=reasoning_parser_name,
        )
    except asyncio.CancelledError:
        dispatcher.cancel(handle)
        logger.info(
            "request_cancel id={} model={} endpoint=responses elapsed_s={:.3f}",
            request_id,
            request.model,
            time.perf_counter() - started,
        )
        raise
    except HTTPException:
        dispatcher.cancel(handle)
        raise
    except Exception:
        dispatcher.cancel(handle)
        raise

    usage = response["usage"]
    elapsed_s = time.perf_counter() - started
    logger.info(
        "request_finish id={} model={} endpoint=responses prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
        request_id,
        request.model,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
        elapsed_s,
        _tokens_per_second(usage["output_tokens"], elapsed_s),
        _tokens_per_second(usage["total_tokens"], elapsed_s),
    )
    _write_request_log(
        request_log_path,
        {
            "status": "success",
            "endpoint": "/v1/responses",
            "request_id": request_id,
            "elapsed_s": elapsed_s,
            "request": _model_dump_json(request),
            "response": response,
        },
    )
    return JSONResponse(response)


def _validate_response_request(request: ResponseRequest, served_model_name: str):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    try:
        resolve_response_chat_template_kwargs(request)
        normalize_tools(request.tools)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _response_response(
    request_id: str,
    created_at: int,
    model: str,
    handle: RequestHandle,
    *,
    reasoning_parser_name: str | None,
) -> dict[str, Any]:
    final = await _wait_final(handle.output_queue)
    try:
        output, incomplete_reasoning = _response_output_items(
            final["text"],
            final["finish_reason"],
            reasoning_parser_name=reasoning_parser_name,
        )
    except (ReasoningParseError, ToolCallParseError) as exc:
        raise HTTPException(status_code=500, detail=f"Responses parse failed: {exc}") from exc

    incomplete = final["finish_reason"] == "length" or incomplete_reasoning
    response: dict[str, Any] = {
        "id": request_id,
        "object": "response",
        "created_at": created_at,
        "status": "incomplete" if incomplete else "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": final["prompt_tokens"],
            "output_tokens": final["completion_tokens"],
            "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
        },
    }
    if incomplete:
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return response


def _response_output_items(
    text: str,
    finish_reason: str | None,
    *,
    reasoning_parser_name: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    parser = get_reasoning_parser(reasoning_parser_name)
    parsed = parser(text, finish_reason) if parser is not None else ParsedReasoning(None, text)
    output: list[dict[str, Any]] = []
    if parsed.reasoning_text is not None:
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "text": parsed.reasoning_text,
                "summary": [],
            }
        )

    tool_calls = parse_tool_calls(parsed.output_text)
    if tool_calls:
        for tool_call in tool_calls:
            output.append(
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "call_id": f"call_{uuid.uuid4().hex}",
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "status": "completed",
                }
            )
    elif parsed.output_text or parsed.reasoning_text is None:
        output.append(_message_output_item(parsed.output_text))
    return output, parsed.incomplete_reasoning


def _message_output_item(text: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }

