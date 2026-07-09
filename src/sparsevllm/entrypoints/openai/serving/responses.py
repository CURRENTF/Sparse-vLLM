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
from sparsevllm.entrypoints.openai.protocol.responses import ResponseRequest
from sparsevllm.entrypoints.openai.render import _response_prompt
from sparsevllm.entrypoints.openai.render import resolve_response_chat_template_kwargs
from sparsevllm.entrypoints.openai.responses import events as response_events
from sparsevllm.entrypoints.openai.responses.reasoning import ParsedReasoning
from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningParseError
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_parser
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_stream_parser
from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningStreamEvent
from sparsevllm.entrypoints.openai.responses.tools import ToolCallParseError
from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamEvent
from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamParser
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

    request_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    started = time.perf_counter()
    logger.info(
        "request_start id={} model={} endpoint=responses stream={} max_output_tokens={} temperature={} top_p={} top_k={}",
        request_id,
        request.model,
        request.stream,
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
    if request.stream:
        _write_request_log(
            request_log_path,
            {
                "status": "stream_started",
                "endpoint": "/v1/responses",
                "request_id": request_id,
                "request": _model_dump_json(request),
            },
        )
        return StreamingResponse(
            _response_stream(
                dispatcher,
                request_id,
                created_at,
                request.model,
                handle,
                started,
                request_log_path,
                request,
                reasoning_parser_name=reasoning_parser_name,
            ),
            media_type="text/event-stream",
        )

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
            "request_cancel id={} model={} endpoint=responses stream=false elapsed_s={:.3f}",
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
        "request_finish id={} model={} endpoint=responses stream=false prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
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
    if request.tool_choice not in (None, "auto"):
        raise HTTPException(status_code=400, detail="Responses tool_choice only supports null or 'auto' in this implementation.")
    if request.parallel_tool_calls not in (None, True):
        raise HTTPException(status_code=400, detail="Responses parallel_tool_calls=false is not implemented yet.")
    if request.reasoning is not None and request.reasoning.summary is not None:
        raise HTTPException(status_code=400, detail="Responses reasoning.summary is not implemented yet.")


async def _response_response(
    request_id: str,
    created_at: int,
    model: str,
    handle: RequestHandle,
    *,
    reasoning_parser_name: str | None,
) -> dict[str, Any]:
    final = await _wait_final(handle.output_queue)
    parser_input = final.get("raw_text", final["text"]) if reasoning_parser_name else final["text"]
    try:
        output, incomplete_reasoning = _response_output_items(
            parser_input,
            final["finish_reason"],
            reasoning_parser_name=reasoning_parser_name,
        )
    except (ReasoningParseError, ToolCallParseError) as exc:
        raise HTTPException(status_code=500, detail=f"Responses parse failed: {exc}") from exc

    incomplete = final["finish_reason"] == "length" or incomplete_reasoning
    return response_events.response_object(
        request_id,
        created_at,
        model,
        "incomplete" if incomplete else "completed",
        output,
        usage=_usage_from_final(final),
        incomplete_reason="max_output_tokens" if incomplete else None,
    )


async def _response_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created_at: int,
    model: str,
    handle: RequestHandle,
    started: float,
    request_log_path: Path | None,
    request: ResponseRequest,
    *,
    reasoning_parser_name: str | None,
):
    state = _ResponseStreamState(request_id, created_at, model)
    reasoning_parser = get_reasoning_stream_parser(reasoning_parser_name)
    tool_parser = ToolCallStreamParser()
    raw_text_len = 0
    visible_text_len = 0
    completion_tokens = 0

    try:
        yield response_events.response_created(state.response("in_progress"))
        while True:
            item = await handle.output_queue.get()
            if item["type"] == "error":
                dispatcher.cancel(handle)
                yield _response_stream_failed(state, item["message"])
                yield "data: [DONE]\n\n"
                _log_response_stream_failure(
                    request_id,
                    model,
                    started,
                    request_log_path,
                    request,
                    item["message"],
                )
                return

            if item["type"] == "token":
                completion_tokens += len(item.get("token_ids", []))
                if reasoning_parser_name:
                    delta = item.get("raw_text_delta", item.get("text", ""))
                    raw_text_len += len(delta)
                else:
                    delta = item.get("text", "")
                    visible_text_len += len(delta)
                for frame in _response_stream_delta(state, reasoning_parser, tool_parser, delta):
                    yield frame
                continue

            if item["type"] == "final":
                parser_text = item.get("raw_text", item["text"]) if reasoning_parser_name else item["text"]
                consumed = raw_text_len if reasoning_parser_name else visible_text_len
                for frame in _response_stream_delta(
                    state,
                    reasoning_parser,
                    tool_parser,
                    parser_text[consumed:],
                ):
                    yield frame
                for frame in _response_stream_reasoning_events(
                    state,
                    tool_parser,
                    reasoning_parser.finish(item["finish_reason"]),
                ):
                    yield frame
                for frame in _response_stream_tool_events(
                    state,
                    tool_parser.finish(item["finish_reason"]),
                ):
                    yield frame
                for frame in state.finish_open_items():
                    yield frame

                usage = _usage_from_final(item)
                incomplete = item["finish_reason"] == "length" or bool(
                    getattr(reasoning_parser, "incomplete_reasoning", False)
                )
                response = state.response(
                    "incomplete" if incomplete else "completed",
                    usage=usage,
                    incomplete_reason="max_output_tokens" if incomplete else None,
                )
                yield response_events.response_completed(response)
                yield "data: [DONE]\n\n"

                elapsed_s = time.perf_counter() - started
                logger.info(
                    "request_finish id={} model={} endpoint=responses stream=true prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
                    request_id,
                    model,
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
                return
    except asyncio.CancelledError:
        dispatcher.cancel(handle)
        elapsed_s = time.perf_counter() - started
        logger.info(
            "request_cancel id={} model={} endpoint=responses stream=true completion_tokens={} elapsed_s={:.3f}",
            request_id,
            model,
            completion_tokens,
            elapsed_s,
        )
        _write_request_log(
            request_log_path,
            {
                "status": "cancelled",
                "endpoint": "/v1/responses",
                "request_id": request_id,
                "elapsed_s": elapsed_s,
                "request": _model_dump_json(request),
            },
        )
        raise
    except Exception as exc:
        dispatcher.cancel(handle)
        message = f"{type(exc).__name__}: {exc}"
        yield _response_stream_failed(state, message)
        yield "data: [DONE]\n\n"
        _log_response_stream_failure(
            request_id,
            model,
            started,
            request_log_path,
            request,
            message,
        )


def _response_stream_delta(
    state: "_ResponseStreamState",
    reasoning_parser: Any,
    tool_parser: ToolCallStreamParser,
    delta: str,
) -> list[str]:
    return _response_stream_reasoning_events(
        state,
        tool_parser,
        reasoning_parser.feed(delta),
    )


def _response_stream_reasoning_events(
    state: "_ResponseStreamState",
    tool_parser: ToolCallStreamParser,
    reasoning_events: list[ReasoningStreamEvent],
) -> list[str]:
    frames: list[str] = []
    for event in reasoning_events:
        if event.kind == "reasoning_delta":
            frames.extend(state.reasoning_delta(event.text))
        elif event.kind == "reasoning_done":
            frames.extend(state.reasoning_done())
        elif event.kind == "answer_delta":
            frames.extend(_response_stream_tool_events(state, tool_parser.feed(event.text)))
        else:
            raise AssertionError(f"unknown reasoning stream event {event.kind!r}")
    return frames


def _response_stream_tool_events(
    state: "_ResponseStreamState",
    tool_events: list[ToolCallStreamEvent],
) -> list[str]:
    frames: list[str] = []
    for event in tool_events:
        if event.kind == "answer_delta":
            frames.extend(state.message_delta(event.text))
        elif event.kind == "tool_call_started":
            frames.extend(state.function_call_started(event.name or ""))
        elif event.kind == "tool_call_arguments_delta":
            frames.extend(state.function_call_arguments_delta(event.arguments_delta))
        elif event.kind == "tool_call_done":
            frames.extend(state.function_call_done())
        else:
            raise AssertionError(f"unknown tool call stream event {event.kind!r}")
    return frames


def _response_stream_failed(state: "_ResponseStreamState", message: str) -> str:
    error = {"type": "server_error", "message": message}
    return response_events.response_failed(
        state.response("failed", error=error),
        error,
    )


def _log_response_stream_failure(
    request_id: str,
    model: str,
    started: float,
    request_log_path: Path | None,
    request: ResponseRequest,
    message: str,
):
    elapsed_s = time.perf_counter() - started
    logger.info(
        "request_failure id={} model={} endpoint=responses stream=true elapsed_s={:.3f} error={}",
        request_id,
        model,
        elapsed_s,
        message,
    )
    _write_request_log(
        request_log_path,
        {
            "status": "failure",
            "endpoint": "/v1/responses",
            "request_id": request_id,
            "elapsed_s": elapsed_s,
            "request": _model_dump_json(request),
            "error": message,
        },
    )


def _usage_from_final(final: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": final["prompt_tokens"],
        "output_tokens": final["completion_tokens"],
        "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
    }


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


class _ResponseStreamState:
    def __init__(self, request_id: str, created_at: int, model: str):
        self.request_id = request_id
        self.created_at = created_at
        self.model = model
        self.output: list[dict[str, Any]] = []
        self._message_item: dict[str, Any] | None = None
        self._message_index: int | None = None
        self._message_text = ""
        self._message_done = False
        self._reasoning_item: dict[str, Any] | None = None
        self._reasoning_index: int | None = None
        self._reasoning_text = ""
        self._reasoning_done = False
        self._function_item: dict[str, Any] | None = None
        self._function_index: int | None = None
        self._function_arguments = ""

    def response(
        self,
        status: str,
        usage: dict[str, int] | None = None,
        incomplete_reason: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return response_events.response_object(
            self.request_id,
            self.created_at,
            self.model,
            status,
            self.output,
            usage=usage,
            incomplete_reason=incomplete_reason,
            error=error,
        )

    def message_delta(self, text: str) -> list[str]:
        if not text:
            return []
        frames = self._ensure_message()
        assert self._message_item is not None
        assert self._message_index is not None
        self._message_text += text
        self._message_item["content"][0]["text"] = self._message_text
        frames.append(response_events.output_text_delta(self._message_item["id"], self._message_index, 0, text))
        return frames

    def reasoning_delta(self, text: str) -> list[str]:
        if not text:
            return []
        frames = self._ensure_reasoning()
        assert self._reasoning_item is not None
        assert self._reasoning_index is not None
        self._reasoning_text += text
        self._reasoning_item["text"] = self._reasoning_text
        frames.append(response_events.reasoning_text_delta(self._reasoning_item["id"], self._reasoning_index, 0, text))
        return frames

    def reasoning_done(self) -> list[str]:
        frames = self._ensure_reasoning()
        if self._reasoning_done:
            return frames
        assert self._reasoning_item is not None
        assert self._reasoning_index is not None
        part = {"type": "reasoning_text", "text": self._reasoning_text}
        frames.append(
            response_events.reasoning_text_done(
                self._reasoning_item["id"],
                self._reasoning_index,
                0,
                self._reasoning_text,
            )
        )
        frames.append(response_events.reasoning_part_done(self._reasoning_item["id"], self._reasoning_index, 0, part))
        frames.append(response_events.output_item_done(self._reasoning_index, self._reasoning_item))
        self._reasoning_done = True
        return frames

    def function_call_started(self, name: str) -> list[str]:
        if not name:
            raise ToolCallParseError("tool call stream event missing function name.")
        if self._function_item is not None:
            raise ToolCallParseError("new tool call started before previous tool call finished.")
        self._function_index = len(self.output)
        self._function_item = {
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "call_id": f"call_{uuid.uuid4().hex}",
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        self._function_arguments = ""
        self.output.append(self._function_item)
        return [response_events.output_item_added(self._function_index, self._function_item)]

    def function_call_arguments_delta(self, arguments_delta: str) -> list[str]:
        if self._function_item is None or self._function_index is None:
            raise ToolCallParseError("tool call arguments arrived before tool call item.")
        if not arguments_delta:
            return []
        self._function_arguments += arguments_delta
        self._function_item["arguments"] = self._function_arguments
        return [
            response_events.function_call_arguments_delta(
                self._function_item["id"],
                self._function_index,
                arguments_delta,
            )
        ]

    def function_call_done(self) -> list[str]:
        if self._function_item is None or self._function_index is None:
            raise ToolCallParseError("tool call done arrived before tool call item.")
        self._function_item["status"] = "completed"
        frames = [
            response_events.function_call_arguments_done(
                self._function_item["id"],
                self._function_index,
                self._function_arguments,
            ),
            response_events.output_item_done(self._function_index, self._function_item),
        ]
        self._function_item = None
        self._function_index = None
        self._function_arguments = ""
        return frames

    def finish_open_items(self) -> list[str]:
        frames: list[str] = []
        frames.extend(self._finish_message())
        if self._reasoning_item is not None and not self._reasoning_done:
            frames.extend(self.reasoning_done())
        if self._function_item is not None:
            frames.extend(self.function_call_done())
        if not self.output:
            frames.extend(self._ensure_message())
            frames.extend(self._finish_message())
        return frames

    def _ensure_message(self) -> list[str]:
        if self._message_item is not None:
            return []
        self._message_index = len(self.output)
        self._message_item = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        self.output.append(self._message_item)
        frames = [response_events.output_item_added(self._message_index, self._message_item)]
        part = {"type": "output_text", "text": "", "annotations": []}
        self._message_item["content"].append(part)
        frames.append(response_events.content_part_added(self._message_item["id"], self._message_index, 0, part))
        return frames

    def _finish_message(self) -> list[str]:
        if self._message_item is None or self._message_done:
            return []
        assert self._message_index is not None
        part = self._message_item["content"][0]
        self._message_item["status"] = "completed"
        self._message_done = True
        return [
            response_events.output_text_done(self._message_item["id"], self._message_index, 0, self._message_text),
            response_events.content_part_done(self._message_item["id"], self._message_index, 0, part),
            response_events.output_item_done(self._message_index, self._message_item),
        ]

    def _ensure_reasoning(self) -> list[str]:
        if self._reasoning_item is not None:
            return []
        self._reasoning_index = len(self.output)
        self._reasoning_item = {
            "id": f"rs_{uuid.uuid4().hex}",
            "type": "reasoning",
            "text": "",
            "summary": [],
        }
        self.output.append(self._reasoning_item)
        part = {"type": "reasoning_text", "text": ""}
        return [
            response_events.output_item_added(self._reasoning_index, self._reasoning_item),
            response_events.reasoning_part_added(self._reasoning_item["id"], self._reasoning_index, 0, part),
        ]
