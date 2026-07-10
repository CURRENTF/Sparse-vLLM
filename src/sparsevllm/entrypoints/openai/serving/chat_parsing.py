import uuid
from dataclasses import dataclass
from typing import Any

from sparsevllm.entrypoints.openai.responses.reasoning import ParsedReasoning
from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningStreamEvent
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_parser
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_stream_parser
from sparsevllm.entrypoints.openai.responses.tools import ParsedToolCall
from sparsevllm.entrypoints.openai.responses.tools import ToolCallParseError
from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamEvent
from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamParser
from sparsevllm.entrypoints.openai.responses.tools import parse_tool_calls


@dataclass(frozen=True)
class ParsedChatOutput:
    reasoning_content: str | None
    content: str
    tool_calls: list[ParsedToolCall]


def parse_chat_output(
    text: str,
    finish_reason: str | None,
    *,
    reasoning_parser_name: str | None,
    parse_tools: bool,
) -> ParsedChatOutput:
    parser = get_reasoning_parser(reasoning_parser_name)
    parsed = parser(text, finish_reason) if parser is not None else ParsedReasoning(None, text)
    tool_calls = parse_tool_calls(parsed.output_text) if parse_tools else None
    return ParsedChatOutput(
        reasoning_content=parsed.reasoning_text,
        content=parsed.output_text,
        tool_calls=tool_calls or [],
    )


class ChatStreamParser:
    def __init__(
        self,
        *,
        reasoning_parser_name: str | None,
        parse_tools: bool,
        buffer_initial_reasoning: bool,
    ):
        self._reasoning_parser = get_reasoning_stream_parser(
            reasoning_parser_name,
            buffer_initial_content=buffer_initial_reasoning,
        )
        self._tool_parser = ToolCallStreamParser() if parse_tools else None
        self._tool_index = 0
        self._active_tool_id: str | None = None
        self.tools_called = False

    def feed(self, text_delta: str) -> list[dict[str, Any]]:
        return self._reasoning_events(self._reasoning_parser.feed(text_delta))

    def finish(self, finish_reason: str | None) -> list[dict[str, Any]]:
        deltas = self._reasoning_events(self._reasoning_parser.finish(finish_reason))
        if self._tool_parser is not None:
            deltas.extend(self._tool_events(self._tool_parser.finish(finish_reason)))
        return deltas

    def _reasoning_events(self, events: list[ReasoningStreamEvent]) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for event in events:
            if event.kind == "reasoning_delta":
                if event.text:
                    deltas.append({"reasoning_content": event.text})
            elif event.kind == "answer_delta":
                if self._tool_parser is None:
                    if event.text:
                        deltas.append({"content": event.text})
                else:
                    deltas.extend(self._tool_events(self._tool_parser.feed(event.text)))
            elif event.kind != "reasoning_done":
                raise AssertionError(f"unknown reasoning stream event {event.kind!r}")
        return deltas

    def _tool_events(self, events: list[ToolCallStreamEvent]) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for event in events:
            if event.kind == "answer_delta":
                if event.text:
                    deltas.append({"content": event.text})
            elif event.kind == "tool_call_started":
                if self._active_tool_id is not None:
                    raise ToolCallParseError("new tool call started before previous tool call finished.")
                if not event.name:
                    raise ToolCallParseError("tool call stream event missing function name.")
                self._active_tool_id = f"call_{uuid.uuid4().hex}"
                self.tools_called = True
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": self._tool_index,
                                "id": self._active_tool_id,
                                "type": "function",
                                "function": {"name": event.name, "arguments": ""},
                            }
                        ]
                    }
                )
            elif event.kind == "tool_call_arguments_delta":
                if self._active_tool_id is None:
                    raise ToolCallParseError("tool call arguments arrived before tool call start.")
                if event.arguments_delta:
                    deltas.append(
                        {
                            "tool_calls": [
                                {
                                    "index": self._tool_index,
                                    "function": {"arguments": event.arguments_delta},
                                }
                            ]
                        }
                    )
            elif event.kind == "tool_call_done":
                if self._active_tool_id is None:
                    raise ToolCallParseError("tool call ended before tool call start.")
                self._active_tool_id = None
                self._tool_index += 1
            else:
                raise AssertionError(f"unknown tool call stream event {event.kind!r}")
        return deltas
