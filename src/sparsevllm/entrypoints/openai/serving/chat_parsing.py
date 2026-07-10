from dataclasses import dataclass

from sparsevllm.entrypoints.openai.responses.reasoning import ParsedReasoning
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_parser
from sparsevllm.entrypoints.openai.responses.tools import ParsedToolCall
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
