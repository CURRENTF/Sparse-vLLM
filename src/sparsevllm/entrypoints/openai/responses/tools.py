import json
import re
from dataclasses import dataclass
from typing import Any


class ToolCallParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: str


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("tools entries must be JSON objects.")
        tool_type = tool.get("type")
        if tool_type != "function":
            raise ValueError(f"Unsupported tool type: {tool_type!r}.")
        function = tool.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ValueError("function tool.function must be a JSON object.")
            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters", {})
            strict = function.get("strict", tool.get("strict", False))
        else:
            name = tool.get("name")
            description = tool.get("description")
            parameters = tool.get("parameters", {})
            strict = tool.get("strict", False)

        if not isinstance(name, str) or not name:
            raise ValueError("function tool name must be a non-empty string.")
        if description is not None and not isinstance(description, str):
            raise ValueError("function tool description must be a string.")
        if not isinstance(parameters, dict):
            raise ValueError("function tool parameters must be a JSON object.")
        if not isinstance(strict, bool):
            raise ValueError("function tool strict must be a bool.")

        item = {
            "type": "function",
            "name": name,
            "parameters": parameters,
            "strict": strict,
        }
        if description is not None:
            item["description"] = description
        normalized.append(item)
    return normalized


def parse_tool_calls(text: str) -> list[ParsedToolCall] | None:
    stripped = text.strip()
    if not stripped.startswith("<tool_call>") and not stripped.startswith("<tool_calls>"):
        return None
    if stripped.startswith("<tool_calls>"):
        return _parse_tool_calls_array(stripped)

    matches = _TOOL_CALL_RE.findall(stripped)
    if not matches:
        raise ToolCallParseError("tool call output opened <tool_call> but did not close </tool_call>.")
    tail = _TOOL_CALL_RE.sub("", stripped).strip()
    if tail:
        raise ToolCallParseError("tool call output contains text outside <tool_call> blocks.")
    return [_parse_tool_call_json(match) for match in matches]


def _parse_tool_calls_array(text: str) -> list[ParsedToolCall]:
    end = text.find("</tool_calls>")
    if end < 0:
        raise ToolCallParseError("tool call output opened <tool_calls> but did not close </tool_calls>.")
    body = text[len("<tool_calls>"):end].strip()
    tail = text[end + len("</tool_calls>"):].strip()
    if tail:
        raise ToolCallParseError("tool call output contains text outside <tool_calls>.")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"tool call JSON parse failed: {exc}") from exc
    if not isinstance(data, list):
        raise ToolCallParseError("<tool_calls> body must be a JSON array.")
    return [_parse_tool_call_object(item) for item in data]


def _parse_tool_call_json(text: str) -> ParsedToolCall:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"tool call JSON parse failed: {exc}") from exc
    return _parse_tool_call_object(data)


def _parse_tool_call_object(data: Any) -> ParsedToolCall:
    if not isinstance(data, dict):
        raise ToolCallParseError("tool call body must be a JSON object.")
    name = data.get("name")
    arguments = data.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ToolCallParseError("tool call name must be a non-empty string.")
    if isinstance(arguments, str):
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(f"tool call arguments JSON string parse failed: {exc}") from exc
        arguments_text = arguments
    elif isinstance(arguments, dict):
        arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    else:
        raise ToolCallParseError("tool call arguments must be a JSON object or JSON string.")
    return ParsedToolCall(name=name, arguments=arguments_text)

