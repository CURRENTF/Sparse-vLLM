import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


_QWEN_RESPONSE_TEMPLATE_BASE = {
    "version": 1,
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {
            "open_pattern": r"<think>\s*",
            "close": "</think>",
            "content": "text",
            "optional": True,
        },
        "content": {
            "close_pattern": r"<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|\Z",
            "content": "text",
            "optional": True,
        },
    },
}

_QWEN_JSON_TOOL_FIELD = {
    "open": "<tool_call>",
    "close": "</tool_call>",
    "content": "json",
    "repeats": True,
    "optional": True,
}

_QWEN_XML_TOOL_FIELD = {
    "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\n]+)>\s*",
    "close_pattern": r"\s*</function>\s*</tool_call>",
    "content": "xml-inline",
    "content_args": {
        "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    },
    "transform": {
        "type": "function",
        "function": {"name": "{name}", "arguments": "{content}"},
    },
    "repeats": True,
    "optional": True,
}


class ResponseParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedModelResponse:
    reasoning_content: str | None
    content: str
    tool_calls: list[dict[str, Any]]


class TransformersResponseParser:
    """Thin OpenAI adapter around Transformers response parsing APIs."""

    def __init__(
        self,
        tokenizer: Any,
        response_template: dict[str, Any],
        *,
        validate_qwen_delimiters: bool = False,
    ):
        if not callable(getattr(tokenizer, "parse_response", None)):
            raise TypeError("Transformers response parsing requires tokenizer.parse_response().")
        if not callable(getattr(tokenizer, "get_response_parser", None)):
            raise TypeError("Transformers response streaming requires tokenizer.get_response_parser().")
        self._tokenizer = tokenizer
        self._response_template = response_template
        self._validate_qwen_delimiters = validate_qwen_delimiters

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "TransformersResponseParser | None":
        response_template = getattr(tokenizer, "response_template", None)
        if response_template is not None:
            return cls(tokenizer, response_template)

        chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, dict):
            chat_template = "\n".join(str(value) for value in chat_template.values())
        if not isinstance(chat_template, str) or "<think>" not in chat_template:
            return None

        response_template = {
            **_QWEN_RESPONSE_TEMPLATE_BASE,
            "fields": dict(_QWEN_RESPONSE_TEMPLATE_BASE["fields"]),
        }
        if "<function=" in chat_template and "<parameter=" in chat_template:
            response_template["fields"]["tool_calls"] = _QWEN_XML_TOOL_FIELD
        elif "<tool_call>" in chat_template:
            response_template["fields"]["tool_calls"] = _QWEN_JSON_TOOL_FIELD
        return cls(tokenizer, response_template, validate_qwen_delimiters=True)

    def parse(
        self,
        text: str,
        *,
        prefix: str,
        parse_tools: bool,
        finish_reason: str | None,
    ) -> ParsedModelResponse:
        try:
            if self._validate_qwen_delimiters and finish_reason != "length":
                _validate_qwen_response_delimiters(prefix, text)
            parsed = self._tokenizer.parse_response(
                text,
                self._response_template,
                prefix=prefix,
            )
            return _normalize_parsed_message(parsed, parse_tools=parse_tools)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc

    def stream(self, *, prefix: str, parse_tools: bool) -> "TransformersResponseStreamParser":
        try:
            parser = self._tokenizer.get_response_parser(
                self._response_template,
                prefix=prefix,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc
        return TransformersResponseStreamParser(
            parser,
            prefix=prefix,
            parse_tools=parse_tools,
            validate_qwen_delimiters=self._validate_qwen_delimiters,
        )


class TransformersResponseStreamParser:
    def __init__(
        self,
        parser: Any,
        *,
        prefix: str,
        parse_tools: bool,
        validate_qwen_delimiters: bool,
    ):
        self._parser = parser
        self._prefix = prefix
        self._parse_tools = parse_tools
        self._validate_qwen_delimiters = validate_qwen_delimiters
        self._initial_events = list(parser.initial_events)
        self._raw_text = ""
        self._tool_index = 0
        self.tools_called = False

    def feed(self, text_delta: str) -> list[dict[str, Any]]:
        events, self._initial_events = self._initial_events, []
        try:
            self._raw_text += text_delta
            events.extend(self._parser.feed(text_delta))
            return self._events_to_deltas(events)
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc

    def finish(self, finish_reason: str | None) -> list[dict[str, Any]]:
        try:
            if self._validate_qwen_delimiters and finish_reason != "length":
                _validate_qwen_response_delimiters(self._prefix, self._raw_text)
            _message, events = self._parser.finalize()
            return self._events_to_deltas(events)
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc

    def _events_to_deltas(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for event in events:
            event_type = event["type"]
            field = event["field"]
            if event_type == "region_chunk" and not event.get("dirty", False):
                if field in {"thinking", "reasoning", "reasoning_content"}:
                    if event["text"]:
                        deltas.append({"reasoning_content": event["text"]})
                elif field == "content" and event["text"]:
                    deltas.append({"content": event["text"]})
            elif event_type == "region_close" and field == "tool_calls" and self._parse_tools:
                tool_call = _normalize_tool_call(event["value"])
                call_id = f"call_{uuid.uuid4().hex}"
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": self._tool_index,
                                "id": call_id,
                                "type": "function",
                                "function": tool_call["function"],
                            }
                        ]
                    }
                )
                self._tool_index += 1
                self.tools_called = True
            elif event_type not in {"region_open", "region_chunk", "region_close"}:
                raise ResponseParseError(f"Unknown Transformers response event {event_type!r}.")
        return deltas


def _normalize_parsed_message(parsed: Any, *, parse_tools: bool) -> ParsedModelResponse:
    if not isinstance(parsed, dict):
        raise ResponseParseError(
            f"Transformers response parser returned {type(parsed).__name__}; expected a message object."
        )
    parsed = dict(parsed)
    role = parsed.pop("role", "assistant")
    if role != "assistant":
        raise ResponseParseError(f"Parsed response role must be 'assistant', got {role!r}.")
    reasoning = None
    for key in ("thinking", "reasoning_content", "reasoning"):
        if key in parsed:
            reasoning = parsed.pop(key)
            break
    content = parsed.pop("content", "")
    raw_tool_calls = parsed.pop("tool_calls", [])
    if raw_tool_calls and not parse_tools:
        raise ResponseParseError("Model returned tool calls when the request did not enable tools.")
    unexpected = {key: value for key, value in parsed.items() if value not in (None, "", [], {})}
    if unexpected:
        raise ResponseParseError(
            f"Unsupported parsed response fields: {sorted(unexpected)}."
        )
    if reasoning is not None and not isinstance(reasoning, str):
        raise ResponseParseError("Parsed reasoning content must be a string.")
    if not isinstance(content, str):
        raise ResponseParseError("Parsed assistant content must be a string.")
    if not isinstance(raw_tool_calls, list):
        raise ResponseParseError("Parsed tool_calls must be a list.")
    return ParsedModelResponse(
        reasoning_content=reasoning,
        content=content,
        tool_calls=[_normalize_tool_call(tool_call) for tool_call in raw_tool_calls],
    )


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ResponseParseError("Parsed tool call must be an object.")
    function = tool_call.get("function", tool_call)
    if not isinstance(function, dict):
        raise ResponseParseError("Parsed tool call function must be an object.")
    name = function.get("name")
    arguments = function.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ResponseParseError("Parsed tool call requires a non-empty function name.")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _validate_qwen_response_delimiters(prefix: str, text: str) -> None:
    assistant_prefix = prefix.rsplit("<|im_start|>assistant\n", 1)[-1]
    response = assistant_prefix + text
    pairs = (
        (response.count("<think>"), response.count("</think>"), "<think>"),
        (response.count("<tool_call>"), response.count("</tool_call>"), "<tool_call>"),
        (len(re.findall(r"<function=[^>\n]+>", response)), response.count("</function>"), "<function>"),
    )
    for opened, closed, name in pairs:
        if opened != closed:
            raise ResponseParseError(
                f"Qwen response did not close all {name} blocks: opened {opened}, closed {closed}."
            )
