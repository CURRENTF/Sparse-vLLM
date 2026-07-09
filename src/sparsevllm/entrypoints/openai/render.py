import inspect
from typing import Any

from sparsevllm.entrypoints.openai.protocol.chat import ChatContentPart
from sparsevllm.entrypoints.openai.protocol.chat import ChatMessage
from sparsevllm.entrypoints.openai.protocol.responses import ResponseRequest
from sparsevllm.entrypoints.openai.responses.tools import normalize_tools


SUPPORTED_CHAT_TEMPLATE_KWARGS = {"enable_thinking"}


def _chat_template_role(role: str) -> str:
    return "system" if role == "developer" else role


def _chat_content_text(content: str | list[ChatContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content)


def validate_chat_template_kwargs(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("chat_template_kwargs must be a JSON object.")
    unknown = sorted(set(value) - SUPPORTED_CHAT_TEMPLATE_KWARGS)
    if unknown:
        raise ValueError(f"Unsupported chat_template_kwargs keys: {unknown}.")
    if "enable_thinking" in value and not isinstance(value["enable_thinking"], bool):
        raise ValueError("chat_template_kwargs.enable_thinking must be a bool.")
    return dict(value)


def resolve_response_chat_template_kwargs(request: ResponseRequest) -> dict[str, Any] | None:
    kwargs = validate_chat_template_kwargs(request.chat_template_kwargs) or {}
    effort = request.reasoning.effort if request.reasoning is not None else None
    if effort is None:
        return kwargs or None

    effort_enable_thinking = effort != "none"
    if "enable_thinking" in kwargs and kwargs["enable_thinking"] != effort_enable_thinking:
        raise ValueError("reasoning.effort conflicts with chat_template_kwargs.enable_thinking.")
    kwargs["enable_thinking"] = effort_enable_thinking
    return kwargs


def _chat_prompt(
    tokenizer: Any,
    messages: list[ChatMessage],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    chat = [
        {
            "role": _chat_template_role(message.role),
            "content": _chat_content_text(message.content),
        }
        for message in messages
    ]
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        kwargs.update(chat_template_kwargs or {})
        return tokenizer.apply_chat_template(chat, **kwargs)
    if chat_template_kwargs:
        raise ValueError("chat_template_kwargs requires a tokenizer chat_template.")

    rendered = []
    for message in chat:
        rendered.append(f"{message['role']}: {message['content']}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _response_prompt(tokenizer: Any, request: ResponseRequest) -> str:
    chat_template_kwargs = resolve_response_chat_template_kwargs(request)
    tools = normalize_tools(request.tools) if request.tools else None
    messages = _response_messages(request)

    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if has_template:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        if tools:
            if not _supports_chat_template_kwarg(tokenizer, "tools"):
                raise ValueError("Tokenizer chat template does not support tools.")
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(messages, **kwargs)

    if chat_template_kwargs:
        raise ValueError("chat_template_kwargs requires a tokenizer chat_template.")
    if tools:
        raise ValueError("tools requires a tokenizer chat_template with tools support.")
    rendered = []
    for message in messages:
        rendered.append(f"{message['role']}: {message.get('content', '')}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _response_messages(request: ResponseRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.instructions is not None:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
        return messages

    if not request.input:
        raise ValueError("responses input must not be empty.")
    for item in request.input:
        if not isinstance(item, dict):
            raise ValueError("responses input items must be JSON objects.")
        messages.extend(_response_input_item_messages(item))
    return messages


def _response_input_item_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    item_type = item.get("type")
    if item_type in (None, "message"):
        return [_response_message_item(item)]
    if item_type == "function_call_output":
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("function_call_output.call_id must be a non-empty string.")
        if not isinstance(output, str):
            raise ValueError("function_call_output.output must be a string.")
        return [{"role": "tool", "content": output, "tool_call_id": call_id}]
    if item_type == "function_call":
        return [_response_function_call_item(item)]
    if item_type == "reasoning":
        return []
    raise ValueError(f"Unsupported responses input item type: {item_type!r}.")


def _response_message_item(item: dict[str, Any]) -> dict[str, Any]:
    role = item.get("role")
    if role not in {"developer", "system", "user", "assistant"}:
        raise ValueError("message.role must be one of developer, system, user, assistant.")
    return {
        "role": _chat_template_role(str(role)),
        "content": _response_content_text(item.get("content")),
    }


def _response_function_call_item(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("function_call.call_id must be a non-empty string.")
    if not isinstance(name, str) or not name:
        raise ValueError("function_call.name must be a non-empty string.")
    if not isinstance(arguments, str):
        raise ValueError("function_call.arguments must be a string.")
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _response_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("message content parts must be JSON objects.")
            part_type = part.get("type")
            if part_type not in {"text", "input_text", "output_text"}:
                raise ValueError(f"Unsupported message content part type: {part_type!r}.")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("message content text parts require a string text field.")
            texts.append(text)
        return "\n".join(texts)
    raise ValueError("message.content must be a string or a text-only content part list.")


def _supports_chat_template_kwarg(tokenizer: Any, name: str) -> bool:
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters

