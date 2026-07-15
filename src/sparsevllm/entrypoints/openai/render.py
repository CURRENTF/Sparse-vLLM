import inspect
import json
from typing import Any

from sparsevllm.entrypoints.openai.protocol.chat import ChatContentPart
from sparsevllm.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparsevllm.entrypoints.openai.protocol.chat import ChatMessage
from sparsevllm.entrypoints.openai.protocol.responses import ResponseRequest


BOOLEAN_CHAT_TEMPLATE_KWARGS = {"enable_thinking", "preserve_thinking"}


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
    for name in BOOLEAN_CHAT_TEMPLATE_KWARGS:
        if name in value and not isinstance(value[name], bool):
            raise ValueError(f"chat_template_kwargs.{name} must be a bool.")
    return dict(value)


def resolve_chat_template_kwargs(request: ChatCompletionRequest) -> dict[str, Any] | None:
    kwargs = validate_chat_template_kwargs(request.chat_template_kwargs) or {}
    _merge_chat_template_kwarg(kwargs, "preserve_thinking", request.preserve_thinking)
    enable_thinking = request.enable_thinking
    if request.reasoning_effort is not None:
        effort_enable_thinking = request.reasoning_effort != "none"
        if enable_thinking is not None and enable_thinking != effort_enable_thinking:
            raise ValueError("reasoning_effort conflicts with enable_thinking.")
        enable_thinking = effort_enable_thinking
    _merge_chat_template_kwarg(kwargs, "enable_thinking", enable_thinking)
    return kwargs or None


def _merge_chat_template_kwarg(kwargs: dict[str, Any], name: str, value: Any):
    if value is None:
        return
    if name in kwargs and kwargs[name] != value:
        raise ValueError(f"{name} conflicts with chat_template_kwargs.{name}.")
    kwargs[name] = value


def resolve_chat_tools(request: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    tools = normalize_tools(request.tools)
    if request.tool_choice not in (None, "auto", "none"):
        raise ValueError("Chat tool_choice only supports null, 'auto', or 'none' in this implementation.")
    if request.parallel_tool_calls not in (None, True):
        raise ValueError("Chat parallel_tool_calls=false is not implemented yet.")
    return None if request.tool_choice == "none" else tools


def _chat_request_prompt(tokenizer: Any, request: ChatCompletionRequest) -> str:
    return _chat_prompt(
        tokenizer,
        request.messages,
        resolve_chat_template_kwargs(request),
        resolve_chat_tools(request),
    )


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
    tools: list[dict[str, Any]] | None = None,
) -> str:
    chat = []
    for message in messages:
        rendered_message = {
            "role": _chat_template_role(message.role),
            "content": None if message.content is None else _chat_content_text(message.content),
        }
        if message.reasoning_content is not None:
            rendered_message["reasoning_content"] = message.reasoning_content
        if message.tool_calls is not None:
            rendered_message["tool_calls"] = _chat_template_tool_calls(message.tool_calls)
        if message.tool_call_id is not None:
            rendered_message["tool_call_id"] = message.tool_call_id
        chat.append(rendered_message)
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        kwargs.update(chat_template_kwargs or {})
        if tools:
            if not _supports_chat_template_kwarg(tokenizer, "tools"):
                raise ValueError("Tokenizer chat template does not support tools.")
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(chat, **kwargs)
    if any("reasoning_content" in message for message in chat):
        raise ValueError("reasoning_content requires a tokenizer chat_template.")
    if chat_template_kwargs:
        raise ValueError("chat_template_kwargs requires a tokenizer chat_template.")
    if tools:
        raise ValueError("tools requires a tokenizer chat_template with tools support.")
    if _messages_require_chat_template(chat):
        raise ValueError("Chat tool-call history requires a tokenizer chat_template.")

    rendered = []
    for message in chat:
        rendered.append(f"{message['role']}: {message['content'] or ''}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _chat_template_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for tool_call in tool_calls:
        normalized = dict(tool_call)
        function = dict(normalized["function"])
        try:
            arguments = json.loads(function["arguments"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"assistant tool call {function['name']!r} arguments is not valid JSON: {exc}"
            ) from exc
        if not isinstance(arguments, dict):
            raise ValueError(
                f"assistant tool call {function['name']!r} arguments must decode to a JSON object."
            )
        function["arguments"] = arguments
        normalized["function"] = function
        rendered.append(normalized)
    return rendered


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
    if _messages_require_chat_template(messages):
        raise ValueError("Responses tool-call history requires a tokenizer chat_template.")
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


def _messages_require_chat_template(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


def _supports_chat_template_kwarg(tokenizer: Any, name: str) -> bool:
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters
