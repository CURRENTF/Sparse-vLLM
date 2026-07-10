import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
from unittest.mock import AsyncMock


class _FastTokenizerAdapter:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.is_fast = True
        self.backend_tokenizer = tokenizer._tokenizer
        self.chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids, skip_special_tokens=True):
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _byte_level_tokenizer(*, special_tokens=()):
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["训练数据：中文，日本語，한국어，café，🙂。"],
        vocab_size=256,
        min_frequency=100,
        special_tokens=list(special_tokens),
    )
    return _FastTokenizerAdapter(tokenizer)


async def _dispatcher_items_for_text(text, *, stop=()):
    from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
    from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

    tokenizer = _byte_level_tokenizer()
    token_ids = tokenizer.encode(text)

    class Engine:
        def __init__(self):
            self.tokenizer = tokenizer
            self.last_step_token_outputs = []
            self.last_step_logprob_outputs = []
            self.aborted = []

        def abort_request(self, seq_id):
            self.aborted.append(seq_id)

        def exit(self):
            pass

    engine = Engine()
    dispatcher = AsyncEngineDispatcher(engine)
    output_queue = asyncio.Queue()
    active = {
        7: _ActiveRequest(
            index=0,
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
            prompt_token_ids=[10],
            max_tokens=len(token_ids),
            stop=list(stop),
            completion_token_ids=[],
            completion_token_logprobs=[],
            completion_top_logprobs=[],
            detokenizer=IncrementalDetokenizer(tokenizer),
        )
    }
    items = []
    try:
        for token_id in token_ids:
            engine.last_step_token_outputs = [(7, [token_id])]
            engine.last_step_logprob_outputs = [(7, [None], [None])]
            dispatcher._publish_token_deltas(active)
            await asyncio.sleep(0)
            while not output_queue.empty():
                items.append(output_queue.get_nowait())
            if 7 not in active:
                break

        if 7 in active:
            dispatcher._publish_finished(
                active,
                [(7, token_ids, [None] * len(token_ids), [None] * len(token_ids))],
            )
            await asyncio.sleep(0)
            while not output_queue.empty():
                items.append(output_queue.get_nowait())
    finally:
        dispatcher.close()
    return items


class _TestRequest:
    def __init__(self, app):
        self.app = app


def _route_endpoint(app, path):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
        if hasattr(route, "effective_route_contexts"):
            for context in route.effective_route_contexts():
                original_route = context.original_route
                if getattr(original_route, "path", None) == path:
                    return original_route.endpoint
    raise AssertionError(f"route not found: {path}")


def _response_sse_events(chunks):
    text = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in chunks)
    events = []
    for frame in text.split("\n\n"):
        if not frame:
            continue
        lines = frame.splitlines()
        data_lines = [line for line in lines if line.startswith("data: ")]
        if not data_lines:
            continue
        data = data_lines[-1].removeprefix("data: ")
        if data == "[DONE]":
            events.append(("[DONE]", None))
            continue
        event_lines = [line for line in lines if line.startswith("event: ")]
        event_type = event_lines[-1].removeprefix("event: ") if event_lines else None
        payload = json.loads(data)
        events.append((event_type, payload))
    return events


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("pydantic") is None,
    "OpenAI API server dependencies are not installed",
)
class OpenAIAPIServerTest(unittest.IsolatedAsyncioTestCase):
    def test_incremental_detokenizer_waits_for_complete_unicode(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")
        detokenizer = IncrementalDetokenizer(tokenizer)

        deltas = [detokenizer.push([token_id]) for token_id in token_ids]
        final = detokenizer.finish(token_ids)

        self.assertEqual("".join(delta.text for delta in deltas), "训练")
        self.assertEqual("".join(delta.raw_text for delta in deltas), "训练")
        self.assertNotIn("�", "".join(delta.text for delta in deltas))
        self.assertEqual(final.text, "训练")
        self.assertEqual(final.raw_text, "训练")
        self.assertEqual(final.text_delta, "")
        self.assertEqual(final.raw_text_delta, "")

    def test_incremental_detokenizer_handles_multilingual_batches(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        text = "中文，日本語，한국어，café，🙂。"
        token_ids = tokenizer.encode(text)
        detokenizer = IncrementalDetokenizer(tokenizer)

        midpoint = len(token_ids) // 2
        deltas = [
            detokenizer.push(token_ids[:midpoint]),
            detokenizer.push(token_ids[midpoint:]),
        ]

        self.assertEqual("".join(delta.text for delta in deltas), text)
        self.assertEqual(detokenizer.finish(token_ids).text, text)

    def test_incremental_detokenizer_keeps_raw_special_tokens(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer(special_tokens=["<special>"])
        token_id = tokenizer._tokenizer.token_to_id("<special>")
        detokenizer = IncrementalDetokenizer(tokenizer)

        delta = detokenizer.push([token_id])
        final = detokenizer.finish([token_id])

        self.assertEqual(delta.text, "")
        self.assertEqual(delta.raw_text, "<special>")
        self.assertEqual(final.text, "")
        self.assertEqual(final.raw_text, "<special>")

    def test_incremental_detokenizer_flushes_invalid_final_bytes(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训")[:2]
        detokenizer = IncrementalDetokenizer(tokenizer)

        delta = detokenizer.push(token_ids)
        final = detokenizer.finish(token_ids)

        self.assertEqual(delta.text, "")
        self.assertEqual(final.text, "�")
        self.assertEqual(final.text_delta, "�")

    def test_incremental_detokenizer_reports_unpublished_final_suffix(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")
        detokenizer = IncrementalDetokenizer(tokenizer)

        final = detokenizer.finish(token_ids)

        self.assertEqual(final.text, "训练")
        self.assertEqual(final.raw_text, "训练")
        self.assertEqual(final.text_delta, "训练")
        self.assertEqual(final.raw_text_delta, "训练")

    def test_incremental_detokenizer_rejects_non_fast_tokenizer(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        class SlowTokenizer:
            is_fast = False

        with self.assertRaisesRegex(TypeError, "fast tokenizer backend"):
            IncrementalDetokenizer(SlowTokenizer())

    async def test_dispatcher_rejects_slow_tokenizer_before_admission(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        class SlowTokenizer:
            is_fast = False

        class Engine:
            tokenizer = SlowTokenizer()

            def __init__(self):
                self.added = False

            def add_request(self, _prompt, _sampling_params):
                self.added = True
                return 1

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 1})(),
                0,
            )
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(item["type"], "error")
        self.assertIn("fast tokenizer backend", item["message"])
        self.assertFalse(engine.added)

    def test_incremental_detokenizer_rejects_final_token_mismatch(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        first = tokenizer.encode("训")
        second = tokenizer.encode("练")
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push(first)

        with self.assertRaisesRegex(RuntimeError, "token history mismatch"):
            detokenizer.finish(second)

    def test_incremental_detokenizers_keep_request_state_isolated(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        first_ids = tokenizer.encode("训练")
        second_ids = tokenizer.encode("中文")
        first = IncrementalDetokenizer(tokenizer)
        second = IncrementalDetokenizer(tokenizer)

        first_delta = first.push(first_ids[:3])
        second_delta = second.push(second_ids)
        final_delta = first.push(first_ids[3:])

        self.assertEqual(first_delta.text, "训")
        self.assertEqual(second_delta.text, "中文")
        self.assertEqual(final_delta.text, "练")
        self.assertEqual(first.finish(first_ids).text, "训练")
        self.assertEqual(second.finish(second_ids).text, "中文")

    def test_incremental_detokenizer_rejects_canonical_text_mismatch(self):
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("a")
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push(token_ids)
        tokenizer.decode = lambda _ids, skip_special_tokens=True: "different"

        with self.assertRaisesRegex(RuntimeError, "not a prefix"):
            detokenizer.finish(token_ids)

    async def test_completion_response_collects_usage_and_sorts_choices(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _completion_response

        queue_0 = asyncio.Queue()
        queue_1 = asyncio.Queue()
        await queue_0.put(
            {
                "type": "final",
                "index": 1,
                "text": "second",
                "finish_reason": "length",
                "prompt_tokens": 2,
                "completion_tokens": 3,
            }
        )
        await queue_1.put(
            {
                "type": "final",
                "index": 0,
                "text": "first",
                "finish_reason": "stop",
                "prompt_tokens": 5,
                "completion_tokens": 7,
            }
        )

        handles = [
            RequestHandle(output_queue=queue_0, cancelled=threading.Event()),
            RequestHandle(output_queue=queue_1, cancelled=threading.Event()),
        ]

        response = await _completion_response("cmpl-test", 123, "model-a", handles)

        self.assertEqual([choice["text"] for choice in response["choices"]], ["first", "second"])
        self.assertEqual(response["usage"], {"prompt_tokens": 7, "completion_tokens": 10, "total_tokens": 17})

    def test_sse_serializes_openai_data_frame(self):
        from sparsevllm.entrypoints.openai.api_server import _sse

        frame = _sse({"text": "hello"})

        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertEqual(json.loads(frame.removeprefix("data: ")), {"text": "hello"})

    def test_deltakv_serving_method_fails_fast(self):
        from sparsevllm.entrypoints.openai.api_server import _validate_serving_method

        with self.assertRaisesRegex(ValueError, "not supported"):
            _validate_serving_method({"vllm_sparse_method": "deltakv"})
        with self.assertRaisesRegex(ValueError, "not supported"):
            _validate_serving_method({"sparse_method": "deltakv-standalone"})

    def test_completion_request_rejects_unknown_fields(self):
        from pydantic import ValidationError

        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, CompletionRequest

        with self.assertRaises(ValidationError):
            CompletionRequest(model="m", prompt="p", suffix="ignored")
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "assistant", "content": "p", "tool_calls": []}],
                metadata={"trace": "x"},
            )

    def test_chat_request_accepts_claw_eval_tool_metadata(self):
        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="m",
            messages=[
                {"role": "user", "content": "p"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
            stream=True,
            stream_options={"include_usage": True},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertTrue(request.stream_options["include_usage"])
        self.assertEqual(ChatMessage(role="assistant").content, None)

    def test_chat_message_tool_fields_are_role_scoped(self):
        from pydantic import ValidationError

        from sparsevllm.entrypoints.openai.api_server import ChatMessage

        with self.assertRaisesRegex(ValidationError, "tool_calls is only valid"):
            ChatMessage(role="user", content="question", tool_calls=[])
        with self.assertRaisesRegex(ValidationError, "tool_call_id is only valid"):
            ChatMessage(role="assistant", content="answer", tool_call_id="call_1")
        with self.assertRaisesRegex(ValidationError, "require tool_call_id"):
            ChatMessage(role="tool", content="result")
        with self.assertRaisesRegex(ValidationError, "non-empty id"):
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            )

    def test_logprob_request_limits_match_openai_bounds(self):
        from pydantic import ValidationError

        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, CompletionRequest

        with self.assertRaises(ValidationError):
            CompletionRequest(model="m", prompt="p", logprobs=6)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                logprobs=True,
                top_logprobs=21,
            )

    def test_stop_with_logprobs_fails_fast(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            CompletionRequest,
            _validate_chat_request,
            _validate_request,
        )

        with self.assertRaises(HTTPException):
            _validate_request(
                CompletionRequest(model="m", prompt="p", stop="END", logprobs=1),
                "m",
            )
        with self.assertRaises(HTTPException):
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    stop="END",
                    logprobs=True,
                ),
                "m",
            )

    def test_chat_prompt_uses_fallback_without_chat_template(self):
        from sparsevllm.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = None

            def apply_chat_template(self, *_args, **_kwargs):
                raise AssertionError("chat_template is not configured")

        prompt = _chat_prompt(Tokenizer(), [ChatMessage(role="user", content="hello")])

        self.assertEqual(prompt, "user: hello\nassistant:")

    def test_chat_prompt_maps_developer_and_text_parts_for_templates(self):
        from sparsevllm.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None

            def apply_chat_template(self, chat, **_kwargs):
                self.chat = chat
                return "rendered"

        tokenizer = Tokenizer()
        prompt = _chat_prompt(
            tokenizer,
            [
                ChatMessage(
                    role="developer",
                    content=[
                        {"type": "text", "text": "policy"},
                        {"type": "text", "text": "details"},
                    ],
                )
            ],
        )

        self.assertEqual(prompt, "rendered")
        self.assertEqual(tokenizer.chat, [{"role": "system", "content": "policy\ndetails"}])

    def test_chat_prompt_preserves_reasoning_content_for_templates(self):
        from sparsevllm.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None

            def apply_chat_template(self, chat, **_kwargs):
                self.chat = chat
                return "rendered"

        tokenizer = Tokenizer()
        prompt = _chat_prompt(
            tokenizer,
            [
                ChatMessage(
                    role="assistant",
                    content="answer",
                    reasoning_content="reason",
                )
            ],
        )

        self.assertEqual(prompt, "rendered")
        self.assertEqual(
            tokenizer.chat,
            [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}],
        )

    def test_reasoning_content_requires_assistant_role_and_chat_template(self):
        from pydantic import ValidationError

        from sparsevllm.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        with self.assertRaisesRegex(ValidationError, "only valid for assistant"):
            ChatMessage(role="user", content="question", reasoning_content="reason")

        class Tokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "requires a tokenizer chat_template"):
            _chat_prompt(
                Tokenizer(),
                [ChatMessage(role="assistant", content="answer", reasoning_content="reason")],
            )

    def test_chat_template_kwargs_enable_thinking_passes_to_tokenizer(self):
        from sparsevllm.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, _chat, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        prompt = _chat_prompt(
            tokenizer,
            [ChatMessage(role="user", content="hello")],
            {"enable_thinking": False},
        )

        self.assertEqual(prompt, "rendered")
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_top_level_enable_thinking_resolves_to_chat_template_kwargs(self):
        from sparsevllm.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            resolve_chat_template_kwargs,
        )

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            enable_thinking=False,
        )

        self.assertEqual(resolve_chat_template_kwargs(request), {"enable_thinking": False})

    def test_top_level_enable_thinking_conflict_fails_fast(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        class Tokenizer:
            chat_template = "template"

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": True},
        )

        with self.assertRaisesRegex(HTTPException, "conflicts") as ctx:
            _validate_chat_request(request, "m", Tokenizer())
        self.assertEqual(ctx.exception.status_code, 400)

    def test_chat_reasoning_effort_controls_thinking(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _validate_chat_request,
            resolve_chat_template_kwargs,
        )

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            reasoning_effort="none",
        )
        self.assertEqual(resolve_chat_template_kwargs(request), {"enable_thinking": False})

        conflicting = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            reasoning_effort="high",
            enable_thinking=False,
        )
        with self.assertRaisesRegex(HTTPException, "conflicts"):
            _validate_chat_request(conflicting, "m")

    def test_chat_prompt_passes_tools_and_tool_history(self):
        from sparsevllm.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
        )

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None
                self.tools = None

            def apply_chat_template(self, chat, tools=None, **_kwargs):
                self.chat = chat
                self.tools = tools
                return "rendered"

        request = ChatCompletionRequest(
            model="m",
            messages=[
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "need a lookup",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        tokenizer = Tokenizer()

        self.assertEqual(_chat_request_prompt(tokenizer, request), "rendered")
        self.assertEqual(tokenizer.chat[1]["reasoning_content"], "need a lookup")
        self.assertEqual(tokenizer.chat[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(tokenizer.chat[2]["tool_call_id"], "call_1")
        self.assertEqual(tokenizer.tools[0]["name"], "get_weather")

    def test_chat_tool_controls_and_template_support_fail_fast(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
            _validate_chat_request,
        )

        tool = {"type": "function", "function": {"name": "search", "parameters": {}}}
        for request in (
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                tools=[tool],
                tool_choice="required",
            ),
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                tools=[tool],
                parallel_tool_calls=False,
            ),
        ):
            with self.assertRaises(HTTPException):
                _validate_chat_request(request, "m")

        class NoTemplateTokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "tools requires"):
            _chat_request_prompt(
                NoTemplateTokenizer(),
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    tools=[tool],
                ),
            )

        class NoToolsTokenizer:
            chat_template = "template"

            def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
                del chat, tokenize, add_generation_prompt
                return "rendered"

        with self.assertRaisesRegex(ValueError, "does not support tools"):
            _chat_request_prompt(
                NoToolsTokenizer(),
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    tools=[tool],
                ),
            )

    def test_chat_template_kwargs_validation_is_explicit(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        class Tokenizer:
            chat_template = "template"

        with self.assertRaises(HTTPException) as unknown_ctx:
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    chat_template_kwargs={"unknown": True},
                ),
                "m",
                Tokenizer(),
            )
        self.assertEqual(unknown_ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as type_ctx:
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    chat_template_kwargs={"enable_thinking": "false"},
                ),
                "m",
                Tokenizer(),
            )
        self.assertEqual(type_ctx.exception.status_code, 400)

        class NoTemplateTokenizer:
            chat_template = None

        with self.assertRaises(HTTPException) as template_ctx:
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    chat_template_kwargs={"enable_thinking": False},
                ),
                "m",
                NoTemplateTokenizer(),
            )
        self.assertEqual(template_ctx.exception.status_code, 400)

    def test_chat_max_completion_tokens_maps_to_sampling_params(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _sampling_params_from_request,
            _validate_chat_request,
        )

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            max_completion_tokens=32,
        )

        _validate_chat_request(request, "m")
        self.assertEqual(_sampling_params_from_request(request).max_tokens, 32)

        with self.assertRaises(HTTPException):
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    max_tokens=16,
                    max_completion_tokens=32,
                ),
                "m",
            )

    def test_missing_non_bool_engine_arg_fails_fast(self):
        from sparsevllm.entrypoints.openai.api_server import _parse_engine_kwargs

        with self.assertRaisesRegex(ValueError, "Missing value"):
            _parse_engine_kwargs(["--max-model-len"])

        with self.assertRaisesRegex(ValueError, "Unknown Sparse-vLLM engine argument"):
            _parse_engine_kwargs(["--observation-layers", "0"])

        with self.assertRaisesRegex(ValueError, "Unknown Sparse-vLLM engine argument"):
            _parse_engine_kwargs(["--obs-layer-ids", "0"])

        self.assertEqual(_parse_engine_kwargs(["--sparse-method", "snapkv"]), {"sparse_method": "snapkv"})

    def test_create_app_disables_periodic_throughput_logs_by_default(self):
        from sparsevllm.entrypoints.openai import api_server

        class Engine:
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def __init__(self, _model, **kwargs):
                self.kwargs = kwargs

            def exit(self):
                pass

        with patch.object(api_server, "LLM", Engine):
            app = api_server.create_app("/tmp/model", served_model_name="model")
            try:
                self.assertEqual(app.state.dispatcher.engine.kwargs["throughput_log_interval_s"], 0.0)
            finally:
                app.state.dispatcher.close()

        with patch.object(api_server, "LLM", Engine):
            app = api_server.create_app(
                "/tmp/model",
                {"throughput_log_interval_s": 5.0},
                served_model_name="model",
            )
            try:
                self.assertEqual(app.state.dispatcher.engine.kwargs["throughput_log_interval_s"], 5.0)
            finally:
                app.state.dispatcher.close()

    async def test_cancel_during_admission_aborts_after_seq_id_exists(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.add_started = threading.Event()
                self.release_add = threading.Event()
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                self.add_started.set()
                self.release_add.wait(timeout=5)
                return 123

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def step(self):
                raise AssertionError("cancelled request should not be stepped")

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit("prompt", object(), 0)
            self.assertTrue(await asyncio.to_thread(engine.add_started.wait, 5))
            dispatcher.cancel(handle)
            engine.release_add.set()
            await asyncio.sleep(0.1)
            self.assertEqual(engine.aborted, [123])
        finally:
            dispatcher.close()

    async def test_prefix_cache_inspect_route_uses_dispatcher_control_queue(self):
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1, 2]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def prefix_cache_inspect(self, token_ids, include_subtree=False):
                return {
                    "token_ids": list(token_ids),
                    "include_subtree": bool(include_subtree),
                    "thread": threading.current_thread().name,
                }

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            response = await endpoint(
                api_server.PrefixCacheInspectRequest(token_ids=[7, 8], include_subtree=True),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        payload = json.loads(response.body)
        self.assertEqual(payload["token_ids"], [7, 8])
        self.assertTrue(payload["include_subtree"])
        self.assertEqual(payload["thread"], "sparsevllm-openai-dispatcher")

    async def test_prefix_cache_match_accepts_chat_messages(self):
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None
            chat_template = "template"

            def apply_chat_template(self, chat, **_kwargs):
                return "|".join(f"{item['role']}:{item['content']}" for item in chat)

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def prefix_cache_match(self, token_ids):
                return {
                    "token_ids": list(token_ids),
                    "thread": threading.current_thread().name,
                    "supported": True,
                    "enabled": True,
                }

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        try:
            response = await endpoint(
                api_server.PrefixCacheMatchRequest(messages=[{"role": "user", "content": "hello"}]),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        payload = json.loads(response.body)
        self.assertEqual(payload["token_ids"], [ord(ch) for ch in "user:hello"])
        self.assertEqual(payload["thread"], "sparsevllm-openai-dispatcher")

    async def test_worker_info_and_load_routes(self):
        from sparsevllm.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def worker_info(self, served_model_name=None, tags=None):
                return {"served_model_name": served_model_name, "tags": list(tags or [])}

            def worker_load(self):
                return {"active_requests": 3, "thread": threading.current_thread().name}

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        info_endpoint = _route_endpoint(app, "/v1/worker/info")
        load_endpoint = _route_endpoint(app, "/v1/worker/load")
        try:
            with patch.dict(os.environ, {"SPARSEVLLM_WORKER_TAGS": "dialog, omnikv"}):
                info_response = info_endpoint(_TestRequest(app))
            load_response = await load_endpoint(_TestRequest(app))
        finally:
            app.state.dispatcher.close()

        self.assertEqual(json.loads(info_response.body), {"served_model_name": "model", "tags": ["dialog", "omnikv"]})
        self.assertEqual(json.loads(load_response.body), {"active_requests": 3, "thread": "sparsevllm-openai-dispatcher"})

    async def test_prefix_cache_text_selector_tokenizes_server_side(self):
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = "<s>"

            def __init__(self):
                self.calls = []

            def encode(self, text, add_special_tokens=False):
                self.calls.append((text, add_special_tokens))
                return [0, 11] if add_special_tokens else [11]

        class Engine:
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def __init__(self):
                self.tokenizer = Tokenizer()

            def prefix_cache_inspect(self, token_ids, include_subtree=False):
                del include_subtree
                return {"token_ids": list(token_ids), "calls": list(self.tokenizer.calls)}

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            response = await endpoint(api_server.PrefixCacheInspectRequest(text="hello"), _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        payload = json.loads(response.body)
        self.assertEqual(payload["token_ids"], [0, 11])
        self.assertEqual(payload["calls"], [["hello", True]])

    async def test_prefix_cache_selector_rejects_both_or_neither(self):
        from fastapi import HTTPException
        from sparsevllm.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            with self.assertRaises(HTTPException):
                await endpoint(api_server.PrefixCacheInspectRequest(), _TestRequest(app))
            with self.assertRaises(HTTPException):
                await endpoint(api_server.PrefixCacheInspectRequest(token_ids=[1], text="x"), _TestRequest(app))
        finally:
            app.state.dispatcher.close()

    async def test_prefix_cache_disabled_error_is_explicit(self):
        from fastapi import HTTPException
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def prefix_cache_inspect(self, token_ids, include_subtree=False):
                del token_ids, include_subtree
                raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            with self.assertRaises(HTTPException) as ctx:
                await endpoint(api_server.PrefixCacheInspectRequest(token_ids=[1]), _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("prefix cache is not enabled", ctx.exception.detail)

    async def test_prefix_cache_delete_and_priority_routes_are_synchronous_controls(self):
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def __init__(self):
                self.calls = []

            def prefix_cache_delete_subtree(self, token_ids):
                self.calls.append(("delete", list(token_ids), threading.current_thread().name))
                return {
                    "deleted_block_ids": ["aa"],
                    "deleted_block_count": 1,
                    "blocked_blocks": [{"block_id": "bb", "reason": "referenced"}],
                }

            def prefix_cache_set_eviction_priority(self, token_ids, priority):
                self.calls.append(("priority", list(token_ids), int(priority), threading.current_thread().name))
                return {
                    "matched": True,
                    "root_block_id": "aa",
                    "updated_block_count": 2,
                    "eviction_priority": int(priority),
                }

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        delete_endpoint = _route_endpoint(app, "/v1/prefix_cache/delete_subtree")
        priority_endpoint = _route_endpoint(app, "/v1/prefix_cache/set_eviction_priority")
        try:
            delete_response = await delete_endpoint(
                api_server.PrefixCacheDeleteSubtreeRequest(text="ab"),
                _TestRequest(app),
            )
            priority_response = await priority_endpoint(
                api_server.PrefixCacheSetEvictionPriorityRequest(token_ids=[7, 8], priority=-5),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        self.assertEqual(json.loads(delete_response.body)["blocked_blocks"][0]["reason"], "referenced")
        self.assertEqual(json.loads(priority_response.body)["eviction_priority"], -5)
        self.assertEqual(
            engine.calls,
            [
                ("delete", [97, 98], "sparsevllm-openai-dispatcher"),
                ("priority", [7, 8], -5, "sparsevllm-openai-dispatcher"),
            ],
        )

    async def test_step_failure_aborts_active_request(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        test_tokenizer = _byte_level_tokenizer()

        class Engine:
            tokenizer = test_tokenizer
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def __init__(self):
                self.step_started = threading.Event()
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 77

            def step(self):
                self.step_started.set()
                raise RuntimeError("boom")

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit("prompt", type("Sampling", (), {"max_tokens": 4})(), 0)
            self.assertTrue(await asyncio.to_thread(engine.step_started.wait, 2))
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            self.assertEqual(item["type"], "error")
            self.assertEqual(engine.aborted, [77])
        finally:
            dispatcher.close()

    async def test_final_detokenizer_error_reaches_client(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        streamed_token_id = tokenizer.encode("a")[0]
        mismatched_final_id = tokenizer.encode("b")[0]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 7

            def step(self):
                self.last_step_token_outputs = [(7, [streamed_token_id])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
                return [(7, [mismatched_final_id], [None], [None])], 0

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 1})(),
                0,
            )
            token_item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            error_item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["type"], "token")
        self.assertEqual(error_item["type"], "error")
        self.assertIn("token history mismatch", error_item["message"])
        self.assertEqual(engine.aborted, [7])

    async def test_stop_detokenizer_error_reaches_client(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        token_id = tokenizer.encode("a")[0]
        tokenizer.decode = lambda _ids, skip_special_tokens=True: "different"

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 7

            def step(self):
                self.last_step_token_outputs = [(7, [token_id])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
                return [], 0

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 2})(),
                0,
                ["a"],
            )
            error_item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(error_item["type"], "error")
        self.assertIn("not a prefix", error_item["message"])
        self.assertEqual(engine.aborted, [7])

    async def test_dispatcher_close_times_out_blocked_step_and_exits_engine(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        test_tokenizer = _byte_level_tokenizer()

        class Engine:
            tokenizer = test_tokenizer

            def __init__(self):
                self.step_started = threading.Event()
                self.release_step = threading.Event()
                self.exited = threading.Event()
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 123

            def step(self):
                self.step_started.set()
                self.release_step.wait()
                return [], 0

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                self.exited.set()

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            await dispatcher.submit("prompt", type("Sampling", (), {"max_tokens": 1000})(), 0)
            self.assertTrue(await asyncio.to_thread(engine.step_started.wait, 2))
            with patch.dict(os.environ, {"SPARSEVLLM_OPENAI_SHUTDOWN_TIMEOUT_S": "0.05"}):
                started = time.perf_counter()
                dispatcher.close()
                elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertTrue(engine.exited.is_set())
            self.assertTrue(dispatcher._thread.is_alive())
        finally:
            engine.release_step.set()
            dispatcher._thread.join(timeout=1.0)

    async def test_completion_route_error_cancels_sibling_handles(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _completion_response

        queue_0 = asyncio.Queue()
        queue_1 = asyncio.Queue()
        handle_0 = RequestHandle(output_queue=queue_0, cancelled=threading.Event())
        handle_1 = RequestHandle(output_queue=queue_1, cancelled=threading.Event())
        await queue_0.put({"type": "error", "message": "failed"})

        with self.assertRaises(HTTPException):
            try:
                await _completion_response("cmpl-test", 123, "model-a", [handle_0, handle_1])
            except Exception:
                for handle in (handle_0, handle_1):
                    handle.cancelled.set()
                raise

        self.assertTrue(handle_0.cancelled.is_set())
        self.assertTrue(handle_1.cancelled.is_set())

    async def test_non_streaming_cancel_logs_request_cancel(self):
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            def encode(self, _prompt):
                return [1]

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(str(token_id) for token_id in token_ids)

        class Engine:
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def __init__(self):
                self.tokenizer = Tokenizer()
                self.last_step_token_outputs = []

            def add_request(self, _prompt, _sampling_params):
                return 1

            def abort_request(self, _seq_id):
                pass

            def step(self):
                return [], 0

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/completions")
        request = api_server.CompletionRequest(model="model", prompt="p")
        try:
            from sparsevllm.entrypoints.openai.serving import completion as completion_serving

            with patch.object(
                completion_serving,
                "_completion_response",
                AsyncMock(side_effect=asyncio.CancelledError),
            ), patch.object(completion_serving.logger, "info") as log_info:
                with self.assertRaises(asyncio.CancelledError):
                    await endpoint(request, _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn("request_cancel id={} model={} stream=false elapsed_s={:.3f}", messages)

    async def test_dispatcher_streaming_delta_uses_cumulative_suffix(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("ab")

        class Engine:
            last_step_token_outputs = [(7, [token_ids[1]])]
            last_step_logprob_outputs = [(7, [None], [None])]

            def __init__(self):
                self.tokenizer = tokenizer

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        output_queue = asyncio.Queue()
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push([token_ids[0]])
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=2,
                stop=[],
                completion_token_ids=[token_ids[0]],
                completion_token_logprobs=[None],
                completion_top_logprobs=[None],
                detokenizer=detokenizer,
                emitted_text_len=1,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(item["text"], "b")
        self.assertEqual(item["raw_text_delta"], "b")
        self.assertEqual(active[7].completion_token_ids, token_ids)

    async def test_dispatcher_streams_complete_unicode_with_pending_logprobs(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
            )
        }
        token_logprobs = [-float(index + 1) for index in range(len(token_ids))]
        top_logprobs = [{token_id: value} for token_id, value in zip(token_ids, token_logprobs)]
        token_items = []
        try:
            for token_id, token_logprob, top_logprob in zip(
                token_ids,
                token_logprobs,
                top_logprobs,
            ):
                engine.last_step_token_outputs = [(7, [token_id])]
                engine.last_step_logprob_outputs = [(7, [token_logprob], [top_logprob])]
                dispatcher._publish_token_deltas(active)
                await asyncio.sleep(0)
                while not output_queue.empty():
                    token_items.append(output_queue.get_nowait())

            dispatcher._publish_finished(
                active,
                [(7, token_ids, token_logprobs, top_logprobs)],
            )
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual([item["text"] for item in token_items], ["训", "练"])
        self.assertEqual("".join(item["text"] for item in token_items), "训练")
        self.assertNotIn("�", "".join(item["text"] for item in token_items))
        self.assertEqual([len(item["token_ids"]) for item in token_items], [3, 3])
        self.assertEqual(
            [value for item in token_items for value in item["token_logprobs"]],
            token_logprobs,
        )
        self.assertEqual(final_item["text"], "训练")
        self.assertEqual(final_item["raw_text"], "训练")
        self.assertEqual(final_item["text_delta"], "")

    async def test_final_reconciliation_publishes_pending_logprobs(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训")[:2]
        token_logprobs = [-1.0, -2.0]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
            )
        }
        try:
            for token_id, logprob in zip(token_ids, token_logprobs):
                engine.last_step_token_outputs = [(7, [token_id])]
                engine.last_step_logprob_outputs = [(7, [logprob], [None])]
                dispatcher._publish_token_deltas(active)
            self.assertTrue(output_queue.empty())

            dispatcher._publish_finished(
                active,
                [(7, token_ids, token_logprobs, [None, None])],
            )
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["type"], "token")
        self.assertEqual(token_item["text"], "�")
        self.assertEqual(token_item["token_ids"], token_ids)
        self.assertEqual(token_item["token_logprobs"], token_logprobs)
        self.assertEqual(final_item["type"], "final")
        self.assertEqual(final_item["text"], "�")
        self.assertEqual(final_item["text_delta"], "")

    async def test_final_suffix_publishes_token_metadata(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")
        token_logprobs = [-float(index + 1) for index in range(len(token_ids))]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
            )
        }
        try:
            dispatcher._publish_finished(
                active,
                [(7, token_ids, token_logprobs, [None] * len(token_ids))],
            )
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["type"], "token")
        self.assertEqual(token_item["text"], "训练")
        self.assertEqual(token_item["token_ids"], token_ids)
        self.assertEqual(token_item["token_logprobs"], token_logprobs)
        self.assertEqual(final_item["type"], "final")
        self.assertEqual(final_item["text_delta"], "")

    async def test_stream_logprobs_include_raw_only_special_token(self):
        from sparsevllm.entrypoints.openai.api_server import (
            RequestHandle,
            _chat_completion_stream,
            _completion_stream,
        )

        tokenizer = _byte_level_tokenizer(special_tokens=["<special>"])
        token_id = tokenizer._tokenizer.token_to_id("<special>")
        items = [
            {
                "type": "token",
                "index": 0,
                "text": "",
                "raw_text_delta": "<special>",
                "token_ids": [token_id],
                "token_logprobs": [-0.5],
                "top_logprobs": [None],
            },
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<special>",
                "text_delta": "",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [token_id],
                "token_logprobs": [-0.5],
                "top_logprobs": [None],
            },
        ]

        def handle():
            queue = asyncio.Queue()
            for item in copy.deepcopy(items):
                queue.put_nowait(item)
            return RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        completion_chunks = [
            chunk
            async for chunk in _completion_stream(
                Dispatcher(),
                "cmpl-test",
                123,
                "model",
                [handle()],
                tokenizer=tokenizer,
            )
        ]
        chat_chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle()],
                tokenizer=tokenizer,
            )
        ]
        completion_payloads = [
            payload
            for event, payload in _response_sse_events(completion_chunks)
            if event != "[DONE]"
        ]
        chat_payloads = [
            payload
            for event, payload in _response_sse_events(chat_chunks)
            if event != "[DONE]"
        ]

        self.assertIsNotNone(completion_payloads[0]["choices"][0]["logprobs"])
        self.assertEqual(completion_payloads[0]["choices"][0]["text"], "")
        self.assertIsNotNone(chat_payloads[1]["choices"][0]["logprobs"])
        self.assertEqual(chat_payloads[1]["choices"][0]["delta"]["content"], "")

    async def test_unicode_streams_match_non_streaming_endpoints(self):
        from sparsevllm.entrypoints.openai.api_server import (
            RequestHandle,
            ResponseRequest,
            _chat_completion_response,
            _chat_completion_stream,
            _completion_response,
            _completion_stream,
            _response_response,
            _response_stream,
        )

        text = "中文，日本語，한국어，café，🙂。"
        items = await _dispatcher_items_for_text(text)

        def handle():
            queue = asyncio.Queue()
            for item in copy.deepcopy(items):
                queue.put_nowait(item)
            return RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        completion_chunks = [
            chunk
            async for chunk in _completion_stream(
                Dispatcher(),
                "cmpl-test",
                123,
                "model",
                [handle()],
            )
        ]
        completion_text = "".join(
            payload["choices"][0]["text"]
            for event, payload in _response_sse_events(completion_chunks)
            if event != "[DONE]"
        )
        completion_final = await _completion_response("cmpl-test", 123, "model", [handle()])

        chat_chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle()],
            )
        ]
        chat_text = "".join(
            (payload["choices"][0].get("delta") or {}).get("content", "")
            for event, payload in _response_sse_events(chat_chunks)
            if event != "[DONE]" and payload.get("choices")
        )
        chat_final = await _chat_completion_response("chatcmpl-test", 123, "model", [handle()])

        request = ResponseRequest(model="model", input="hello", stream=True)
        response_chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                handle(),
                time.perf_counter(),
                None,
                request,
                reasoning_parser_name=None,
            )
        ]
        response_events = _response_sse_events(response_chunks)
        response_text = "".join(
            payload["delta"]
            for event, payload in response_events
            if event == "response.output_text.delta"
        )
        response_final = await _response_response(
            "resp_test",
            123,
            "model",
            handle(),
            reasoning_parser_name=None,
        )

        self.assertEqual(completion_text, text)
        self.assertEqual(completion_text, completion_final["choices"][0]["text"])
        self.assertEqual(chat_text, text)
        self.assertEqual(chat_text, chat_final["choices"][0]["message"]["content"])
        self.assertEqual(response_text, text)
        self.assertEqual(response_text, response_final["output"][0]["content"][0]["text"])
        self.assertNotIn(b"\xef\xbf\xbd", "".join(response_chunks).encode("utf-8"))

    async def test_unicode_response_reasoning_and_tool_call_stream(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        raw_text = (
            '<think>中文推理</think><tool_call>{"name":"查询",'
            '"arguments":{"城市":"北京"}}</tool_call>'
        )
        items = await _dispatcher_items_for_text(raw_text)
        queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                RequestHandle(output_queue=queue, cancelled=threading.Event()),
                time.perf_counter(),
                None,
                ResponseRequest(model="model", input="hello", stream=True),
                reasoning_parser_name="qwen3",
            )
        ]
        events = _response_sse_events(chunks)
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][0]["text"], "中文推理")
        self.assertEqual(completed["output"][1]["type"], "function_call")
        self.assertEqual(completed["output"][1]["name"], "查询")
        self.assertEqual(completed["output"][1]["arguments"], '{"城市":"北京"}')
        self.assertNotIn(b"\xef\xbf\xbd", "".join(chunks).encode("utf-8"))

    async def test_unicode_stop_is_not_exposed(self):
        items = await _dispatcher_items_for_text("回答训练结束忽略", stop=["结束"])
        streamed_text = "".join(item.get("text", "") for item in items if item["type"] == "token")
        final = [item for item in items if item["type"] == "final"][0]

        self.assertEqual(streamed_text, "回答训练")
        self.assertEqual(final["text"], "回答训练")
        self.assertNotIn("结束", streamed_text)

    async def test_streaming_finish_log_includes_tps_metrics(self):
        from sparsevllm.entrypoints.openai import api_server

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "x",
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "x",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )
        handle = api_server.RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        with patch.object(api_server.logger, "info") as log_info:
            chunks = [
                chunk
                async for chunk in api_server._completion_stream(
                    Dispatcher(),
                    "cmpl-test",
                    123,
                    "model",
                    [handle],
                    started=time.perf_counter() - 1.0,
                )
            ]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn(
            "request_finish id={} model={} stream=true prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
            messages,
        )

    async def test_dispatcher_stop_buffers_partial_stop_prefix(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparsevllm.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("abSTOP")

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = [(7, [token_ids[1]])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
                self.aborted = []

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push([token_ids[0]])
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=["bSTOP"],
                completion_token_ids=[token_ids[0]],
                completion_token_logprobs=[None],
                completion_top_logprobs=[None],
                detokenizer=detokenizer,
                emitted_text_len=0,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            self.assertEqual(token_item["text"], "a")
            engine.last_step_token_outputs = [(7, token_ids[2:])]
            engine.last_step_logprob_outputs = [
                (7, [None] * len(token_ids[2:]), [None] * len(token_ids[2:]))
            ]
            dispatcher._publish_token_deltas(active)
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(final_item["type"], "final")
        self.assertEqual(final_item["text"], "a")
        self.assertEqual(final_item["text_delta"], "")
        self.assertEqual(engine.aborted, [7])

    async def test_chat_completion_response_shape(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(response["usage"], {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})

    async def test_chat_completion_parses_qwen3_reasoning(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": "reason</think>\n\nanswer<|im_end|>",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "token_ids": [1, 2, 3],
                "token_logprobs": [None, None, None],
                "top_logprobs": [None, None, None],
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            reasoning_parser_name="qwen3",
        )

        choice = response["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "reason")
        self.assertEqual(choice["message"]["content"], "answer")
        self.assertEqual(choice["finish_reason"], "stop")

    async def test_chat_completion_parses_reasoning_and_tool_calls(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        raw_text = (
            "<think>need weather</think>"
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        )
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": raw_text,
                "raw_text": raw_text,
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "token_ids": [1, 2, 3, 4, 5],
                "token_logprobs": [None] * 5,
                "top_logprobs": [None] * 5,
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            reasoning_parser_name="qwen3",
            parse_tools=True,
        )

        choice = response["choices"][0]
        message = choice["message"]
        self.assertEqual(message["reasoning_content"], "need weather")
        self.assertIsNone(message["content"])
        self.assertTrue(message["tool_calls"][0]["id"].startswith("call_"))
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"city":"Paris"}')
        self.assertEqual(choice["finish_reason"], "tool_calls")

    async def test_chat_completion_parses_multiple_tool_calls(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        text = (
            '<tool_call>{"name":"first","arguments":{"x":1}}</tool_call>'
            '<tool_call>{"name":"second","arguments":{"y":2}}</tool_call>'
        )
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": text,
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            parse_tools=True,
        )

        calls = response["choices"][0]["message"]["tool_calls"]
        self.assertEqual([call["function"]["name"] for call in calls], ["first", "second"])

    async def test_chat_completion_reasoning_length_remains_explicit(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "length",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            reasoning_parser_name="qwen3",
        )

        choice = response["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "partial")
        self.assertEqual(choice["message"]["content"], "")
        self.assertEqual(choice["finish_reason"], "length")

    async def test_chat_completion_parse_failures_are_explicit(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        async def parse(text, *, reasoning_parser_name=None, parse_tools=False):
            queue = asyncio.Queue()
            await queue.put(
                {
                    "type": "final",
                    "index": 0,
                    "text": text,
                    "raw_text": text,
                    "finish_reason": "stop",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
            return await _chat_completion_response(
                "chatcmpl-test",
                123,
                "model",
                [RequestHandle(output_queue=queue, cancelled=threading.Event())],
                reasoning_parser_name=reasoning_parser_name,
                parse_tools=parse_tools,
            )

        with self.assertRaisesRegex(HTTPException, "did not close"):
            await parse("<think>partial", reasoning_parser_name="qwen3")
        with self.assertRaisesRegex(HTTPException, "JSON parse failed"):
            await parse('<tool_call>{"name":</tool_call>', parse_tools=True)

    def test_chat_logprobs_reject_parsed_outputs(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        base = {
            "model": "m",
            "messages": [{"role": "user", "content": "p"}],
            "logprobs": True,
        }
        with self.assertRaisesRegex(HTTPException, "cannot be aligned"):
            _validate_chat_request(
                ChatCompletionRequest(**base),
                "m",
                reasoning_parser_name="qwen3",
            )
        with self.assertRaisesRegex(HTTPException, "cannot be aligned"):
            _validate_chat_request(
                ChatCompletionRequest(
                    **base,
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "search", "parameters": {}},
                        }
                    ],
                ),
                "m",
            )

    async def test_chat_stream_starts_with_assistant_role(self):
        from sparsevllm.entrypoints.openai import api_server

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "text_delta": "",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 0,
                "token_ids": [],
                "token_logprobs": [],
                "top_logprobs": [],
            }
        )
        handle = api_server.RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in api_server._chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle],
            )
        ]

        first = json.loads(chunks[0].removeprefix("data: "))
        self.assertEqual(first["choices"][0]["delta"], {"role": "assistant"})

    def test_completion_logprobs_serializes_sampled_tokens(self):
        from sparsevllm.entrypoints.openai.api_server import _completion_logprobs

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return {1: "a", 2: "b", 3: "c"}[token_ids[0]]

        logprobs = _completion_logprobs(
            Tokenizer(),
            [1, 2],
            [-0.1, -0.2],
            [{1: -0.1, 3: -1.0}, None],
        )

        self.assertEqual(logprobs["tokens"], ["a", "b"])
        self.assertEqual(logprobs["token_logprobs"], [-0.1, -0.2])
        self.assertEqual(logprobs["top_logprobs"][0], {"a": -0.1, "c": -1.0})

    def test_chat_logprobs_true_requests_sampled_logprobs(self):
        from sparsevllm.entrypoints.openai.api_server import ChatCompletionRequest, _sampling_params_from_request

        request = ChatCompletionRequest(
            model="model",
            messages=[{"role": "user", "content": "hello"}],
            logprobs=True,
        )

        self.assertEqual(_sampling_params_from_request(request).logprobs, 0)

    def test_response_prompt_renders_string_input_and_instructions(self):
        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None
                self.kwargs = None

            def apply_chat_template(self, chat, **kwargs):
                self.chat = chat
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        request = ResponseRequest(
            model="model",
            instructions="policy",
            input="hello",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(_response_prompt(tokenizer, request), "rendered")
        self.assertEqual(
            tokenizer.chat,
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_response_prompt_rejects_unsupported_item_type(self):
        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "Unsupported responses input item type"):
            _response_prompt(
                Tokenizer(),
                ResponseRequest(model="model", input=[{"type": "image", "image_url": "x"}]),
            )

    def test_response_prompt_passes_tools_and_tool_outputs(self):
        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None
                self.tools = None

            def apply_chat_template(self, chat, tools=None, **_kwargs):
                self.chat = chat
                self.tools = tools
                return "rendered"

        tokenizer = Tokenizer()
        request = ResponseRequest(
            model="model",
            input=[{"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertEqual(_response_prompt(tokenizer, request), "rendered")
        self.assertEqual(tokenizer.chat, [{"role": "tool", "content": '{"ok":true}', "tool_call_id": "call_1"}])
        self.assertEqual(tokenizer.tools[0]["name"], "get_weather")

    def test_response_prompt_rejects_tools_without_template_support(self):
        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
                del chat, tokenize, add_generation_prompt
                return "rendered"

        with self.assertRaisesRegex(ValueError, "does not support tools"):
            _response_prompt(
                Tokenizer(),
                ResponseRequest(
                    model="model",
                    input="hello",
                    tools=[{"type": "function", "name": "tool", "parameters": {}}],
                ),
            )

    def test_response_prompt_rejects_tool_history_without_template(self):
        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = None

        for item in [
            {"type": "function_call", "call_id": "call_1", "name": "tool", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
        ]:
            with self.assertRaisesRegex(ValueError, "tool-call history requires"):
                _response_prompt(Tokenizer(), ResponseRequest(model="model", input=[item]))

    def test_response_reasoning_effort_conflicts_fail_fast(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _validate_response_request

        with self.assertRaises(HTTPException) as ctx:
            _validate_response_request(
                ResponseRequest(
                    model="model",
                    input="hello",
                    reasoning={"effort": "none"},
                    chat_template_kwargs={"enable_thinking": True},
                ),
                "model",
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_response_unimplemented_control_fields_fail_fast(self):
        from fastapi import HTTPException

        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _validate_response_request

        for request in [
            ResponseRequest(model="model", input="hello", tool_choice="required"),
            ResponseRequest(model="model", input="hello", parallel_tool_calls=False),
            ResponseRequest(model="model", input="hello", reasoning={"summary": "auto"}),
        ]:
            with self.assertRaises(HTTPException) as ctx:
                _validate_response_request(request, "model")
            self.assertEqual(ctx.exception.status_code, 400)

    def test_response_max_output_tokens_maps_to_sampling_params(self):
        from sparsevllm.entrypoints.openai.api_server import ResponseRequest, _sampling_params_from_response_request

        request = ResponseRequest(model="model", input="hello", max_output_tokens=7)

        self.assertEqual(_sampling_params_from_response_request(request).max_tokens, 7)

    async def test_response_response_shape_and_usage(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _response_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 2,
            }
        )

        response = await _response_response(
            "resp_test",
            123,
            "model",
            RequestHandle(output_queue=queue, cancelled=threading.Event()),
            reasoning_parser_name=None,
        )

        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(response["usage"], {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})

    async def test_response_reasoning_parser_uses_raw_text(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, _response_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": "<think>reason</think>answer",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 3,
            }
        )

        response = await _response_response(
            "resp_test",
            123,
            "model",
            RequestHandle(output_queue=queue, cancelled=threading.Event()),
            reasoning_parser_name="qwen3",
        )

        self.assertEqual(response["output"][0]["type"], "reasoning")
        self.assertEqual(response["output"][0]["text"], "reason")
        self.assertEqual(response["output"][1]["content"][0]["text"], "answer")

    async def test_response_stream_true_returns_responses_sse(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest
        from sparsevllm.entrypoints.openai.serving.responses import serve_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "hel",
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            async def submit(self, prompt, _sampling_params, index, stop):
                self.prompt = prompt
                self.index = index
                self.stop = stop
                return RequestHandle(output_queue=queue, cancelled=threading.Event())

            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        class Tokenizer:
            chat_template = None

        response = await serve_response(
            ResponseRequest(model="model", input="hello", stream=True),
            dispatcher=Dispatcher(),
            tokenizer=Tokenizer(),
            served_model_name="model",
            request_log_path=None,
            reasoning_parser_name=None,
        )
        chunks = [chunk async for chunk in response.body_iterator]
        events = _response_sse_events(chunks)
        event_types = [event for event, _payload in events]

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("response.created", event_types)
        self.assertIn("response.output_item.added", event_types)
        self.assertIn("response.content_part.added", event_types)
        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("response.output_text.done", event_types)
        self.assertIn("response.content_part.done", event_types)
        self.assertIn("response.output_item.done", event_types)
        self.assertEqual(event_types[-2:], ["response.completed", "[DONE]"])
        completed = events[-2][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(completed["usage"], {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})

    async def test_response_stream_qwen3_reasoning_uses_raw_delta(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        for delta in ["<thi", "nk>rea", "son</thi", "nk>answer"]:
            await queue.put(
                {
                    "type": "token",
                    "index": 0,
                    "text": "",
                    "raw_text_delta": delta,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": "<think>reason</think>answer",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "token_ids": [1, 2, 3, 4],
                "token_logprobs": [None, None, None, None],
                "top_logprobs": [None, None, None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                RequestHandle(output_queue=queue, cancelled=threading.Event()),
                time.perf_counter() - 1.0,
                None,
                ResponseRequest(model="model", input="hello", stream=True),
                reasoning_parser_name="qwen3",
            )
        ]

        events = _response_sse_events(chunks)
        reasoning_deltas = [
            payload["delta"]
            for event, payload in events
            if event == "response.reasoning_text.delta"
        ]
        output_deltas = [
            payload["delta"]
            for event, payload in events
            if event == "response.output_text.delta"
        ]
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertEqual("".join(reasoning_deltas), "reason")
        self.assertEqual("".join(output_deltas), "answer")
        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][1]["content"][0]["text"], "answer")

    async def test_response_stream_parser_disabled_returns_raw_visible_text(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "<think>reason</think>answer",
                "raw_text_delta": "<think>reason</think>answer",
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "<think>reason</think>answer",
                "raw_text": "<think>reason</think>answer",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                RequestHandle(output_queue=queue, cancelled=threading.Event()),
                time.perf_counter(),
                None,
                ResponseRequest(model="model", input="hello", stream=True),
                reasoning_parser_name=None,
            )
        ]

        output_deltas = [
            payload["delta"]
            for event, payload in _response_sse_events(chunks)
            if event == "response.output_text.delta"
        ]
        self.assertEqual("".join(output_deltas), "<think>reason</think>answer")

    async def test_response_stream_qwen3_thinking_off_streams_plain_answer(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "hel",
                "raw_text_delta": "hel",
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "raw_text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True, reasoning={"effort": "none"}),
                    reasoning_parser_name="qwen3",
                )
            ]
        )

        event_types = [event for event, _payload in events]
        output_deltas = [
            payload["delta"]
            for event, payload in events
            if event == "response.output_text.delta"
        ]
        self.assertLess(
            event_types.index("response.output_text.delta"),
            event_types.index("response.completed"),
        )
        self.assertEqual("".join(output_deltas), "hello")

    async def test_response_stream_reasoning_length_finishes_incomplete(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "",
                "raw_text_delta": "<think>partial",
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "length",
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True),
                    reasoning_parser_name="qwen3",
                )
            ]
        )

        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]
        self.assertEqual(completed["status"], "incomplete")
        self.assertEqual(completed["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(completed["output"][0]["text"], "partial")

    async def test_response_stream_unclosed_reasoning_stop_fails_fast(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "stop",
                "prompt_tokens": 2,
                "completion_tokens": 1,
            }
        )

        cancelled = threading.Event()

        class Dispatcher:
            def cancel(self, _handle):
                cancelled.set()

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True),
                    reasoning_parser_name="qwen3",
                )
            ]
        )

        failed = [payload for event, payload in events if event == "response.failed"][0]
        self.assertIn("did not close", failed["error"]["message"])
        self.assertTrue(cancelled.is_set())

    def test_response_route_returns_non_streaming_response(self):
        from fastapi.testclient import TestClient
        from sparsevllm.entrypoints.openai import api_server

        test_tokenizer = _byte_level_tokenizer()
        completion_token_ids = test_tokenizer.encode("hello")

        class Engine:
            tokenizer = test_tokenizer
            config = type("Config", (), {"vllm_sparse_method": ""})()
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def add_request(self, prompt, sampling_params):
                self.prompt = prompt
                self.sampling_params = sampling_params
                return 1

            def step(self):
                return [
                    (
                        1,
                        completion_token_ids,
                        [None] * len(completion_token_ids),
                        [None] * len(completion_token_ids),
                    )
                ], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        try:
            response = TestClient(app).post(
                "/v1/responses",
                json={"model": "model", "input": "hello", "max_output_tokens": 4},
            )
        finally:
            app.state.dispatcher.close()

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(engine.sampling_params.max_tokens, 4)

    def test_qwen3_reasoning_parser_builds_reasoning_item(self):
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        output, incomplete = _response_output_items(
            "<think>reason</think>answer",
            "stop",
            reasoning_parser_name="qwen3",
        )

        self.assertFalse(incomplete)
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(output[0]["text"], "reason")
        self.assertEqual(output[1]["content"][0]["text"], "answer")

    def test_qwen3_reasoning_parser_handles_template_opened_think(self):
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        output, incomplete = _response_output_items(
            "reason</think>\n\nanswer<|im_end|>",
            "stop",
            reasoning_parser_name="qwen3",
        )

        self.assertFalse(incomplete)
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(output[0]["text"], "reason")
        self.assertEqual(output[1]["content"][0]["text"], "answer")

    def test_qwen3_reasoning_parser_handles_incomplete_length(self):
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        output, incomplete = _response_output_items(
            "<think>partial",
            "length",
            reasoning_parser_name="qwen3",
        )

        self.assertTrue(incomplete)
        self.assertEqual(output, [{"id": output[0]["id"], "type": "reasoning", "text": "partial", "summary": []}])

    def test_qwen3_reasoning_parser_rejects_unclosed_stop(self):
        from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningParseError
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        with self.assertRaises(ReasoningParseError):
            _response_output_items("<think>partial", "stop", reasoning_parser_name="qwen3")

    def test_reasoning_parser_disabled_returns_raw_text(self):
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        output, incomplete = _response_output_items(
            "<think>reason</think>answer",
            "stop",
            reasoning_parser_name=None,
        )

        self.assertFalse(incomplete)
        self.assertEqual(output[0]["content"][0]["text"], "<think>reason</think>answer")

    def test_tool_call_output_item_is_parsed(self):
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        output, incomplete = _response_output_items(
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            "stop",
            reasoning_parser_name=None,
        )

        self.assertFalse(incomplete)
        self.assertEqual(output[0]["type"], "function_call")
        self.assertEqual(output[0]["name"], "get_weather")
        self.assertEqual(output[0]["arguments"], '{"city":"Paris"}')

    def test_malformed_tool_call_json_fails_fast(self):
        from sparsevllm.entrypoints.openai.responses.tools import ToolCallParseError
        from sparsevllm.entrypoints.openai.api_server import _response_output_items

        with self.assertRaises(ToolCallParseError):
            _response_output_items('<tool_call>{"name":</tool_call>', "stop", reasoning_parser_name=None)

    def test_qwen3_reasoning_stream_parser_handles_split_tags(self):
        from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_stream_parser

        parser = get_reasoning_stream_parser("qwen3")
        events = []
        for delta in ["<thi", "nk>rea", "son</thi", "nk>\n\nanswer<|im_end|>"]:
            events.extend(parser.feed(delta))
        events.extend(parser.finish("stop"))

        reasoning = "".join(event.text for event in events if event.kind == "reasoning_delta")
        answer = "".join(event.text for event in events if event.kind == "answer_delta")

        self.assertEqual(reasoning, "reason")
        self.assertEqual(answer, "answer")
        self.assertIn("reasoning_done", [event.kind for event in events])

    def test_qwen3_reasoning_stream_parser_handles_template_opened_think(self):
        from sparsevllm.entrypoints.openai.responses.reasoning import Qwen3ReasoningStreamParser

        parser = Qwen3ReasoningStreamParser(buffer_initial_content=True)
        events = []
        for delta in ["reason", "</thi", "nk>\n\nanswer"]:
            events.extend(parser.feed(delta))
        events.extend(parser.finish("stop"))

        reasoning = "".join(event.text for event in events if event.kind == "reasoning_delta")
        answer = "".join(event.text for event in events if event.kind == "answer_delta")

        self.assertEqual(reasoning, "reason")
        self.assertEqual(answer, "answer")

    def test_qwen3_reasoning_stream_parser_rejects_unclosed_stop(self):
        from sparsevllm.entrypoints.openai.responses.reasoning import ReasoningParseError
        from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_stream_parser

        parser = get_reasoning_stream_parser("qwen3")
        parser.feed("<think>partial")

        with self.assertRaises(ReasoningParseError):
            parser.finish("stop")

    def test_qwen3_tool_call_stream_parser_handles_split_json(self):
        from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamParser

        parser = ToolCallStreamParser()
        events = []
        for delta in [
            "<tool_",
            'call>{"name":"get_weather",',
            '"arguments":{"city":"Paris"}}</tool_call>',
        ]:
            events.extend(parser.feed(delta))
        events.extend(parser.finish("stop"))

        self.assertEqual([event.kind for event in events], [
            "tool_call_started",
            "tool_call_arguments_delta",
            "tool_call_done",
        ])
        self.assertEqual(events[0].name, "get_weather")
        self.assertEqual(events[1].arguments_delta, '{"city":"Paris"}')

    def test_qwen3_tool_call_stream_parser_handles_sequential_split_blocks(self):
        from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamParser

        parser = ToolCallStreamParser()
        events = []
        events.extend(
            parser.feed(
                '<tool_call>{"name":"first","arguments":{"x":1}}</tool_call>'
            )
        )
        events.extend(
            parser.feed(
                '<tool_call>{"name":"second","arguments":{"y":2}}</tool_call>'
            )
        )
        events.extend(parser.finish("stop"))

        started = [event.name for event in events if event.kind == "tool_call_started"]
        arguments = [
            event.arguments_delta
            for event in events
            if event.kind == "tool_call_arguments_delta"
        ]

        self.assertEqual(started, ["first", "second"])
        self.assertEqual(arguments, ['{"x":1}', '{"y":2}'])

    def test_qwen3_tool_call_stream_parser_fails_malformed_json(self):
        from sparsevllm.entrypoints.openai.responses.tools import ToolCallParseError
        from sparsevllm.entrypoints.openai.responses.tools import ToolCallStreamParser

        parser = ToolCallStreamParser()

        with self.assertRaises(ToolCallParseError):
            parser.feed('<tool_call>{"name":</tool_call>')

    async def test_response_stream_tool_call_outputs_function_events(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                "raw_text_delta": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True),
                    reasoning_parser_name=None,
                )
            ]
        )
        event_types = [event for event, _payload in events]
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(completed["output"][0]["type"], "function_call")
        self.assertEqual(completed["output"][0]["name"], "get_weather")
        self.assertEqual(completed["output"][0]["arguments"], '{"city":"Paris"}')

    async def test_response_stream_reasoning_then_tool_call(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        raw = '<think>reason</think><tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        queue = asyncio.Queue()
        for delta in ["<think>reason</think><tool_", 'call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>']:
            await queue.put(
                {
                    "type": "token",
                    "index": 0,
                    "text": "",
                    "raw_text_delta": delta,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": raw,
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True),
                    reasoning_parser_name="qwen3",
                )
            ]
        )
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertEqual([item["type"] for item in completed["output"]], ["reasoning", "function_call"])
        self.assertEqual(completed["output"][0]["text"], "reason")
        self.assertEqual(completed["output"][1]["arguments"], '{"city":"Paris"}')

    async def test_response_stream_cancel_releases_dispatcher_request(self):
        from sparsevllm.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        handle = RequestHandle(output_queue=queue, cancelled=threading.Event())
        cancelled = threading.Event()
        outer = self

        class Dispatcher:
            def cancel(self, observed_handle):
                outer.assertIs(observed_handle, handle)
                cancelled.set()

        stream = _response_stream(
            Dispatcher(),
            "resp_test",
            123,
            "model",
            handle,
            time.perf_counter(),
            None,
            ResponseRequest(model="model", input="hello", stream=True),
            reasoning_parser_name=None,
        )

        first = await stream.__anext__()
        self.assertIn("response.created", first)
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        self.assertTrue(cancelled.is_set())

    async def test_prefix_cache_match_accepts_response_selector(self):
        from sparsevllm.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None
            chat_template = "template"

            def apply_chat_template(self, chat, **_kwargs):
                return "|".join(f"{item['role']}:{item['content']}" for item in chat)

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def prefix_cache_match(self, token_ids):
                return {"token_ids": list(token_ids), "supported": True, "enabled": True}

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        try:
            response = await endpoint(
                api_server.PrefixCacheMatchRequest(
                    response={"model": "model", "instructions": "policy", "input": "hello"}
                ),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        self.assertEqual(
            json.loads(response.body)["token_ids"],
            [ord(ch) for ch in "system:policy|user:hello"],
        )

    async def test_prefix_cache_match_rejects_multiple_selectors_with_response(self):
        from fastapi import HTTPException
        from sparsevllm.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"vllm_sparse_method": ""})()

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        try:
            with self.assertRaises(HTTPException):
                await endpoint(
                    api_server.PrefixCacheMatchRequest(text="hello", response={"model": "model", "input": "hello"}),
                    _TestRequest(app),
                )
        finally:
            app.state.dispatcher.close()


class OpenAIClientTest(unittest.TestCase):
    def test_stream_client_prints_text_without_sse_frames(self):
        client_path = Path(__file__).resolve().parents[1] / "src/sparsevllm/entrypoints/openai/client.py"
        spec = importlib.util.spec_from_file_location("sparsevllm_openai_client_test", client_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        lines = [
            'data: {"choices": [{"text": " local"}]}\n',
            "\n",
            'data: {"choices": [{"text": "_attention"}]}\n',
            "data: [DONE]\n",
        ]

        with patch("builtins.print") as mocked_print:
            output = module.print_stream_text(lines)

        self.assertEqual(output, " local_attention")
        self.assertEqual(mocked_print.call_args_list[0].args, (" local",))
        self.assertEqual(mocked_print.call_args_list[1].args, ("_attention",))


if __name__ == "__main__":
    unittest.main()
