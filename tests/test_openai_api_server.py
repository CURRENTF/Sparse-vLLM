import asyncio
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
from unittest.mock import AsyncMock


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("pydantic") is None,
    "OpenAI API server dependencies are not installed",
)
class OpenAIAPIServerTest(unittest.IsolatedAsyncioTestCase):
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

        from sparsevllm.entrypoints.openai.api_server import CompletionRequest

        with self.assertRaises(ValidationError):
            CompletionRequest(model="m", prompt="p", suffix="ignored")

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

    def test_missing_non_bool_engine_arg_fails_fast(self):
        from sparsevllm.entrypoints.openai.api_server import _parse_engine_kwargs

        with self.assertRaisesRegex(ValueError, "Missing value"):
            _parse_engine_kwargs(["--max-model-len"])

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

        class Tokenizer:
            def encode(self, _prompt):
                return [1]

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(str(token_id) for token_id in token_ids)

        class Engine:
            def __init__(self):
                self.tokenizer = Tokenizer()
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

    async def test_dispatcher_close_times_out_blocked_step_and_exits_engine(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher

        class Tokenizer:
            def encode(self, _prompt):
                return [1]

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(str(token_id) for token_id in token_ids)

        class Engine:
            tokenizer = Tokenizer()

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
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/completions")
        request = api_server.CompletionRequest(model="model", prompt="p")
        try:
            with patch.object(
                api_server,
                "_completion_response",
                AsyncMock(side_effect=asyncio.CancelledError),
            ), patch.object(api_server.logger, "info") as log_info:
                with self.assertRaises(asyncio.CancelledError):
                    await endpoint(request)
        finally:
            app.state.dispatcher.close()

        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn("request_cancel id={} model={} stream=false elapsed_s={:.3f}", messages)

    async def test_dispatcher_streaming_delta_uses_cumulative_suffix(self):
        from sparsevllm.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return {(): "", (1,): "a", (1, 2): "ab"}[tuple(token_ids)]

        class Engine:
            tokenizer = Tokenizer()
            last_step_token_outputs = [(7, [2])]
            last_step_logprob_outputs = [(7, [None], [None])]

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
                max_tokens=2,
                stop=[],
                completion_token_ids=[1],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                emitted_text_len=1,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(item["text"], "b")
        self.assertEqual(active[7].completion_token_ids, [1, 2])

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

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return {(): "", (1,): "a", (1, 2): "ab", (1, 2, 3): "abSTOP"}[tuple(token_ids)]

        class Engine:
            tokenizer = Tokenizer()

            def __init__(self):
                self.last_step_token_outputs = [(7, [2])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
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
                max_tokens=4,
                stop=["bSTOP"],
                completion_token_ids=[1],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                emitted_text_len=0,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            self.assertEqual(token_item["text"], "a")
            engine.last_step_token_outputs = [(7, [3])]
            engine.last_step_logprob_outputs = [(7, [None], [None])]
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
