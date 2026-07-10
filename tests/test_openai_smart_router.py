import asyncio
import importlib.util
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None,
    "OpenAI smart router dependencies are not installed",
)
class OpenAISmartRouterTest(unittest.TestCase):
    def test_model_cards_use_smallest_worker_context(self):
        from sparsevllm.entrypoints.openai.smart_router import _router_model_cards
        from sparsevllm.entrypoints.openai.smart_router import WorkerState

        cards = _router_model_cards(
            [
                WorkerState(url="http://a", info={"served_model_name": "model", "max_model_len": 128000}),
                WorkerState(url="http://b", info={"served_model_name": "model", "max_model_len": 64000}),
                WorkerState(
                    url="http://unhealthy",
                    info={"served_model_name": "model", "max_model_len": 32000},
                    healthy=False,
                ),
            ],
            123,
        )

        self.assertEqual(cards[0]["max_model_len"], 64000)

    def test_choose_worker_prefers_prefix_match_when_load_is_close(self):
        from sparsevllm.entrypoints.openai.smart_router import WorkerProbe, WorkerState, choose_worker

        cache_worker = WorkerState(url="http://worker-a", info={"sparse_method": "omnikv"})
        load_worker = WorkerState(url="http://worker-b", info={"sparse_method": "omnikv"})
        worker, reason = choose_worker(
            [
                WorkerProbe(
                    worker=cache_worker,
                    load={"active_requests": 1},
                    match={"supported": True, "enabled": True, "matched_tokens": 128, "match_ratio": 0.75},
                ),
                WorkerProbe(
                    worker=load_worker,
                    load={"active_requests": 0},
                    match={"supported": True, "enabled": True, "matched_tokens": 0, "match_ratio": 0.0},
                ),
            ],
            overload_load_factor=1.5,
            load_abs_threshold=1,
        )

        self.assertIs(worker, cache_worker)
        self.assertEqual(reason, "best_prefix_match")

    def test_choose_worker_falls_back_to_lowest_load_when_match_worker_is_overloaded(self):
        from sparsevllm.entrypoints.openai.smart_router import WorkerProbe, WorkerState, choose_worker

        cache_worker = WorkerState(url="http://worker-a", info={"sparse_method": "omnikv"})
        load_worker = WorkerState(url="http://worker-b", info={"sparse_method": "omnikv"})
        worker, reason = choose_worker(
            [
                WorkerProbe(
                    worker=cache_worker,
                    load={"active_requests": 10},
                    match={"supported": True, "enabled": True, "matched_tokens": 128, "match_ratio": 0.75},
                ),
                WorkerProbe(
                    worker=load_worker,
                    load={"active_requests": 0},
                    match={"supported": True, "enabled": True, "matched_tokens": 0, "match_ratio": 0.0},
                ),
            ],
            overload_load_factor=1.5,
            load_abs_threshold=1,
        )

        self.assertIs(worker, load_worker)
        self.assertEqual(reason, "prefix_match_overloaded_lowest_load")

    def test_route_profiles_filter_heterogeneous_workers_by_sparse_method(self):
        from sparsevllm.entrypoints.openai.smart_router import SmartRouter

        router = SmartRouter(
            worker_urls=["http://omni", "http://snap"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={
                "conversation": {"methods": ["omnikv"]},
                "bulk": {"methods": ["snapkv"]},
            },
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv", "tags": ["dialog"]}
        router.workers[1].info = {"served_model_name": "model", "sparse_method": "snapkv", "tags": ["bulk"]}

        conversation = router._candidate_workers(
            "/v1/chat/completions",
            {"model": "model", "messages": [{"role": "user", "content": "x"}]},
            {"route_profile": "conversation"},
        )
        bulk = router._candidate_workers(
            "/v1/completions",
            {"model": "model", "prompt": "x"},
            {"route_profile": "bulk"},
        )

        self.assertEqual([worker.url for worker in conversation], ["http://omni"])
        self.assertEqual([worker.url for worker in bulk], ["http://snap"])

    def test_route_profiles_treat_empty_method_as_vanilla(self):
        from sparsevllm.entrypoints.openai.smart_router import SmartRouter

        router = SmartRouter(
            worker_urls=["http://vanilla", "http://snap"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={"default": {"methods": ["vanilla"]}},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "", "tags": []}
        router.workers[1].info = {"served_model_name": "model", "sparse_method": "snapkv", "tags": []}

        candidates = router._candidate_workers(
            "/v1/completions",
            {"model": "model", "prompt": "x"},
            {},
        )

        self.assertEqual([worker.url for worker in candidates], ["http://vanilla"])

    def test_route_hints_are_stripped_before_forwarding(self):
        from sparsevllm.entrypoints.openai.smart_router import strip_route_hints

        payload, hints = strip_route_hints(
            {
                "model": "model",
                "prompt": "hello",
                "svllm_route_profile": "bulk",
                "svllm_method_preference": ["snapkv"],
            }
        )

        self.assertEqual(payload, {"model": "model", "prompt": "hello"})
        self.assertEqual(hints["route_profile"], "bulk")
        self.assertEqual(hints["method_preference"], ["snapkv"])

    def test_match_payload_for_chat_and_completion_requests(self):
        from sparsevllm.entrypoints.openai.smart_router import match_payload_for_request

        chat_payload = {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {}},
                }
            ],
            "reasoning_effort": "none",
        }
        self.assertEqual(
            match_payload_for_request(
                "/v1/chat/completions",
                chat_payload,
            ),
            {"chat": chat_payload},
        )
        self.assertEqual(
            match_payload_for_request("/v1/completions", {"prompt": [1, 2, 3]}),
            {"token_ids": [1, 2, 3]},
        )
        self.assertEqual(
            match_payload_for_request("/v1/completions", {"prompt": ["a", "b"]}),
            {"text": "a"},
        )
        response_payload = {"model": "model", "input": "hello", "reasoning": {"effort": "none"}}
        self.assertEqual(
            match_payload_for_request("/v1/responses", response_payload),
            {"response": response_payload},
        )

    def test_responses_route_profile_inference_stays_default(self):
        from sparsevllm.entrypoints.openai.smart_router import infer_route_profile

        self.assertEqual(
            infer_route_profile(
                "/v1/responses",
                {"input": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            ),
            "default",
        )

    def test_responses_route_forwards_through_router(self):
        from sparsevllm.entrypoints.openai import smart_router

        class JsonRequest:
            async def json(self):
                return {"model": "model", "input": "hello"}

        app = smart_router.create_app(["http://worker-a"])
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/responses")
        app.state.router.route_openai_request = AsyncMock(return_value="ok")

        response = asyncio.run(endpoint(JsonRequest()))

        self.assertEqual(response, "ok")
        app.state.router.route_openai_request.assert_awaited_once_with(
            "/v1/responses",
            {"model": "model", "input": "hello"},
        )

    def test_responses_route_hints_are_stripped_before_forwarding(self):
        from fastapi.responses import Response
        from sparsevllm.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info

        async def forward_json(_worker, _endpoint, payload):
            self.assertEqual(payload, {"model": "model", "input": "hello"})
            return Response(content=b"{}", media_type="application/json")

        router.forward_json = forward_json
        response = asyncio.run(
            router.route_openai_request(
                "/v1/responses",
                {
                    "model": "model",
                    "input": "hello",
                    "svllm_target_worker": "0",
                    "svllm_method_preference": ["omnikv"],
                },
            )
        )

        self.assertEqual(response.headers["x-sparsevllm-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparsevllm-route-reason"], "target_worker")

    def test_streaming_upstream_http_error_is_returned_before_sse_response(self):
        from sparsevllm.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info
        upstream_error = smart_router.UpstreamError(
            status=400,
            headers={"Content-Type": "application/json"},
            body=b'{"error":"bad request"}',
        )

        with patch.object(smart_router, "_open_stream_response", return_value=upstream_error):
            response = asyncio.run(
                router.route_openai_request(
                    "/v1/completions",
                    {
                        "model": "model",
                        "prompt": "hello",
                        "stream": True,
                        "svllm_target_worker": "0",
                    },
                )
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body, b'{"error":"bad request"}')
        self.assertEqual(response.headers["x-sparsevllm-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparsevllm-route-reason"], "target_worker")
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_responses_streaming_transparently_forwards_sse_bytes(self):
        from sparsevllm.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        class Upstream:
            def __init__(self):
                self.chunks = [
                    b"event: response.created\n",
                    b'data: {"type":"response.created"}\n\n',
                    b"data: [DONE]\n\n",
                ]
                self.closed = False

            def read(self, _size):
                if not self.chunks:
                    return b""
                return self.chunks.pop(0)

            def close(self):
                self.closed = True

        upstream_response = Upstream()
        router.refresh_worker_info = refresh_worker_info

        async def run_request():
            with patch.object(
                smart_router,
                "_open_stream_response",
                return_value=smart_router.UpstreamStream(
                    response=upstream_response,
                    headers={"Content-Type": "text/event-stream"},
                ),
            ):
                response = await router.route_openai_request(
                    "/v1/responses",
                    {
                        "model": "model",
                        "input": "hello",
                        "stream": True,
                        "svllm_target_worker": "0",
                    },
                )
                chunks = [chunk async for chunk in response.body_iterator]
                return response, chunks

        response, chunks = asyncio.run(run_request())

        self.assertEqual(b"".join(chunks), b'event: response.created\ndata: {"type":"response.created"}\n\ndata: [DONE]\n\n')
        self.assertEqual(response.headers["x-sparsevllm-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparsevllm-route-reason"], "target_worker")
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertTrue(upstream_response.closed)
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_responses_streaming_upstream_http_error_is_returned(self):
        from sparsevllm.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info
        upstream_error = smart_router.UpstreamError(
            status=400,
            headers={"Content-Type": "application/json"},
            body=b'{"error":"bad request"}',
        )

        with patch.object(smart_router, "_open_stream_response", return_value=upstream_error):
            response = asyncio.run(
                router.route_openai_request(
                    "/v1/responses",
                    {
                        "model": "model",
                        "input": "hello",
                        "stream": True,
                        "svllm_target_worker": "0",
                    },
                )
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body, b'{"error":"bad request"}')
        self.assertEqual(response.headers["x-sparsevllm-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparsevllm-route-reason"], "target_worker")
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_streaming_open_exception_releases_local_inflight(self):
        from sparsevllm.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info
        with patch.object(smart_router, "_open_stream_response", side_effect=ValueError("boom")):
            with self.assertRaisesRegex(ValueError, "boom"):
                asyncio.run(
                    router.route_openai_request(
                        "/v1/completions",
                        {
                            "model": "model",
                            "prompt": "hello",
                            "stream": True,
                            "svllm_target_worker": "0",
                        },
                    )
                )

        self.assertEqual(router.workers[0].local_inflight, 0)


if __name__ == "__main__":
    unittest.main()
