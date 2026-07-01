import importlib.util
import unittest


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None,
    "OpenAI smart router dependencies are not installed",
)
class OpenAISmartRouterTest(unittest.TestCase):
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

        self.assertEqual(
            match_payload_for_request(
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hello"}]},
            ),
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(
            match_payload_for_request("/v1/completions", {"prompt": [1, 2, 3]}),
            {"token_ids": [1, 2, 3]},
        )
        self.assertEqual(
            match_payload_for_request("/v1/completions", {"prompt": ["a", "b"]}),
            {"text": "a"},
        )


if __name__ == "__main__":
    unittest.main()
