import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.simulated_deep_research import run


def _json_response(
    payload,
    *,
    status=200,
    headers=None,
):
    return run.HttpResponse(
        status=status,
        headers=headers or {},
        body=json.dumps(payload).encode("utf-8"),
    )


class FakeService:
    def __init__(
        self,
        *,
        fail_sample_number=None,
        wrong_subagent_method=False,
    ):
        self.active = 0
        self.max_active = 0
        self.completion_calls = 0
        self.payloads = []
        self.fail_sample_number = fail_sample_number
        self.wrong_subagent_method = wrong_subagent_method

    async def get(self, url, _timeout_s):
        if url.endswith("/models"):
            return _json_response(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "sim-model",
                            "owned_by": "sparsevllm-router",
                            "max_model_len": 65536,
                        }
                    ],
                }
            )
        if url.endswith("/health"):
            return _json_response(
                {
                    "status": "ok",
                    "healthy_workers": [
                        "http://snap-worker",
                        "http://main-worker",
                    ],
                    "router_policy": {
                        "request_timeout_s": 900.0,
                        "overload_load_factor": 1.0,
                        "load_abs_threshold": 0,
                        "profiles": {},
                    },
                }
            )
        if url.endswith("/v1/worker/info"):
            method = "snapkv" if "snap-worker" in url else "omnikv"
            return _json_response(
                {
                    "served_model_name": "sim-model",
                    "sparse_method": method,
                    "benchmark_config": {
                        "gpu_memory_utilization": 0.9,
                        "prefill_schedule_policy": "all_chunked",
                        "chunk_prefill_size": 8192,
                        "decode_cuda_graph": True,
                        "enable_prefix_caching": method == "omnikv",
                    },
                }
            )
        raise AssertionError(f"Unexpected GET URL: {url}")

    async def post(self, url, payload, _timeout_s):
        self.completion_calls += 1
        self.payloads.append(payload)
        call_number = self.completion_calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.001)
            methods = payload["svllm_method_preference"].split(",")
            if methods == ["snapkv"]:
                worker = "http://snap-worker"
                method = (
                    "omnikv"
                    if self.wrong_subagent_method
                    else "snapkv"
                )
            else:
                worker = "http://main-worker"
                method = "omnikv"
            headers = {
                "x-sparsevllm-worker": worker,
                "x-sparsevllm-route-reason": "lowest_load_no_prefix_match",
                "x-sparsevllm-sparse-method": method,
            }
            if call_number == self.fail_sample_number:
                return _json_response(
                    {"error": "synthetic failure"},
                    status=500,
                    headers=headers,
                )
            prompt_tokens = len(payload["prompt"])
            completion_tokens = payload["max_tokens"]
            return _json_response(
                {
                    "id": f"cmpl-{call_number}",
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic output",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
                headers=headers,
            )
        finally:
            self.active -= 1


class SimulatedDeepResearchTest(unittest.TestCase):
    def _config(self, output_dir):
        return run.BenchmarkConfig(
            base_url="http://router.test/v1",
            model="sim-model",
            output_dir=output_dir,
            rounds=2,
            articles_per_round=3,
            article_token_buckets=((1, 10, 20),),
            query_tokens=4,
            subagent_output_token_buckets=((1, 2, 5),),
            main_overhead_tokens=3,
            min_round_summary_tokens=4,
            max_round_summary_tokens=4,
            final_overhead_tokens=2,
            min_final_output_tokens=6,
            max_final_output_tokens=6,
            seed=7,
        )

    def test_required_model_len_covers_each_request_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            self.assertEqual(run.required_model_len(config), 29)

    def test_default_profile_is_long_tailed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = run.BenchmarkConfig(
                base_url="http://router.test/v1",
                model="sim-model",
                output_dir=Path(tmp) / "run",
            )
            self.assertEqual(
                config.article_token_buckets,
                (
                    (60, 1_000, 8_000),
                    (25, 8_001, 16_000),
                    (10, 16_001, 32_000),
                    (5, 32_001, 64_000),
                ),
            )
            self.assertEqual(
                config.subagent_output_token_buckets,
                ((90, 100, 600), (10, 800, 1_500)),
            )
            self.assertEqual(config.request_timeout_s, 930.0)
            self.assertEqual(config.router_timeout_margin_s, 30.0)
            self.assertEqual(run.required_model_len(config), 65_564)
            cli_config = run.config_from_args(
                run.build_arg_parser().parse_args(
                    [
                        "--model",
                        "sim-model",
                        "--output-dir",
                        str(Path(tmp) / "cli-run"),
                    ]
                )
            )
            self.assertEqual(cli_config.request_timeout_s, 930.0)
            self.assertEqual(cli_config.router_timeout_margin_s, 30.0)

    def test_rejects_invalid_token_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "article_token_buckets": ((0, 10, 20),),
                }
            )
            with self.assertRaisesRegex(ValueError, "weight must be positive"):
                run.validate_config(config)

    def test_required_model_len_includes_accumulated_main_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "rounds": 4,
                    "articles_per_round": 2,
                    "article_token_buckets": ((1, 1, 1),),
                    "query_tokens": 1,
                    "subagent_output_token_buckets": ((1, 1, 1),),
                    "main_overhead_tokens": 3,
                    "min_round_summary_tokens": 5,
                    "max_round_summary_tokens": 5,
                    "final_overhead_tokens": 2,
                    "min_final_output_tokens": 1,
                    "max_final_output_tokens": 1,
                }
            )
            self.assertEqual(run.required_model_len(config), 26)

    def test_rejects_overlapping_main_and_subagent_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "main_agent_methods": ("snapkv",),
                }
            )
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                run.validate_config(config)

    def test_direct_server_allows_the_same_method_for_both_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "subagent_methods": ("vanilla",),
                    "main_agent_methods": ("vanilla",),
                    "require_router": False,
                }
            )
            run.validate_config(config)

    def test_runs_parallel_subagents_and_writes_auditable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService()

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(aggregate["expected_requests"], 9)
            self.assertEqual(aggregate["successful_requests"], 9)
            self.assertEqual(aggregate["phase_counts"]["subagent"], 6)
            self.assertEqual(aggregate["phase_counts"]["round_summary"], 2)
            self.assertEqual(aggregate["phase_counts"]["final_summary"], 1)
            self.assertEqual(
                aggregate["route_method_counts"],
                {"omnikv": 3, "snapkv": 6},
            )
            self.assertEqual(
                aggregate["phase_metrics"]["subagent"][
                    "route_method_counts"
                ],
                {"snapkv": 6},
            )
            self.assertEqual(
                aggregate["phase_metrics"]["round_summary"][
                    "route_method_counts"
                ],
                {"omnikv": 2},
            )
            self.assertEqual(aggregate["distinct_route_workers"], 2)
            self.assertGreaterEqual(service.max_active, 3)
            main_payloads = [
                payload
                for payload in service.payloads
                if payload["svllm_method_preference"] == "omnikv,vanilla"
            ]
            self.assertEqual(len(main_payloads), 3)
            first_round_prompt = main_payloads[0]["prompt"]
            second_round_prompt = main_payloads[1]["prompt"]
            final_prompt = main_payloads[2]["prompt"]
            self.assertEqual(
                second_round_prompt[: config.main_overhead_tokens],
                first_round_prompt[: config.main_overhead_tokens],
            )
            reusable_before_final = (
                config.main_overhead_tokens
                + config.max_round_summary_tokens
            )
            self.assertEqual(
                final_prompt[:reusable_before_final],
                second_round_prompt[:reusable_before_final],
            )

            for name in (
                "run_info.json",
                "raw_outputs.jsonl",
                "parsed_outputs.jsonl",
                "per_sample_results.jsonl",
                "round_metrics.jsonl",
                "aggregate_metrics.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            sample_rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(sample_rows), 9)
            self.assertTrue(all(row["status"] == "success" for row in sample_rows))
            self.assertTrue(
                all(
                    row["article_tokens"] is not None
                    for row in sample_rows
                    if row["phase"] == "subagent"
                )
            )
            self.assertEqual(
                {
                    row["route_method"]
                    for row in sample_rows
                    if row["phase"] == "subagent"
                },
                {"snapkv"},
            )
            self.assertEqual(
                {
                    row["route_method"]
                    for row in sample_rows
                    if row["phase"] != "subagent"
                },
                {"omnikv"},
            )
            main_rows = [
                row for row in sample_rows if row["phase"] != "subagent"
            ]
            self.assertEqual(
                [
                    row["expected_reusable_prefix_tokens"]
                    for row in main_rows
                ],
                [0, config.main_overhead_tokens, reusable_before_final],
            )
            run_info = json.loads(
                (output_dir / "run_info.json").read_text(encoding="utf-8")
            )
            worker_configs = run_info["preflight"]["workers"]
            self.assertEqual(
                [worker["info"]["sparse_method"] for worker in worker_configs],
                ["snapkv", "omnikv"],
            )
            self.assertTrue(
                worker_configs[1]["info"]["benchmark_config"][
                    "enable_prefix_caching"
                ]
            )
            self.assertEqual(
                run_info["preflight"]["health"]["router_policy"][
                    "request_timeout_s"
                ],
                900.0,
            )

    def test_http_failure_is_recorded_before_run_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(fail_sample_number=2)

            with self.assertRaisesRegex(run.BenchmarkFailed, "model_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "model_failed")
            self.assertEqual(aggregate["status_counts"]["model_failed"], 1)
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                sum(row["status"] == "model_failed" for row in rows),
                1,
            )

    def test_non_json_http_error_is_model_failure(self):
        async def fail_post(_url, _payload, _timeout_s):
            return run.HttpResponse(
                status=502,
                headers={},
                body=b"<html>bad gateway</html>",
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "model_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=fail_post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "model_failed")
            self.assertEqual(aggregate["status_counts"]["model_failed"], 3)

    def test_malformed_choice_text_is_parse_failure(self):
        async def malformed_post(_url, payload, _timeout_s):
            prompt_tokens = len(payload["prompt"])
            completion_tokens = payload["max_tokens"]
            return _json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": None,
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
                headers={
                    "x-sparsevllm-worker": "http://snap-worker",
                    "x-sparsevllm-route-reason": "test",
                    "x-sparsevllm-sparse-method": "snapkv",
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "parse_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=malformed_post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "parse_failed")
            self.assertEqual(aggregate["status_counts"]["parse_failed"], 3)

    def test_malformed_usage_is_parse_failure(self):
        async def malformed_post(_url, payload, _timeout_s):
            return _json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic output",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": None,
                        "completion_tokens": payload["max_tokens"],
                    },
                },
                headers={
                    "x-sparsevllm-worker": "http://snap-worker",
                    "x-sparsevllm-route-reason": "test",
                    "x-sparsevllm-sparse-method": "snapkv",
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "parse_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=malformed_post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "parse_failed")
            self.assertEqual(aggregate["status_counts"]["parse_failed"], 3)

    def test_preflight_malformed_json_is_parse_failure(self):
        async def malformed_get(url, timeout_s):
            if url.endswith("/models"):
                return run.HttpResponse(
                    status=200,
                    headers={},
                    body=b"not-json",
                )
            return await FakeService().get(url, timeout_s)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "parse_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=malformed_get,
                        post_fn=FakeService().post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "parse_failed")

    def test_reusable_prefix_uses_actual_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "synthetic_token_id_low": 7,
                    "synthetic_token_id_high": 7,
                }
            )
            service = FakeService()
            asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )
            main_payloads = [
                payload
                for payload in service.payloads
                if payload["svllm_method_preference"] == "omnikv,vanilla"
            ]
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            main_rows = [
                row for row in rows if row["phase"] != "subagent"
            ]
            expected = [
                0,
                min(
                    len(main_payloads[0]["prompt"]),
                    len(main_payloads[1]["prompt"]),
                ),
                min(
                    len(main_payloads[1]["prompt"]),
                    len(main_payloads[2]["prompt"]),
                ),
            ]
            self.assertEqual(
                [
                    row["expected_reusable_prefix_tokens"]
                    for row in main_rows
                ],
                expected,
            )

    def test_default_http_executor_reaches_configured_concurrency(self):
        concurrency = 40
        barrier = threading.Barrier(concurrency)
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_request(_url, *, payload, timeout_s):
            nonlocal active, max_active
            self.assertGreater(timeout_s, 0)
            methods = payload["svllm_method_preference"].split(",")
            if methods == ["snapkv"]:
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait(timeout=5)
                finally:
                    with lock:
                        active -= 1
                worker = "http://snap-worker"
                method = "snapkv"
            else:
                worker = "http://main-worker"
                method = "omnikv"
            prompt_tokens = len(payload["prompt"])
            completion_tokens = payload["max_tokens"]
            return _json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic output",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
                headers={
                    "x-sparsevllm-worker": worker,
                    "x-sparsevllm-route-reason": "test",
                    "x-sparsevllm-sparse-method": method,
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "rounds": 1,
                    "articles_per_round": concurrency,
                }
            )
            with patch.object(run, "_request_bytes", side_effect=fake_request):
                aggregate = asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                    )
                )
            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(max_active, concurrency)

    def test_wrong_subagent_method_is_metric_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(wrong_subagent_method=True)

            with self.assertRaisesRegex(run.BenchmarkFailed, "metric_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "metric_failed")
            self.assertEqual(aggregate["status_counts"]["metric_failed"], 3)

    def test_fails_when_router_has_only_one_worker(self):
        async def one_worker_get(url, _timeout_s):
            if url.endswith("/models"):
                return _json_response(
                    {
                        "data": [
                            {
                                "id": "sim-model",
                                "owned_by": "sparsevllm-router",
                                "max_model_len": 65536,
                            }
                        ]
                    }
                )
            if url.endswith("/v1/worker/info"):
                return _json_response(
                    {
                        "served_model_name": "sim-model",
                        "benchmark_config": {},
                    }
                )
            return _json_response(
                {
                    "status": "ok",
                    "healthy_workers": ["http://only-worker"],
                    "router_policy": {
                        "request_timeout_s": 900.0,
                        "overload_load_factor": 1.0,
                        "load_abs_threshold": 0,
                        "profiles": {},
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "invalid_input"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=one_worker_get,
                        post_fn=FakeService().post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "invalid_input")

    def test_preflight_ignores_healthy_workers_for_other_models(self):
        service = FakeService()

        async def multi_model_get(url, timeout_s):
            if url.endswith("/health"):
                response = await service.get(url, timeout_s)
                payload = json.loads(response.body)
                payload["healthy_workers"].append("http://other-worker")
                return _json_response(payload)
            if url == "http://other-worker/v1/worker/info":
                return _json_response(
                    {
                        "served_model_name": "other-model",
                        "benchmark_config": {},
                    }
                )
            return await service.get(url, timeout_s)

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            preflight_info = asyncio.run(
                run.preflight(config, get_fn=multi_model_get)
            )

        self.assertEqual(
            [worker["url"] for worker in preflight_info["workers"]],
            ["http://snap-worker", "http://main-worker"],
        )

    def test_preflight_rejects_insufficient_client_timeout_margin(self):
        service = FakeService()

        async def insufficient_margin_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url.endswith("/health"):
                payload = json.loads(response.body)
                payload["router_policy"]["request_timeout_s"] = 920.0
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "must exceed the router upstream timeout",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=insufficient_margin_get)
                )

    def test_preflight_rejects_missing_main_agent_method(self):
        service = FakeService()

        async def snapkv_only_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url.endswith("/v1/worker/info"):
                payload = json.loads(response.body)
                payload["sparse_method"] = "snapkv"
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "configured main agent methods",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=snapkv_only_get)
                )

    def test_git_dirty_is_unknown_when_probe_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with patch.object(run, "_git_command", return_value=None):
                run_info = run._run_info(
                    config,
                    None,
                    status="running",
                )

        self.assertIsNone(run_info["git_dirty"])

    def test_direct_server_payload_omits_router_only_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "require_router": False,
                }
            )
            spec = run.RequestSpec(
                sample_id="direct",
                phase="subagent",
                round_index=0,
                request_index=0,
                prompt_tokens=8,
                completion_tokens=2,
                prompt_seed=1,
                method_preferences=("snapkv",),
            )
            payload = run.build_payload(spec, config)
            self.assertNotIn("svllm_method_preference", payload)


if __name__ == "__main__":
    unittest.main()
