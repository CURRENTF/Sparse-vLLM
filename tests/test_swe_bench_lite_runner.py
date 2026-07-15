import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from benchmark.swe_bench_lite.run import (
    RunnerError,
    SweBenchLiteRunner,
    _canonical_hash,
    _instance_image_names,
    _reject_secrets,
    _require_local_images,
    assert_runtime_provenance_matches,
    build_official_run_id,
    merge_batch_predictions,
    normalize_results,
    render_mini_config,
    validate_completed_batch,
    validate_predictions,
    main,
)


def _prediction(instance_id: str, patch: str = "diff --git a/a b/a\n") -> dict:
    return {
        "instance_id": instance_id,
        "model_name_or_path": "openai/sparsevllm-swe",
        "model_patch": patch,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_trajectory(batch_dir: Path, instance_id: str, exit_status: str) -> None:
    _write_json(
        batch_dir / instance_id / f"{instance_id}.traj.json",
        {
            "info": {
                "exit_status": exit_status,
                "model_stats": {"api_calls": 3, "instance_cost": 0.0},
            }
        },
    )


class SweBenchLiteRunnerTest(unittest.TestCase):
    def test_instance_image_names_use_official_swebench_key_format(self):
        self.assertEqual(
            _instance_image_names([{"instance_id": "astropy__astropy-12907"}]),
            ["swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"],
        )

    def test_extra_mini_config_is_snapshotted_and_rejects_api_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "deepseek.yaml"
            source.write_text(
                "model:\n"
                "  model_kwargs:\n"
                "    extra_body:\n"
                "      thinking:\n"
                "        type: disabled\n",
                encoding="utf-8",
            )
            runner = object.__new__(SweBenchLiteRunner)
            runner.run_dir = root / "run"
            runner.extra_mini_configs = [source]

            runner._prepare_extra_mini_configs()

            snapshot = runner.extra_mini_config_snapshots[0]
            self.assertEqual(
                snapshot.read_text(encoding="utf-8"),
                source.read_text(encoding="utf-8"),
            )
            self.assertEqual(len(runner.extra_mini_config_records[0]["sha256"]), 64)

            source.write_text("model:\n  model_kwargs:\n    api_key: secret\n", encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "secret"):
                runner._prepare_extra_mini_configs()

    def test_rendered_local_config_has_api_base_but_no_api_key_or_provider_extra_body(self):
        config = render_mini_config(
            step_limit=80,
            cost_limit=0.0,
            wall_time_limit_seconds=1800,
            cost_tracking="ignore_errors",
            max_tokens=4096,
            temperature=0.0,
            top_p=1.0,
            api_base="http://127.0.0.1:18000/v1",
        )

        self.assertIn('api_base: "http://127.0.0.1:18000/v1"', config)
        self.assertIn('cost_tracking: "ignore_errors"', config)
        self.assertIn(
            "model_class: benchmark.swe_bench_lite.model.SparseVLLMLitellmModel",
            config,
        )
        self.assertIn("step_limit: 80", config)
        self.assertNotIn("api_key", config)
        self.assertNotIn("thinking", config)

    def test_model_environment_makes_benchmark_adapter_importable(self):
        runner = object.__new__(SweBenchLiteRunner)
        runner.repo_root = Path("/adapter").resolve()
        runner.args = Namespace(api_proxy_from_environment=False, offline_dataset=False)

        with mock.patch.dict("os.environ", {"PYTHONPATH": "/existing"}, clear=True):
            env = runner._model_env()

        self.assertEqual(env["PYTHONPATH"], f"{runner.repo_root}:/existing")

    def test_validate_predictions_rejects_missing_and_extra_ids(self):
        with self.assertRaisesRegex(RunnerError, "missing=.*b.*extra=.*c"):
            validate_predictions(
                {"a": _prediction("a"), "c": _prediction("c")},
                ["a", "b"],
                source=Path("preds.json"),
            )

    def test_merge_only_reads_declared_numeric_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first = run_dir / "batches" / "batch_000"
            second = run_dir / "batches" / "batch_001"
            backup = run_dir / "batches" / "batch_001.before_resume"
            _write_json(first / "preds.json", {"a": _prediction("a")})
            _write_json(second / "preds.json", {"b": _prediction("b")})
            _write_json(backup / "preds.json", {"a": _prediction("a")})
            _write_trajectory(first, "a", "Submitted")
            _write_trajectory(second, "b", "LimitsExceeded")

            combined, generation = merge_batch_predictions(run_dir, [["a"], ["b"]])

        self.assertEqual(set(combined), {"a", "b"})
        self.assertEqual(
            [row["exit_status"] for row in generation],
            ["Submitted", "LimitsExceeded"],
        )
        self.assertEqual(
            [row["status"] for row in generation],
            ["success", "success"],
        )

    def test_completed_batch_marker_must_match_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = Path(tmp)
            _write_json(batch_dir / "preds.json", {"a": _prediction("a")})
            _write_json(
                batch_dir / "batch_done.json",
                {"instances": 1, "predictions_sha256": "stale"},
            )

            with self.assertRaisesRegex(RunnerError, "marker"):
                validate_completed_batch(batch_dir, ["a"])

    def test_official_run_id_is_bound_to_prediction_hash(self):
        first = build_official_run_id("lite300", "a" * 64)
        second = build_official_run_id("lite300", "b" * 64)

        self.assertEqual(first, "lite300-pred-aaaaaaaaaaaa")
        self.assertNotEqual(first, second)

    def test_runtime_provenance_drift_is_rejected(self):
        expected = {"adapter_git": {"commit": "a"}, "packages": {"swebench": "1"}}
        current = {"adapter_git": {"commit": "b"}, "packages": {"swebench": "1"}}

        with self.assertRaisesRegex(RunnerError, "provenance drift"):
            assert_runtime_provenance_matches(expected, current)

    def test_secret_validation_rejects_provider_tokens_and_url_credentials(self):
        bad_values = (
            {"access_token": "plain-token-value"},
            {"header": "Bearer abcdefghijklmnop"},
            {"model": {"value": "hf_abcdefghijklmnopqrstuvwxyz"}},
            {"endpoint": "https://user:password@example.com/v1"},
            {"endpoint": "https://example.com/v1?api_key=secret"},
        )

        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(RunnerError, "secret"):
                    _reject_secrets(value, source=Path("config.yaml"))

    @mock.patch("benchmark.swe_bench_lite.run.subprocess.run")
    def test_docker_daemon_failure_is_not_reported_as_missing_images(self, run_mock):
        run_mock.return_value = subprocess.CompletedProcess(
            ["docker", "info"], 1, stdout="", stderr="daemon unavailable"
        )

        with self.assertRaisesRegex(RunnerError, "Docker daemon"):
            _require_local_images(["swebench/image:latest"])

    def test_summarize_stage_does_not_call_prepare(self):
        runner = object.__new__(SweBenchLiteRunner)
        runner.args = Namespace(stage="summarize")
        runner.prepare = mock.Mock(side_effect=AssertionError("prepare must not run"))
        runner._load_artifact_context = mock.Mock()
        runner.summarize = mock.Mock()

        runner.run()

        runner._load_artifact_context.assert_called_once_with()
        runner.summarize.assert_called_once_with()

    @mock.patch("benchmark.swe_bench_lite.run._run_logged")
    def test_official_evaluation_runs_from_artifact_directory(self, run_logged):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = object.__new__(SweBenchLiteRunner)
            runner.args = Namespace(
                dataset="SWE-bench/SWE-bench_Lite",
                split="test",
                eval_workers=1,
                eval_timeout=60,
            )
            runner.run_dir = root
            runner.official_dir = root / "official"
            runner.predictions_path = root / "preds_all.json"
            runner.manifest_path = root / "run_manifest.json"
            runner.evaluation_identity_path = root / "evaluation_identity.json"
            runner.status_path = root / "status.jsonl"
            runner.instance_ids = ["a"]
            runner.run_id = "logical-run"
            runner.official_run_id = None
            runner._model_env = mock.Mock(return_value={})
            predictions = {"a": _prediction("a")}
            _write_json(runner.predictions_path, predictions)
            _write_json(
                root / "prediction_merge_summary.json",
                {"predictions_sha256": _canonical_hash(predictions)},
            )
            _write_json(runner.manifest_path, {"runtime_provenance": {"commit": "a"}})

            def fake_run(command, *, cwd, env, log_path):
                self.assertEqual(cwd, runner.official_dir)
                official_run_id = command[command.index("--run_id") + 1]
                _write_json(
                    cwd / f"openai__sparsevllm-swe.{official_run_id}.json",
                    {"total_instances": 1},
                )

            run_logged.side_effect = fake_run
            runner.evaluate()

            self.assertTrue(runner._official_report_path().is_file())
            marker = (
                runner.official_dir
                / "logs"
                / "run_evaluation"
                / str(runner.official_run_id)
                / ".sparsevllm_adapter_identity.json"
            )
            self.assertTrue(marker.is_file())

    def test_summarize_requires_only_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = {"a": _prediction("a")}
            predictions_hash = _canonical_hash(predictions)
            official_run_id = build_official_run_id("artifact-run", predictions_hash)
            _write_json(
                root / "run_config.json",
                {
                    "run_id": "artifact-run",
                    "instance_ids": ["a"],
                    "batch_size": 1,
                    "model": "openai/sparsevllm-swe",
                    "dataset": "SWE-bench/SWE-bench_Lite",
                    "split": "test",
                },
            )
            _write_json(root / "preds_all.json", predictions)
            _write_json(
                root / "evaluation_identity.json",
                {
                    "logical_run_id": "artifact-run",
                    "official_run_id": official_run_id,
                    "predictions_sha256": predictions_hash,
                    "runtime_provenance_sha256": "a" * 64,
                },
            )
            batch = root / "batches" / "batch_000"
            _write_json(batch / "preds.json", predictions)
            _write_trajectory(batch, "a", "Submitted")
            _write_json(
                root / "official" / f"model.{official_run_id}.json",
                {
                    "total_instances": 1,
                    "submitted_instances": 1,
                    "completed_instances": 1,
                    "resolved_instances": 1,
                    "unresolved_instances": 0,
                    "empty_patch_instances": 0,
                    "error_instances": 0,
                    "completed_ids": ["a"],
                    "resolved_ids": ["a"],
                    "unresolved_ids": [],
                    "empty_patch_ids": [],
                    "error_ids": [],
                },
            )

            result = main(["--stage", "summarize", "--run-dir", str(root)])

            self.assertEqual(result, 0)
            summary = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["resolved_instances"], 1)

    def test_generation_failure_accounts_for_unattempted_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = object.__new__(SweBenchLiteRunner)
            runner.run_dir = root
            runner.generation_results_path = root / "generation_results.jsonl"
            runner.instance_ids = ["a", "b"]
            runner.args = Namespace(batch_size=1)

            runner._write_generation_failure_results(
                failed_batch_index=0, error=RunnerError("model failed")
            )

            rows = [
                json.loads(line)
                for line in runner.generation_results_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                {row["instance_id"]: row["status"] for row in rows},
                {"a": "model_failed", "b": "skipped_by_policy"},
            )

    def test_normalize_results_separates_model_and_metric_failures(self):
        expected = ["a", "b", "c", "d"]
        predictions = {
            "a": _prediction("a"),
            "b": _prediction("b"),
            "c": _prediction("c", patch=""),
            "d": _prediction("d"),
        }
        generation = [
            {
                "instance_id": instance_id,
                "status": "model_failed" if instance_id == "c" else "success",
                "exit_status": exit_status,
                "has_patch": bool(predictions[instance_id]["model_patch"]),
                "model_patch_len": len(predictions[instance_id]["model_patch"]),
                "model_stats": {"api_calls": 2, "instance_cost": 0.1},
                "trajectory_path": f"/{instance_id}.traj.json",
            }
            for instance_id, exit_status in (
                ("a", "Submitted"),
                ("b", "Submitted"),
                ("c", "LimitsExceeded"),
                ("d", "Submitted"),
            )
        ]
        official = {
            "total_instances": 4,
            "submitted_instances": 4,
            "completed_instances": 2,
            "resolved_instances": 1,
            "unresolved_instances": 1,
            "empty_patch_instances": 1,
            "error_instances": 1,
            "completed_ids": ["a", "b"],
            "resolved_ids": ["a"],
            "unresolved_ids": ["b"],
            "empty_patch_ids": ["c"],
            "error_ids": ["d"],
        }

        rows, summary = normalize_results(
            expected_ids=expected,
            predictions=predictions,
            generation_rows=generation,
            official_report=official,
        )

        self.assertEqual(
            {row["instance_id"]: row["status"] for row in rows},
            {"a": "success", "b": "success", "c": "model_failed", "d": "metric_failed"},
        )
        self.assertEqual(summary["resolved_instances"], 1)
        self.assertEqual(summary["score"], 0.25)
        self.assertEqual(summary["total_api_calls"], 8)
        self.assertAlmostEqual(summary["total_instance_cost"], 0.4)

    def test_missing_trajectory_is_parse_failed(self):
        predictions = {"a": _prediction("a")}
        generation = [
            {
                "instance_id": "a",
                "status": "parse_failed",
                "exit_status": None,
                "has_patch": True,
                "model_patch_len": 10,
                "model_stats": {},
                "trajectory_path": None,
            }
        ]
        official = {
            "total_instances": 1,
            "submitted_instances": 1,
            "completed_instances": 0,
            "resolved_instances": 0,
            "unresolved_instances": 0,
            "empty_patch_instances": 0,
            "error_instances": 0,
            "completed_ids": [],
            "resolved_ids": [],
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
        }

        rows, _ = normalize_results(
            expected_ids=["a"],
            predictions=predictions,
            generation_rows=generation,
            official_report=official,
        )

        self.assertEqual(rows[0]["status"], "parse_failed")


if __name__ == "__main__":
    unittest.main()
