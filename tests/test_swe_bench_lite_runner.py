import json
import tempfile
import unittest
from pathlib import Path

from benchmark.swe_bench_lite.run import (
    RunnerError,
    SweBenchLiteRunner,
    merge_batch_predictions,
    normalize_results,
    render_mini_config,
    validate_predictions,
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
            with self.assertRaisesRegex(RunnerError, "API key"):
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
        self.assertIn("step_limit: 80", config)
        self.assertNotIn("api_key", config)
        self.assertNotIn("thinking", config)

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
