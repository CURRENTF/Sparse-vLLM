import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmark.sparsevllm_regression import run_suite
from benchmark.sparsevllm_regression.manifest import REQUIRED_ARTIFACTS


class SparseVLLMRegressionSuiteTest(unittest.TestCase):
    def test_validate_layer_runs_as_a_standard_test_and_writes_required_artifacts(self):
        # Keep the default tests on the cheap validate layer only. Full
        # quality/logits/perf/stress regression runs require model paths,
        # checkpoints, datasets, and GPUs, so they must be launched explicitly
        # through benchmark/sparsevllm_regression/run_suite.py.
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run_suite.py",
                "--layer",
                "validate",
                "--models",
                "qwen25_7b",
                "--methods",
                "vanilla",
                "--run_id",
                "unit_validate",
                "--output_root",
                tmp,
            ]

            stdout = io.StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                self.assertEqual(run_suite.main(), 0)
            self.assertIn("[validate] manifest ok:", stdout.getvalue())

            run_root = Path(tmp) / "sparsevllm_regression" / "unit_validate"
            missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (run_root / name).exists()]
            self.assertEqual(missing_artifacts, [])

            with (run_root / "grade_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["layer"], "validate")
            self.assertEqual(summary["models"], ["qwen25_7b"])
            self.assertEqual(summary["methods"], ["vanilla"])
            self.assertEqual(summary["commands"], [])
            self.assertEqual(summary["skipped"], [])

            for jsonl_name in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "perf.jsonl"):
                self.assertEqual((run_root / jsonl_name).read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
