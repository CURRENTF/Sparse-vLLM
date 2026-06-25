import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark.math_bench.eval import evaluate_record, extract_pred_answer


class MathBenchEvalTest(unittest.TestCase):
    def test_trailing_boxed_marker_is_parse_failed_not_exception(self):
        self.assertIsNone(extract_pred_answer("reasoning...\n\\boxed"))

    def test_boxed_without_braces_is_parsed_by_math_verify_expression_extractor(self):
        self.assertEqual(extract_pred_answer("Final answer: \\boxed 42\n"), "42")

    def test_equivalent_math_expression_is_correct(self):
        _, result = evaluate_record(
            {
                "id": "frac",
                "pred": "Therefore, the final answer is: $\\boxed{0.5}$.",
                "gold": {"solution": "The answer is $\\boxed{\\frac{1}{2}}$.", "answer": "\\frac{1}{2}"},
            },
            dataset="math500",
            parse_timeout=5.0,
            verify_timeout=5.0,
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["correct"])

    def test_cli_writes_result_per_sample_and_parsed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred_path = tmp_path / "math500.jsonl"
            pred_path.write_text(
                json.dumps(
                    {
                        "id": "1",
                        "pred": "Therefore, the final answer is: $\\boxed{42}$.",
                        "gold": {"solution": "The answer is $\\boxed{42}$.", "answer": "42"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, "benchmark/math_bench/eval.py", "--path", str(tmp_path)],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn('"pass@1": 100.0', completed.stdout)
            result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["math500"]["metric"], "math_verify")
            self.assertEqual(result["math500"]["correct"], 1)
            self.assertEqual(len((tmp_path / "math500_parsed_outputs.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((tmp_path / "math500_per_sample_results.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
