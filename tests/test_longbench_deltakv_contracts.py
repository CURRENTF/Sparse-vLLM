import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.long_bench import pred as longbench_pred
from benchmark.long_bench.metrics import classification_score, qa_f1_score
from deltakv.configs.runtime_params import normalize_runtime_params
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_COMPRESSED_LATENT_WO_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
)


class LongBenchDeltaKVContractsTest(unittest.TestCase):
    def test_no_chat_datasets_remain_raw_for_every_thinking_mode(self):
        for dataset in longbench_pred.NO_CHAT_TEMPLATE_DATASETS:
            for thinking_mode in ("off", "on", "on_strip"):
                with self.subTest(dataset=dataset, thinking_mode=thinking_mode):
                    self.assertFalse(
                        longbench_pred.should_use_chat_template(dataset, thinking_mode=thinking_mode)
                    )

    def test_chat_template_policy_matches_regular_and_kvzip_prompt_paths(self):
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="off")
        )
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="on_strip")
        )
        self.assertFalse(
            longbench_pred.should_use_chat_template("hotpotqa", no_chat_template=True)
        )

    def test_hotpotqa_and_trec_metric_contracts(self):
        self.assertEqual(qa_f1_score("Paris", "Paris"), 1.0)
        self.assertEqual(qa_f1_score("Paris", "London"), 0)
        self.assertEqual(
            classification_score(
                "DESC",
                "DESC",
                all_classes=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
            ),
            1.0,
        )

    def test_longbench_data_validation_fails_fast_for_missing_hotpotqa_and_trec(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = longbench_pred.DATA_PREFIX_PATH
            longbench_pred.DATA_PREFIX_PATH = str(Path(tmp) / "missing")
            try:
                with self.assertRaisesRegex(FileNotFoundError, "LongBench data root"):
                    longbench_pred.validate_longbench_data_paths(["hotpotqa", "trec"], use_longbench_e=False)
            finally:
                longbench_pred.DATA_PREFIX_PATH = old_root

    def test_longbench_data_validation_requires_explicit_root(self):
        old_root = longbench_pred.DATA_PREFIX_PATH
        longbench_pred.DATA_PREFIX_PATH = None
        try:
            with self.assertRaisesRegex(FileNotFoundError, "DELTAKV_LONGBENCH_DATA_DIR"):
                longbench_pred.validate_longbench_data_paths(["hotpotqa"], use_longbench_e=False)
        finally:
            longbench_pred.DATA_PREFIX_PATH = old_root

    def test_longbench_prompt_budget_reserves_generation_tokens(self):
        self.assertEqual(
            longbench_pred._prompt_token_budget(204_800, 121_000, 64),
            120_936,
        )
        with self.assertRaisesRegex(ValueError, "smaller than"):
            longbench_pred._prompt_token_budget(32, 64, 32)

    def test_longbench_eval_rejects_skipped_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "hotpotqa.jsonl").write_text(
                json.dumps(
                    {
                        "status": "skipped_by_policy",
                        "pred": "",
                        "answers": None,
                        "all_classes": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmark/long_bench/eval.py",
                    "--path",
                    str(output_dir),
                ],
                cwd=longbench_pred.REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            metrics = json.loads((output_dir / "result.json").read_text())
            self.assertEqual(metrics["status"], "failed")
            self.assertEqual(
                metrics["task_statuses"]["hotpotqa"]["status"],
                "skipped_by_policy",
            )

    def test_new_hf_sparse_methods_route_without_legacy_names(self):
        for sparse_method in (
            DELTA_COMPRESSED_LATENT_WO_FULL,
            DELTA_COMPRESSED_LATENT_W_FULL,
            DELTA_ORIGIN_WO_FULL,
            DELTA_ORIGIN_W_FULL,
        ):
            with self.subTest(sparse_method=sparse_method):
                normalized = normalize_runtime_params({"sparse_method": sparse_method}, backend="hf")
                self.assertEqual(normalized.hf_model_cls, sparse_method)
                self.assertNotIn("origin_residual", sparse_method)
                self.assertNotIn("full_deltakv", sparse_method)


if __name__ == "__main__":
    unittest.main()
