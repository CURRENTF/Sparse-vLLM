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
    def _quality_cfg(self):
        return {
            "tasks": ["hotpotqa"],
            "batch_size": 4,
            "sparsevllm_max_num_seqs_in_batch": 4,
            "sparsevllm_max_decoding_seqs": 4,
            "min_prompt_tokens": 0,
            "samples_per_task": 2,
            "min_required_samples": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
        }

    def test_quality_command_uses_engine_tp_without_changing_longbench_workers(self):
        cmd = run_suite._quality_command(
            model_id="qwen25_7b",
            method_id="snapkv",
            model={"model_path": "/models/qwen", "tokenizer_path": "/models/qwen"},
            method={"sparse_method": "snapkv", "config": {"sparse_method": "snapkv"}},
            quality=self._quality_cfg(),
            performance={
                "decode_cuda_graph": True,
                "enforce_eager": False,
                "tensor_parallel_size": 2,
            },
            output_root=Path("/tmp/sparsevllm-quality"),
        )

        self.assertEqual(cmd[cmd.index("--ws") + 1], "1")
        hyper_params = json.loads(cmd[cmd.index("--hyper_param") + 1])
        self.assertEqual(hyper_params["tensor_parallel_size"], 2)
        self.assertTrue(hyper_params["decode_cuda_graph"])
        self.assertFalse(hyper_params["decode_cuda_graph_capture_sampling"])

    def test_perf_command_uses_tp_decode_graph_hyper_params(self):
        cmd = run_suite._perf_command(
            model_id="qwen25_7b",
            model={"model_path": "/models/qwen", "tokenizer_path": "/models/qwen"},
            method_id="snapkv",
            method={"sparse_method": "snapkv", "config": {"sparse_method": "snapkv"}},
            performance={
                "lengths": [1024],
                "batch_sizes": [2],
                "output_len": 8,
                "decode_cuda_graph": True,
                "enforce_eager": False,
                "tensor_parallel_size": 2,
            },
            output_jsonl=Path("/tmp/perf.jsonl"),
        )

        hyper_params = json.loads(cmd[cmd.index("--hyper_params") + 1])
        self.assertEqual(hyper_params["tensor_parallel_size"], 2)
        self.assertTrue(hyper_params["decode_cuda_graph"])
        self.assertFalse(hyper_params["decode_cuda_graph_capture_sampling"])

    def test_tp_decode_graph_command_rejects_quest_v1(self):
        with self.assertRaisesRegex(ValueError, "v1 gate"):
            run_suite._quality_command(
                model_id="qwen25_7b",
                method_id="quest",
                model={"model_path": "/models/qwen", "tokenizer_path": "/models/qwen"},
                method={"sparse_method": "quest", "config": {"sparse_method": "quest"}},
                quality=self._quality_cfg(),
                performance={
                    "decode_cuda_graph": True,
                    "enforce_eager": False,
                    "tensor_parallel_size": 2,
                },
                output_root=Path("/tmp/sparsevllm-quality"),
            )

    def test_scbench_command_can_enable_decode_graph_without_changing_default(self):
        base = {
            "tasks": ["scbench_kv"],
            "num_eval_examples": 1,
            "max_turns": 2,
            "max_seq_length": 1024,
            "batch_size": 2,
            "prefix_cache_block_size": 16,
        }

        default_cmd = run_suite._scbench_command(
            manifest_path=Path("/tmp/manifest.json"),
            model_id="qwen3_4b",
            method_ids=["vanilla"],
            scbench=base,
            output_dir=Path("/tmp/scbench"),
        )
        self.assertNotIn("--decode_cuda_graph", default_cmd)
        self.assertNotIn("--no-enforce_eager", default_cmd)

        graph_cmd = run_suite._scbench_command(
            manifest_path=Path("/tmp/manifest.json"),
            model_id="qwen3_4b",
            method_ids=["vanilla"],
            scbench={**base, "decode_cuda_graph": True, "enforce_eager": False},
            output_dir=Path("/tmp/scbench"),
        )
        self.assertIn("--decode_cuda_graph", graph_cmd)
        self.assertIn("--no-enforce_eager", graph_cmd)

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
