import io
import json
import sys
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

    def test_quality_command_can_enable_tp_prefix_graph_for_supported_methods(self):
        cmd = run_suite._quality_command(
            model_id="qwen25_7b",
            method_id="quest",
            model={"model_path": "/models/qwen", "tokenizer_path": "/models/qwen"},
            method={
                "sparse_method": "quest",
                "config": {"sparse_method": "quest", "quest_chunk_size": 16},
            },
            quality={
                **self._quality_cfg(),
                "enable_prefix_caching": True,
                "prefix_cache_block_size": 16,
                "enable_profiler": True,
            },
            performance={
                "decode_cuda_graph": True,
                "enforce_eager": False,
                "tensor_parallel_size": 2,
            },
            output_root=Path("/tmp/sparsevllm-quality"),
        )

        hyper_params = json.loads(cmd[cmd.index("--hyper_param") + 1])
        self.assertEqual(hyper_params["tensor_parallel_size"], 2)
        self.assertTrue(hyper_params["decode_cuda_graph"])
        self.assertTrue(hyper_params["enable_prefix_caching"])
        self.assertTrue(hyper_params["enable_profiler"])
        self.assertEqual(hyper_params["prefix_cache_block_size"], 16)
        self.assertFalse(hyper_params["decode_cuda_graph_capture_sampling"])

    def test_quality_command_rejects_prefix_cache_for_unsupported_methods(self):
        with self.assertRaisesRegex(ValueError, "enable_prefix_caching"):
            run_suite._quality_command(
                model_id="qwen25_7b",
                method_id="snapkv",
                model={"model_path": "/models/qwen", "tokenizer_path": "/models/qwen"},
                method={"sparse_method": "snapkv", "config": {"sparse_method": "snapkv"}},
                quality={**self._quality_cfg(), "enable_prefix_caching": True},
                performance={
                    "decode_cuda_graph": True,
                    "enforce_eager": False,
                    "tensor_parallel_size": 2,
                },
                output_root=Path("/tmp/sparsevllm-quality"),
            )

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

    def test_tp_decode_graph_command_accepts_quest_v11(self):
        cmd = run_suite._quality_command(
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

        hyper_params = json.loads(cmd[cmd.index("--hyper_param") + 1])
        self.assertEqual(hyper_params["tensor_parallel_size"], 2)
        self.assertTrue(hyper_params["decode_cuda_graph"])
        self.assertFalse(hyper_params["decode_cuda_graph_capture_sampling"])

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
            scbench={**base, "decode_cuda_graph": True, "enforce_eager": False, "tensor_parallel_size": 2},
            output_dir=Path("/tmp/scbench"),
        )
        self.assertIn("--decode_cuda_graph", graph_cmd)
        self.assertIn("--no-enforce_eager", graph_cmd)
        self.assertEqual(graph_cmd[graph_cmd.index("--tensor_parallel_size") + 1], "2")

    def test_stress_command_can_require_prefix_cache_hit(self):
        cmd = run_suite._stress_command(
            model_id="qwen25_7b",
            model={"model_path": "/models/qwen", "tokenizer_path": "/models/qwen"},
            method_id="omnikv",
            method={
                "sparse_method": "omnikv",
                "config": {"sparse_method": "omnikv"},
            },
            performance={
                "decode_cuda_graph": True,
                "enforce_eager": False,
                "tensor_parallel_size": 2,
            },
            stress={
                "length": 1024,
                "request_counts": [4],
                "output_len": 8,
                "max_decode_steps_after_full": 2,
                "enable_prefix_caching": True,
                "prefix_cache_block_size": 16,
                "require_prefix_cache_hit": True,
                "enable_profiler": True,
            },
            output_jsonl=Path("/tmp/stress.jsonl"),
        )

        hyper_params = json.loads(cmd[cmd.index("--hyper_params") + 1])
        self.assertTrue(hyper_params["enable_prefix_caching"])
        self.assertTrue(hyper_params["enable_profiler"])
        self.assertTrue(hyper_params["decode_cuda_graph"])
        self.assertEqual(hyper_params["tensor_parallel_size"], 2)
        self.assertIn("--require_prefix_cache_hit", cmd)
        self.assertEqual(cmd[cmd.index("--admission_wave_size") + 1], "2")

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

    def test_validate_layer_records_quick_regression_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run_suite.py",
                "--layer",
                "validate",
                "--models",
                "qwen3_4b",
                "--methods",
                "vanilla",
                "--quality_tasks",
                "qasper,hotpotqa",
                "--quality_batch_size",
                "2",
                "--quality_samples_per_task",
                "2",
                "--quality_min_required_samples",
                "2",
                "--quality_sparsevllm_max_num_seqs_in_batch",
                "2",
                "--quality_sparsevllm_max_decoding_seqs",
                "2",
                "--scbench_tasks",
                "scbench_kv",
                "--scbench_num_eval_examples",
                "1",
                "--scbench_batch_size",
                "1",
                "--stress_length",
                "512",
                "--stress_request_counts",
                "2",
                "--stress_output_len",
                "2",
                "--stress_max_num_seqs_in_batch",
                "2",
                "--stress_max_decoding_seqs",
                "2",
                "--stress_max_decode_steps_after_full",
                "1",
                "--enable_profiler",
                "--run_id",
                "unit_quick_overrides",
                "--output_root",
                tmp,
            ]

            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(run_suite.main(), 0)

            run_root = Path(tmp) / "sparsevllm_regression" / "unit_quick_overrides"
            with (run_root / "resolved_manifest.json").open("r", encoding="utf-8") as handle:
                resolved = json.load(handle)

            self.assertEqual(resolved["quality"]["tasks"], ["qasper", "hotpotqa"])
            self.assertEqual(resolved["quality"]["batch_size"], 2)
            self.assertEqual(resolved["quality"]["samples_per_task"], 2)
            self.assertEqual(resolved["quality"]["min_required_samples"], 2)
            self.assertEqual(resolved["quality"]["sparsevllm_max_num_seqs_in_batch"], 2)
            self.assertEqual(resolved["quality"]["sparsevllm_max_decoding_seqs"], 2)
            self.assertEqual(resolved["scbench"]["tasks"], ["scbench_kv"])
            self.assertEqual(resolved["scbench"]["num_eval_examples"], 1)
            self.assertEqual(resolved["scbench"]["batch_size"], 1)
            self.assertEqual(resolved["stress"]["length"], 512)
            self.assertEqual(resolved["stress"]["request_counts"], [2])
            self.assertEqual(resolved["stress"]["output_len"], 2)
            self.assertEqual(resolved["stress"]["max_num_seqs_in_batch"], 2)
            self.assertEqual(resolved["stress"]["max_decoding_seqs"], 2)
            self.assertEqual(resolved["stress"]["max_decode_steps_after_full"], 1)
            self.assertTrue(resolved["quality"]["enable_profiler"])
            self.assertTrue(resolved["performance"]["enable_profiler"])
            self.assertTrue(resolved["stress"]["enable_profiler"])

    def test_run_command_timeout_records_and_terminates(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "timeout.log"
            cmd = [sys.executable, "-c", "import time; time.sleep(10)"]

            with self.assertRaises(run_suite.CommandExecutionError) as raised:
                run_suite._run_command(
                    cmd,
                    cwd=Path.cwd(),
                    dry_run=False,
                    log_path=log_path,
                    timeout_s=0.1,
                )

            self.assertEqual(raised.exception.record["status"], "timeout")
            self.assertEqual(raised.exception.record["timeout_s"], 0.1)
            self.assertIn("command exceeded timeout_s=0.1", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
