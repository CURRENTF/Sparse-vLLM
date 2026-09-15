"""CPU contracts: preserve failures, distinguish cost skips, reject lost samples."""
import importlib.util
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe = load("run")
timed = load("timed_mini")


def fake_instance(instance, output_dir, fail=False):
    if fail:
        raise ValueError("original failure")
    return {"unchanged": instance}


class RecipeContracts(unittest.TestCase):
    def test_stop_server_terminates_only_recorded_process_group(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "snapkv-chain/smoke"
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            reaper = threading.Thread(target=process.wait)
            reaper.start()
            recipe.write(directory / "server_process.json", {
                "pid": process.pid,
                "process_group": os.getpgid(process.pid),
                "start_ticks": recipe.process_start_ticks(process.pid),
                "hostname": socket.gethostname(),
                "boot_id": recipe.boot_id(),
                "command": "test sleep",
            })
            try:
                with patch("builtins.print"):
                    recipe.stop_server(SimpleNamespace(
                        root=root, method="snapkv-chain", phase="smoke"
                    ))
            finally:
                if process.poll() is None:
                    process.kill()
                reaper.join(timeout=5)
            self.assertEqual(process.returncode, -15)
            self.assertEqual(
                recipe.read(directory / "server_stop.json")["status"],
                "stop_requested",
            )

    def test_server_manifest_serializes_command_for_driver_validation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipe.write(root / "snapkv-chain/engine.json", {
                "sparse_method": "snapkv", "max_model_len": 1024})
            recipe.write(root / "model/config.json", {"dtype": "bfloat16"})
            args = SimpleNamespace(
                root=root, method="snapkv-chain", phase="smoke",
                python=sys.executable, model=root / "model", gpus="4,5", port=0,
                timeout=1)
            with patch.object(recipe, "environment", return_value=(sys.executable, {})), \
                    patch.object(recipe, "idle_pair", return_value={"gpus": "", "processes": ""}), \
                    patch.object(recipe, "capture", return_value="test-value"), \
                    patch.object(recipe, "run_logged"):
                recipe.serve(args)
            manifest = recipe.read(root / "snapkv-chain/smoke/server_manifest.json")
            self.assertIsInstance(manifest["command"], str)
            self.assertIn("sparsevllm.entrypoints.openai.api_server", manifest["command"])

    def test_timing_preserves_return_and_exception(self):
        with tempfile.TemporaryDirectory() as root:
            wrapped = timed.instrument(fake_instance)
            with patch.object(timed.time, "perf_counter", side_effect=[10., 13.]):
                result = wrapped({"instance_id": "a"}, root)
            self.assertEqual(result, {"unchanged": {"instance_id": "a"}})
            self.assertEqual(recipe.read(Path(root) / "a/a.wall_time.json")["elapsed_s"], 3)
            with self.assertRaisesRegex(ValueError, "original failure"):
                wrapped({"instance_id": "b"}, root, fail=True)
            row = recipe.read(Path(root) / "b/b.wall_time.json")
            self.assertEqual(row["uncaught_exception"], "ValueError")
            with self.assertRaises(FileExistsError):
                wrapped({"instance_id": "a"}, root)

    def test_timeout_record_is_distinct_from_crash(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaises(RuntimeError):
                recipe.run_logged([sys.executable, "-c", "import time; time.sleep(10)"],
                                  recipe.environment(sys.executable)[1], root, "generate", .05)
            self.assertEqual(recipe.read(root / "generate.result.json")["status"], "timeout")

    def test_slow_gate_does_not_classify_crash_as_slow(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipe.write(root / "setting.json", {"slow_baseline_policy": {
                "pilot_instances": 64, "full_generation_limit_seconds": 43200, "slowdown_ratio": 4}})
            recipe.write(root / "snapkv-chain/pilot/generate.result.json",
                         {"status": "success", "elapsed_seconds": 100})
            path = root / "snapkv-no-chain/pilot/generate.result.json"
            recipe.write(path, {"status": "failed", "elapsed_seconds": 900})
            with self.assertRaisesRegex(ValueError, "implementation failure"):
                recipe.gate(root)
            path.write_text(json.dumps({"status": "timeout", "elapsed_seconds": 900}))
            self.assertEqual(recipe.gate(root)["status"], "skipped_by_policy")

    def test_collector_rejects_missing_or_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "snapkv-chain/smoke/benchmark"
            recipe.write(root / "setting.json", {"slow_baseline_policy": {"pilot_instances": 64}})
            recipe.write(directory / "final_summary.json", {"total_instances": 1})
            (directory / "instances.txt").write_text("expected\n")
            (directory / "generation_results.jsonl").write_text('{"instance_id":"wrong"}\n')
            with self.assertRaisesRegex(ValueError, "unexpected samples"):
                recipe.collect(SimpleNamespace(root=root, method="snapkv-chain", phase="smoke"))

    def test_collection_preserves_e2e_boundaries_and_cache_accounting(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "snapkv-chain/smoke"
            recipe.write(root / "setting.json", {"slow_baseline_policy": {"pilot_instances": 64}})
            recipe.write(directory / "benchmark/final_summary.json", {"total_instances": 1, "resolved_instances": 1})
            (directory / "benchmark/instances.txt").write_text("a\n")
            for name in ("generation_results", "per_sample_results"):
                (directory / f"benchmark/{name}.jsonl").write_text(json.dumps(
                    {"instance_id": "a", "status": "success", "resolved": True}) + "\n")
            recipe.write(directory / "benchmark/batches/batch_000/a/a.wall_time.json",
                         {"instance_id": "a", "elapsed_s": 12})
            for stage, duration in (("generate", 20), ("evaluate", 40)):
                recipe.write(directory / f"{stage}.result.json", {"status": "success", "elapsed_seconds": duration})
            recipe.write(directory / "server_requests/a.json", {
                "status": "success", "request_id": "a", "elapsed_s": 3,
                "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                       "prompt_tokens_details": {"cached_tokens": 60}}}})
            with patch("builtins.print"):
                recipe.collect(SimpleNamespace(root=root, method="snapkv-chain", phase="smoke"))
            report = recipe.read(directory / "report.json")
            self.assertEqual(report["requests"]["uncached_prompt_tokens"], 40)
            self.assertEqual(report["task_completion"]["resolved"]["p50_s"], 12)
            self.assertEqual(report["generation_stage"]["elapsed_seconds"], 20)
            self.assertEqual(report["official_evaluation_stage"]["elapsed_seconds"], 40)
            row = json.loads((directory / "request_samples.jsonl").read_text())
            self.assertIsNone(row["ttft_ms"])


if __name__ == "__main__":
    unittest.main()
