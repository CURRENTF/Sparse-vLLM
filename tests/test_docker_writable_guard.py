import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.swe_bench_lite.docker_writable_guard import (
    DockerWritableLayerGuard,
    DockerWritableLayerGuardError,
    DockerWritableLayerLimitExceeded,
)


class DockerWritableLayerGuardTest(unittest.TestCase):
    def _guard(self, root: Path, **kwargs) -> DockerWritableLayerGuard:
        return DockerWritableLayerGuard(
            executable="docker",
            limit_bytes=4 * 1024**3,
            poll_seconds=1.0,
            events_path=root / "events.jsonl",
            run_id="test-run",
            autostart=False,
            **kwargs,
        )

    @mock.patch("benchmark.swe_bench_lite.docker_writable_guard.subprocess.run")
    def test_only_container_over_limit_is_killed(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = self._guard(root)
            below = guard.register("below", image="image-below")
            above = guard.register(
                "above",
                image="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
            )
            records = "\n".join(
                (
                    json.dumps({"Id": "below", "SizeRw": 4 * 1024**3}),
                    json.dumps({"Id": "above", "SizeRw": 4 * 1024**3 + 1}),
                )
            )
            run_mock.side_effect = (
                subprocess.CompletedProcess([], 0, stdout=records, stderr=""),
                subprocess.CompletedProcess([], 0, stdout="above", stderr=""),
            )

            guard._poll_once()

            below.raise_if_failed()
            with self.assertRaises(DockerWritableLayerLimitExceeded):
                above.raise_if_failed()
            self.assertEqual(
                run_mock.call_args_list[1].args[0], ["docker", "kill", "above"]
            )
            event = json.loads((root / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "limit_exceeded")
            self.assertEqual(event["container_id"], "above")
            self.assertEqual(event["instance_id"], "astropy__astropy-12907")
            self.assertEqual(event["writable_bytes"], 4 * 1024**3 + 1)

    @mock.patch("benchmark.swe_bench_lite.docker_writable_guard.subprocess.run")
    def test_repeated_monitor_failure_fails_closed(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            guard = self._guard(Path(tmp), max_monitor_failures=3)
            state = guard.register("sample", image="image")
            run_mock.side_effect = (
                subprocess.CompletedProcess([], 1, stdout="", stderr="daemon error"),
                subprocess.CompletedProcess([], 1, stdout="", stderr="daemon error"),
                subprocess.CompletedProcess([], 1, stdout="", stderr="daemon error"),
                subprocess.CompletedProcess([], 0, stdout="sample", stderr=""),
            )

            guard._poll_once()
            guard._poll_once()
            state.raise_if_failed()
            guard._poll_once()

            with self.assertRaises(DockerWritableLayerGuardError):
                state.raise_if_failed()
            self.assertEqual(
                run_mock.call_args_list[-1].args[0], ["docker", "kill", "sample"]
            )

    @mock.patch("benchmark.swe_bench_lite.docker_writable_guard.subprocess.run")
    def test_invalid_size_terminates_the_affected_sample(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            guard = self._guard(Path(tmp))
            state = guard.register("sample", image="image")
            run_mock.side_effect = (
                subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=json.dumps({"Id": "sample", "SizeRw": None}),
                    stderr="",
                ),
                subprocess.CompletedProcess([], 0, stdout="sample", stderr=""),
            )

            guard._poll_once()

            with self.assertRaises(DockerWritableLayerGuardError):
                state.raise_if_failed()


if __name__ == "__main__":
    unittest.main()
