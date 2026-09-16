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
    @mock.patch("benchmark.swe_bench_lite.docker_writable_guard.subprocess.run")
    def test_expired_container_does_not_kill_healthy_samples_or_skip_size_checks(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = self._guard(root)
            expired = guard.register("expired", image="image")
            healthy = guard.register("healthy", image="image")
            over = guard.register("over", image="image")
            run_mock.side_effect = [
                subprocess.CompletedProcess([], 1, stdout="\n".join([
                    json.dumps({"Id": "healthy", "SizeRw": 1024}),
                    json.dumps({"Id": "over", "SizeRw": guard.limit_bytes + 1}),
                ]), stderr="error: no such object: expired"),
                subprocess.CompletedProcess([], 0, stdout="over", stderr=""),
            ]
            guard._poll_once()
            healthy.raise_if_failed()
            with self.assertRaisesRegex(DockerWritableLayerGuardError, "disappeared"):
                expired.raise_if_failed()
            with self.assertRaises(DockerWritableLayerLimitExceeded):
                over.raise_if_failed()
            self.assertEqual(set(guard._snapshot()), {"healthy"})
            self.assertEqual([c.args[0] for c in run_mock.call_args_list[1:]], [["docker", "kill", "over"]])
            events = [json.loads(l) for l in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual({e["event"] for e in events}, {"container_missing", "limit_exceeded"})

    def test_cleanup_timeout_does_not_interrupt_fail_closed_for_other_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = self._guard(root)
            states = [guard.register(name, image="image") for name in ("first", "second")]
            with mock.patch.object(guard, "_kill", side_effect=[
                subprocess.TimeoutExpired(["docker", "kill", "first"], 30), None,
            ]) as kill:
                guard._fail_closed(guard._snapshot(), "inspect timed out")
            for state in states:
                with self.assertRaises(DockerWritableLayerGuardError):
                    state.raise_if_failed()
            self.assertEqual([call.args[0] for call in kill.call_args_list], ["first", "second"])
            self.assertEqual(guard._snapshot(), {})
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            cleanup = [event for event in events if event["event"] == "container_cleanup_failed"]
            self.assertEqual(len(cleanup), 1)
            self.assertEqual(cleanup[0]["container_id"], "first")
            self.assertTrue(cleanup[0]["container_may_still_be_running"])

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
    def test_cleanup_race_does_not_fail_surviving_containers(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            guard = self._guard(Path(tmp), max_monitor_failures=3)
            survivor = guard.register("survivor", image="image-survivor")

            def run_side_effect(command, **_kwargs):
                if command[1] == "kill":
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
                gone = next(item for item in command if item.startswith("gone-"))
                guard.unregister(gone)
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout=json.dumps({"Id": "survivor", "SizeRw": 1024}),
                    stderr=f"Error: No such object: {gone}",
                )

            run_mock.side_effect = run_side_effect
            for index in range(3):
                guard.register(f"gone-{index}", image=f"image-gone-{index}")
                guard._poll_once()

            survivor.raise_if_failed()
            self.assertEqual(guard._monitor_failures, 0)
            self.assertFalse(
                any(call.args[0][1] == "kill" for call in run_mock.call_args_list)
            )

    @mock.patch("benchmark.swe_bench_lite.docker_writable_guard.subprocess.run")
    def test_limit_exceeded_does_not_fail_surviving_containers(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = self._guard(root, max_monitor_failures=3)
            exceeded = guard.register(
                "exceeded", image="sweb.eval.x86_64.astropy_1776_astropy-12907"
            )
            survivor = guard.register("survivor", image="image-survivor")
            live_containers = {"exceeded", "survivor"}

            def run_side_effect(command, **_kwargs):
                if command[1] == "kill":
                    live_containers.discard(command[2])
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

                inspected = command[5:]
                records = []
                missing = []
                for container_id in inspected:
                    if container_id not in live_containers:
                        missing.append(container_id)
                        continue
                    writable_bytes = (
                        4 * 1024**3 + 1 if container_id == "exceeded" else 1024
                    )
                    records.append(
                        json.dumps({"Id": container_id, "SizeRw": writable_bytes})
                    )
                return subprocess.CompletedProcess(
                    command,
                    1 if missing else 0,
                    stdout="\n".join(records),
                    stderr="\n".join(
                        f"Error: No such object: {container_id}"
                        for container_id in missing
                    ),
                )

            run_mock.side_effect = run_side_effect
            for _ in range(4):
                guard._poll_once()

            with self.assertRaises(DockerWritableLayerLimitExceeded):
                exceeded.raise_if_failed()
            survivor.raise_if_failed()
            self.assertEqual(guard._monitor_failures, 0)
            self.assertEqual(
                [
                    call.args[0][2]
                    for call in run_mock.call_args_list
                    if call.args[0][1] == "kill"
                ],
                ["exceeded"],
            )
            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["event"] for event in events], ["limit_exceeded"])

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
