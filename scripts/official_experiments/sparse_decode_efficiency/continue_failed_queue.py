"""One-shot continuation of untouched lanes after a legacy fail-fast queue stops.

Never retries a failed lane or changes the running queue's frozen source.
Linux pidfds keep reservation handoff signals bound to verified processes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read_status(path):
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue  # The producer may currently be appending its final row.
        timestamp, stage, status, gpus, extra = line.rstrip("\n").split("\t", 4)
        rows.append(dict(time=timestamp, stage=stage, status=status, gpus=gpus, **json.loads(extra)))
    return rows


def remaining_lanes(root, lanes, rows, *, explicit_followup=False, new_campaign=False,
                    allow_failed_methods=False):
    """Return only unstarted formal lanes after an explicitly local failure."""
    terminal = [row for row in rows if row["stage"] == "queue"][-1]
    if explicit_followup:
        allowed_failure = False
        if allow_failed_methods and terminal["status"] == "failed":
            error = terminal.get("error", "")
            prefix = "Lane failures retained after continuing other methods: "
            if error.startswith(("RuntimeError('" + prefix, 'RuntimeError("' + prefix)):
                summary = json.loads((root / "queue_summary.json").read_text())
                failures = summary.get("failures", {})
                allowed_failure = summary.get("status") == "failed" and bool(failures)
                for lane, failure in failures.items():
                    evidence = json.loads((root / lane / "lane_failure.json").read_text())
                    allowed_failure = (allowed_failure and lane in summary["lanes"]
                                       and evidence == failure and failure["phase"] in ("smoke", "sweep"))
        if terminal["status"] not in ("completed", "probe_completed") and not allowed_failure:
            raise RuntimeError("Explicit follow-up requires a successfully completed previous queue")
        if not new_campaign and any(list((root / lane).glob("bs*")) for lane in lanes):
            raise RuntimeError("Explicit follow-up must not repeat a previous lane")
        return lanes
    if terminal["status"] == "completed":
        return []
    error = terminal.get("error", "")
    local_errors = ("RuntimeError('Benchmark failed, inspect ", "RuntimeError('Case timeout: ",
                    "RuntimeError('Missing actual concurrency/completion evidence: ",
                    "RuntimeError('Invalid stage metric: ",
                    "RuntimeError('No validated nonzero capacity boundary for ")
    cases = [row for row in rows if "/" in row["stage"] and row["stage"].split("/")[0] in lanes]
    if not cases:
        raise RuntimeError("Legacy queue failed before any method case")
    failed_lane = cases[-1]["stage"].split("/")[0]
    case_path = str(root / cases[-1]["stage"])
    local_validation = error.startswith("ValueError(") and case_path + "/" in error
    if terminal["status"] != "failed" or not (error.startswith(local_errors) or local_validation):
        raise RuntimeError(f"Legacy failure is not an explicitly classified method error: {error}")
    return [lane for lane in lanes if lane != failed_lane and not list((root / lane).glob("bs*"))]


def pin_guard(pid, parent, devices=None):
    """Pin the verified same-user guard, never a benchmark or a foreign PID."""
    fd = os.pidfd_open(pid)
    try:
        proc = Path(f"/proc/{pid}")
        argv = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
        if (proc.stat().st_uid != os.getuid()
                or not any(Path(arg).name == "decode_capacity_guard.py" for arg in argv)
                or argv[argv.index("--parent") + 1] != str(parent)):
            raise RuntimeError("Reservation identity mismatch")
        if devices is not None:
            pids = subprocess.check_output(["nvidia-smi", "-i", devices,
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
            if set(map(int, pids.split())) != {pid}:
                raise RuntimeError("Reservation handoff requires no active model or foreign GPU processes")
        signal.pidfd_send_signal(fd, 0)
        return fd
    except BaseException:
        os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--previous-parent", type=int,
                        help="Optional while a predecessor is still queued; otherwise derived from its verified guard")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--orchestrator", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--lanes", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-measurements-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=172800)
    parser.add_argument("--explicit-followup", action="store_true",
                        help="After successful prior queue, run the explicitly listed new lanes")
    parser.add_argument("--new-campaign", action="store_true",
                        help="Explicit follow-up with a different recorded campaign; never reuse its measurements")
    parser.add_argument("--allow-failed-methods", action="store_true",
                        help="Continue explicit new lanes after recorded isolated method failures; never after resource failures")
    parser.add_argument("--probe-concurrency", type=int)
    parser.add_argument("--additional-probe-concurrency", type=int, action="append", default=[])
    parser.add_argument("--complete-from-capacity", type=Path)
    parser.add_argument("--completion-note")
    args = parser.parse_args()
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Use a Linux system Python with pidfd_open and pidfd_send_signal; no PID-only fallback")
    if not 0 < args.timeout_seconds <= 172800:
        raise ValueError("Wait must be bounded to at most 48 hours")
    if args.new_campaign and not args.explicit_followup:
        raise ValueError("A new campaign requires an explicit follow-up")
    if args.allow_failed_methods and not args.explicit_followup:
        raise ValueError("--allow-failed-methods requires --explicit-followup")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def status(state, **extra):
        row = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), status=state, **extra)
        print(json.dumps(row), flush=True)
        with (args.output_dir / "status.tsv").open("a") as out:
            out.write(f'{row["time"]}\tqueue\t{state}\t{args.gpus}\t{json.dumps(extra)}\n')

    guard_fd, child = None, None

    def stop(signum, frame):
        raise InterruptedError(f"Continuation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        config = json.loads(args.config.read_text())
        status("waiting_for_legacy_queue")
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            path = args.previous_root / "status.tsv"
            rows = read_status(path) if path.exists() else []
            terminal = [r for r in rows if r["stage"] == "queue" and r["status"] in ("completed", "probe_completed", "failed")]
            if terminal:
                break
            if args.previous_parent is not None:
                os.kill(args.previous_parent, 0)
            time.sleep(10)
        else:
            raise RuntimeError("Legacy queue wait timed out")
        if not args.new_campaign and json.loads((args.previous_root / "campaign.json").read_text()) != config:
            raise ValueError("Continuation campaign differs from the original")
        lanes = remaining_lanes(args.previous_root, args.lanes.split(","), rows,
                                explicit_followup=args.explicit_followup, new_campaign=args.new_campaign,
                                allow_failed_methods=args.allow_failed_methods)
        if not lanes:
            if terminal[-1]["status"] == "failed":
                raise RuntimeError("Legacy method failed; no untouched methods remain")
            status("completed", continuation_needed=False)
            return
        if (args.previous_root / "guard.contention.json").exists():
            raise RuntimeError("Legacy queue has GPU contention evidence")
        deadline = time.monotonic() + 30
        while not any(r["stage"] == "resource" and r["status"] == "holding_after_run" for r in rows):
            if time.monotonic() >= deadline:
                raise RuntimeError("Legacy queue is not safely holding after exit")
            time.sleep(1)
            rows = read_status(args.previous_root / "status.tsv")
        guard = json.loads((args.previous_root / "guard.json").read_text())
        if guard["devices"] != args.gpus:
            raise ValueError("GPU assignment changed")
        if args.previous_parent is None:
            argv = Path(f'/proc/{guard["pid"]}/cmdline').read_bytes().decode().strip("\0").split("\0")
            args.previous_parent = int(argv[argv.index("--parent") + 1])
        guard_fd = pin_guard(guard["pid"], args.previous_parent, args.gpus)
        signal.pidfd_send_signal(guard_fd, signal.SIGSTOP)
        command = [sys.executable, str(args.orchestrator), "--config", str(args.config),
            "--repo", str(args.repo), "--model", args.model, "--gpus", args.gpus,
            "--lanes", ",".join(lanes), "--attempt", args.attempt,
            "--export-measurements-dir", str(args.export_measurements_dir),
            "--handoff-reservation-pid", str(guard["pid"]), "--hold-reservation-seconds", "86400"]
        if args.probe_concurrency is not None:
            command += ["--probe-concurrency", str(args.probe_concurrency), "--probe-only"]
            for batch in args.additional_probe_concurrency:
                command += ["--additional-probe-concurrency", str(batch)]
        if args.complete_from_capacity:
            if not args.completion_note:
                raise ValueError("Completing old measurements requires a provenance note")
            command += ["--complete-from-capacity", str(args.complete_from_capacity),
                        "--completion-note", args.completion_note]
        (args.output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        next_root = Path(config["output_root"]) / args.model / ("_".join(lanes) + "-" + args.attempt)
        if next_root.exists():
            raise FileExistsError(next_root)
        with (args.output_dir / "sweep.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status("handoff", remaining_lanes=lanes, next_root=str(next_root))
        deadline = time.monotonic() + 55
        while not (next_root / "guard.json").is_file():
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("New reservation did not attach before handoff deadline")
            time.sleep(1)
        new_guard = json.loads((next_root / "guard.json").read_text())
        if new_guard["devices"] != args.gpus:
            raise RuntimeError("New reservation attached the wrong GPUs")
        new_guard_fd = pin_guard(new_guard["pid"], child.pid)
        os.close(new_guard_fd)
        signal.pidfd_send_signal(guard_fd, signal.SIGTERM)
        signal.pidfd_send_signal(guard_fd, signal.SIGCONT)
        os.close(guard_fd)
        guard_fd = None
        status("continuing", remaining_lanes=lanes, next_root=str(next_root))
        while time.monotonic() < deadline + args.timeout_seconds:
            next_status = read_status(next_root / "status.tsv")
            next_terminal = [r for r in next_status if r["stage"] == "queue" and r["status"] in ("completed", "probe_completed", "failed")]
            if next_terminal:
                if terminal[-1]["status"] == "failed" or next_terminal[-1]["status"] == "failed":
                    raise RuntimeError("Continuation finished with a failed method; inspect both summaries")
                status("completed", next_root=str(next_root), probe_only=args.probe_concurrency is not None)
                return
            if child.poll() is not None:
                raise RuntimeError(f"Continuation exited before terminal status: {child.returncode}")
            time.sleep(10)
        raise RuntimeError("Continuation wait timed out; inspect live sweep before intervention")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        if guard_fd is not None:
            try:
                if child is not None and child.poll() is None:
                    ready = next_root / "guard.json"
                    if ready.exists():
                        pid = json.loads(ready.read_text())["pid"]
                        fd = pin_guard(pid, child.pid)
                        try:
                            signal.pidfd_send_signal(fd, signal.SIGTERM)
                        finally:
                            os.close(fd)
                    child.terminate()
                    child.wait(timeout=30)
            finally:
                signal.pidfd_send_signal(guard_fd, signal.SIGCONT)
                os.close(guard_fd)


if __name__ == "__main__":
    main()
