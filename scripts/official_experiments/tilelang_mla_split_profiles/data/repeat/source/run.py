"""Reserve an idle GPU continuously across smoke and the paired sweep."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    p.add_argument("--case-ids", type=Path)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    ready = args.output / "guard-ready.json"
    children = []

    def stop(signum, frame):
        raise RuntimeError(f"Controller interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = args.output / "status.tsv"

    def record(stage, state):
        with status.open("a") as f:
            f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")}\t{stage}\t{state}\t{os.environ["CUDA_VISIBLE_DEVICES"]}\t{args.config}\t{args.output / (stage + ".log")}\t{args.output / stage / "cases.jsonl"}\n')

    try:
        with (args.output / "guard.log").open("x") as guard_log:
            guard = subprocess.Popen([sys.executable, str(script.parent.parent / "decode_capacity_128k2k/decode_capacity_guard.py"),
                "--parent", str(os.getpid()), "--ready", str(ready), "--max-seconds", "21600"],
                stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(guard)
            deadline = time.monotonic() + 90
            while not ready.exists():
                if guard.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("GPU guard failed to attach; inspect guard.log")
                time.sleep(0.5)
            for stage in (["smoke"] if args.smoke_only else ["smoke", "full"]):
                if guard.poll() is not None:
                    raise RuntimeError("GPU guard exited")
                command = [sys.executable, "-u", str(script.with_name("benchmark.py")),
                           "--config", str(args.config), "--output", str(args.output / stage)]
                if stage == "smoke":
                    command.append("--smoke")
                elif args.case_ids:
                    command += ["--case-ids", str(args.case_ids)]
                (args.output / (stage + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
                record(stage, "running")
                with (args.output / (stage + ".log")).open("x") as log:
                    worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children.append(worker)
                    while worker.poll() is None:
                        if guard.poll() is not None:
                            raise RuntimeError("GPU guard exited during measurement")
                        time.sleep(1)
                    if worker.returncode:
                        record(stage, "failed")
                        raise RuntimeError(f"{stage} failed: exit {worker.returncode}")
                manifest = json.loads((args.output / stage / "run_manifest.json").read_text())
                rows = [json.loads(s) for s in (args.output / stage / "cases.jsonl").read_text().splitlines()]
                if manifest["status"] != "completed" or len(rows) != len(manifest["cases"]) or any(r["status"] != "success" for r in rows):
                    raise RuntimeError("Incomplete measurement artifacts")
                record(stage, "completed")
    except BaseException:
        record("controller", "failed")
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    main()
