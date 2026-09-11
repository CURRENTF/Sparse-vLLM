"""Freeze a boundary-sync campaign, then orchestrate the existing sweeps in tmux."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent
RELATIVE_PACKAGE = PACKAGE.relative_to(PACKAGE.parents[2])


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(args):
    repo = PACKAGE.parents[2]
    text = os.path.expandvars(args.config.read_text())
    if re.search(r"\$\{[^}]+\}", text):
        raise ValueError("Unresolved environment variables in config")
    config = json.loads(text)
    if config.get("measurement_protocol") != "boundary_sync_v2":
        raise ValueError("This launcher requires boundary_sync_v2")
    if Path(config["output_root"]).resolve() != args.run_root:
        raise ValueError("DECODE_OUTPUT_ROOT must equal --run-root")
    args.run_root.mkdir(parents=True, exist_ok=False)
    snapshot = args.run_root / "source"
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo
    ).decode().split("\0")
    for name in sorted(set(names)):
        path = Path(name)
        if not name or not (repo / path).is_file():
            continue
        if path.parts[0] not in ("src", "benchmark", "configs", "scripts") and len(path.parts) != 1:
            continue
        if "data" in path.parts or "tmp" in path.parts or (repo / path).stat().st_size > 5_000_000:
            continue
        source = repo / path
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write(args.run_root / "manifest.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
        "repo": str(repo),
    })
    write(args.run_root / "config.json", config)
    write(args.run_root / "launch.json", {"data_root": str(args.data_root)})
    for model in config["models"]:
        subprocess.run([sys.executable, str(snapshot / RELATIVE_PACKAGE / "sweep_decode_capacity.py"),
                        "--config", str(args.run_root / "config.json"), "--repo", str(snapshot),
                        "--model", model, "--gpus", f"auto:{config['models'][model]['tp']}",
                        "--check-only"], check=True)
    print(f"Prepared {args.run_root}; run this frozen script with --run-root ... --run", flush=True)


def run(args):
    root = args.run_root
    config = json.loads((root / "config.json").read_text())
    data_root = Path(json.loads((root / "launch.json").read_text())["data_root"])
    snapshot = root / "source"
    package = snapshot / RELATIVE_PACKAGE
    processes = []
    with (root / "status.tsv").open("x", buffering=1) as status:
        def record(stage, result):
            status.write(f"{datetime.now(timezone.utc).isoformat()}\t{stage}\t{result}\n")
        record("queue", "started")
        for model, spec in config["models"].items():
            lanes = ["svllm-vanilla", "svllm-snapkv", "svllm-quest", "svllm-omnikv", "vllm-vanilla"]
            lanes += [lane for lane in config["external_lanes"] if lane not in config["unsupported"].get(model, {})]
            command = [sys.executable, str(package / "sweep_decode_capacity.py"),
                       "--config", str(root / "config.json"), "--repo", str(snapshot),
                       "--model", model, "--gpus", f"auto:{spec['tp']}", "--lanes", ",".join(lanes),
                       "--attempt", "boundary-v2", "--export-measurements-dir", str(data_root / "measurements")]
            write(root / f"{model}.command.json", command)
            with (root / f"{model}.run.log").open("x") as log:
                processes.append((model, subprocess.Popen(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT)))
        codes = []
        for model, process in processes:
            code = process.wait()
            codes.append(code)
            record(model, f"exit={code}")
        if any(codes):
            record("queue", "failed; preserve completed lanes and inspect model logs")
            raise RuntimeError(f"Sweep exit codes: {codes}")
        record("plot", "started")
        subprocess.run([config["conda"], "run", "--no-capture-output", "-p", config["native_env"],
                        "python", str(package / "plot_decode_capacity.py"), "--config", str(root / "config.json"),
                        "--output-dir", str(root / "plots"), "--export-data-dir", str(data_root / "plot")], check=True)
        record("queue", "completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.boundary-sync.json")
    parser.add_argument("--data-root", type=lambda p: Path(p).resolve())
    parser.add_argument("--run", action="store_true", help="Execute an already prepared campaign")
    args = parser.parse_args()
    if not args.run and args.data_root is None:
        parser.error("Preparation requires --data-root")
    (run if args.run else prepare)(args)
