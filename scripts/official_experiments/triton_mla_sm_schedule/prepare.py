"""Prepare jobs using the selected checkout and the existing guarded queue."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--conda", type=Path, required=True)
p.add_argument("--env", type=Path, required=True)
args = p.parse_args()
root, repo = args.root.resolve(), args.repo.resolve()
root.mkdir(parents=True, exist_ok=False)
source = repo
manifest = {"source_repo": str(repo), "source": str(source),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
            "formula": "head_tile=min(16,next_power_of_two(heads)); target_splits=ceil(SM/ceil(heads/head_tile)); target_blocks=active_rows*target_splits; programs=4*SM"}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
jobs = root / "queue/jobs"
jobs.mkdir(parents=True)
prefix = [str(args.conda), "run", "--no-capture-output", "-p", str(args.env), "python"]
environment = {"PYTHONPATH": str(source) + ":" + str(source / "src"),
               "TRITON_CACHE_DIR": str(root / "cache/triton"), "CUDA_HOME": "/usr/local/cuda-13.0"}
commands = {
    "00_tests": prefix + ["-m", "pytest", "-q", "tests/test_mla_attention_operator.py", "tests/test_mla_kernels.py", "tests/test_mla_sm_schedule.py", "--junitxml", str(root / "tests.xml")],
    "01_micro": prefix + ["scripts/official_experiments/triton_mla_sm_schedule/micro.py", "--config", "scripts/official_experiments/triton_mla_sm_schedule/micro.json", "--output-dir", str(root / "micro")],
}
for name, command in commands.items():
    (jobs / (name + ".json")).write_text(json.dumps({"command": command, "cwd": str(source), "env": environment, "timeout": 7200}, indent=2))
(root / "queue/STOP").touch()
print(json.dumps({"root": str(root), "source": str(source), "jobs": list(commands)}))
