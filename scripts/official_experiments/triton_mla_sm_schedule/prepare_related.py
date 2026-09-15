"""Run historical H2O benchmark jobs against an explicitly prepared checkout."""
import argparse
import subprocess
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--updated-source", type=Path, required=True, help="Checkout containing the intended schedule change")
p.add_argument("--root", type=Path, required=True)
p.add_argument("--gpu", required=True)
args = p.parse_args()
root, base = args.root.resolve(), args.baseline_root.resolve()
root.mkdir(parents=True, exist_ok=False)
source = args.updated_source.resolve(strict=True)
(root / "manifest.json").write_text(json.dumps({"baseline_root": str(base), "source": str(source),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()),
    "protocol": "historical 32K/512 H2O TP1, 8 warmup +256 measured full-residency edge-synchronized steps, 3 iterations; not final128K plot"}, indent=2))
jobs = root / "queue/jobs"
jobs.mkdir(parents=True)
for name in ["04_glm_common8", "06_glm_wave64"]:
    spec = json.loads((base / "queue" / name / "command.json").read_text())
    command = spec["command"]
    command[command.index("--output-dir") + 1] = str(root / "queue" / name)
    command[command.index("--monitor-gpus") + 1] = args.gpu
    hp_index = command.index("--hyper-params") + 1
    hp = Path(command[hp_index][1:])
    dest = root / hp.name
    shutil.copy2(hp, dest)
    command[hp_index] = "@" + str(dest)
    spec["cwd"] = str(source)
    spec["env"]["PYTHONPATH"] = str(source) + ":" + str(source / "src")
    (jobs / (name + ".json")).write_text(json.dumps(spec, indent=2))
(root / "queue/STOP").touch()
