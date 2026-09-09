"""Isolate the schedule change on the historical H2O model benchmark source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--updated-source", type=Path, required=True)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--gpu", required=True)
args = p.parse_args()
root, base = args.root.resolve(), args.baseline_root.resolve()
root.mkdir(parents=True, exist_ok=False)
shutil.copytree(base / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
changed = ["src/sparsevllm/kernels/triton/mla/decode_schedule.py",
           "src/sparsevllm/kernels/triton/mla/__init__.py", "src/sparsevllm/operators/mla_attention.py"]
for name in changed:
    shutil.copy2(args.updated_source / name, root / "source" / name)
hashes = {str(p.relative_to(root / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (root / "source").rglob("*") if p.is_file()}
(root / "manifest.json").write_text(json.dumps({"baseline_root": str(base), "changes": changed,
    "sha256": hashes, "protocol": "historical 32K/512 H2O TP1, 8 warmup +256 measured full-residency edge-synchronized steps, 3 iterations; not final128K plot"}, indent=2))
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
    spec["cwd"] = str(root / "source")
    spec["env"]["PYTHONPATH"] = str(root / "source") + ":" + str(root / "source/src")
    (jobs / (name + ".json")).write_text(json.dumps(spec, indent=2))
(root / "queue/STOP").touch()
