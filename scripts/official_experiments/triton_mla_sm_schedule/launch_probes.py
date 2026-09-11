"""Launch finite guarded capacity probes on distinct explicitly idle GPUs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--orchestrator-repo", type=Path, required=True)
p.add_argument("--campaign", type=Path, required=True)
p.add_argument("--smoke-from", type=Path, required=True)
p.add_argument("--gpu-batches", required=True, help="GPU:batch pairs separated by commas")
p.add_argument("--manifest-name", default="parallel_probes.json")
args = p.parse_args()
manifest = args.campaign / args.manifest_name
if manifest.exists():
    raise FileExistsError(manifest)
pairs = [tuple(map(int, value.split(":"))) for value in args.gpu_batches.split(",")]
if len({gpu for gpu, _ in pairs}) != len(pairs) or any(gpu < 0 or batch <= 0 for gpu, batch in pairs):
    raise ValueError("Probe pairs require distinct nonnegative GPUs and positive batches")
commands = []
for gpu, batch in pairs:
    name = f"probe{batch}-gpu{gpu}"
    command = ["python3", str(args.orchestrator_repo / "scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py"),
        "--repo", str(args.repo), "--config", str(args.campaign / "config.json"),
        "--model", "qwen3-30b-fp8", "--lanes", "tangram-snapkv", "--gpus", str(gpu),
        "--attempt", name, "--reuse-smoke-from", str(args.smoke_from),
        "--probe-concurrency", str(batch), "--probe-only"]
    session = f"tangram-{args.campaign.name}-{name}"
    log = args.campaign / f"{name}.log"
    if log.exists():
        raise FileExistsError(log)
    subprocess.run(["tmux", "new-session", "-d", "-s", session,
                    shlex.join(command) + " > " + shlex.quote(str(log)) + " 2>&1"], check=True)
    commands.append({"session": session, "gpu": gpu, "batch": batch, "command": command})
with manifest.open("x") as handle:
    json.dump(commands, handle, indent=2)
print(json.dumps(commands))
