"""Resolve external capacity configs and launch the canonical sweeper in tmux."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--baseline-config", type=Path, required=True)
p.add_argument("--tangram-env", type=Path, required=True)
p.add_argument("--hisparse-env", type=Path, required=True)
p.add_argument("--scratch", type=Path, required=True)
p.add_argument("--gpus", required=True, help="Tangram,HiSparse physical GPU IDs")
p.add_argument("--lanes", default="tangram-snapkv,hisparse-quest")
p.add_argument("--input-len", type=int)
p.add_argument("--output-len", type=int)
args = p.parse_args()
root, repo = args.root.resolve(), args.repo.resolve()
root.mkdir(parents=True, exist_ok=False)
base = json.loads(args.baseline_config.read_text())
template = json.loads(Path(__file__).with_name("external_capacity.json").read_text())
for key in ("conda", "native_env", "vllm_env", "input_len", "output_len", "gpu_memory_utilization"):
    template[key] = base[key]
if args.input_len is not None or args.output_len is not None:
    if args.input_len is None or args.output_len is None or min(args.input_len, args.output_len) <= 0:
        raise ValueError("Specify positive input-len and output-len together")
    if args.input_len + args.output_len != base["input_len"] + base["output_len"]:
        raise ValueError("Extended-output protocol must preserve the original total context span")
    template.update(input_len=args.input_len, output_len=args.output_len,
                    protocol_variant="user_approved_extended_output_constant_total_context")
template["output_root"] = str(root)
template["scratch_root"] = str(args.scratch.resolve())
template["models"]["qwen3-30b-fp8"] = base["models"]["qwen3-30b-fp8"]
template["external_lanes"]["tangram-snapkv"]["env"] = str(args.tangram_env.resolve())
template["external_lanes"]["hisparse-quest"]["env"] = str(args.hisparse_env.resolve())
config_path = root / "config.json"
config_path.write_text(json.dumps(template, indent=2) + "\n")
gpus = args.gpus.split(",")
if len(gpus) != 2 or gpus[0] == gpus[1]:
    raise ValueError("Specify two distinct GPU IDs")
commands = []
for lane, gpu in zip(template["external_lanes"], gpus):
    if lane not in args.lanes.split(","):
        continue
    command = ["python3", str(repo / "scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py"),
               "--repo", str(repo), "--config", str(config_path), "--model", "qwen3-30b-fp8", "--lanes", lane,
               "--gpus", gpu, "--attempt", "initial"]
    session = f"{lane}-{root.name}"
    shell = shlex.join(command) + " > " + shlex.quote(str(root / (lane + ".log"))) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    commands.append({"session": session, "gpu": gpu, "command": command})
(root / "launch.json").write_text(json.dumps(commands, indent=2) + "\n")
print(json.dumps(commands))
