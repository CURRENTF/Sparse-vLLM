"""Revalidate raw stage data and add external curves without rewriting history."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1] / "sparse_decode_efficiency"
sys.path.insert(0, str(PACKAGE))
from plot_decode_capacity import validate_measurement


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-plot", type=Path, required=True)
    p.add_argument("--portable-base", type=Path, required=True)
    p.add_argument("--external-root", type=Path, required=True)
    p.add_argument("--tangram-root", type=Path, help="Optional user-approved constant-span, longer-output Tangram campaign")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    base, portable = read(args.base_plot), read(args.portable_base)
    config = read(args.external_root / "config.json")
    for key in ("input_len", "output_len", "gpu_memory_utilization"):
        if base["config"][key] != config[key]:
            raise ValueError(f"External workload differs: {key}")
    if base["config"]["models"]["qwen3-30b-fp8"] != config["models"]["qwen3-30b-fp8"]:
        raise ValueError("External model/checkpoint/topology mismatch")
    if len(base["curves"]) != len(portable["curves"]):
        raise ValueError("Portable base does not match validated base")
    # Do not accept a previous plotted aggregate as substitute for raw evidence.
    for curve, archived in zip(base["curves"], portable["curves"]):
        if (curve["model"], curve["lane"]) != (archived["model"], archived["lane"]):
            raise ValueError("Portable/base curve ordering differs")
        for point, old in zip(curve["points"], archived["points"], strict=True):
            actual = validate_measurement(Path(point["artifact"]), point["concurrency"], base["config"])
            if actual != point or {k:v for k,v in point.items() if k != "artifact"} != {k:v for k,v in old.items() if k != "artifact"}:
                raise ValueError("Reused point does not match its raw evidence")
    combined = copy.deepcopy(base)
    combined["config"]["external_lanes"] = config["external_lanes"]
    unsupported = {"glm4.7-flash": {
        "tangram-snapkv": "Pinned Tangram compression.validate_model_support rejects Glm4MoeLiteForCausalLM; supported architecture list has no MLA model.",
        "hisparse-quest": "Pinned HiSparse QuEST hisparse_hook validates AttentionArch.MHA only (standard MHA/GQA), rejecting MLA."}}
    combined["config"]["unsupported"] = unsupported
    provenance = {"base_plot_sha256": hashlib.sha256(args.base_plot.read_bytes()).hexdigest(),
                  "external_config": config, "unsupported": unsupported, "curves": {}}
    public_provenance = {
        "base_plot_sha256": provenance["base_plot_sha256"],
        "preserved_original_points": sum(len(c["points"]) for c in base["curves"]),
        "unsupported": unsupported, "external_backends": {},
        "caveats": [
            "Original points were raw-revalidated and reused; the Triton H2O schedule is not on their measured attention paths",
            "External sparse algorithms and selection granularity are not equal-quality implementations",
            "HiSparse means the full-GPU-KV QuEST PR series, not the CPU-offload paper system",
            "Synchronized scheduler/engine stages exclude prefill; HiSparse excludes IPC receive, while vLLM includes its worker-sync RPC",
            "Tangram uses its native interleaved scheduler, not the native SnapKV wave-admission extension",
            "Capacity is conditional on this complete-output/full-residency protocol, not a theoretical memory limit",
        ],
    }
    lane_roots = {lane: args.external_root for lane in config["external_lanes"]}
    protocols = {}
    if args.tangram_root:
        extended = read(args.tangram_root / "config.json")
        if (extended.get("protocol_variant") != "user_approved_extended_output_constant_total_context"
                or extended["input_len"] + extended["output_len"] != config["input_len"] + config["output_len"]
                or extended["output_len"] <= config["output_len"]
                or extended["gpu_memory_utilization"] != config["gpu_memory_utilization"]
                or extended["models"] != config["models"]
                or extended["external_lanes"]["tangram-snapkv"] != config["external_lanes"]["tangram-snapkv"]):
            raise ValueError("Extended Tangram campaign must preserve checkpoint, topology, memory and sparse parameters")
        lane_roots["tangram-snapkv"] = args.tangram_root
        protocols = {"qwen3-30b-fp8": {"tangram-snapkv": {
            "input_len": extended["input_len"], "output_len": extended["output_len"],
            "label_suffix": "*", "minimum_measured_steps": 2015,
            "reason": "User-approved longer generation and equally reduced prompt preserve total span and avoid admission-limited short windows",
        }}}
        combined["config"]["curve_protocols"] = protocols
        public_provenance["curve_protocols"] = protocols
    for lane in config["external_lanes"]:
        lane_root = lane_roots[lane]
        lane_config = read(lane_root / "config.json")
        matches = [path for path in (lane_root / "qwen3-30b-fp8").glob(f"*/{lane}/capacity.json")
                   if read(path).get("status") == "completed"]
        if len(matches) != 1:
            raise ValueError(f"Expected one external capacity artifact: {lane}")
        boundary = read(matches[0])
        if boundary["status"] != "completed":
            raise ValueError(f"External sweep incomplete: {lane}")
        points = []
        identities = []
        for attempt in boundary["attempts"]:
            if attempt["status"] != "success":
                continue
            path = Path(attempt["artifact"])
            row = json.loads(path.read_text().strip())
            expected = config["external_lanes"][lane]
            if row["engine"] != expected["engine"] or row["method"] != expected["method"] or row["backend_label"] != expected["backend_label"]:
                raise ValueError("External engine/method identity mismatch")
            if row["decode_warmup_steps_after_full"] != 32:
                raise ValueError("External warmup does not match the original stage protocol")
            identity = read(path.parent / "identity.json")
            if identities and identity["source_sha256"] != identities[0]["source_sha256"]:
                raise ValueError("Source changed inside the external sweep")
            identities.append(identity)
            if lane_root != args.external_root and row["measured_decode_steps_after_full"] < 2015:
                raise ValueError("Extended-output point still lacks the requested full-batch decode window")
            points.append(validate_measurement(path, attempt["concurrency"], lane_config))
        curve = {"model": "qwen3-30b-fp8", "lane": lane,
                 "max_concurrency": boundary["max_concurrency"],
                 "first_failed_concurrency": boundary["first_failed_concurrency"],
                 "attempts": boundary["attempts"], "points": sorted(points, key=lambda x:x["concurrency"]),
                 "capacity_artifact": str(matches[0])}
        combined["curves"].append(curve)
        provenance["curves"][lane] = {"capacity": str(matches[0]), "identity": identities[0]}
        public_provenance["external_backends"][lane] = {
            "engine": row["engine"], "backend_label": row["backend_label"],
            "version": row.get("sglang_version", row.get("vllm_version")),
            "source_sha256": identities[0]["source_sha256"],
            "max_concurrency": boundary["max_concurrency"],
        }
        combined["curves"].append({"model": "glm4.7-flash", "lane": lane, "status": "unsupported",
                                   "reason": unsupported["glm4.7-flash"][lane], "points": [], "attempts": []})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / "input.json", combined)
    subprocess.run([sys.executable, str(PACKAGE / "plot_decode_capacity.py"), "--plot-data", str(args.output_dir / "input.json"),
                    "--output-dir", str(args.output_dir / "figures")], check=True)
    export = read(args.output_dir / "figures/plot_data.json")
    # Original 44 portable points are retained unchanged, including artifact IDs.
    export["curves"][:len(portable["curves"])] = portable["curves"]
    export["config"] = copy.deepcopy(portable["config"])
    export["config"]["external_lanes"] = copy.deepcopy(config["external_lanes"])
    export["config"]["unsupported"] = unsupported
    if protocols:
        export["config"]["curve_protocols"] = protocols
    for lane, lane_config in export["config"]["external_lanes"].items():
        lane_config["env"] = "${TANGRAM_ENV}" if lane == "tangram-snapkv" else "${HISPARSE_ENV}"
    for curve in export["curves"][len(portable["curves"]):]:
        lane_root = lane_roots[curve["lane"]]
        prefix = "external/" if lane_root == args.external_root else "external-extended/"
        for item in curve["attempts"] + curve["points"]:
            if "artifact" in item:
                item["artifact"] = prefix + str(Path(item["artifact"]).relative_to(lane_root))
        if "capacity_artifact" in curve:
            curve["capacity_artifact"] = prefix + str(Path(curve["capacity_artifact"]).relative_to(lane_root))
    write(args.output_dir / "plot_data.json", export)
    write(args.output_dir / "provenance.json", provenance)
    write(args.output_dir / "public_provenance.json", public_provenance)
    subprocess.run([sys.executable, str(PACKAGE / "plot_decode_capacity.py"), "--plot-data", str(args.output_dir / "plot_data.json"),
                    "--output-dir", str(args.output_dir / "portable-replot")], check=True)
    for name in ("decode_capacity_128k2k.png", "decode_capacity_128k2k_logy.png"):
        if (args.output_dir / "figures" / name).read_bytes() != (args.output_dir / "portable-replot" / name).read_bytes():
            raise ValueError("Portable replot changed the figure")


if __name__ == "__main__":
    main()
