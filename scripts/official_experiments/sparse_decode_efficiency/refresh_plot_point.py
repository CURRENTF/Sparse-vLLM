"""Replace one measured plot point after raw validation; never extrapolate rates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from plot_decode_capacity import (
    curve_config, validate_grid_identity, validate_measurement,
    without_source_fingerprints,
)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-data", type=Path, required=True)
    parser.add_argument("--measurement-export", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = read(args.plot_data)
    replacement = without_source_fingerprints(read(args.measurement_export))
    config = curve_config(data["config"], args.panel)
    if config["measurement_protocol"] != "boundary_sync_v2" or replacement["smoke_only"]:
        raise ValueError("Only formal boundary-sync measurements can replace a point")
    curve, = [c for c in data["curves"]
              if c["model"] == args.panel and c["lane"] == replacement["lane"]]
    batch = replacement["point"]["concurrency"]
    old, = [p for p in curve["points"] if p["concurrency"] == batch]
    artifact = Path(replacement["point"]["artifact"])
    previous = Path(old["artifact"]).parent
    current = artifact.parent
    old_manifest, new_manifest = read(previous / "run_manifest.json"), read(current / "run_manifest.json")
    for key in ("protocol", "workload", "seed", "prompt_token_ids_sha256", "repetition_lifecycle"):
        if old_manifest[key] != new_manifest[key]:
            raise ValueError(f"Measurement contract changed: {key}")
    for key in old_manifest["args"].keys() | new_manifest["args"].keys():
        if key not in {"output_dir", "hyper_params", "hardware", "monitor_gpus"}:
            if old_manifest["args"].get(key) != new_manifest["args"].get(key):
                raise ValueError(f"Measurement argument changed: {key}")
    if read(previous / "hyper_params.json") != read(current / "hyper_params.json"):
        raise ValueError("Resolved hyperparameters changed")
    if read(current / "run_status.json")["status"] != "completed":
        raise ValueError("Replacement run is not completed")
    point = validate_measurement(artifact, batch, config)
    if point != replacement["point"]:
        raise ValueError("Replacement export disagrees with raw validated measurement")
    validate_grid_identity(point, config["source_model"], config)
    if point["model_identity"]["model_config_sha256"] != old["model_identity"]["model_config_sha256"]:
        raise ValueError("Model config changed")
    curve["points"][curve["points"].index(old)] = point
    attempt, = [a for a in curve["attempts"] if a["concurrency"] == batch and a["status"] == "success"]
    attempt["artifact"] = str(artifact)
    record = dict(panel=args.panel, lane=curve["lane"], concurrency=batch,
                  reason=args.reason, old_point=old, new_point=point,
                  previous_plot=str(args.plot_data),
                  previous_plot_sha256=hashlib.sha256(args.plot_data.read_bytes()).hexdigest(),
                  measurement_export=str(args.measurement_export),
                  capacity_evidence="Original boundary evidence retained; no new capacity claim.")
    data["config"].setdefault("point_refreshes", []).append({
        key: record[key] for key in ("panel", "lane", "concurrency", "reason", "capacity_evidence")})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("plot_data.json", data), ("refresh.json", record),
                        ("measurement.json", replacement)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(f"{args.panel}/{curve['lane']}/BS{batch}: "
          f"{old['decode_throughput_tps']:.6f} -> {point['decode_throughput_tps']:.6f} tok/s")


if __name__ == "__main__":
    main()
