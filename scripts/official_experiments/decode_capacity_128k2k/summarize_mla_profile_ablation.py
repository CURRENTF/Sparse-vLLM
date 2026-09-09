"""Revalidate raw ABBA measurements and export portable, unsmoothed results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from plot_decode_capacity import read_rows, validate_measurement


def find_plans(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "tilelang_launch_plan":
                yield item
            else:
                yield from find_plans(item)
    elif isinstance(value, list):
        for item in value:
            yield from find_plans(item)


def without_split_counts(value):
    if isinstance(value, dict):
        return {key: without_split_counts(item) for key, item in value.items() if key != "num_split"}
    if isinstance(value, list):
        return [without_split_counts(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "config.json").read_text())
    cases = [case for case in config["cases"] if not case.get("smoke")]
    if summary["status"] != "completed" or [r["name"] for r in summary["rows"]] != [c["name"] for c in cases]:
        raise ValueError("Incomplete or mismatched experiment queue")
    rows, hashes, tokens, runtime_stats = [], {}, {}, {}
    for case in cases:
        name, batch, variant = case["name"], case["batch"], case["variant"]
        path = root / name
        point = validate_measurement(path / "performance.jsonl", batch,
                                     {"input_len": 131072, "output_len": 2048})
        point["artifact"] = f"{name}/performance.jsonl"
        record = read_rows(path / "performance.jsonl")[0]
        runtime_stats[name] = without_split_counts(record["operator_runtime_stats"])
        if not record["decode_graph_active"] or record["decode_graph_force_eager_count"]:
            raise ValueError(f"Graph replay contract changed: {name}")
        plans = list(find_plans(json.loads((path / "aggregate_metrics.json").read_text())))
        expected_split = 8 if batch == 5 and variant == "profile" else 32
        if len(plans) != 2:
            raise ValueError(f"Expected launch plans from both ranks: {name}")
        active = []
        for plan in plans:
            selected = [p for p in plan["batch_configs"] if p["batch_size"] == batch]
            if len(selected) != 1 or selected[0]["num_split"] != expected_split:
                raise ValueError(f"Ablation variant was not applied: {name}")
            active.append({k: v for k, v in plan.items() if k != "batch_configs"} | selected[0])
        rows.append(dict(name=name, variant=variant, **point, rank_launch_plans=active))
        token_file = path / f"omnikv-131072-{batch}" / "raw_outputs.jsonl"
        tokens[name] = [r["token_ids"] for r in sorted(read_rows(token_file), key=lambda r: r["request_id"])]
        for artifact in path.rglob("*.json*"):
            hashes[str(artifact.relative_to(root))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    comparisons = []
    for batch in sorted({r["concurrency"] for r in rows}):
        group = [r for r in rows if r["concurrency"] == batch]
        if [r["variant"] for r in group] != ["profile", "fixed32", "fixed32", "profile"]:
            raise ValueError(f"Not an ABBA sequence: BS{batch}")
        if any(runtime_stats[r["name"]] != runtime_stats[group[0]["name"]] for r in group):
            raise ValueError(f"Operator metadata differs beyond split counts: BS{batch}")
        means = {v: statistics.mean(r["decode_throughput_tps"] for r in group if r["variant"] == v)
                 for v in ("profile", "fixed32")}
        comparisons.append({"concurrency": batch, "mean_tps": means, "other_operator_metadata_identical": True,
            "fixed32_change_percent": (means["fixed32"] / means["profile"] - 1) * 100,
            "tokens_equal_to_first_run": {r["name"]: tokens[r["name"]] == tokens[group[0]["name"]] for r in group},
            "mismatch_positions_per_request": {r["name"]: [
                [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
                for left, right in zip(tokens[group[0]["name"]], tokens[r["name"]], strict=True)] for r in group}})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "success", "metric": "full_batch_pure_decode_steps",
        "aggregation": "arithmetic mean of two independent runs per variant; no confidence interval",
        "rows": rows, "comparisons": comparisons, "artifact_sha256": hashes}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ["name", "variant", "concurrency", "decode_throughput_tps", "decode_tokens",
               "decode_elapsed_s", "measured_steps", "artifact"]
    with (args.output_dir / "points.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
