"""Validate saved paired measurements, aggregate and render without CUDA."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import shutil


def load_run(root):
    manifest = json.loads((root / "run_manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    raw = [json.loads(line) for line in (root / "raw_samples.jsonl").read_text().splitlines()]
    ids = [row["id"] for row in rows]
    if manifest["status"] != "completed" or len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in manifest["cases"]}:
        raise ValueError(f"Incomplete or duplicate cases: {root}")
    expected = manifest["config"]["abba_rounds"] * 2
    if len(raw) != len(rows) * 2 * expected:
        raise ValueError("Raw sample count mismatch")
    for row in rows:
        if row["status"] != "success" or row["oracle"]["status"] != "success":
            raise ValueError(f"Invalid numerical result: {row['id']}")
        old, new = row["configs"]["legacy"], row["configs"]["formula"]
        if {k: v for k, v in old.items() if k != "num_split"} != {k: v for k, v in new.items() if k != "num_split"}:
            raise ValueError("The comparison changed more than splits")
        if max(row["actual_lengths"]) != row["context"]:
            raise ValueError("Formula context differs from maximum actual context")
        target = manifest["gpu"]["sm_count"] * max(math.log2(row["context"] / 64), 1)
        raw_split = target / row["base_ctas"]
        legal = manifest["config"]["legal_splits"]
        expected_split = min([s for s in legal if s >= math.ceil(raw_split)] or [max(legal)])
        if not math.isclose(row["raw_splits"], raw_split) or new["num_split"] != expected_split:
            raise ValueError("Serialized candidate does not follow its formula")
        if row["identical_config"] != (old == new):
            raise ValueError("Identical-config control is mislabeled")
        for arm in ["legacy", "formula"]:
            matching = [r for r in raw if r["case_id"] == row["id"] and r["arm"] == arm]
            values = [r["latency_us"] for r in matching]
            if len(values) != expected or any(r["status"] != "success" for r in matching) or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Invalid samples: {row['id']} {arm}")
            if values != row["samples_us"][arm] or statistics.median(values) != row["median_us"][arm]:
                raise ValueError("Raw samples do not reconstruct aggregate")
        if not math.isclose(row["speedup"], row["median_us"]["legacy"] / row["median_us"]["formula"], rel_tol=1e-12):
            raise ValueError("Speedup does not reconstruct")
    return manifest, rows


def aggregate(rows):
    speedups = [r["speedup"] for r in rows]
    return {"count": len(rows), "geomean_speedup": math.exp(statistics.mean(map(math.log, speedups))),
            "median_speedup": statistics.median(speedups),
            "rule_wins_gt_5pct": sum(s > 1.05 for s in speedups),
            "rule_losses_gt_5pct": sum(s < 1 / 1.05 for s in speedups),
            "within_5pct": sum(1 / 1.05 <= s <= 1.05 for s in speedups),
            "min_speedup": min(speedups), "max_speedup": max(speedups)}


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="white", context="paper", font_scale=1.15)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    df = pd.DataFrame(rows)
    uniform = df[(df.distribution == "uniform") & (df.batch <= 128)]
    bound = max(abs(math.log2(v)) for v in df.speedup)
    cmap = sns.diverging_palette(350, 155, s=65, l=65, as_cmap=True)

    def heatmap(matrix, name, xlabel, figsize, cbar_label="Speedup (Legacy / Rule)"):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        sns.heatmap(np.log2(matrix), annot=matrix.map(lambda v: f"{v:.2f}"), fmt="",
                    cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                    linewidths=0.6, linecolor="white", ax=ax,
                    cbar_kws={"label": cbar_label})
        ticks = [v for v in [0.25, 0.5, 1, 2, 4, 8] if abs(math.log2(v)) <= bound]
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([math.log2(v) for v in ticks], labels=[f"{v:g}×" for v in ticks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Context length")
        ax.set_yticklabels([str(v) if v < 1024 else f"{v // 1024}K" for v in matrix.index], rotation=0)
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{extension}", dpi=220)
        plt.close(fig)

    ragged = df[df.distribution == "ragged"].pivot(index="context", columns=["heads", "batch"], values="speedup").sort_index(axis=1)
    ragged.columns = [f"{h}/{b}" for h, b in ragged.columns]
    heatmap(ragged, "speedup_ragged", "Local heads / batch size", (7.5, 1.9), "Speedup")
    large = df[(df.distribution == "uniform") & (df.batch == 256)].pivot(index="context", columns="heads", values="speedup")
    heatmap(large, "speedup_bs256", "Local heads", (3.6, 1.9), "Speedup")
    for heads in sorted(uniform.heads.unique()):
        matrix = uniform[uniform.heads == heads].pivot(index="context", columns="batch", values="speedup")
        heatmap(matrix, f"speedup_h{heads}", "Batch size", (5.4, 3.25))
        # Long-context latency curves make absolute cost visible alongside ratios.
        selected = [r for r in rows if r["heads"] == heads and r["context"] == 131072 and r["distribution"] == "uniform"]
        selected.sort(key=lambda r: r["batch"])
        for scale in ["linear", "log"]:
            fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
            for arm, label, color, marker in [("legacy", "Legacy", "#6B9AC4", "o"), ("formula", "Rule", "#54BFA3", "s")]:
                ax.plot([r["batch"] for r in selected], [r["median_us"][arm] / 1000 for r in selected],
                        label=label, color=color, marker=marker, markersize=4, linewidth=1.5)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 32, 128], labels=[1, 4, 16, 32, 128])
            ax.set_yscale(scale)
            if scale == "linear":
                ax.set_ylim(bottom=0)
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Kernel latency (ms)")
            ax.legend(frameon=False)
            ax.grid(alpha=0.15)
            sns.despine(ax=ax)
            for extension in ["png", "pdf"]:
                fig.savefig(output / f"latency_128k_h{heads}_{scale}.{extension}", dpi=220)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--repeat", type=Path)
    p.add_argument("--confirmation", type=Path, help="Post-sweep extrema/control repeat; never replaces primary rows")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--export-data", type=Path, help="Fresh portable artifact directory with machine paths removed")
    args = p.parse_args()
    manifest, rows = load_run(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"metric": "legacy median kernel latency / rule median kernel latency",
               "aggregation": "unweighted geometric mean over sampled shapes; not a serving workload mix",
               "all": aggregate(rows),
               "changed": aggregate([r for r in rows if not r["identical_config"]]),
               "identical_controls": aggregate([r for r in rows if r["identical_config"]]),
               "by_heads": {str(h): aggregate([r for r in rows if r["heads"] == h]) for h in sorted({r["heads"] for r in rows})},
               "by_context": {str(n): aggregate([r for r in rows if r["context"] == n]) for n in sorted({r["context"] for r in rows})},
               "uniform_grid_by_context": {str(n): aggregate([r for r in rows if r["context"] == n and r["distribution"] == "uniform" and r["batch"] <= 128]) for n in sorted({r["context"] for r in rows})},
               "by_batch": {str(b): aggregate([r for r in rows if r["batch"] == b]) for b in sorted({r["batch"] for r in rows})},
               "by_distribution": {d: aggregate([r for r in rows if r["distribution"] == d]) for d in sorted({r["distribution"] for r in rows})},
               "max_within_arm_sample_range_fraction": max((max(v) - min(v)) / statistics.median(v) for r in rows for v in r["samples_us"].values()),
               "worst": sorted(rows, key=lambda r: r["speedup"])[:10],
               "best": sorted(rows, key=lambda r: r["speedup"], reverse=True)[:10]}
    for name, path in [("repeat", args.repeat), ("confirmation", args.confirmation)]:
        if path is None:
            continue
        repeat_manifest, repeat_rows = load_run(path)
        if manifest["source_sha256"] != repeat_manifest["source_sha256"] or manifest["versions"] != repeat_manifest["versions"]:
            raise ValueError("Repeat source or toolchain differs from primary")
        reference = {r["id"]: r for r in rows}
        if any(r["id"] not in reference or r["configs"] != reference[r["id"]]["configs"] for r in repeat_rows):
            raise ValueError("Repeat plans differ or are absent from primary")
        summary[name] = {"all": aggregate(repeat_rows), "gpu": repeat_manifest["gpu"],
            "max_paired_speedup_relative_difference": max(abs(r["speedup"] / reference[r["id"]]["speedup"] - 1) for r in repeat_rows),
            "paired": [{"id": r["id"], "primary_speedup": reference[r["id"]]["speedup"],
                        "repeat_speedup": r["speedup"]} for r in repeat_rows]}
    for filename, value in [("summary.json", summary), ("results.json", {"manifest": manifest, "cases": rows})]:
        (args.output / filename).write_text(json.dumps(value, indent=2) + "\n")
    with (args.output / "points.csv").open("w") as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=["id", "heads", "batch", "context", "distribution", "legacy_splits", "formula_splits", "legacy_us", "formula_us", "speedup", "identical_config"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in ["id", "heads", "batch", "context", "distribution", "speedup", "identical_config"]},
                "legacy_splits": r["configs"]["legacy"]["num_split"], "formula_splits": r["configs"]["formula"]["num_split"],
                "legacy_us": r["median_us"]["legacy"], "formula_us": r["median_us"]["formula"]})
    plot(rows, args.output)
    if args.export_data:
        args.export_data.mkdir(parents=True, exist_ok=False)
        for name, path in [("primary", args.run), ("repeat", args.repeat), ("confirmation", args.confirmation)]:
            if path is None:
                continue
            identity, _ = load_run(path)
            public = {k: identity[k] for k in ["status", "config", "git_head", "versions", "cuda", "gpu", "protocol", "cases"]}
            public["source_sha256"] = {Path(k).name: v for k, v in identity["source_sha256"].items()}
            if len(public["source_sha256"]) != len(identity["source_sha256"]):
                raise ValueError("Source snapshot basename collision")
            target = args.export_data / name
            target.mkdir()
            (target / "run_manifest.json").write_text(json.dumps(public, indent=2) + "\n")
            for filename in ["cases.jsonl", "raw_samples.jsonl", "generated_profiles.json"]:
                shutil.copy2(path / filename, target / filename)
            shutil.copytree(path / "source", target / "source")
        for filename in ["summary.json", "points.csv"]:
            shutil.copy2(args.output / filename, args.export_data / filename)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["worst", "best", "repeat"]}, indent=2))


if __name__ == "__main__":
    main()
