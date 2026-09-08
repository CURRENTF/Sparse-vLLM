"""Plot recorded E2E comparisons, not decode-stage throughput.

Usage: python benchmark/efficiency/plot_vortex_comparison.py
    --data <RECORDED_DATA.json> --output-dir <FIGURE_DIR>

Input carries protocol/provenance plus per-iteration output tokens, elapsed
time and throughput. No model inference or new metric aggregation is performed.
Requires Matplotlib >= 3.7 and Seaborn >= 0.13 for automatic layout and bar gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {"Vortex": "#87AED4", "Sparse-vLLM": "#49B6A3", "Sparse-vLLM (wave2)": "#ECAB83"}


def validated_rows(payload: dict) -> list[dict]:
    if payload["metric"] != "output_token_throughput_tps":
        raise ValueError("This plot requires E2E output throughput, not decode throughput")
    rows, seen_cases = [], set()
    for case in payload["cases"]:
        identity = (case["model"], case["method"], case["series"])
        if identity in seen_cases:
            raise ValueError(f"Duplicate case: {identity}")
        seen_cases.add(identity)
        if case["series"] not in PALETTE or case["concurrency"] <= 0:
            raise ValueError(f"Invalid series or concurrency: {identity}")
        iterations = set()
        for sample in case["samples"]:
            if sample["status"] != "success":
                raise ValueError(f"Cannot plot a failed sample: {identity}")
            if sample["iteration"] in iterations:
                raise ValueError(f"Duplicate iteration: {identity}")
            iterations.add(sample["iteration"])
            tokens, elapsed, value = (sample[k] for k in ("output_tokens", "elapsed_s", "value"))
            if not all(math.isfinite(x) and x > 0 for x in (tokens, elapsed, value)):
                raise ValueError(f"Nonpositive or nonfinite measurement: {identity}")
            if not math.isclose(value, tokens / elapsed, rel_tol=1e-9):
                raise ValueError(f"Throughput disagrees with recorded tokens/time: {identity}")
            rows.append({**dict(zip(("model", "method", "series"), identity)),
                         "concurrency": case["concurrency"], "value": value})
        if len(iterations) < 2:
            raise ValueError(f"At least two iterations are needed for variability: {identity}")
    if not rows:
        raise ValueError("No measurements to plot")
    return rows


def draw_panel(ax, frame, models, series, *, show_batch):
    for model in models:
        for engine in series:
            values = frame.loc[(frame.model == model) & (frame.series == engine), "concurrency"].unique()
            if len(values) != 1:
                raise ValueError(f"Missing or inconsistent case: {model}, {engine}")
        if not show_batch and frame.loc[frame.model == model, "concurrency"].nunique() != 1:
            raise ValueError(f"Common-concurrency panel contains unmatched batches: {model}")
    sns.barplot(data=frame, x="model", y="value", hue="series", order=models,
                hue_order=series, palette=PALETTE, saturation=1, errorbar="sd",
                capsize=0.08, gap=0.12, edgecolor="none", err_kws={"linewidth": 1}, ax=ax)
    for container, engine in zip(ax.containers, series):
        labels = []
        for model, value in zip(models, container.datavalues):
            batch = frame.loc[(frame.model == model) & (frame.series == engine), "concurrency"].iloc[0]
            labels.append(f"{value:.1f}" + (f"\nB={batch}" if show_batch else ""))
        ax.bar_label(container, labels=labels, padding=4, fontsize=9, color="#364152")
    if not show_batch:
        ax.set_xticks(ax.get_xticks(), [f"{model}\nB={frame.loc[frame.model == model, 'concurrency'].iloc[0]}"
                                       for model in models])
    ax.set_ylabel("End-to-end output throughput (token/s)")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=float, default=10.0, help="Width in inches; height is width / 2.5")
    args = parser.parse_args()
    if not math.isfinite(args.width) or args.width <= 0:
        raise ValueError("Width must be positive and finite")
    source = args.data.read_bytes()
    payload = json.loads(source)
    frame = pd.DataFrame(validated_rows(payload))
    models = list(dict.fromkeys(frame.model))
    series = ["Vortex", "Sparse-vLLM"]
    if set(frame.method) != {"QuEST", "H2O / H2O-like"}:
        raise ValueError("Expected QuEST and the explicitly approximate H2O comparison")
    names = ["common_concurrency", "h2o_capacity"]
    targets = [args.output_dir / f"{name}.{ext}" for name in names for ext in ("png", "pdf", "svg")]
    targets.append(args.output_dir / "plot_manifest.json")
    if any(p.exists() for p in targets):
        raise FileExistsError("Use a fresh output directory; refusing to overwrite figures")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", font_scale=1.15,
                  rc={"axes.edgecolor": "#D6DDE5", "grid.color": "#E9EEF3", "grid.linewidth": 0.7,
                      "text.color": "#364152", "axes.labelcolor": "#364152", "svg.fonttype": "none",
                      "pdf.fonttype": 42, "savefig.bbox": None})
    for name in names:
        methods = ["QuEST", "H2O / H2O-like"] if name == "common_concurrency" else ["H2O / H2O-like"]
        hues = series if name == "common_concurrency" else list(PALETTE)
        fig, axes = plt.subplots(1, len(methods), figsize=(args.width, args.width / 2.5),
                                 layout="constrained", squeeze=False, sharey=True)
        for ax, method in zip(axes.flat, methods):
            selected = frame[(frame.method == method) & frame.series.isin(hues)]
            draw_panel(ax, selected, models, hues, show_batch=name == "h2o_capacity")
            ax.set_xlabel(method)
        # Autoscale only after all shared-axis panels have supplied their data.
        axes.flat[0].margins(y=0.20)
        axes.flat[0].set_ylim(bottom=0)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        for index, ax in enumerate(axes.flat):
            ax.get_legend().remove()
            if index:
                ax.set_ylabel("")
        fig.legend(handles, labels, loc="outside upper center", ncols=len(hues), frameon=False)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"{name}.{ext}", dpi=300)
        plt.close(fig)
    manifest = {"command": [sys.executable, *sys.argv], "data_sha256": hashlib.sha256(source).hexdigest(),
                "metric": payload["metric"], "error_bars": "sample standard deviation across iterations",
                "figsize_inches": [args.width, args.width / 2.5], "title": None,
                "versions": {"matplotlib": matplotlib.__version__, "seaborn": sns.__version__, "pandas": pd.__version__},
                "outputs": [str(p) for p in targets[:-1]]}
    targets[-1].write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
