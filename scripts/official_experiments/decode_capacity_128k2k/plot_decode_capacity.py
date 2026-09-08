"""Validate completed capacity sweeps and plot synchronized pure decode rates.

Validate raw runs: python plot_decode_capacity.py --config campaign.json
Replot exported data: python plot_decode_capacity.py --plot-data plot_data.json
Override method colors: add --palette path/to/palette.json
Only observed results are accepted. Incomplete curves fail before rendering.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_PALETTE = Path(__file__).resolve().parent / "palettes" / "fresh_modern.json"

LANES = {
    "vllm-vanilla": ("vLLM vanilla", "o"),
    "svllm-vanilla": ("Sparse-vLLM vanilla", "s"),
    "svllm-snapkv": ("SnapKV · 8K context", "D"),
    "svllm-quest": ("QuEST · budget 2048", "^"),
    "svllm-omnikv": ("OmniKV · auto / 2048", "P"),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_measurement(path, concurrency, config):
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one measurement: {path}")
    row = rows[0]
    expected = {"status": "success", "stage_metrics_status": "success",
                "measurement_scope": "full_batch_pure_decode_steps",
                "actual_decode_peak": concurrency, "completed_requests": concurrency,
                "length": config["input_len"], "output_len": config["output_len"],
                "scheduler_preemptions": 0, "synchronize_step_timing": True}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{path}: {key}={row.get(key)!r}, expected {value!r}")
    case = path.parent / f"{row['method']}-{row['length']}-{concurrency}"
    outputs = read_rows(case / "raw_outputs.jsonl")
    if len(outputs) != concurrency or len({item["request_id"] for item in outputs}) != concurrency:
        raise ValueError(f"Missing or duplicate requests: {case}")
    if any(item["status"] != "success" or len(item["token_ids"]) != config["output_len"] for item in outputs):
        raise ValueError(f"Incomplete generation: {case}")
    steps = [item for item in read_rows(case / "steps.jsonl") if item["measured"]]
    if not steps or any(abs(item["tokens"]) != concurrency or not math.isfinite(item["elapsed_s"]) or item["elapsed_s"] <= 0 for item in steps):
        raise ValueError(f"Measured step is not a full batch: {case}")
    if row["engine"] == "sparsevllm" and any(item["tokens"] >= 0 for item in steps):
        raise ValueError(f"Prefill included in native decode rate: {case}")
    if row["engine"] == "vllm" and any(not item["pure_decode"] for item in steps):
        raise ValueError(f"Mixed or prefill step included in vLLM decode rate: {case}")
    tokens = sum(abs(item["tokens"]) for item in steps)
    elapsed = sum(item["elapsed_s"] for item in steps)
    if tokens != row["decode_stage_tokens"] or not math.isclose(elapsed, row["decode_stage_elapsed_s"], rel_tol=1e-10):
        raise ValueError(f"Stage accumulation disagrees with raw steps: {case}")
    if not math.isclose(tokens / elapsed, row["decode_stage_throughput_tps"], rel_tol=1e-10):
        raise ValueError(f"Throughput disagrees with raw steps: {case}")
    return {"concurrency": concurrency, "decode_throughput_tps": row["decode_stage_throughput_tps"],
            "decode_tokens": tokens, "decode_elapsed_s": elapsed,
            "artifact": str(path), "measured_steps": len(steps)}


def load_campaign(config):
    root = Path(config["output_root"])
    curves = []
    for model in config["models"]:
        for lane in LANES:
            matches = [path for path in (root / model).glob(f"*/{lane}/capacity.json")
                       if json.loads(path.read_text()).get("status") == "completed"]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one completed {model}/{lane} sweep, found {len(matches)}")
            boundary = json.loads(matches[0].read_text())
            if boundary["status"] != "completed":
                raise ValueError(f"Incomplete sweep: {matches[0]}")
            maximum = boundary["max_concurrency"]
            if boundary["first_failed_concurrency"] != maximum + 1:
                raise ValueError(f"Integer maximum is unbounded: {matches[0]}")
            attempts = boundary["attempts"]
            if not any(item["concurrency"] == maximum + 1 and item["status"] == "capacity_exceeded" for item in attempts):
                raise ValueError(f"Missing capacity failure evidence: {matches[0]}")
            needed = {maximum}
            power = 1
            while power <= maximum:
                needed.add(power)
                power *= 2
            points = {}
            for attempt in attempts:
                if attempt["status"] == "success":
                    batch = attempt["concurrency"]
                    points[batch] = validate_measurement(Path(attempt["artifact"]), batch, config)
            if not needed <= points.keys():
                raise ValueError(f"Missing required concurrency points: {matches[0]}")
            curves.append({"model": model, "lane": lane, "max_concurrency": maximum,
                           "first_failed_concurrency": boundary["first_failed_concurrency"],
                           "attempts": attempts,
                           "points": [points[batch] for batch in sorted(points)],
                           "capacity_artifact": str(matches[0])})
    return curves


def render(curves, config, output, palette_path=DEFAULT_PALETTE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    from matplotlib.ticker import LogLocator, StrMethodFormatter
    import seaborn as sns

    palette = json.loads(Path(palette_path).read_text())
    colors = palette["colors"]
    for lane in LANES:
        if lane not in colors or not is_color_like(colors[lane]):
            raise ValueError(f"Palette {palette_path} has a missing or invalid color for {lane}")

    sns.set_theme(context="notebook", style="whitegrid", font="DejaVu Sans",
                  rc={"axes.spines.top": False, "axes.spines.right": False,
                      "grid.alpha": 0.25, "axes.titleweight": "bold"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "palette.json").write_text(json.dumps(palette, indent=2) + "\n")

    def panel(ax, model, log_y):
        for curve in curves:
            if curve["model"] != model:
                continue
            label, marker = LANES[curve["lane"]]
            color = colors[curve["lane"]]
            xs = [point["concurrency"] for point in curve["points"]]
            ys = [point["decode_throughput_tps"] for point in curve["points"]]
            sns.lineplot(x=xs, y=ys, ax=ax, label=label, color=color, marker=marker,
                         markersize=5 if curve["lane"] == "svllm-vanilla" else 6,
                         linestyle="--" if curve["lane"] == "svllm-vanilla" else "-",
                         linewidth=2, estimator=None, errorbar=None)
        topology = config["models"][model]
        ax.set_title(f"{topology.get('display_name', model)}\nTP{topology['tp']} · EP{topology['ep']}")
        ax.set(xlabel="Concurrency", ylabel="Pure decode throughput (tokens/s)")
        ax.set_xscale("log", base=2)
        # Integer-boundary probes can be adjacent on the log axis; keep native
        # major ticks at powers of two instead of overlapping endpoint labels.
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.set_ylabel("Pure decode throughput (tokens/s, log scale)")
        else:
            ax.set_ylim(bottom=0)
        ax.legend_.remove()

    models = list(config["models"])
    for log_y in (False, True):
        suffix = "_logy" if log_y else ""
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.6),
                                 layout="constrained", squeeze=False)
        for ax, model in zip(axes.flat, models):
            panel(ax, model, log_y)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
        fig.supxlabel("128K input · 2K output · full-batch synchronized decode steps · first 32 steps excluded")
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"decode_capacity_128k2k{suffix}.{extension}", dpi=220)
        plt.close(fig)
        for model in models:
            fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
            panel(ax, model, log_y)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output / f"{model}{suffix}.{extension}", dpi=220)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--plot-data", type=Path, help="Replot a previously raw-validated portable export")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE,
                        help="Method-color JSON; defaults to palettes/fresh_modern.json beside this script")
    args = parser.parse_args()
    if args.plot_data:
        exported = json.loads(args.plot_data.read_text())
        if exported["schema_version"] != 1:
            raise ValueError("Unsupported plot-data schema")
        config, curves = exported["config"], exported["curves"]
        expected = {(model, lane) for model in config["models"] for lane in LANES}
        if len(curves) != len(expected) or {(c["model"], c["lane"]) for c in curves} != expected:
            raise ValueError("Portable export is missing curves or contains duplicates")
        for curve in curves:
            points = curve["points"]
            batches = [p["concurrency"] for p in points]
            maximum = curve["max_concurrency"]
            if (type(maximum) is not int or maximum < 1
                    or curve["first_failed_concurrency"] != maximum + 1
                    or not any(item["concurrency"] == maximum + 1
                               and item["status"] == "capacity_exceeded"
                               for item in curve["attempts"])):
                raise ValueError("Portable export is missing an integer capacity boundary")
            successful = {item["concurrency"] for item in curve["attempts"]
                          if item["status"] == "success"}
            if set(batches) != successful:
                raise ValueError("Portable export points disagree with successful attempts")
            needed = {curve["max_concurrency"]}
            power = 1
            while power <= curve["max_concurrency"]:
                needed.add(power)
                power *= 2
            if (batches != sorted(set(batches)) or not needed <= set(batches)
                    or max(batches) != curve["max_concurrency"]):
                raise ValueError("Portable export has an incomplete concurrency curve")
            for point in points:
                tokens, elapsed, rate = (point[key] for key in
                                        ("decode_tokens", "decode_elapsed_s", "decode_throughput_tps"))
                if (not all(math.isfinite(v) and v > 0 for v in (tokens, elapsed, rate))
                        or not math.isclose(tokens / elapsed, rate, rel_tol=1e-10)):
                    raise ValueError("Portable export has inconsistent stage throughput")
        output = args.output_dir or args.plot_data.parent / "plots"
    else:
        config = json.loads(args.config.read_text())
        curves = load_campaign(config)
        output = args.output_dir or Path(config["output_root"]) / "plots"
    render(curves, config, output, palette_path=args.palette)
    (output / "validated_curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    (output / "plot_data.json").write_text(json.dumps({
        "schema_version": 1,
        "metric": "actual_decode_tokens / sum_synchronized_full_batch_decode_step_seconds",
        "validation": "raw steps and full outputs verified at export; replot checks exported sums only",
        "config": config, "curves": curves,
    }, indent=2) + "\n")
    fields = ["model", "lane", "concurrency", "decode_throughput_tps", "decode_tokens",
              "decode_elapsed_s", "measured_steps", "artifact"]
    with (output / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for curve in curves:
            for point in curve["points"]:
                writer.writerow({"model": curve["model"], "lane": curve["lane"], **point})


if __name__ == "__main__":
    main()
