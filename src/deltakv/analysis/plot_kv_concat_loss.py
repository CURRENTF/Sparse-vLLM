#!/usr/bin/env python3
"""Plot step-aligned K/V split vs. concatenation training loss from W&B."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import wandb

from deltakv.analysis.colors import (
    COLOR_GRID,
    COLOR_BLACK,
    TEXT_HIGHLIGHT_2,
)


ENTITY = "concentrate_42"
PROJECT = "ReKV"
SPLIT_RUN_ID = "4860matu"
CONCAT_RUN_ID = "qc952pgw"
DEFAULT_MAX_STEP = 6960
DEFAULT_METRIC = "train/loss"
DEFAULT_OUTPUT = Path("outputs") / "analysis" / "kv_concat_loss_step_aligned.pdf"
RULER_VT_COUNT_AXIS_COLOR = "#1f5f8f"
RULER_VT_SCORE_AXIS_COLOR = TEXT_HIGHLIGHT_2

HISTORY_KEYS = [
    "_runtime",
    "_step",
    "train/global_step",
    "train/loss",
    "train/ntp_loss",
    "train/mse_loss",
    "train/epoch",
    "train/learning_rate",
]

MATCHED_CONFIG_KEYS = [
    "model_name_or_path",
    "dataset_path",
    "recon_mode",
    "ref_mode",
    "seq_chunk_size",
    "layer_chunk_size",
    "learning_rate",
    "use_nonlinear_compressor",
    "compressor_intermediate_size",
    "collect_kv_before_rope",
    "cluster_on_kv",
    "cluster_ratio",
    "cluster_metric",
    "compression_mode",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "num_train_epochs",
    "num_compressors",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "torch_dtype",
    "bf16",
    "fp16",
    "lr_scheduler_type",
    "warmup_ratio",
    "weight_decay",
    "optim",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--split-run-id", default=SPLIT_RUN_ID)
    parser.add_argument("--concat-run-id", default=CONCAT_RUN_ID)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--max-step", type=int, default=DEFAULT_MAX_STEP)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window in training steps. Use 1 to disable smoothing.",
    )
    parser.add_argument("--ylim-min", type=float, default=None)
    parser.add_argument("--ylim-max", type=float, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Raw CSV output path. Defaults to output path with .csv suffix.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Metadata JSON path. Defaults to output path with .json suffix.",
    )
    return parser.parse_args()


def almost_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) < 1e-12
        except (TypeError, ValueError):
            return left == right
    return left == right


def load_run(api: wandb.Api, entity: str, project: str, run_id: str):
    return api.run(f"{entity}/{project}/{run_id}")


def validate_matched_configs(split_run, concat_run) -> list[dict[str, Any]]:
    mismatches = []
    split_config = split_run.config
    concat_config = concat_run.config
    for key in MATCHED_CONFIG_KEYS:
        split_value = split_config.get(key)
        concat_value = concat_config.get(key)
        if not almost_equal(split_value, concat_value):
            mismatches.append(
                {
                    "key": key,
                    "split": split_value,
                    "concat": concat_value,
                }
            )
    if mismatches:
        details = json.dumps(mismatches, indent=2, ensure_ascii=False, default=str)
        raise ValueError(f"Runs are not matched on required config keys:\n{details}")
    return [
        {
            "key": "split_kv",
            "split": split_config.get("split_kv"),
            "concat": concat_config.get("split_kv"),
        },
        {
            "key": "kv_compressed_size",
            "split": split_config.get("kv_compressed_size"),
            "concat": concat_config.get("kv_compressed_size"),
        },
    ]


def load_history(run, metric: str, max_step: int) -> list[dict[str, Any]]:
    keys = sorted(set(HISTORY_KEYS + [metric]))
    rows = []
    for row in run.scan_history(keys=keys, page_size=1000):
        step = row.get("train/global_step", row.get("_step"))
        value = row.get(metric)
        if step is None or value is None:
            continue
        step = int(step)
        if step > max_step:
            continue
        rows.append(
            {
                "step": step,
                "runtime": row.get("_runtime"),
                "metric": float(value),
                "loss": row.get("train/loss"),
                "ntp_loss": row.get("train/ntp_loss"),
                "mse_loss": row.get("train/mse_loss"),
                "epoch": row.get("train/epoch"),
                "learning_rate": row.get("train/learning_rate"),
            }
        )
    rows.sort(key=lambda item: item["step"])
    if not rows:
        raise ValueError(f"No history rows found for {run.name} and metric {metric}")
    return rows


def smooth_series(rows: list[dict[str, Any]], window_steps: int) -> list[float]:
    if window_steps <= 1:
        return [float(row["metric"]) for row in rows]
    half_window = window_steps / 2.0
    values = []
    left = 0
    right = 0
    running_sum = 0.0
    for row in rows:
        center = row["step"]
        while right < len(rows) and rows[right]["step"] <= center + half_window:
            running_sum += float(rows[right]["metric"])
            right += 1
        while left < right and rows[left]["step"] < center - half_window:
            running_sum -= float(rows[left]["metric"])
            left += 1
        values.append(running_sum / max(1, right - left))
    return values


def write_csv(path: Path, series: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "step",
        "runtime",
        "metric",
        "loss",
        "ntp_loss",
        "mse_loss",
        "epoch",
        "learning_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, rows in series.items():
            for row in rows:
                writer.writerow({"label": label, **row})


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def plot(
    series: dict[str, list[dict[str, Any]]],
    metric: str,
    output: Path,
    smooth_window: int,
    ylim_min: float | None,
    ylim_max: float | None,
) -> None:
    configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    styles = {
        "Joint K/V": {
            "color": RULER_VT_COUNT_AXIS_COLOR,
            "linewidth": 1.55,
            "marker": "o",
            "markevery": 80,
            "zorder": 4,
        },
        "Split K/V": {
            "color": RULER_VT_SCORE_AXIS_COLOR,
            "linewidth": 1.45,
            "marker": "s",
            "markevery": 80,
            "zorder": 3,
        },
    }

    for label, rows in series.items():
        steps = [row["step"] for row in rows]
        values = smooth_series(rows, smooth_window)
        style = styles[label]
        ax.plot(
            steps,
            values,
            label=label,
            color=style["color"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=2.4,
            markevery=style["markevery"],
            markeredgewidth=0.0,
            zorder=style["zorder"],
        )

    ax.set_xlabel("Training Step")
    ylabel = {
        "train/loss": r"Training Loss ($\downarrow$)",
        "train/ntp_loss": r"NTP Loss ($\downarrow$)",
        "train/mse_loss": r"MSE Loss ($\downarrow$)",
    }.get(metric, f"{metric} ($\\downarrow$)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(row["step"] for rows in series.values() for row in rows))
    if ylim_min is not None or ylim_max is not None:
        ax.set_ylim(bottom=ylim_min, top=ylim_max)
    ax.grid(True, which="major", color=COLOR_GRID, linewidth=0.45)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.25)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.55, pad=2)
    ax.tick_params(axis="both", which="minor", length=1.4, width=0.35)
    ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=0.92)

    fig.tight_layout(pad=0.25)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    raw_csv = args.raw_csv or args.output.with_suffix(".csv")
    metadata_json = args.metadata_json or args.output.with_suffix(".json")

    api = wandb.Api(timeout=90)
    split_run = load_run(api, args.entity, args.project, args.split_run_id)
    concat_run = load_run(api, args.entity, args.project, args.concat_run_id)
    allowed_differences = validate_matched_configs(split_run, concat_run)

    series = {
        "Joint K/V": load_history(concat_run, args.metric, args.max_step),
        "Split K/V": load_history(split_run, args.metric, args.max_step),
    }
    write_csv(raw_csv, series)
    plot(series, args.metric, args.output, args.smooth_window, args.ylim_min, args.ylim_max)

    metadata = {
        "entity": args.entity,
        "project": args.project,
        "metric": args.metric,
        "max_step": args.max_step,
        "smooth_window": args.smooth_window,
        "ylim_min": args.ylim_min,
        "ylim_max": args.ylim_max,
        "output": str(args.output),
        "raw_csv": str(raw_csv),
        "allowed_config_differences": allowed_differences,
        "runs": {
            "Joint K/V": {
                "id": concat_run.name,
                "display_name": concat_run.display_name,
                "url": concat_run.url,
                "state": concat_run.state,
                "final_step": series["Joint K/V"][-1]["step"],
                "final_metric": series["Joint K/V"][-1]["metric"],
            },
            "Split K/V": {
                "id": split_run.name,
                "display_name": split_run.display_name,
                "url": split_run.url,
                "state": split_run.state,
                "final_step": series["Split K/V"][-1]["step"],
                "final_metric": series["Split K/V"][-1]["metric"],
            },
        },
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    run()
