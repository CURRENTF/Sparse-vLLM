#!/usr/bin/env python3
"""Plot dynamic-stride reference counts with RULER-VT scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from deltakv.analysis.colors import (
    COLOR_PRIMARY,
    COLOR_PRIMARY_LIGHT,
    COLOR_SECONDARY,
    COLOR_SECONDARY_LIGHT,
    COLOR_GRID,
    TEXT_HIGHLIGHT_1,
    TEXT_HIGHLIGHT_2,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "dynamic_stride_ruler_vt"
DEFAULT_OUTPUT = Path("outputs") / "analysis" / "reference_count_similarity.pdf"

CONTEXT = [4096, 8192, 16384, 32768, 65536, 98304]
CTX_LABELS = ["4K", "8K", "16K", "32K", "64K", "96K"]
SINK_TOKENS = 8
INITIAL_STRIDE = 10
COUNT_AXIS_COLOR = TEXT_HIGHLIGHT_1
SCORE_AXIS_COLOR = TEXT_HIGHLIGHT_2
SETTING_LINESTYLE = "-"
COUNT_MARKER_SIZE = 4.8
SCORE_MARKER_SIZE = 5.2


COUNT_SERIES = [
    {
        "id": "fixed_s10",
        "label": "Fixed $s=10$",
        "marker": "o",
        "color": COUNT_AXIS_COLOR,
        "kind": "fixed",
        "stride": 10,
    },
    {
        "id": "alpha_0p001",
        "label": r"Dynamic $\alpha=0.001$",
        "marker": "s",
        "color": COLOR_PRIMARY,
        "kind": "dynamic",
        "alpha": 0.001,
    },
    {
        "id": "alpha_0p02",
        "label": r"Dynamic $\alpha=0.02$",
        "marker": "^",
        "color": COLOR_PRIMARY_LIGHT,
        "kind": "dynamic",
        "alpha": 0.02,
    },
]

RULER_VT_SERIES = [
    {
        "id": "fixed_s10",
        "filename": "fixed_s10_aggregate_metrics.json",
        "marker": "o",
        "color": TEXT_HIGHLIGHT_2,
    },
    {
        "id": "alpha_0p001",
        "filename": "alpha0p001_aggregate_metrics.json",
        "marker": "s",
        "color": COLOR_SECONDARY,
    },
    {
        "id": "alpha_0p02",
        "filename": "alpha0p02_aggregate_metrics.json",
        "marker": "^",
        "color": COLOR_SECONDARY_LIGHT,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing RULER-VT aggregate_metrics JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    return parser.parse_args()


def fixed_count(context_length: int, stride: int) -> int:
    tail = max(0, context_length - SINK_TOKENS)
    return min(SINK_TOKENS, context_length) + math.ceil(tail / stride)


def dynamic_count(context_length: int, alpha: float) -> int:
    count = min(SINK_TOKENS, context_length)
    position = SINK_TOKENS
    while position < context_length:
        count += 1
        step = INITIAL_STRIDE + int(alpha * max(0, position - SINK_TOKENS))
        position += max(1, step)
    return count


def format_ref_count(value, _pos):
    if value >= 1000:
        return f"{value / 1000:g}K"
    return f"{int(value)}"


def load_ruler_vt_scores(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing RULER-VT aggregate file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_length = data["score_by_context_length"]
    return [float(by_length[str(length)]["score"]) for length in CONTEXT]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.0,
        }
    )


def plot(data_dir: Path, output: Path) -> None:
    configure_matplotlib()

    fig, ax_count = plt.subplots(figsize=(3.35, 2.51))
    ax_score = ax_count.twinx()

    for series in COUNT_SERIES:
        if series["kind"] == "fixed":
            counts = [fixed_count(length, int(series["stride"])) for length in CONTEXT]
        else:
            counts = [dynamic_count(length, float(series["alpha"])) for length in CONTEXT]
        ax_count.plot(
            CONTEXT,
            counts,
            color=str(series["color"]),
            linestyle=SETTING_LINESTYLE,
            linewidth=1.45,
            marker=str(series["marker"]),
            markersize=COUNT_MARKER_SIZE,
            markeredgewidth=0.7,
            markerfacecolor=str(series["color"]),
            markeredgecolor=str(series["color"]),
            zorder=3,
        )

    for series in RULER_VT_SERIES:
        scores = load_ruler_vt_scores(data_dir / str(series["filename"]))
        ax_score.plot(
            CONTEXT,
            scores,
            color=str(series["color"]),
            linestyle=SETTING_LINESTYLE,
            linewidth=1.35,
            marker=str(series["marker"]),
            markersize=SCORE_MARKER_SIZE,
            markerfacecolor=str(series["color"]),
            markeredgecolor=str(series["color"]),
            markeredgewidth=0.85,
            alpha=0.98,
            zorder=4,
        )

    ax_count.set_xscale("log", base=2)
    ax_count.set_yscale("log")
    ax_count.set_xlim(CONTEXT[0], CONTEXT[-1])
    ax_count.set_ylim(45, 13000)
    ax_score.set_ylim(78, 101.5)

    ax_count.set_xlabel("Context Length")
    ax_count.set_ylabel(r"Reference Token Count ($\downarrow$)")
    ax_score.set_ylabel(r"RULER-VT Score ($\uparrow$)")
    ax_count.yaxis.label.set_color(COUNT_AXIS_COLOR)
    ax_score.yaxis.label.set_color(SCORE_AXIS_COLOR)

    ax_count.set_xticks(CONTEXT, CTX_LABELS)
    ax_count.set_yticks([100, 300, 1000, 3000, 10000])
    ax_count.yaxis.set_major_formatter(FuncFormatter(format_ref_count))
    ax_score.set_yticks([85, 90, 95, 100])

    ax_count.tick_params(axis="x", which="major", length=2.5, width=0.55, pad=2)
    ax_count.tick_params(axis="x", which="minor", length=1.4, width=0.35)
    ax_count.tick_params(axis="y", which="major", length=2.5, width=0.55, pad=2, colors=COUNT_AXIS_COLOR)
    ax_count.tick_params(axis="y", which="minor", length=1.4, width=0.35, colors=COUNT_AXIS_COLOR)
    ax_score.tick_params(axis="y", which="major", length=2.5, width=0.55, pad=2, colors=SCORE_AXIS_COLOR)
    ax_score.tick_params(axis="y", which="minor", length=1.4, width=0.35, colors=SCORE_AXIS_COLOR)
    ax_count.spines["left"].set_color(COUNT_AXIS_COLOR)
    ax_score.spines["right"].set_color(SCORE_AXIS_COLOR)

    ax_count.grid(True, which="major", color=COLOR_GRID, linewidth=0.45)
    ax_count.grid(True, which="minor", color="#eeeeee", linewidth=0.25)
    ax_count.set_axisbelow(True)

    schedule_handles = [
        Line2D(
            [0],
            [0],
            color="#424242",
            linestyle=SETTING_LINESTYLE,
            marker=str(series["marker"]),
            markersize=4.6,
            markerfacecolor="#424242",
            markeredgewidth=0.65,
            linewidth=1.4,
            label=str(series["label"]),
        )
        for series in COUNT_SERIES
    ]
    schedule_legend = ax_count.legend(
        handles=schedule_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.78,
        borderpad=0.14,
        handlelength=1.55,
        handletextpad=0.28,
        columnspacing=0.45,
        labelspacing=0.12,
    )
    schedule_legend.get_frame().set_linewidth(0.35)
    schedule_legend.get_frame().set_edgecolor("#d8d8d8")
    ax_count.add_artist(schedule_legend)

    fig.tight_layout(pad=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01)


def main() -> None:
    args = parse_args()
    plot(args.data_dir, args.output)


if __name__ == "__main__":
    main()
