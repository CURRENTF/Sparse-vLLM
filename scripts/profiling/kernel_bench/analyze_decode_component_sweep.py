#!/usr/bin/env python3
"""Analyze matched attention and routed-MoE decode component sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ratio_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty ratio list")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _matched_attention_rows(
    summary: dict[str, Any], profile: str
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in summary["attention_comparisons"]
        if row["profile"] == profile
    ]
    if not rows:
        raise ValueError(f"attention profile not found: {profile}")
    missing = [
        (row["context_len"], row["batch_size"])
        for row in rows
        if row["triton_ms"] is None or row["flashinfer_ms"] is None
    ]
    if missing:
        raise ValueError(
            f"attention profile {profile} has unmatched providers: {missing}"
        )
    return rows


def _moe_index(
    summary: dict[str, Any], profile: str
) -> dict[int, dict[str, Any]]:
    rows = [
        row for row in summary["moe_comparisons"] if row["profile"] == profile
    ]
    if not rows:
        raise ValueError(f"MoE profile not found: {profile}")
    return {int(row["batch_size"]): row for row in rows}


def _attention_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(row["flashinfer_over_triton"]) for row in rows]
    per_context = []
    for context_len in sorted({int(row["context_len"]) for row in rows}):
        context_rows = [
            row for row in rows if int(row["context_len"]) == context_len
        ]
        context_ratios = [
            float(row["flashinfer_over_triton"]) for row in context_rows
        ]
        per_context.append(
            {
                "context_len": context_len,
                "case_count": len(context_rows),
                "flashinfer_win_count": sum(x < 1.0 for x in context_ratios),
                "triton_win_count": sum(x > 1.0 for x in context_ratios),
                "ratio": _ratio_summary(context_ratios),
            }
        )
    return {
        "case_count": len(rows),
        "flashinfer_win_count": sum(x < 1.0 for x in ratios),
        "triton_win_count": sum(x > 1.0 for x in ratios),
        "exact_tie_count": sum(x == 1.0 for x in ratios),
        "flashinfer_over_triton": _ratio_summary(ratios),
        "per_context": per_context,
    }


def _component_rows(
    attention_rows: list[dict[str, Any]],
    moe_by_batch: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for attention in attention_rows:
        batch_size = int(attention["batch_size"])
        moe = moe_by_batch.get(batch_size)
        if moe is None or moe["triton_ms"] is None:
            continue
        triton_attention_ms = float(attention["triton_ms"])
        flashinfer_attention_ms = float(attention["flashinfer_ms"])
        triton_moe_ms = float(moe["triton_ms"])
        triton_total_ms = triton_attention_ms + triton_moe_ms
        hybrid_total_ms = flashinfer_attention_ms + triton_moe_ms
        result.append(
            {
                "context_len": int(attention["context_len"]),
                "batch_size": batch_size,
                "triton_attention_ms": triton_attention_ms,
                "flashinfer_attention_ms": flashinfer_attention_ms,
                "triton_moe_ms": triton_moe_ms,
                "triton_total_ms": triton_total_ms,
                "hybrid_total_ms": hybrid_total_ms,
                "hybrid_total_over_triton_total": (
                    hybrid_total_ms / triton_total_ms
                ),
                "triton_attention_share": triton_attention_ms / triton_total_ms,
                "triton_moe_share": triton_moe_ms / triton_total_ms,
                "hybrid_attention_share": (
                    flashinfer_attention_ms / hybrid_total_ms
                ),
                "hybrid_moe_share": triton_moe_ms / hybrid_total_ms,
            }
        )
    if not result:
        raise ValueError("attention and MoE profiles have no matched batch sizes")
    return result


def _component_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "triton_attention_share",
        "triton_moe_share",
        "hybrid_attention_share",
        "hybrid_moe_share",
        "hybrid_total_over_triton_total",
    )
    aggregate = {
        field: _ratio_summary([float(row[field]) for row in rows])
        for field in fields
    }
    per_context = []
    for context_len in sorted({int(row["context_len"]) for row in rows}):
        context_rows = [
            row for row in rows if int(row["context_len"]) == context_len
        ]
        per_context.append(
            {
                "context_len": context_len,
                "case_count": len(context_rows),
                **{
                    field: _ratio_summary(
                        [float(row[field]) for row in context_rows]
                    )
                    for field in fields
                },
            }
        )
    return {"case_count": len(rows), "aggregate": aggregate, "per_context": per_context}


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _build_report(analysis: dict[str, Any]) -> str:
    attention = analysis["attention"]
    component = analysis["component_composition"]
    lines = [
        "# Decode component sweep analysis",
        "",
        "## Attention provider trend",
        "",
        (
            f"FlashInfer wins {attention['flashinfer_win_count']}/"
            f"{attention['case_count']} matched cases; Triton wins "
            f"{attention['triton_win_count']}/{attention['case_count']}."
        ),
        "",
        "| KV length | Cases | FI wins | Triton wins | Median FI/Tri | Range |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in attention["per_context"]:
        ratio = row["ratio"]
        lines.append(
            f"| {row['context_len']} | {row['case_count']} | "
            f"{row['flashinfer_win_count']} | {row['triton_win_count']} | "
            f"{ratio['median']:.3f} | {ratio['min']:.3f}-{ratio['max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Attention and routed-MoE kernel-only composition",
            "",
            (
                "Hybrid means FlashInfer attention plus Triton routed MoE. "
                "Shares use the sum of only those two direct component calls."
            ),
            "",
            (
                "| KV length | Cases | Triton attention share | "
                "Hybrid attention share | Hybrid/Triton total |"
            ),
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in component["per_context"]:
        lines.append(
            f"| {row['context_len']} | {row['case_count']} | "
            f"{_pct(row['triton_attention_share']['median'])} | "
            f"{_pct(row['hybrid_attention_share']['median'])} | "
            f"{row['hybrid_total_over_triton_total']['median']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Excluded: QKV/O projections, router logits/top-k, collectives, "
                "normalization, sampling, cache selection/compaction, and host work."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    summary_path = args.summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "component_analysis.json"
    report_path = output_dir / "component_analysis.md"
    if analysis_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis in {output_dir}")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    attention_rows = _matched_attention_rows(summary, args.attention_profile)
    moe_by_batch = _moe_index(summary, args.moe_profile)
    component_rows = _component_rows(attention_rows, moe_by_batch)
    analysis = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(summary_path),
        "source_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "attention_profile": args.attention_profile,
        "moe_profile": args.moe_profile,
        "attention": _attention_analysis(attention_rows),
        "component_composition": {
            **_component_analysis(component_rows),
            "rows": component_rows,
        },
        "limitations": [
            "These are direct CUDA-Graph component callable medians.",
            (
                "Composition denominators include only attention and routed-MoE; "
                "they are not full-layer or serving-level shares."
            ),
        ],
    }
    _write_json(analysis_path, analysis)
    report_path.write_text(_build_report(analysis), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--attention-profile", required=True)
    parser.add_argument("--moe-profile", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
