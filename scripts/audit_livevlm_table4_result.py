#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("/data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla")
EXPECTED_VISIBLE_SUBTASKS = {
    "OP": ("Object Perception", 369),
    "CR": ("Causal Reasoning", 128),
    "CS": ("Clips Summarize", 317),
    "ATP": ("Attribute Perception", 312),
    "EU": ("Event Understanding", 159),
    "TR": ("Text-Rich Understanding", 321),
    "PR": ("Prospective Reasoning", 108),
    "SU": ("Spatial Understanding", 246),
    "ACP": ("Action Perception", 352),
    "CT": ("Counting", 188),
    "ER": ("Emotion Recognition", 250),
    "SCU": ("Scene Understanding", 250),
    "SD": ("Source Discrimination", 250),
    "MA": ("Multimodal Alignment", 250),
}
EXPECTED_EXTRA_SUBTASKS = {
    "ACU": ("Anomaly Context Understanding", 250),
    "MCU": ("Misleading Context Recognition", 250),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a LLaVA-OneVision StreamingBench LiveVLM Table 4 run.")
    parser.add_argument("--metrics_path", default="", help="Path to vanilla_aggregate_metrics.json.")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory containing vanilla_aggregate_metrics.json when --metrics_path is omitted.",
    )
    parser.add_argument("--json_out", default="", help="Optional path to write the audit summary JSON.")
    parser.add_argument(
        "--require_overall_delta_within_pct",
        type=float,
        default=None,
        help="Optional fail-fast threshold for |observed overall - expected overall| in percentage points.",
    )
    return parser.parse_args()


def resolve_metrics_path(metrics_path: str, output_dir: str) -> Path:
    if metrics_path:
        path = Path(metrics_path)
    else:
        path = Path(output_dir) / "vanilla_aggregate_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics file: {path}")
    return path


def require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def require_number(value, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    return float(value)


def weighted_accuracy_pct(items: list[dict]) -> float | None:
    total = sum(int(item.get("total", 0)) for item in items)
    if total <= 0:
        return None
    correct = sum(int(item.get("correct", 0)) for item in items)
    return 100.0 * correct / total


def audit_subtask_items(items: list[dict], expected: dict[str, tuple[str, int]], group_name: str) -> None:
    abbrs = [str(item.get("abbr", "")) for item in items]
    duplicates = sorted({abbr for abbr in abbrs if abbrs.count(abbr) > 1})
    if duplicates:
        raise RuntimeError(f"{group_name} subtask mismatch: duplicate abbreviations={duplicates}")

    by_abbr = {abbr: item for abbr, item in zip(abbrs, items)}
    missing = [abbr for abbr in expected if abbr not in by_abbr]
    unexpected = sorted(abbr for abbr in by_abbr if abbr not in expected)
    if missing or unexpected:
        raise RuntimeError(f"{group_name} subtask mismatch: missing={missing}, unexpected={unexpected}")

    mismatched = {}
    for abbr, (task_type, expected_rows) in expected.items():
        item = by_abbr[abbr]
        got_task_type = item.get("task_type")
        got_rows = int(require_number(item.get("total"), f"{abbr}.total"))
        recorded_expected_rows = int(require_number(item.get("expected_rows"), f"{abbr}.expected_rows"))
        if (
            got_task_type != task_type
            or got_rows != expected_rows
            or recorded_expected_rows != expected_rows
            or item.get("matches_expected_rows") is not True
        ):
            mismatched[abbr] = {
                "task_type": got_task_type,
                "expected_task_type": task_type,
                "total": got_rows,
                "expected_rows": expected_rows,
                "recorded_expected_rows": recorded_expected_rows,
                "matches_expected_rows": item.get("matches_expected_rows"),
            }
    if mismatched:
        raise RuntimeError(f"{group_name} subtask row-count mismatch: {mismatched}")


def audit_metrics(metrics: dict, require_overall_delta_within_pct: float | None = None) -> dict:
    stats = require_mapping(metrics.get("livevlm_table4_stats"), "livevlm_table4_stats")
    overall = require_mapping(stats.get("overall"), "livevlm_table4_stats.overall")
    subtasks = stats.get("subtasks")
    extra_subtasks = stats.get("overall_extra_subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != 14:
        raise RuntimeError(f"Expected 14 visible LiveVLM Table 4 subtasks, got {len(subtasks) if isinstance(subtasks, list) else type(subtasks).__name__}")
    if not isinstance(extra_subtasks, list) or len(extra_subtasks) != 2:
        raise RuntimeError(
            "Expected 2 overall-only LiveVLM Table 4 subtasks for ACU/MCU, "
            f"got {len(extra_subtasks) if isinstance(extra_subtasks, list) else type(extra_subtasks).__name__}"
        )
    audit_subtask_items(subtasks, EXPECTED_VISIBLE_SUBTASKS, "visible")
    audit_subtask_items(extra_subtasks, EXPECTED_EXTRA_SUBTASKS, "overall-only")

    expected_rows = int(require_number(stats.get("expected_overall_row_count"), "expected_overall_row_count"))
    overall_total = int(require_number(overall.get("total"), "overall.total"))
    if overall_total != expected_rows:
        raise RuntimeError(f"Overall row count mismatch: got {overall_total}, expected {expected_rows}")
    if overall.get("matches_expected_row_count") is not True:
        raise RuntimeError("overall.matches_expected_row_count must be true for a baseline audit")

    missing_visible = [item.get("abbr", item.get("task_type", "?")) for item in subtasks if int(item.get("total", 0)) <= 0]
    missing_extra = [item.get("abbr", item.get("task_type", "?")) for item in extra_subtasks if int(item.get("total", 0)) <= 0]
    if missing_visible or missing_extra:
        raise RuntimeError(f"Missing evaluated rows for subtasks: visible={missing_visible}, extra={missing_extra}")

    observed_overall = require_number(overall.get("accuracy_pct"), "overall.accuracy_pct")
    expected_overall = require_number(stats.get("expected_llava_onevision_7b_overall_pct"), "expected_llava_onevision_7b_overall_pct")
    overall_delta = observed_overall - expected_overall
    if require_overall_delta_within_pct is not None and abs(overall_delta) > require_overall_delta_within_pct:
        raise RuntimeError(
            f"Overall delta {overall_delta:.4f} pct exceeds threshold "
            f"{require_overall_delta_within_pct:.4f} pct"
        )

    visible_deltas = [
        require_number(item.get("delta_vs_expected_pct"), f"{item.get('abbr')}.delta_vs_expected_pct")
        for item in subtasks
    ]
    summary = {
        "method": metrics.get("method"),
        "num_samples": metrics.get("num_samples"),
        "overall_accuracy_pct": observed_overall,
        "expected_overall_accuracy_pct": expected_overall,
        "overall_delta_vs_expected_pct": overall_delta,
        "overall_total": overall_total,
        "expected_overall_row_count": expected_rows,
        "status_counts": overall.get("status_counts", {}),
        "visible_subtask_count": len(subtasks),
        "overall_extra_subtask_count": len(extra_subtasks),
        "max_abs_visible_subtask_delta_pct": max(abs(delta) for delta in visible_deltas),
        "expected_display_weighted_accuracy_pct": stats.get("expected_display_weighted_accuracy_pct"),
        "implied_expected_extra_subtasks_accuracy_pct": stats.get("implied_expected_extra_subtasks_accuracy_pct"),
        "observed_extra_subtasks_accuracy_pct": weighted_accuracy_pct(extra_subtasks),
        "subtasks": subtasks,
        "overall_extra_subtasks": extra_subtasks,
    }
    return summary


def print_summary(summary: dict) -> None:
    print(
        "overall "
        f"observed={summary['overall_accuracy_pct']:.4f} "
        f"expected={summary['expected_overall_accuracy_pct']:.4f} "
        f"delta={summary['overall_delta_vs_expected_pct']:.4f} "
        f"rows={summary['overall_total']}"
    )
    print(
        "extra "
        f"observed={summary['observed_extra_subtasks_accuracy_pct']:.4f} "
        f"implied_expected={summary['implied_expected_extra_subtasks_accuracy_pct']:.4f}"
    )
    for item in summary["subtasks"]:
        print(
            f"{item['abbr']:>3} {item['accuracy_pct']:7.3f} "
            f"expected={item['expected_llava_onevision_7b_pct']:7.3f} "
            f"delta={item['delta_vs_expected_pct']:7.3f} "
            f"total={item['total']:4d} {item['task_type']}"
        )
    for item in summary["overall_extra_subtasks"]:
        print(f"{item['abbr']:>3} {item['accuracy_pct']:7.3f} total={item['total']:4d} {item['task_type']}")


def main():
    args = parse_args()
    metrics_path = resolve_metrics_path(args.metrics_path, args.output_dir)
    metrics = require_mapping(json.loads(metrics_path.read_text(encoding="utf-8")), str(metrics_path))
    summary = audit_metrics(metrics, require_overall_delta_within_pct=args.require_overall_delta_within_pct)
    print_summary(summary)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
