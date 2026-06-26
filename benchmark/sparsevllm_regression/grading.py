from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GRADE_ORDER = {"A": 3, "B": 2, "C": 1, "D": 0, "N/A": -1}


@dataclass(frozen=True)
class GateGrade:
    name: str
    grade: str
    status: str
    metrics: dict[str, Any]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "grade": self.grade,
            "status": self.status,
            "metrics": self.metrics,
            "reason": self.reason,
        }


def _require_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    return float(value)


def grade_quality(vanilla_score: float, sparse_score: float) -> GateGrade:
    vanilla = _require_number(vanilla_score, "vanilla_score")
    sparse = _require_number(sparse_score, "sparse_score")
    score_loss = max(0.0, vanilla - sparse)
    if score_loss < 0.1:
        grade = "A"
    elif score_loss <= 0.5:
        grade = "B"
    elif score_loss <= 1.0:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        name="quality",
        grade=grade,
        status="success" if grade != "D" else "failed",
        metrics={
            "vanilla_score": vanilla,
            "sparse_score": sparse,
            "score_loss": score_loss,
        },
    )


def grade_logits(metrics: dict[str, Any] | None, *, p99_threshold: float | None = None) -> GateGrade:
    if metrics is None:
        return GateGrade("logits", "N/A", "skipped_by_policy", {}, "HF logits reference is not available.")

    steps = metrics.get("decode_steps") or []
    if not steps:
        return GateGrade("logits", "D", "failed", metrics, "No decode step metrics.")
    top1_ok = all(bool(step.get("argmax_match")) for step in steps)
    top5 = [
        float((step.get("topk_overlap") or {}).get("5", {}).get("ratio", 0.0))
        for step in steps
    ]
    top10 = [
        float((step.get("topk_overlap") or {}).get("10", {}).get("ratio", 0.0))
        for step in steps
    ]
    top5_mean = sum(top5) / len(top5)
    top10_mean = sum(top10) / len(top10)
    p99_values = [float(step.get("p99_abs_diff", float("inf"))) for step in steps]
    p99_max = max(p99_values) if p99_values else float("inf")

    if top1_ok and top5_mean >= 0.8 and top10_mean >= 0.9 and (
        p99_threshold is None or p99_max <= float(p99_threshold)
    ):
        grade = "A"
    elif top1_ok and top5_mean >= 0.8:
        grade = "B"
    elif top1_ok:
        grade = "C"
    else:
        grade = "D"

    return GateGrade(
        name="logits",
        grade=grade,
        status="success" if grade != "D" else "failed",
        metrics={
            "top1_all_match": top1_ok,
            "top5_overlap_mean": top5_mean,
            "top10_overlap_mean": top10_mean,
            "p99_abs_diff_max": p99_max,
            "p99_threshold": p99_threshold,
        },
    )


def grade_perf(
    speedup: float,
    *,
    graph_expected: bool = True,
    graph_active: bool = True,
    require_speedup: bool = True,
) -> GateGrade:
    speedup = _require_number(speedup, "speedup")
    if graph_expected and not graph_active:
        return GateGrade(
            "performance",
            "D",
            "failed",
            {
                "speedup": speedup,
                "graph_expected": graph_expected,
                "graph_active": graph_active,
                "require_speedup": require_speedup,
            },
            "decode CUDA graph was expected but not active.",
        )
    if not require_speedup:
        return GateGrade(
            "performance",
            "A",
            "success",
            {
                "speedup": speedup,
                "graph_expected": graph_expected,
                "graph_active": graph_active,
                "require_speedup": require_speedup,
            },
            "Speedup is recorded but not required by this performance gate.",
        )
    if speedup >= 2.0:
        grade = "A"
    elif speedup >= 1.5:
        grade = "B"
    elif speedup > 1.0:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        "performance",
        grade,
        "success" if grade != "D" else "failed",
        {
            "speedup": speedup,
            "graph_expected": graph_expected,
            "graph_active": graph_active,
            "require_speedup": require_speedup,
        },
    )


def grade_memory(*, expected_savings: float | None, observed_savings: float | None) -> GateGrade:
    if expected_savings is None or observed_savings is None:
        return GateGrade(
            "memory",
            "D",
            "failed",
            {"expected_savings": expected_savings, "observed_savings": observed_savings},
            "Memory accounting is incomplete.",
        )
    expected = _require_number(expected_savings, "expected_savings")
    observed = _require_number(observed_savings, "observed_savings")
    error = abs(expected - observed)
    if observed <= 0:
        grade = "D"
    elif error <= 0.05:
        grade = "A"
    elif error <= 0.10:
        grade = "B"
    elif error <= 0.20:
        grade = "C"
    else:
        grade = "D"
    return GateGrade(
        "memory",
        grade,
        "success" if grade != "D" else "failed",
        {"expected_savings": expected, "observed_savings": observed, "abs_error": error},
    )


def grade_stress(
    *,
    completed: bool,
    crashed: bool,
    preemptions: int,
    full_admission_window: bool,
    utilization_ok: bool,
) -> GateGrade:
    metrics = {
        "completed": bool(completed),
        "crashed": bool(crashed),
        "preemptions": int(preemptions),
        "full_admission_window": bool(full_admission_window),
        "utilization_ok": bool(utilization_ok),
    }
    if not completed or crashed:
        return GateGrade("stress", "D", "failed", metrics, "Run crashed, stuck, or did not finish.")
    if preemptions == 0 and full_admission_window and utilization_ok:
        grade = "A"
    elif preemptions == 0:
        grade = "B"
    else:
        grade = "C"
    return GateGrade("stress", grade, "success", metrics)


def worst_required_grade(grades: list[GateGrade]) -> str:
    required = [grade.grade for grade in grades if grade.grade != "N/A"]
    if not required:
        return "N/A"
    return min(required, key=lambda grade: GRADE_ORDER[grade])
