#!/usr/bin/env python3
"""Compare two GLM checkpoint parallel-validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


LOGIT_KEYS = ("prefill", "final_decode")
MATCHED_RUN_FIELDS = (
    "model_path",
    "seed",
    "prompt_len",
    "prompt_sha256",
    "max_tokens",
    "sparse_method",
    "tensor_parallel_size",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real-checkpoint GLM parallel validation artifacts."
    )
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_abs_diff", type=float, required=True)
    parser.add_argument("--mean_abs_diff", type=float, required=True)
    parser.add_argument("--p99_abs_diff", type=float, required=True)
    parser.add_argument(
        "--allow_first_token_tie",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accept a first generated-token mismatch when the candidate token "
            "is tied for the maximum in the reference prefill logits and all "
            "later generated tokens match."
        ),
    )
    parser.add_argument(
        "--tie_atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for the reference-logit tie check.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifact(run_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    summary_path = run_dir / "summary.json"
    logits_path = run_dir / "logits.pt"
    if not summary_path.is_file() or not logits_path.is_file():
        raise FileNotFoundError(
            f"Expected summary.json and logits.pt under {run_dir}."
        )
    summary = _load_json(summary_path)
    if summary.get("status") != "success":
        raise RuntimeError(
            f"Parallel validation did not succeed in {run_dir}: "
            f"status={summary.get('status')!r}."
        )
    raw_logits = torch.load(logits_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_logits, dict):
        raise TypeError(f"Expected a tensor dictionary in {logits_path}.")
    logits: dict[str, torch.Tensor] = {}
    for key in LOGIT_KEYS:
        tensor = raw_logits.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Missing tensor {key!r} in {logits_path}.")
        logits[key] = tensor.detach().float().cpu().contiguous()
    return summary, logits


def _compare_tensor(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    max_abs_limit: float,
    mean_abs_limit: float,
    p99_abs_limit: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            "Logit shape mismatch: "
            f"reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}."
        )
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Logit artifacts must contain only finite values.")
    diff = (reference - candidate).abs().flatten()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(torch.quantile(diff, 0.99).item())
    reference_argmax = reference.argmax(dim=-1).tolist()
    candidate_argmax = candidate.argmax(dim=-1).tolist()
    gates = {
        "max_abs_diff": max_abs <= float(max_abs_limit),
        "mean_abs_diff": mean_abs <= float(mean_abs_limit),
        "p99_abs_diff": p99_abs <= float(p99_abs_limit),
    }
    return {
        "shape": list(reference.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "p99_abs_diff": p99_abs,
        "reference_argmax": reference_argmax,
        "candidate_argmax": candidate_argmax,
        "argmax_match": reference_argmax == candidate_argmax,
        "limits": {
            "max_abs_diff": float(max_abs_limit),
            "mean_abs_diff": float(mean_abs_limit),
            "p99_abs_diff": float(p99_abs_limit),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _validate_generated_tokens(
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    reference_prefill_logits: torch.Tensor,
    *,
    allow_first_token_tie: bool,
    tie_atol: float,
) -> dict[str, Any]:
    reference = [int(value) for value in reference_summary["generated_token_ids"]]
    candidate = [int(value) for value in candidate_summary["generated_token_ids"]]
    exact_match = reference == candidate
    tie_accepted = False
    reference_tie_gap = None
    if (
        not exact_match
        and allow_first_token_tie
        and reference
        and len(reference) == len(candidate)
        and reference[1:] == candidate[1:]
    ):
        row = reference_prefill_logits.reshape(
            -1, reference_prefill_logits.shape[-1]
        )[-1]
        candidate_first = candidate[0]
        if not 0 <= candidate_first < int(row.numel()):
            raise ValueError(
                f"Candidate token {candidate_first} is outside the logit vocabulary."
            )
        reference_tie_gap = float((row.max() - row[candidate_first]).item())
        tie_accepted = reference_tie_gap <= float(tie_atol)
    return {
        "reference": reference,
        "candidate": candidate,
        "exact_match": exact_match,
        "first_token_tie_accepted": tie_accepted,
        "reference_tie_gap": reference_tie_gap,
        "tie_atol": float(tie_atol),
        "passed": exact_match or tie_accepted,
    }


def compare_runs(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    max_abs_limit: float,
    mean_abs_limit: float,
    p99_abs_limit: float,
    allow_first_token_tie: bool,
    tie_atol: float,
) -> dict[str, Any]:
    limits = (max_abs_limit, mean_abs_limit, p99_abs_limit, tie_atol)
    if any(float(value) < 0.0 for value in limits):
        raise ValueError(f"Comparison tolerances must be non-negative, got {limits}.")
    reference_summary, reference_logits = _load_artifact(reference_dir)
    candidate_summary, candidate_logits = _load_artifact(candidate_dir)

    mismatches = {
        field: {
            "reference": reference_summary.get(field),
            "candidate": candidate_summary.get(field),
        }
        for field in MATCHED_RUN_FIELDS
        if reference_summary.get(field) != candidate_summary.get(field)
    }
    if mismatches:
        raise ValueError(f"Parallel runs are not comparable: {mismatches}.")

    comparisons = {
        key: _compare_tensor(
            reference_logits[key],
            candidate_logits[key],
            max_abs_limit=max_abs_limit,
            mean_abs_limit=mean_abs_limit,
            p99_abs_limit=p99_abs_limit,
        )
        for key in LOGIT_KEYS
    }
    generated_tokens = _validate_generated_tokens(
        reference_summary,
        candidate_summary,
        reference_logits["prefill"],
        allow_first_token_tie=allow_first_token_tie,
        tie_atol=tie_atol,
    )
    gates = {
        **{f"{key}_logits": value["passed"] for key, value in comparisons.items()},
        "generated_tokens": generated_tokens["passed"],
    }
    return {
        "status": "success" if all(gates.values()) else "metric_failed",
        "reference": {
            "directory": str(reference_dir.resolve()),
            "tensor_parallel_size": reference_summary["tensor_parallel_size"],
            "expert_parallel_size": reference_summary["expert_parallel_size"],
            "summary_sha256": _file_sha256(reference_dir / "summary.json"),
            "logits_sha256": _file_sha256(reference_dir / "logits.pt"),
        },
        "candidate": {
            "directory": str(candidate_dir.resolve()),
            "tensor_parallel_size": candidate_summary["tensor_parallel_size"],
            "expert_parallel_size": candidate_summary["expert_parallel_size"],
            "summary_sha256": _file_sha256(candidate_dir / "summary.json"),
            "logits_sha256": _file_sha256(candidate_dir / "logits.pt"),
        },
        "logits": comparisons,
        "generated_tokens": generated_tokens,
        "gates": gates,
    }


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = compare_runs(
            Path(args.reference_dir).expanduser().resolve(),
            Path(args.candidate_dir).expanduser().resolve(),
            max_abs_limit=float(args.max_abs_diff),
            mean_abs_limit=float(args.mean_abs_diff),
            p99_abs_limit=float(args.p99_abs_diff),
            allow_first_token_tie=bool(args.allow_first_token_tie),
            tie_atol=float(args.tie_atol),
        )
    except Exception as exc:
        result = {
            "status": "invalid_input",
            "reference_dir": str(
                Path(args.reference_dir).expanduser().resolve()
            ),
            "candidate_dir": str(
                Path(args.candidate_dir).expanduser().resolve()
            ),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stream = sys.stdout if result["status"] == "success" else sys.stderr
    print(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        file=stream,
    )
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
