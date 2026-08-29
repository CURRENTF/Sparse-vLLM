#!/usr/bin/env python3
"""Run isolated multi-shape decode attention and routed-MoE benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK = Path(__file__).with_name("benchmark_decode_components.py")
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_positive_int(value: object, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive, got {parsed}")
    return parsed


def _validate_name(value: object, label: str) -> str:
    name = str(value)
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"{label} has invalid name {name!r}")
    return name


def _attention_cases(profile: dict[str, Any]) -> list[tuple[int, int]]:
    cases = []
    for context_len, batch_size in profile.get("cases", []):
        cases.append(
            (
                _require_positive_int(context_len, "context_len"),
                _require_positive_int(batch_size, "batch_size"),
            )
        )
    for matrix in profile.get("matrices", []):
        context_lengths = [
            _require_positive_int(value, "context_len")
            for value in matrix["context_lengths"]
        ]
        batch_sizes = [
            _require_positive_int(value, "batch_size")
            for value in matrix["batch_sizes"]
        ]
        cases.extend(
            (context_len, batch_size)
            for context_len in context_lengths
            for batch_size in batch_sizes
        )
    if not cases:
        raise ValueError(f"attention profile {profile['name']} has no cases")
    if len(set(cases)) != len(cases):
        raise ValueError(f"attention profile {profile['name']} has duplicate cases")
    return cases


def _timing_args(timing: dict[str, Any]) -> list[str]:
    return [
        "--warmup",
        str(_require_positive_int(timing["warmup"], "warmup")),
        "--samples",
        str(_require_positive_int(timing["samples"], "samples")),
        "--iterations",
        str(_require_positive_int(timing["iterations"], "iterations")),
        "--seed",
        str(int(timing["seed"])),
        "--graph" if timing["graph"] else "--no-graph",
    ]


def _run_child(
    *,
    command: list[str],
    case_root: Path,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    case_root.mkdir(parents=True, exist_ok=False)
    stdout_path = case_root / "stdout.log"
    stderr_path = case_root / "stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    summary_path = case_root / "summary.json"
    failure_path = case_root / "failure.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    return result.returncode, summary, failure


def _copy_raw_rows(
    *,
    source: Path,
    destination: Path,
    annotations: dict[str, object],
) -> None:
    if not source.exists():
        return
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        row.update(annotations)
        _append_jsonl(destination, row)


def _shape_args(shape: dict[str, Any]) -> list[str]:
    mapping = {
        "head_dim": "--head-dim",
        "num_query_heads": "--num-query-heads",
        "num_kv_heads": "--num-kv-heads",
        "hidden_size": "--hidden-size",
        "intermediate_size": "--intermediate-size",
        "num_experts": "--num-experts",
        "top_k": "--top-k",
    }
    result = []
    for key, flag in mapping.items():
        if key in shape:
            result.extend(
                [flag, str(_require_positive_int(shape[key], f"shape.{key}"))]
            )
    return result


def _format_ms(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _format_ratio(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Decode component multi-shape sweep",
        "",
        "FI/Tri is FlashInfer latency divided by Triton latency; below 1 "
        "favors FlashInfer.",
        "",
        "## Attention",
        "",
        "| Profile | KV length | Batch | Triton ms | FlashInfer ms | FI/Tri |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["attention_comparisons"]:
        lines.append(
            f"| {row['profile']} | {row['context_len']} | {row['batch_size']} | "
            f"{_format_ms(row['triton_ms'])} | "
            f"{_format_ms(row['flashinfer_ms'])} | "
            f"{_format_ratio(row['flashinfer_over_triton'])} |"
        )
    lines.extend(
        [
            "",
            "## Routed MoE",
            "",
            "| Profile | Batch | Triton ms | FlashInfer ms | FI/Tri |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["moe_comparisons"]:
        lines.append(
            f"| {row['profile']} | {row['batch_size']} | "
            f"{_format_ms(row['triton_ms'])} | "
            f"{_format_ms(row['flashinfer_ms'])} | "
            f"{_format_ratio(row['flashinfer_over_triton'])} |"
        )
    lines.extend(
        [
            "",
            "These are direct CUDA-Graph component callables, not TTFT, TPOT, "
            "request latency, or end-to-end throughput.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite suite root {run_root}")
    run_root.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError(f"unsupported config version {config.get('version')}")
    timing = config["timing"]
    timing_args = _timing_args(timing)
    benchmark = args.benchmark.resolve()
    if not benchmark.is_file():
        raise FileNotFoundError(f"benchmark script not found: {benchmark}")

    raw_path = run_root / "raw_samples.jsonl"
    status_path = run_root / "case_status.jsonl"
    raw_path.touch()
    status_path.touch()
    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": config,
        "benchmark": str(benchmark),
    }
    _write_json(run_root / "suite_manifest.json", manifest)

    attention_rows = []
    case_failures = []
    attention_case_index = 0
    for profile in config["attention_profiles"]:
        profile_name = _validate_name(profile["name"], "attention profile")
        shape = profile["shape"]
        for context_len, batch_size in _attention_cases(profile):
            backend_order = (
                ("triton", "flashinfer")
                if attention_case_index % 2 == 0
                else ("flashinfer", "triton")
            )
            attention_case_index += 1
            for backend in backend_order:
                case_id = f"{profile_name}_c{context_len}_b{batch_size}_{backend}"
                case_root = run_root / "cases" / "attention" / case_id
                command = [
                    sys.executable,
                    str(benchmark),
                    "--run-root",
                    str(case_root),
                    "--components",
                    "attention",
                    "--attention-backends",
                    backend,
                    "--context-lengths",
                    str(context_len),
                    "--batch-sizes",
                    str(batch_size),
                    *timing_args,
                    *_shape_args(shape),
                ]
                print(f"[attention] {case_id}", flush=True)
                return_code, child_summary, failure = _run_child(
                    command=command, case_root=case_root
                )
                status = "success" if return_code == 0 else "model_failed"
                status_row = {
                    "case_id": case_id,
                    "component": "attention",
                    "profile": profile_name,
                    "context_len": context_len,
                    "batch_size": batch_size,
                    "backend": backend,
                    "status": status,
                    "return_code": return_code,
                    "case_root": str(case_root),
                    "failure": failure,
                }
                _append_jsonl(status_path, status_row)
                _copy_raw_rows(
                    source=case_root / "raw_samples.jsonl",
                    destination=raw_path,
                    annotations={"profile": profile_name, "case_id": case_id},
                )
                if status == "success" and child_summary is not None:
                    result = child_summary["attention"][0]
                    stats = result[backend]
                    attention_rows.append(
                        {
                            **status_row,
                            "shape": shape,
                            "provenance": profile["provenance"],
                            "median_ms": stats["median_ms"],
                            "min_ms": stats["min_ms"],
                            "p90_ms": stats["p90_ms"],
                            "correctness": result["correctness"][
                                f"{backend}_vs_torch"
                            ],
                        }
                    )
                else:
                    case_failures.append(status_row)

    moe_rows = []
    flashinfer_moe_probe = None
    for profile in config["moe_profiles"]:
        profile_name = _validate_name(profile["name"], "MoE profile")
        shape = profile["shape"]
        for batch_index, batch_size_value in enumerate(profile["batch_sizes"]):
            batch_size = _require_positive_int(batch_size_value, "batch_size")
            probe = bool(profile.get("probe_flashinfer_on_first_case")) and (
                batch_index == 0
            )
            case_id = f"{profile_name}_b{batch_size}"
            case_root = run_root / "cases" / "moe" / case_id
            command = [
                sys.executable,
                str(benchmark),
                "--run-root",
                str(case_root),
                "--components",
                "moe",
                "--flashinfer-moe",
                "probe" if probe else "skip",
                "--batch-sizes",
                str(batch_size),
                *timing_args,
                *_shape_args(shape),
            ]
            print(f"[moe] {case_id}", flush=True)
            return_code, child_summary, failure = _run_child(
                command=command, case_root=case_root
            )
            status = "success" if return_code == 0 else "model_failed"
            status_row = {
                "case_id": case_id,
                "component": "moe",
                "profile": profile_name,
                "batch_size": batch_size,
                "backend": "triton",
                "status": status,
                "return_code": return_code,
                "case_root": str(case_root),
                "failure": failure,
            }
            _append_jsonl(status_path, status_row)
            _copy_raw_rows(
                source=case_root / "raw_samples.jsonl",
                destination=raw_path,
                annotations={"profile": profile_name, "case_id": case_id},
            )
            if status == "success" and child_summary is not None:
                result = child_summary["moe"][str(batch_size)]
                triton_stats = result["triton"]
                if probe:
                    flashinfer_moe_probe = child_summary["flashinfer_moe_probe"]
                moe_rows.append(
                    {
                        **status_row,
                        "shape": shape,
                        "provenance": profile["provenance"],
                        "median_ms": triton_stats["median_ms"],
                        "min_ms": triton_stats["min_ms"],
                        "p90_ms": triton_stats["p90_ms"],
                        "correctness": result["correctness"][
                            "triton_vs_torch"
                        ],
                    }
                )
            else:
                case_failures.append(status_row)

    attention_index = {
        (
            row["profile"],
            row["context_len"],
            row["batch_size"],
            row["backend"],
        ): row
        for row in attention_rows
    }
    attention_keys = sorted(
        {
            (row["profile"], row["context_len"], row["batch_size"])
            for row in attention_rows
        }
    )
    attention_comparisons = []
    for profile_name, context_len, batch_size in attention_keys:
        triton = attention_index.get(
            (profile_name, context_len, batch_size, "triton")
        )
        flashinfer = attention_index.get(
            (profile_name, context_len, batch_size, "flashinfer")
        )
        triton_ms = triton["median_ms"] if triton else None
        flashinfer_ms = flashinfer["median_ms"] if flashinfer else None
        attention_comparisons.append(
            {
                "profile": profile_name,
                "context_len": context_len,
                "batch_size": batch_size,
                "triton_ms": triton_ms,
                "flashinfer_ms": flashinfer_ms,
                "flashinfer_over_triton": (
                    flashinfer_ms / triton_ms
                    if triton_ms is not None and flashinfer_ms is not None
                    else None
                ),
            }
        )

    moe_comparisons = [
        {
            "profile": row["profile"],
            "batch_size": row["batch_size"],
            "triton_ms": row["median_ms"],
            "flashinfer_ms": None,
            "flashinfer_over_triton": None,
        }
        for row in moe_rows
    ]
    summary = {
        "status": "success" if not case_failures else "completed_with_failures",
        "attention_case_count": attention_case_index,
        "attention_provider_result_count": len(attention_rows),
        "moe_case_count": len(moe_rows) + sum(
            row["component"] == "moe" for row in case_failures
        ),
        "moe_result_count": len(moe_rows),
        "flashinfer_moe_probe": flashinfer_moe_probe,
        "attention_results": attention_rows,
        "attention_comparisons": attention_comparisons,
        "moe_results": moe_rows,
        "moe_comparisons": moe_comparisons,
        "case_failures": case_failures,
        "limitations": [
            (
                "Each provider/case runs in an isolated process; inputs are "
                "seed-matched but providers are not timed in the same CUDA context."
            ),
            (
                "The suite measures direct CUDA-Graph component callables, "
                "not serving-level latency or throughput."
            ),
            (
                "FlashInfer routed-MoE performance remains absent when its "
                "suite-level eligibility probe fails."
            ),
        ],
    }
    _write_json(run_root / "summary.json", summary)
    (run_root / "report.md").write_text(
        _build_report(summary), encoding="utf-8"
    )
    manifest["status"] = summary["status"]
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_root / "suite_manifest.json", manifest)
    print(
        f"completed attention_cases={attention_case_index} "
        f"attention_results={len(attention_rows)} moe_results={len(moe_rows)} "
        f"failures={len(case_failures)}",
        flush=True,
    )
    if case_failures:
        raise RuntimeError(
            f"{len(case_failures)} benchmark cases failed; see {status_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.run_root.exists():
            _write_json(
                args.run_root / "suite_failure.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise


if __name__ == "__main__":
    main()
