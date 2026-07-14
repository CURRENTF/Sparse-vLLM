#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "scripts" / "validation" / "validate_qwen3_moe_sparse_ep.py"
ALL_METHODS = (
    "vanilla",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
)
PREFIX_METHODS = {"vanilla", "omnikv", "quest"}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _physical_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _active_compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _wait_for_idle_devices(*, timeout_s: float, poll_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while True:
        active = _active_compute_processes()
        if not active:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU devices remained busy for {timeout_s:.1f}s: {active}."
            )
        time.sleep(float(poll_s))


def _case_command(
    args: argparse.Namespace,
    *,
    method: str,
    ep_size: int,
    output_dir: Path,
    reference: Path | None,
    prefix: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(VALIDATOR),
        "--model",
        str(Path(args.model).resolve()),
        "--output-dir",
        str(output_dir),
        "--method",
        method,
        "--expert-parallel-size",
        str(ep_size),
        "--prompt-len",
        str(args.prompt_len),
        "--output-tokens",
        str(args.output_tokens),
        "--chunk-prefill-size",
        str(args.chunk_prefill_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
    ]
    if reference is not None:
        command.extend(("--reference", str(reference)))
    if prefix:
        command.append("--enable-prefix-caching")
    return command


def _run_case(
    args: argparse.Namespace,
    *,
    method: str,
    ep_size: int,
    output_dir: Path,
    reference: Path | None,
    prefix: bool,
    port: int,
) -> dict[str, Any]:
    _wait_for_idle_devices(
        timeout_s=args.idle_timeout_s,
        poll_s=args.idle_poll_s,
    )
    command = _case_command(
        args,
        method=method,
        ep_size=ep_size,
        output_dir=output_dir,
        reference=reference,
        prefix=prefix,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" if ep_size == 1 else "0,1"
    env["SPARSEVLLM_MASTER_PORT"] = str(port)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    case_name = f"{method}-{'prefix-' if prefix else ''}ep{ep_size}"
    log_path = output_dir.parent / f"{case_name}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    aggregate_path = output_dir / "aggregate_metrics.json"
    aggregate = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.is_file()
        else None
    )
    record = {
        "case_name": case_name,
        "method": method,
        "prefix_cache": prefix,
        "expert_parallel_size": ep_size,
        "status": (
            aggregate.get("status", "model_failed")
            if aggregate is not None
            else "model_failed"
        ),
        "returncode": int(result.returncode),
        "command": command,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "master_port": port,
        "reference": str(reference) if reference is not None else None,
        "output_dir": str(output_dir),
        "log": str(log_path),
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
    }
    if result.returncode != 0 or record["status"] != "success":
        record["log_tail"] = log_path.read_text(encoding="utf-8").splitlines()[-40:]
    return record


def _require_case_success(record: dict[str, Any]) -> None:
    if record["returncode"] == 0 and record["status"] == "success":
        return
    raise RuntimeError(
        f"Validation case {record['case_name']} failed with "
        f"returncode={record['returncode']}, status={record['status']}. Log tail:\n"
        + "\n".join(record.get("log_tail", []))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Qwen3MoE sparse/prefix EP1-vs-EP2 matrix."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--output-tokens", type=int, default=12)
    parser.add_argument("--chunk-prefill-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--master-port-base", type=int, default=24800)
    parser.add_argument("--idle-timeout-s", type=float, default=120.0)
    parser.add_argument("--idle-poll-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    invalid = sorted(set(methods) - set(ALL_METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}; supported={ALL_METHODS}.")
    if len(methods) != len(set(methods)):
        raise ValueError(f"--methods contains duplicates: {methods}.")
    if _physical_gpu_count() < 2:
        raise RuntimeError("The EP validation matrix requires at least two visible physical GPUs.")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {output_root}.")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    matrix_info = {
        "status": "model_failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "model": str(Path(args.model).resolve()),
        "methods": list(methods),
        "hardware_validated_ep_sizes": [1, 2],
        "unavailable_hardware_topologies": [4, 8],
        "unavailable_reason": "The requested validation host exposes two H20 GPUs.",
        "cases": records,
    }
    _write_json(output_root / "matrix_results.json", matrix_info)
    port = int(args.master_port_base)
    try:
        for prefix in (False, True):
            if prefix and args.skip_prefix:
                continue
            for method in methods:
                if prefix and method not in PREFIX_METHODS:
                    continue
                label = f"{method}-{'prefix-' if prefix else ''}ep1"
                ep1_dir = output_root / label
                print(f"[matrix] starting {label}", flush=True)
                ep1 = _run_case(
                    args,
                    method=method,
                    ep_size=1,
                    output_dir=ep1_dir,
                    reference=None,
                    prefix=prefix,
                    port=port,
                )
                port += 1
                records.append(ep1)
                _write_json(output_root / "matrix_results.json", matrix_info)
                _require_case_success(ep1)

                label = f"{method}-{'prefix-' if prefix else ''}ep2"
                ep2_dir = output_root / label
                print(f"[matrix] starting {label}", flush=True)
                ep2 = _run_case(
                    args,
                    method=method,
                    ep_size=2,
                    output_dir=ep2_dir,
                    reference=ep1_dir / "raw_outputs.pt",
                    prefix=prefix,
                    port=port,
                )
                port += 1
                records.append(ep2)
                _write_json(output_root / "matrix_results.json", matrix_info)
                _require_case_success(ep2)
    except BaseException as exc:
        matrix_info["failure"] = repr(exc)
        _write_json(output_root / "matrix_results.json", matrix_info)
        raise

    matrix_info["status"] = "success"
    matrix_info["failure"] = None
    matrix_info["num_cases"] = len(records)
    _write_json(output_root / "matrix_results.json", matrix_info)
    print(f"[matrix] completed {len(records)} cases", flush=True)


if __name__ == "__main__":
    main()
