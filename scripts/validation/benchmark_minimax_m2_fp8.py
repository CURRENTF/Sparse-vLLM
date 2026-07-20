#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISION = "8e095dcb5d87d55e261ea10fef7fc5f4a596f9a8"
SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}


def _parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}.")
    return values


def _parse_str_csv(value: str) -> list[str]:
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected a non-empty comma-separated list, got {value!r}.")
    return values


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"]),
        "branch": _command_output(["git", "branch", "--show-current"]),
        "dirty": bool(_command_output(["git", "status", "--porcelain"])),
    }


def _query_gpus() -> list[dict[str, Any]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        index, name, memory_used, utilization = [
            part.strip() for part in line.split(",")
        ]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mib": int(memory_used),
                "utilization_percent": int(utilization),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows.")
    return rows


def _select_idle_gpu(
    requested_index: int | None,
    *,
    max_memory_used_mib: int,
    max_utilization_percent: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _query_gpus()
    idle = [
        row
        for row in rows
        if row["memory_used_mib"] <= max_memory_used_mib
        and row["utilization_percent"] <= max_utilization_percent
    ]
    if requested_index is not None:
        matches = [row for row in rows if row["index"] == requested_index]
        if not matches:
            raise ValueError(f"GPU index {requested_index} does not exist.")
        selected = matches[0]
        if selected not in idle:
            raise RuntimeError(
                f"Requested GPU {requested_index} is busy: {selected}; all devices={rows}."
            )
        return selected, rows
    if not idle:
        raise RuntimeError(f"All GPUs are busy; refusing to start benchmark: {rows}.")
    return idle[0], rows


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty list.")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _tensor_summary(tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "abs_mean": float(values.abs().mean()),
        "finite": bool(values.isfinite().all()),
    }


def _aggregate_status(records: list[dict[str, Any]]) -> str:
    statuses = [record["status"] for record in records]
    if statuses and set(statuses) == {"success"}:
        return "success"
    for status in (
        "model_failed",
        "invalid_input",
        "parse_failed",
        "metric_failed",
        "skipped_by_policy",
    ):
        if status in statuses:
            return status
    return "model_failed"


def _parallel_context(local_experts: int):
    from sparsevllm.distributed import ParallelContext, ParallelGroup

    ep_size = 256 // int(local_experts)
    ranks = tuple(range(ep_size))
    return ParallelContext(
        world=ParallelGroup(None, ranks, 0, ep_size),
        tensor=ParallelGroup(None, (0,), 0, 1),
        expert=ParallelGroup(None, ranks, 0, ep_size),
        data=ParallelGroup(None, (0,), 0, 1),
    )


def _expert_config(backend: str):
    from sparsevllm.config import QuantizationConfig

    return SimpleNamespace(
        hidden_size=3072,
        intermediate_size=1536,
        num_local_experts=256,
        num_experts_per_tok=8,
        moe_backend=backend,
        quantization_config=QuantizationConfig(
            enabled=True,
            quant_method="fp8",
            weight_dtype="e4m3",
            activation_scheme="dynamic",
            weight_block_size=(128, 128),
            backend="auto",
            model_name="MiniMax M2.7",
        ),
    )


def _make_experts(backend: str, local_experts: int, source: dict[str, Any]):
    from sparsevllm.models.minimax_m2 import MiniMaxM2PackedExperts

    with patch(
        "sparsevllm.models.minimax_m2.get_parallel_context",
        return_value=_parallel_context(local_experts),
    ):
        experts = MiniMaxM2PackedExperts(_expert_config(backend)).cuda()
    experts.w13_weight.data.copy_(source["w13_weight"])
    experts.w13_scale_inv.copy_(source["w13_scale_inv"])
    experts.w2_weight.data.copy_(source["w2_weight"])
    experts.w2_scale_inv.copy_(source["w2_scale_inv"])
    return experts


def _random_case(torch, *, token_count: int, local_experts: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    hidden_states = torch.randn(
        token_count,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    w13_weight = (
        torch.randn(
            local_experts,
            3072,
            3072,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).to(torch.float8_e4m3fn)
    w2_weight = (
        torch.randn(
            local_experts,
            3072,
            1536,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).to(torch.float8_e4m3fn)
    w13_scale_inv = torch.rand(
        local_experts,
        24,
        24,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    w13_scale_inv.mul_(0.02).add_(0.99)
    w2_scale_inv = torch.rand(
        local_experts,
        24,
        12,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    w2_scale_inv.mul_(0.02).add_(0.99)
    topk_ids = torch.randint(
        0,
        256,
        (token_count, 8),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    topk_weights = torch.rand(
        token_count,
        8,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return {
        "hidden_states": hidden_states,
        "w13_weight": w13_weight,
        "w13_scale_inv": w13_scale_inv,
        "w2_weight": w2_weight,
        "w2_scale_inv": w2_scale_inv,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }


def _forward(experts, case):
    return experts(
        case["hidden_states"],
        case["topk_ids"],
        case["topk_weights"],
    )


def _kernel_launch_count(torch, forward, trace_path: Path | None) -> int:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        forward()
        torch.cuda.synchronize()
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(trace_path))
    return sum(
        1
        for event in profile.events()
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    )


def _measure(
    torch,
    forward,
    *,
    warmup: int,
    iterations: int,
    trace_path: Path | None,
) -> dict[str, Any]:
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward()
        end.record()
        end.synchronize()
        timings_ms.append(float(start.elapsed_time(end)))
    return {
        "timings_ms": timings_ms,
        "median_ms": statistics.median(timings_ms),
        "p95_ms": _percentile(timings_ms, 0.95),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "kernel_launches": _kernel_launch_count(torch, forward, trace_path),
        "profiler_trace": str(trace_path) if trace_path is not None else None,
    }


def _correctness(torch, actual, expected) -> dict[str, Any]:
    error = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float())
    if float(denominator) == 0.0:
        raise RuntimeError("Reference output has zero norm; relative error is undefined.")
    relative_l2 = torch.linalg.vector_norm(error) / denominator
    return {
        "max_abs_error": float(error.abs().max()),
        "mean_abs_error": float(error.abs().mean()),
        "relative_l2_error": float(relative_l2),
    }


def _software_versions(torch) -> dict[str, Any]:
    import kernels
    import transformers
    import triton

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "kernels": getattr(kernels, "__version__", "unknown"),
        "nvidia_smi": _command_output(["nvidia-smi"]),
    }


def _write_report(
    path: Path,
    *,
    aggregate_status: str,
    run_config: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    lines = [
        "# MiniMax M2.7 FP8 component benchmark",
        "",
        f"- status: `{aggregate_status}`",
        f"- model revision: `{run_config['model_revision']}`",
        f"- git commit: `{run_config['git']['commit']}`",
        f"- seed: `{run_config['seed']}`",
        "",
        "| case | backend | status | cold total (ms) | median (ms) | p95 (ms) "
        "| peak memory (GiB) | launches | relative L2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        if "backend" not in record:
            continue
        correctness = record.get("correctness_vs_reference", {})
        relative_l2 = correctness.get("relative_l2_error")
        lines.append(
            "| {case_id} | {backend} | {status} | {cold:.3f} | {median:.3f} | "
            "{p95:.3f} | {memory:.3f} | {launches} | {relative_l2} |".format(
                case_id=record["case_id"],
                backend=record["backend"],
                status=record["status"],
                cold=record["cold_start_ms"],
                median=record["median_ms"],
                p95=record["p95_ms"],
                memory=record["peak_memory_bytes"] / 1024**3,
                launches=record["kernel_launches"],
                relative_l2=(
                    "-" if relative_l2 is None else f"{relative_l2:.6e}"
                ),
            )
        )
    failures = [record for record in records if record["status"] != "success"]
    if failures:
        lines.extend(("", "## Failures", ""))
        for failure in failures:
            lines.append(
                f"- `{failure.get('case_id', 'unknown')}`: "
                f"`{failure['status']}` — `{failure.get('error', 'metric gate failed')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MiniMax M2.7 local FP8 experts on an idle AutoDL GPU."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--token-counts", default="1,2,4,8,16,32,128,512,2048,8192")
    parser.add_argument("--local-experts", default="32,64")
    parser.add_argument("--backends", default="reference,native,routed")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("--max-memory-used-mib", type=int, default=512)
    parser.add_argument("--max-utilization-percent", type=int, default=5)
    parser.add_argument(
        "--trace-case",
        default="tokens_32_local_experts_32/routed",
        help=(
            "Export a profiler trace for this case/backend key; use an empty "
            "value to disable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    parsed_outputs: dict[str, Any] = {}
    raw_outputs: dict[str, Any] = {}
    run_config: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "git": _git_metadata(),
        "model": "MiniMaxAI/MiniMax-M2.7",
        "model_revision": MODEL_REVISION,
        "seed": args.seed,
        "token_counts": _parse_int_csv(args.token_counts),
        "local_experts": _parse_int_csv(args.local_experts),
        "backends": _parse_str_csv(args.backends),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "max_relative_l2": args.max_relative_l2,
        "trace_case": args.trace_case.strip(),
        "shape": {
            "num_experts": 256,
            "top_k": 8,
            "hidden_size": 3072,
            "intermediate_size": 1536,
            "weight_dtype": "torch.float8_e4m3fn",
            "scale_dtype": "torch.float32",
            "activation_dtype": "torch.bfloat16",
            "weight_block_size": [128, 128],
        },
        "input_distribution": {
            "hidden_states": "normal(mean=0,std=1), BF16",
            "weights": "normal(mean=0,std=0.02), cast to FP8 E4M3",
            "weight_scale_inv": "uniform[0.99,1.01), FP32",
            "topk_ids": "uniform integers in [0,256), with replacement",
            "topk_weights": "uniform[0,1), normalized per token",
        },
    }
    exit_code = 0
    torch = None
    try:
        if args.warmup < 0 or args.iterations <= 0:
            raise ValueError("warmup must be >= 0 and iterations must be > 0.")
        supported_backends = {"reference", "native", "routed"}
        unknown = sorted(set(run_config["backends"]) - supported_backends)
        if unknown:
            raise ValueError(f"Unsupported backends: {unknown}.")
        if "reference" not in run_config["backends"]:
            raise ValueError("The reference backend is required for correctness gating.")
        if not 0.0 < args.max_relative_l2 < 1.0:
            raise ValueError("max-relative-l2 must be between 0 and 1.")
        requested_backends = set(run_config["backends"])
        run_config["backends"] = [
            backend
            for backend in ("reference", "native", "routed")
            if backend in requested_backends
        ]
        for local_experts in run_config["local_experts"]:
            if local_experts not in {32, 64}:
                raise ValueError(
                    "MiniMax milestone benchmark supports local_experts 32/64, "
                    f"got {local_experts}."
                )
        scheduled_keys = {
            f"tokens_{token_count}_local_experts_{local_experts}/{backend}"
            for token_count in run_config["token_counts"]
            for local_experts in run_config["local_experts"]
            for backend in run_config["backends"]
        }
        if run_config["trace_case"] and run_config["trace_case"] not in scheduled_keys:
            raise ValueError(
                f"trace-case {run_config['trace_case']!r} is not scheduled; "
                f"available keys include {sorted(scheduled_keys)[:8]}."
            )

        selected_gpu, all_gpus = _select_idle_gpu(
            args.gpu_index,
            max_memory_used_mib=args.max_memory_used_mib,
            max_utilization_percent=args.max_utilization_percent,
        )
        run_config["gpu_preflight"] = {
            "selected": selected_gpu,
            "all_devices": all_gpus,
            "max_memory_used_mib": args.max_memory_used_mib,
            "max_utilization_percent": args.max_utilization_percent,
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "src"))
        import torch as torch_module

        torch = torch_module
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Benchmark must expose exactly the selected CUDA device after preflight."
            )
        from sparsevllm.quantization.fp8 import (
            FINEGRAINED_FP8_KERNEL_REPO,
            FINEGRAINED_FP8_KERNEL_REVISION,
            FINEGRAINED_FP8_KERNEL_SOURCE_SHA256,
            FINEGRAINED_FP8_KERNEL_VERSION,
        )

        run_config["fp8_kernel"] = {
            "repo": FINEGRAINED_FP8_KERNEL_REPO,
            "version": FINEGRAINED_FP8_KERNEL_VERSION,
            "revision": FINEGRAINED_FP8_KERNEL_REVISION,
            "source_sha256": FINEGRAINED_FP8_KERNEL_SOURCE_SHA256,
            "local_override": os.getenv("SPARSEVLLM_FINEGRAINED_FP8_KERNEL_PATH"),
        }
        run_config["software"] = _software_versions(torch)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        for local_experts in run_config["local_experts"]:
            for token_count in run_config["token_counts"]:
                case_id = f"tokens_{token_count}_local_experts_{local_experts}"
                case = _random_case(
                    torch,
                    token_count=token_count,
                    local_experts=local_experts,
                    seed=args.seed + token_count + local_experts,
                )
                reference_output = None
                for backend in run_config["backends"]:
                    implementation_backend = {
                        "reference": "pytorch",
                        "native": "native",
                        "routed": "triton",
                    }[backend]
                    torch.cuda.synchronize()
                    construction_started = time.perf_counter()
                    experts = _make_experts(
                        implementation_backend,
                        local_experts,
                        case,
                    )
                    torch.cuda.synchronize()
                    construction_ms = (
                        time.perf_counter() - construction_started
                    ) * 1000.0
                    forward = lambda experts=experts, case=case: _forward(experts, case)
                    cold_forward_started = time.perf_counter()
                    output = forward()
                    torch.cuda.synchronize()
                    cold_forward_ms = (
                        time.perf_counter() - cold_forward_started
                    ) * 1000.0
                    if backend == "reference":
                        reference_output = output.detach().clone()
                    trace_path = None
                    benchmark_key = f"{case_id}/{backend}"
                    if benchmark_key == run_config["trace_case"]:
                        trace_path = args.output_dir / "profiler" / "trace.json"
                    metrics = _measure(
                        torch,
                        forward,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        trace_path=trace_path,
                    )
                    record = {
                        "case_id": case_id,
                        "status": "success",
                        "backend": backend,
                        "token_count": token_count,
                        "local_experts": local_experts,
                        "ep_size": 256 // local_experts,
                        "construction_ms": construction_ms,
                        "cold_forward_ms": cold_forward_ms,
                        "cold_start_ms": construction_ms + cold_forward_ms,
                        **metrics,
                    }
                    if reference_output is not None and backend != "reference":
                        record["correctness_vs_reference"] = _correctness(
                            torch,
                            output,
                            reference_output,
                        )
                        relative_l2 = record["correctness_vs_reference"][
                            "relative_l2_error"
                        ]
                        if (
                            not math.isfinite(relative_l2)
                            or relative_l2 > args.max_relative_l2
                        ):
                            record["status"] = "metric_failed"
                    records.append(record)
                    parsed_outputs[f"{case_id}/{backend}"] = _tensor_summary(output)
                    raw_outputs[f"{case_id}/{backend}"] = output[:2].detach().cpu()
                    del experts, output
                    torch.cuda.empty_cache()
                if reference_output is not None:
                    del reference_output
                del case
                torch.cuda.empty_cache()
        if any(record["status"] == "metric_failed" for record in records):
            exit_code = 1
    except (ValueError, FileNotFoundError) as exc:
        exit_code = 2
        records.append(
            {
                "case_id": "preflight_or_input",
                "status": "invalid_input",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:
        exit_code = 1
        records.append(
            {
                "case_id": "runtime",
                "status": "model_failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        statuses = [record["status"] for record in records]
        if any(status not in SAMPLE_STATUSES for status in statuses):
            raise RuntimeError(f"Invalid result statuses: {statuses}.")
        aggregate_status = _aggregate_status(records)
        run_config["completed_at"] = datetime.now().astimezone().isoformat()
        _write_json(args.output_dir / "run_config.json", run_config)
        _write_json(args.output_dir / "parsed_outputs.json", parsed_outputs)
        _write_json(args.output_dir / "per_case_results.json", records)
        _write_json(
            args.output_dir / "aggregate_metrics.json",
            {
                "status": aggregate_status,
                "num_cases": len(records),
                "success_cases": sum(record["status"] == "success" for record in records),
                "failed_cases": sum(record["status"] != "success" for record in records),
                "records": records,
            },
        )
        _write_report(
            args.output_dir / "report.md",
            aggregate_status=aggregate_status,
            run_config=run_config,
            records=records,
        )
        if torch is None:
            import torch as torch_for_artifacts

            torch_for_artifacts.save(raw_outputs, args.output_dir / "raw_outputs.pt")
        else:
            torch.save(raw_outputs, args.output_dir / "raw_outputs.pt")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
