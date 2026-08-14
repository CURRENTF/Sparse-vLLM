"""Reproducible vLLM latency baseline for Sparse-vLLM comparisons.

Run this script with an isolated vLLM environment. It intentionally imports
vLLM inside ``main`` so the Sparse-vLLM project environment does not need vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _parse_positive_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return values


def _env_snapshot() -> dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_ALL2ALL_BACKEND",
        "VLLM_USE_V1",
        "NCCL_DEBUG",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-token vLLM baseline compatible with microbench.py."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-sizes", type=_parse_positive_ints, default=[1, 2, 4])
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=1252)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "input_len",
        "output_len",
        "num_warmups",
        "num_iters",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        raise ValueError(
            "max_model_len must cover input_len + output_len: "
            f"{args.max_model_len} < {args.input_len + args.output_len}"
        )
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = list(args.batch_sizes)
    engine_config = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "enable_expert_parallel": bool(args.enable_expert_parallel),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": max(batch_sizes),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enable_prefix_caching": False,
        "language_model_only": True,
        "seed": 0,
        "disable_log_stats": True,
        "compilation_config": {
            "cudagraph_capture_sizes": batch_sizes,
            "max_cudagraph_capture_size": max(batch_sizes),
        },
    }
    run_info = {
        "benchmark": "vllm_microbench",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
        "engine_config": engine_config,
        "input_len": int(args.input_len),
        "output_len": int(args.output_len),
        "batch_sizes": batch_sizes,
        "num_warmups": int(args.num_warmups),
        "num_iters": int(args.num_iters),
        "prompt_token_id": int(args.prompt_token_id),
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        },
        "env": _env_snapshot(),
    }
    _write_json(output_dir / "run_info.json", run_info)

    performance_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    raw_output_rows: list[dict[str, Any]] = []
    llm = None
    try:
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams

        run_info["versions"] = {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "flashinfer_python": version("flashinfer-python"),
        }
        _write_json(output_dir / "run_info.json", run_info)

        llm = LLM(**engine_config)
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=int(args.output_len),
            detokenize=False,
        )

        for batch_size in batch_sizes:
            prompts = [
                {"prompt_token_ids": [int(args.prompt_token_id)] * args.input_len}
                for _ in range(batch_size)
            ]
            for _ in range(args.num_warmups):
                warmup_outputs = llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                if len(warmup_outputs) != batch_size:
                    raise RuntimeError(
                        f"warmup returned {len(warmup_outputs)} requests, "
                        f"expected {batch_size}"
                    )

            latencies: list[float] = []
            for iteration in range(args.num_iters):
                started = perf_counter()
                outputs = llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                latency = perf_counter() - started
                latencies.append(latency)
                if len(outputs) != batch_size:
                    raise RuntimeError(
                        f"iteration {iteration} returned {len(outputs)} requests, "
                        f"expected {batch_size}"
                    )

                for sample_index, output in enumerate(outputs):
                    if len(output.outputs) != 1:
                        raise RuntimeError(
                            f"iteration {iteration} sample {sample_index} returned "
                            f"{len(output.outputs)} sequences, expected 1"
                        )
                    token_ids = list(output.outputs[0].token_ids)
                    status = (
                        "success"
                        if len(token_ids) == args.output_len
                        else "model_failed"
                    )
                    sample_row = {
                        "batch_size": batch_size,
                        "iteration": iteration,
                        "sample_index": sample_index,
                        "status": status,
                        "input_tokens": int(args.input_len),
                        "output_tokens": len(token_ids),
                    }
                    per_sample_rows.append(sample_row)
                    raw_output_rows.append({**sample_row, "token_ids": token_ids})
                    if status != "success":
                        raise RuntimeError(
                            f"iteration {iteration} sample {sample_index} produced "
                            f"{len(token_ids)} tokens, expected {args.output_len}"
                        )

            mean_latency = statistics.fmean(latencies)
            performance_rows.append(
                {
                    "batch_size": batch_size,
                    "status": "success",
                    "latencies_s": latencies,
                    "e2e_latency_s_mean": mean_latency,
                    "e2e_latency_s_median": statistics.median(latencies),
                    "input_tok_s": batch_size * args.input_len / mean_latency,
                    "output_tok_s": batch_size * args.output_len / mean_latency,
                    "total_tok_s": (
                        batch_size * (args.input_len + args.output_len) / mean_latency
                    ),
                }
            )

        aggregate = {
            "benchmark": "vllm_microbench",
            "status": "success",
            "num_cases": len(performance_rows),
            "records": performance_rows,
        }
    except Exception as error:
        aggregate = {
            "benchmark": "vllm_microbench",
            "status": "model_failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "records": performance_rows,
        }
        _write_jsonl(output_dir / "raw_outputs.jsonl", raw_output_rows)
        _write_jsonl(output_dir / "per_sample_results.jsonl", per_sample_rows)
        _write_jsonl(output_dir / "performance.jsonl", performance_rows)
        _write_json(output_dir / "aggregate_metrics.json", aggregate)
        raise
    finally:
        if llm is not None:
            del llm

    _write_jsonl(output_dir / "raw_outputs.jsonl", raw_output_rows)
    _write_jsonl(output_dir / "per_sample_results.jsonl", per_sample_rows)
    _write_jsonl(output_dir / "performance.jsonl", performance_rows)
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
