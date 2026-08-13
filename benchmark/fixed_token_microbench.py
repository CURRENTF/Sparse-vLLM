"""Reproducible fixed-token latency benchmark for Sparse-vLLM and vLLM."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sparsevllm", "vllm"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--nsys-iteration",
        type=int,
        default=-1,
        help="Wrap one timed iteration in an NVTX range for nsys capture.",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        check=False,
        cwd=Path(__file__).parents[1],
        text=True,
    )
    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _validate(args: argparse.Namespace) -> Path:
    for name in (
        "input_len",
        "output_len",
        "batch_size",
        "num_warmups",
        "num_iters",
        "tensor_parallel_size",
        "expert_parallel_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not -1 <= args.nsys_iteration < args.num_iters:
        raise ValueError("nsys_iteration must be -1 or a timed iteration index")
    if args.backend == "vllm" and args.expert_parallel_size != 1:
        raise ValueError(
            "vLLM uses --expert-parallel-size=1 or --enable-expert-parallel"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_engine(args: argparse.Namespace):
    common = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "max_model_len": args.input_len + args.output_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": False,
        "enable_prefix_caching": False,
    }
    if args.backend == "vllm":
        from vllm import LLM

        return LLM(
            **common,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=max(4096, args.batch_size * args.input_len),
            enable_expert_parallel=args.expert_parallel_size > 1,
            enable_flashinfer_autotune=False,
            language_model_only=True,
            trust_remote_code=True,
        )
    from sparsevllm import LLM

    return LLM(
        common.pop("model"),
        **common,
        expert_parallel_size=args.expert_parallel_size,
        max_num_seqs_in_batch=args.batch_size,
        max_decoding_seqs=args.batch_size,
        max_num_seqs_in_gpu=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
        decode_cuda_graph=True,
    )


def _sampling_params(args: argparse.Namespace):
    if args.backend == "vllm":
        from vllm import SamplingParams
    else:
        from sparsevllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)


def _token_ids(args: argparse.Namespace, output: Any) -> list[int]:
    if args.backend == "vllm":
        return list(output.outputs[0].token_ids)
    return list(output["token_ids"])


def main() -> int:
    args = _parser().parse_args()
    output_dir = _validate(args)
    run_info = {
        "benchmark": "fixed_token_microbench",
        "backend": args.backend,
        "command": shlex.join(sys.argv),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": {
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_warmups": args.num_warmups,
            "num_iters": args.num_iters,
            "prompt": "[2] + [100 + (request + position) % 1000]",
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "topology": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel_size": args.expert_parallel_size,
            "cuda_graph": True,
            "prefix_cache": False,
        },
        "profiler": {
            "kind": "nsys_nvtx" if args.nsys_iteration >= 0 else None,
            "iteration": args.nsys_iteration if args.nsys_iteration >= 0 else None,
        },
        "environment": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "python": sys.version,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "flashinfer_python": _package_version("flashinfer-python"),
            "triton": _package_version("triton"),
            "vllm": _package_version("vllm"),
        },
    }
    _write_json(output_dir / "run_info.json", run_info)
    prompts = [
        [
            2,
            *(
                100 + (request + position) % 1000
                for position in range(args.input_len - 1)
            ),
        ]
        for request in range(args.batch_size)
    ]
    engine = None
    raw_outputs: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    try:
        engine = _build_engine(args)
        params = _sampling_params(args)
        for _ in range(args.num_warmups):
            outputs = engine.generate(prompts, params, use_tqdm=False)
            if len(outputs) != args.batch_size:
                raise RuntimeError(f"warmup returned {len(outputs)} requests")
        for iteration in range(args.num_iters):
            profiling = iteration == args.nsys_iteration
            if profiling:
                import torch

                torch.cuda.nvtx.range_push(f"fixed_token_iteration_{iteration}")
            started = perf_counter()
            try:
                outputs = engine.generate(prompts, params, use_tqdm=False)
                elapsed = perf_counter() - started
            finally:
                if profiling:
                    torch.cuda.nvtx.range_pop()
            if len(outputs) != args.batch_size:
                raise RuntimeError(
                    f"iteration {iteration} returned {len(outputs)} requests"
                )
            generated = 0
            for sample_index, output in enumerate(outputs):
                token_ids = _token_ids(args, output)
                status = (
                    "success" if len(token_ids) == args.output_len else "model_failed"
                )
                row = {
                    "iteration": iteration,
                    "sample_index": sample_index,
                    "status": status,
                    "input_tokens": args.input_len,
                    "output_tokens": len(token_ids),
                }
                sample_results.append(row)
                raw_outputs.append({**row, "token_ids": token_ids})
                if status != "success":
                    raise RuntimeError(
                        f"iteration {iteration} sample {sample_index} produced {len(token_ids)} tokens"
                    )
                generated += len(token_ids)
            performance.append(
                {
                    "iteration": iteration,
                    "status": "success",
                    "elapsed_s": elapsed,
                    "output_tokens": generated,
                    "output_tok_s": generated / elapsed,
                }
            )
        rates = [row["output_tok_s"] for row in performance]
        aggregate = {
            "benchmark": "fixed_token_microbench",
            "backend": args.backend,
            "status": "success",
            "output_tok_s_mean": statistics.fmean(rates),
            "output_tok_s_median": statistics.median(rates),
            "samples": len(rates),
        }
    except Exception as error:
        aggregate = {
            "benchmark": "fixed_token_microbench",
            "backend": args.backend,
            "status": "model_failed",
            "error": repr(error),
        }
        raise
    finally:
        _write_jsonl(output_dir / "raw_outputs.jsonl", raw_outputs)
        _write_jsonl(output_dir / "per_sample_results.jsonl", sample_results)
        _write_jsonl(output_dir / "performance.jsonl", performance)
        _write_json(output_dir / "aggregate_metrics.json", aggregate)
        if args.backend == "sparsevllm" and engine is not None:
            engine.exit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
