import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton

from sparsevllm.kernels.triton.context_flashattention_nopad import (
    context_attention_fwd,
)
from sparsevllm.kernels.triton.prefill_score import (
    PrefillScoreWorkspace,
    prefill_score_from_lse_fwd,
    prefill_score_fwd,
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={Path.cwd()}", *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_case(
    length: int,
    query_length: int,
    device: torch.device,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, object]:
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError(
            "query heads must be divisible by KV heads: "
            f"query_heads={query_heads} kv_heads={kv_heads}"
        )
    if not 0 < query_length <= length:
        raise ValueError(
            "query length must be within the physical context: "
            f"query={query_length} context={length}"
        )
    prompt_cache_len = length - query_length
    q = torch.randn(
        query_length, query_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(length, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    metadata = {
        "req_indices": torch.tensor([0], dtype=torch.int32, device=device),
        "q_starts": torch.tensor([0], dtype=torch.int32, device=device),
        "context_lens": torch.tensor([length], dtype=torch.int32, device=device),
        "prompt_lens": torch.tensor(
            [prompt_cache_len], dtype=torch.int32, device=device
        ),
        "page_table": torch.arange(
            length, dtype=torch.int32, device=device
        ).unsqueeze(0),
        "score_starts": torch.tensor(
            [prompt_cache_len], dtype=torch.int32, device=device
        ),
        "score_ends": torch.tensor([length], dtype=torch.int32, device=device),
    }
    return {
        "q": q,
        "k": k,
        "v": v,
        **metadata,
        "baseline_output": torch.empty_like(q),
        "candidate_output": torch.empty_like(q),
        "baseline_score": torch.empty(
            1, length, dtype=torch.float32, device=device
        ),
        "candidate_score": torch.empty(
            1, length, dtype=torch.float32, device=device
        ),
        "softmax_lse": torch.empty(
            query_heads, query_length, dtype=torch.float32, device=device
        ),
        "baseline_workspace": PrefillScoreWorkspace(),
        "candidate_workspace": PrefillScoreWorkspace(),
        "query_length": query_length,
    }


@torch.inference_mode()
def _run_baseline_score(case: dict[str, object]) -> None:
    prefill_score_fwd(
        case["q"],
        case["k"],
        case["baseline_score"],
        case["req_indices"],
        case["q_starts"],
        case["context_lens"],
        case["prompt_lens"],
        case["query_length"],
        case["page_table"],
        case["score_starts"],
        case["score_ends"],
        candidate_start=0,
        recent_keep_tokens=0,
        score_mode="probability",
        workspace=case["baseline_workspace"],
    )


@torch.inference_mode()
def _run_candidate_score(case: dict[str, object]) -> None:
    prefill_score_from_lse_fwd(
        case["q"],
        case["k"],
        case["softmax_lse"],
        case["candidate_score"],
        case["req_indices"],
        case["q_starts"],
        case["context_lens"],
        case["prompt_lens"],
        case["query_length"],
        case["page_table"],
        case["score_starts"],
        case["score_ends"],
        workspace=case["candidate_workspace"],
        _block_m=case["candidate_block_m"],
        _block_n=case["candidate_block_n"],
        _num_warps=case["candidate_num_warps"],
        _num_stages=case["candidate_num_stages"],
    )


@torch.inference_mode()
def _run_baseline(case: dict[str, object]) -> None:
    context_attention_fwd(
        case["q"],
        case["k"],
        case["v"],
        case["baseline_output"],
        case["req_indices"],
        case["q_starts"],
        case["context_lens"],
        case["prompt_lens"],
        case["query_length"],
        case["page_table"],
    )
    _run_baseline_score(case)


@torch.inference_mode()
def _run_candidate(case: dict[str, object]) -> None:
    context_attention_fwd(
        case["q"],
        case["k"],
        case["v"],
        case["candidate_output"],
        case["req_indices"],
        case["q_starts"],
        case["context_lens"],
        case["prompt_lens"],
        case["query_length"],
        case["page_table"],
        softmax_lse=case["softmax_lse"],
    )
    _run_candidate_score(case)


def _time_ms(function, case: dict[str, object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function(case)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=(4096, 16384))
    parser.add_argument(
        "--query-lengths",
        nargs="+",
        type=int,
        help="Query chunk length for each physical context length.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=32,
        help="Compatibility shorthand used when --query-lengths is omitted.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--candidate-block-m", type=int)
    parser.add_argument("--candidate-block-n", type=int)
    parser.add_argument("--candidate-num-warps", type=int)
    parser.add_argument("--candidate-num-stages", type=int)
    parser.add_argument("--query-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    query_lengths = (
        [int(args.window)] * len(args.lengths)
        if args.query_lengths is None
        else [int(value) for value in args.query_lengths]
    )
    if len(query_lengths) != len(args.lengths):
        raise ValueError(
            "--query-lengths must contain one value per --lengths entry: "
            f"contexts={args.lengths} queries={query_lengths}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(20260830)
    sources = (
        Path("src/sparsevllm/kernels/triton/context_flashattention_nopad.py"),
        Path("src/sparsevllm/kernels/triton/prefill_score.py"),
        Path("scripts/profiling/bench_prefill_attention_probability_pipeline.py"),
    )
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "command": [sys.executable, *sys.argv],
        "seed": 20260830,
        "shape": (
            f"B1 Hq{args.query_heads} Hkv{args.kv_heads} "
            f"D{args.head_dim} BF16"
        ),
        "lengths": args.lengths,
        "query_lengths": query_lengths,
        "h2o_prefill_score_window": 0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "baseline": "Triton attention without LSE plus full probability scorer",
        "candidate": "Triton attention with LSE store plus from-LSE scorer",
        "candidate_launch_override": {
            "block_m": args.candidate_block_m,
            "block_n": args.candidate_block_n,
            "num_warps": args.candidate_num_warps,
            "num_stages": args.candidate_num_stages,
        },
        "measurement": "matched attention-plus-score CUDA latency; allocation excluded",
        "graph_mode": "eager",
        "topology": "TP1",
        "source_sha256": {str(path): _sha256(path) for path in sources},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    rows = []
    with (args.output_dir / "raw_samples.jsonl").open("w") as raw_file:
        for length, query_length in zip(args.lengths, query_lengths):
            case = _make_case(
                length,
                query_length,
                device,
                query_heads=int(args.query_heads),
                kv_heads=int(args.kv_heads),
                head_dim=int(args.head_dim),
            )
            case.update(
                candidate_block_m=args.candidate_block_m,
                candidate_block_n=args.candidate_block_n,
                candidate_num_warps=args.candidate_num_warps,
                candidate_num_stages=args.candidate_num_stages,
            )
            for _ in range(args.warmup):
                _run_baseline(case)
                _run_candidate(case)
            torch.cuda.synchronize()
            output_max_abs = float(
                (
                    case["baseline_output"].float()
                    - case["candidate_output"].float()
                )
                .abs()
                .max()
                .item()
            )
            score_max_abs = float(
                (case["baseline_score"] - case["candidate_score"])
                .abs()
                .max()
                .item()
            )
            if output_max_abs > 2e-2 or score_max_abs > 2e-2:
                raise AssertionError(
                    f"pipeline oracle failed at L={length}: "
                    f"output={output_max_abs} score={score_max_abs}"
                )
            samples = {"baseline": [], "candidate": []}
            score_samples = {"baseline": [], "candidate": []}
            for repeat in range(args.repeats):
                order = (
                    (("baseline", _run_baseline), ("candidate", _run_candidate))
                    if repeat % 2 == 0
                    else (("candidate", _run_candidate), ("baseline", _run_baseline))
                )
                for name, function in order:
                    latency_ms = _time_ms(function, case, args.iterations)
                    samples[name].append(latency_ms)
                    raw_file.write(
                        json.dumps(
                            {
                                "length": length,
                                "query_length": query_length,
                                "repeat": repeat,
                                "implementation": name,
                                "latency_ms": latency_ms,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    raw_file.flush()
                score_order = (
                    (
                        ("baseline", _run_baseline_score),
                        ("candidate", _run_candidate_score),
                    )
                    if repeat % 2 == 0
                    else (
                        ("candidate", _run_candidate_score),
                        ("baseline", _run_baseline_score),
                    )
                )
                for name, function in score_order:
                    latency_ms = _time_ms(function, case, args.iterations)
                    score_samples[name].append(latency_ms)
                    raw_file.write(
                        json.dumps(
                            {
                                "length": length,
                                "query_length": query_length,
                                "repeat": repeat,
                                "implementation": name,
                                "component": "score_only",
                                "latency_ms": latency_ms,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    raw_file.flush()
            baseline_median = statistics.median(samples["baseline"])
            candidate_median = statistics.median(samples["candidate"])
            baseline_score_median = statistics.median(score_samples["baseline"])
            candidate_score_median = statistics.median(score_samples["candidate"])
            row = {
                "length": length,
                "query_length": query_length,
                "output_max_abs": output_max_abs,
                "score_max_abs": score_max_abs,
                "samples_ms": samples,
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "speedup": baseline_median / candidate_median,
                "score_samples_ms": score_samples,
                "baseline_score_median_ms": baseline_score_median,
                "candidate_score_median_ms": candidate_score_median,
                "score_component_speedup": (
                    baseline_score_median / candidate_score_median
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "rows": rows}, indent=2, sort_keys=True)
        + "\n"
    )
    (args.output_dir / "SUCCESS").touch()


if __name__ == "__main__":
    main()
