import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from sparsevllm.kernels.triton.qwen3_5.fla.ops.l2norm import (
    fused_qk_l2norm_fwd,
    l2norm_fwd,
)


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    heads: int
    head_dim: int


QUICK_CASES = (
    Case("tp2_t32", 32, 8, 128),
    Case("tp1_t128", 128, 16, 128),
    Case("tp2_t2048", 2048, 8, 128),
)

DEFAULT_CASES = (
    Case("tp2_t1", 1, 8, 128),
    Case("tp2_t32", 32, 8, 128),
    Case("tp2_t128", 128, 8, 128),
    Case("tp2_t512", 512, 8, 128),
    Case("tp2_t2048", 2048, 8, 128),
    Case("tp1_t32", 32, 16, 128),
    Case("tp1_t128", 128, 16, 128),
    Case("tp1_t512", 512, 16, 128),
    Case("d64_t128", 128, 16, 64),
    Case("d256_t128", 128, 8, 256),
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _make_case(case: Case, device: torch.device) -> dict[str, torch.Tensor]:
    width = case.heads * case.head_dim
    packed = torch.randn(
        (case.tokens, width * 3), dtype=torch.bfloat16, device=device
    )
    q = packed[:, :width].view(1, case.tokens, case.heads, case.head_dim)
    k = packed[:, width : 2 * width].view(
        1, case.tokens, case.heads, case.head_dim
    )
    return {"packed": packed, "q": q, "k": k}


def _baseline(data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        l2norm_fwd(data["q"].contiguous()).squeeze(0),
        l2norm_fwd(data["k"].contiguous()).squeeze(0),
    )


def _candidate(data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_qk_l2norm_fwd(data["q"], data["k"])


def _time_ms(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], iterations: int
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _correctness(device: torch.device) -> dict[str, float | bool]:
    data = _make_case(Case("oracle", 37, 5, 128), device)
    q_out, k_out = _candidate(data)
    q = data["q"].squeeze(0).float()
    k = data["k"].squeeze(0).float()
    q_ref = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k_ref = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q_diff = float((q_out.float() - q_ref).abs().max().item())
    k_diff = float((k_out.float() - k_ref).abs().max().item())
    return {
        "q_max_abs_diff": q_diff,
        "k_max_abs_diff": k_diff,
        "passed": q_diff <= 5e-3 and k_diff <= 5e-3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(20260824)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "working_dir": str(Path.cwd()),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "seed": 20260824,
        "input_layout": "column views of [tokens, 3 * heads * head_dim] packed projection",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    correctness = _correctness(device)
    if not correctness["passed"]:
        raise AssertionError(f"correctness failed: {correctness}")

    cases = QUICK_CASES if args.quick else DEFAULT_CASES
    methods = {"baseline": _baseline, "candidate": _candidate}
    rows = []
    with (args.output_dir / "raw_samples.jsonl").open("w") as raw_file:
        for case in cases:
            data = _make_case(case, device)
            for fn in methods.values():
                for _ in range(args.warmup):
                    fn(data)
            torch.cuda.synchronize()
            baseline = _baseline(data)
            candidate = _candidate(data)
            max_abs_diff = max(
                float((lhs - rhs).abs().max().item())
                for lhs, rhs in zip(baseline, candidate)
            )
            samples = {name: [] for name in methods}
            for repeat in range(args.repeats):
                order = ("baseline", "candidate") if repeat % 2 == 0 else (
                    "candidate",
                    "baseline",
                )
                for name in order:
                    latency = _time_ms(
                        lambda name=name: methods[name](data), args.iterations
                    )
                    samples[name].append(latency)
                    raw_file.write(
                        json.dumps(
                            {
                                "case": case.name,
                                "implementation": name,
                                "repeat": repeat,
                                "latency_ms": latency,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    raw_file.flush()
            baseline_ms = statistics.median(samples["baseline"])
            candidate_ms = statistics.median(samples["candidate"])
            row = {
                **asdict(case),
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "max_abs_diff": max_abs_diff,
                "correct": max_abs_diff <= 5e-3,
                "baseline_samples_ms": samples["baseline"],
                "candidate_samples_ms": samples["candidate"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del data, baseline, candidate
            torch.cuda.empty_cache()

    summary = {
        "manifest": manifest,
        "correctness": correctness,
        "rows": rows,
        "all_correct": all(row["correct"] for row in rows),
        "all_candidate_faster": all(row["speedup"] > 1.0 for row in rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
