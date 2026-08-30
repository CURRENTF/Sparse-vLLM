import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import sparsevllm.kernels.triton.context_flashattention_nopad as triton_prefill
from sparsevllm.kernels.triton.context_flashattention_nopad import (
    context_attention_fwd,
)


@dataclass(frozen=True)
class Case:
    name: str
    context: int
    query: int


CASES = (
    Case("l4k_q1", 4096, 1),
    Case("l4k_q8", 4096, 8),
    Case("l4k_q16", 4096, 16),
    Case("l4k_q32", 4096, 32),
    Case("l16k_q32", 16384, 32),
    Case("l4k_q128", 4096, 128),
    Case("l4k_q256", 4096, 256),
)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", *args],
        cwd=repo,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preheat_gpu(device: torch.device, seconds: float = 0.5) -> None:
    left = torch.randn((2048, 2048), dtype=torch.bfloat16, device=device)
    right = torch.randn_like(left)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            torch.mm(left, right)
        torch.cuda.synchronize()


def _make_case(
    case: Case,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompt = case.context - case.query
    q = torch.randn(
        case.query,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        case.context,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    return {
        "q": q,
        "k": k,
        "v": torch.randn_like(k),
        "active_slots": torch.randperm(
            case.context, dtype=torch.int64, device=device
        ).to(torch.int32).unsqueeze(0),
        "req_indices": torch.tensor([0], dtype=torch.int32, device=device),
        "b_start_loc": torch.tensor([0], dtype=torch.int32, device=device),
        "context_lens": torch.tensor(
            [case.context], dtype=torch.int32, device=device
        ),
        "prompt_cache_lens": torch.tensor(
            [prompt], dtype=torch.int32, device=device
        ),
        "output": torch.empty_like(q),
        "score": torch.full(
            (1, case.context), -torch.inf, dtype=torch.float32, device=device
        ),
    }


def _run(
    case: Case,
    data: dict[str, torch.Tensor],
    *,
    use_grouped_score: bool,
) -> None:
    data["score"].fill_(-torch.inf)
    context_attention_fwd(
        data["q"],
        data["k"],
        data["v"],
        data["output"],
        data["req_indices"],
        data["b_start_loc"],
        data["context_lens"],
        data["prompt_cache_lens"],
        case.query,
        data["active_slots"],
        attn_score=data["score"],
        max_context_len=case.context,
        _force_grouped_score=use_grouped_score,
    )


def _time_ms(function, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _oracle(
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, float]:
    case = Case("oracle", 73, 17)
    data = _make_case(case, query_heads, kv_heads, head_dim, device)
    _run(case, data, use_grouped_score=True)
    torch.cuda.synchronize()
    slots = data["active_slots"][0].long()
    prompt = case.context - case.query
    q_positions = torch.arange(prompt, case.context, device=device)
    k_positions = torch.arange(case.context, device=device)
    causal = q_positions[:, None] >= k_positions[None, :]
    expected_output = torch.empty_like(data["q"])
    expected_score = torch.full_like(data["score"], -torch.inf)
    group_size = query_heads // kv_heads
    scale = head_dim**-0.5
    for head in range(query_heads):
        keys = data["k"][slots, head // group_size].float()
        values = data["v"][slots, head // group_size].float()
        logits = torch.matmul(data["q"][:, head].float(), keys.T)
        logits = logits.masked_fill(~causal, -torch.inf)
        expected_output[:, head] = torch.matmul(
            torch.softmax(logits * scale, dim=-1), values
        ).to(data["q"].dtype)
        expected_score[0] = torch.maximum(
            expected_score[0], logits.max(dim=0).values
        )
    return {
        "output_max_abs_diff": float(
            (data["output"].float() - expected_output.float()).abs().max()
        ),
        "score_max_abs_diff": float(
            (data["score"] - expected_score).abs().max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(20260830)
    _preheat_gpu(device)
    oracle = _oracle(
        args.query_heads, args.kv_heads, args.head_dim, device
    )
    if oracle["output_max_abs_diff"] > 5e-2 or oracle["score_max_abs_diff"] > 2e-1:
        raise AssertionError(f"oracle failed: {oracle}")
    implementation_source = Path(triton_prefill.__file__).resolve()
    implementation_repo = implementation_source.parents[4]
    script_source = Path(__file__).resolve()
    sources = (implementation_source, script_source)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_output(implementation_repo, "rev-parse", "HEAD"),
        "git_status": _git_output(implementation_repo, "status", "--short"),
        "implementation_repo": str(implementation_repo),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": 20260830,
        "shape": {
            "batch": 1,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "baseline": "portable per-query-head fused attention plus score kernel",
        "candidate": "grouped GQA fused attention plus score kernel",
        "measurement": (
            "same-process matched fused prefill attention plus reduced raw-QK "
            "score latency"
        ),
        "source_sha256": {str(path): _sha256(path) for path in sources},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    rows = []
    with (args.output_dir / "raw_samples.jsonl").open("w") as raw_file:
        for case in CASES:
            data = _make_case(
                case, args.query_heads, args.kv_heads, args.head_dim, device
            )
            for _ in range(args.warmup):
                _run(case, data, use_grouped_score=False)
                _run(case, data, use_grouped_score=True)
            torch.cuda.synchronize()
            samples = {"baseline": [], "candidate": []}
            for repeat in range(args.repeats):
                order = (
                    ("baseline", "candidate")
                    if repeat % 2 == 0
                    else ("candidate", "baseline")
                )
                for implementation in order:
                    latency = _time_ms(
                        lambda implementation=implementation: _run(
                            case,
                            data,
                            use_grouped_score=implementation == "candidate",
                        ),
                        args.iterations,
                    )
                    samples[implementation].append(latency)
                    raw_file.write(
                        json.dumps(
                            {
                                "case": case.name,
                                "implementation": implementation,
                                "repeat": repeat,
                                "latency_ms": latency,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    raw_file.flush()
            baseline_median = statistics.median(samples["baseline"])
            candidate_median = statistics.median(samples["candidate"])
            row = {
                **asdict(case),
                "samples_ms": samples,
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "speedup": baseline_median / candidate_median,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "oracle": oracle,
                "case_count": len(rows),
                "cases_at_least_1_5x": sum(
                    row["speedup"] >= 1.5 for row in rows
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "SUCCESS").touch()


if __name__ == "__main__":
    main()
