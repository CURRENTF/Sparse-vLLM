"""Benchmark eager decode attention plus production token-score normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sparsevllm.kernels.triton.decode_score import decode_softmax_token_scores
from sparsevllm.kernels.triton.flash_decoding_stage1 import (
    flash_decode_stage1_with_score as mha_stage1_with_score,
)
from sparsevllm.kernels.triton.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.kernels.triton.gqa_flash_decoding_stage1 import (
    flash_decode_stage1_with_score as gqa_stage1_with_score,
)


@dataclass(frozen=True)
class Case:
    batch: int
    context: int

    @property
    def name(self) -> str:
        return f"b{self.batch}_l{self.context}"


CASES = (
    Case(1, 1024),
    Case(1, 4096),
    Case(1, 16384),
    Case(8, 4096),
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


def _shape(family: str) -> tuple[int, int, int]:
    return (28, 4, 128) if family == "gqa" else (32, 32, 128)


def _preheat_gpu(device: torch.device, seconds: float = 0.5) -> None:
    left = torch.randn((2048, 2048), dtype=torch.bfloat16, device=device)
    right = torch.randn_like(left)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            torch.mm(left, right)
        torch.cuda.synchronize(device)


def _make_case(family: str, case: Case, device: torch.device) -> dict[str, object]:
    query_heads, kv_heads, head_dim = _shape(family)
    tokens = case.batch * case.context
    splits = (case.context + 255) // 256
    q = torch.randn(
        case.batch,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        tokens,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    states = {}
    for implementation in ("baseline", "candidate"):
        states[implementation] = {
            "mid_o": torch.empty(
                case.batch,
                query_heads,
                splits,
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
            "mid_lse": torch.empty(
                case.batch,
                query_heads,
                splits,
                dtype=torch.float32,
                device=device,
            ),
            "attention_output": torch.empty_like(q),
            "raw_score": torch.empty(
                case.batch,
                query_heads,
                case.context,
                dtype=torch.float32,
                device=device,
            ),
            "score_lse": torch.empty(
                case.batch,
                query_heads,
                dtype=torch.float32,
                device=device,
            ),
            "normalized_score": torch.empty(
                case.batch,
                case.context,
                dtype=torch.bfloat16,
                device=device,
            ),
        }
    candidate_lens = torch.full(
        (case.batch,),
        case.context - 4 - 32,
        dtype=torch.int32,
        device=device,
    )
    if case.batch > 1:
        candidate_lens[-1] -= 17
    return {
        "q": q,
        "k": k,
        "v": v,
        "active_slots": torch.arange(
            tokens, dtype=torch.int32, device=device
        ).view(case.batch, case.context),
        "req_indices": torch.arange(
            case.batch, dtype=torch.int32, device=device
        ),
        "context_lens": torch.full(
            (case.batch,), case.context, dtype=torch.int32, device=device
        ),
        "candidate_lens": candidate_lens,
        "states": states,
    }


def _attention(
    family: str,
    case: Case,
    data: dict[str, object],
    implementation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = data["states"][implementation]
    stage1 = gqa_stage1_with_score if family == "gqa" else mha_stage1_with_score
    stage1(
        data["q"],
        data["k"],
        data["v"],
        data["active_slots"],
        data["req_indices"],
        data["context_lens"],
        case.context,
        state["mid_o"],
        state["mid_lse"],
        state["raw_score"],
        256,
    )
    flash_decode_stage2(
        state["mid_o"],
        state["mid_lse"],
        data["context_lens"],
        state["attention_output"],
        256,
    )
    return state["attention_output"], state["raw_score"]


def _reference_score(
    raw_score: torch.Tensor,
    candidate_lens: torch.Tensor,
) -> torch.Tensor:
    candidate = raw_score[:, :, 4:]
    lengths = candidate_lens.long().clamp(min=0, max=candidate.shape[-1])
    positions = torch.arange(candidate.shape[-1], device=raw_score.device)
    mask = positions[None, :] < lengths[:, None]
    logits = (candidate * (128**-0.5)).masked_fill(
        ~mask[:, None, :], torch.finfo(torch.float32).min
    )
    reduced = torch.softmax(logits, dim=-1).amax(dim=1).to(torch.bfloat16)
    reduced.masked_fill_(~mask, torch.finfo(torch.bfloat16).min)
    output = torch.full(
        (raw_score.shape[0], raw_score.shape[2]),
        torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16,
        device=raw_score.device,
    )
    output[:, 4:] = reduced
    return output


def _run_baseline(
    family: str,
    case: Case,
    data: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_output, raw_score = _attention(family, case, data, "baseline")
    return attention_output, _reference_score(raw_score, data["candidate_lens"])


def _run_candidate(
    family: str,
    case: Case,
    data: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_output, raw_score = _attention(family, case, data, "candidate")
    state = data["states"]["candidate"]
    score = decode_softmax_token_scores(
        raw_score,
        data["candidate_lens"],
        candidate_start=4,
        softmax_scale=128**-0.5,
        output_dtype=torch.bfloat16,
        lse_workspace=state["score_lse"],
        output=state["normalized_score"],
    )
    return attention_output, score


def _time_ms(function, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _oracle(family: str, device: torch.device) -> dict[str, float]:
    case = Case(2, 73)
    data = _make_case(family, case, device)
    query_heads, kv_heads, head_dim = _shape(family)
    group_size = query_heads // kv_heads
    expected_attention = torch.empty_like(data["q"])
    expected_heads = torch.empty(
        case.batch,
        query_heads,
        case.context,
        dtype=torch.float32,
        device=device,
    )
    scale = head_dim**-0.5
    for batch in range(case.batch):
        token_slice = slice(batch * case.context, (batch + 1) * case.context)
        for head in range(query_heads):
            kv_head = head // group_size
            logits = torch.matmul(
                data["k"][token_slice, kv_head].float(),
                data["q"][batch, head].float(),
            )
            expected_heads[batch, head] = logits
            expected_attention[batch, head] = torch.matmul(
                torch.softmax(logits * scale, dim=0),
                data["v"][token_slice, kv_head].float(),
            ).to(torch.bfloat16)
    expected_score = _reference_score(expected_heads, data["candidate_lens"])
    attention, score = _run_candidate(family, case, data)
    torch.cuda.synchronize(device)
    return {
        "attention_max_abs_diff": float(
            (attention.float() - expected_attention.float()).abs().max()
        ),
        "score_max_abs_diff": float(
            (score.float() - expected_score.float()).abs().max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--families", nargs="+", choices=("gqa", "mha"), default=("gqa", "mha")
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(20260830)
    _preheat_gpu(device)

    script_source = Path(__file__).resolve()
    repo = script_source.parents[2]
    sources = (
        repo / "src/sparsevllm/kernels/triton/gqa_flash_decoding_stage1.py",
        repo / "src/sparsevllm/kernels/triton/flash_decoding_stage1.py",
        repo / "src/sparsevllm/kernels/triton/flash_decoding_stage2.py",
        repo / "src/sparsevllm/kernels/triton/decode_score.py",
        script_source,
    )
    oracle = {family: _oracle(family, device) for family in args.families}
    for family, error in oracle.items():
        if error["attention_max_abs_diff"] > 5e-2 or error["score_max_abs_diff"] > 2e-3:
            raise AssertionError(f"{family} oracle failed: {error}")
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_output(repo, "rev-parse", "HEAD"),
        "git_status": _git_output(repo, "status", "--short"),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": 20260830,
        "families": args.families,
        "cases": [asdict(case) for case in CASES],
        "shape": "D128 BF16, GQA=Hq28/Hkv4, MHA=Hq32/Hkv32",
        "candidate_range": "[4, context-32), with final batch row shortened by 17",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "baseline": "unchanged eager attention score plus production Torch normalization",
        "candidate": "unchanged eager attention score plus source-level Triton normalization",
        "measurement": "matched attention stage1, stage2, and normalization latency",
        "oracle": oracle,
        "source_sha256": {str(path): _sha256(path) for path in sources},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    with (args.output_dir / "raw_samples.jsonl").open("w") as raw_file:
        for family in args.families:
            for case in CASES:
                data = _make_case(family, case, device)
                baseline = lambda: _run_baseline(family, case, data)
                candidate = lambda: _run_candidate(family, case, data)
                baseline_output, baseline_score = baseline()
                candidate_output, candidate_score = candidate()
                torch.cuda.synchronize(device)
                output_diff = float(
                    (baseline_output.float() - candidate_output.float()).abs().max()
                )
                score_diff = float(
                    (baseline_score.float() - candidate_score.float()).abs().max()
                )
                if output_diff > 5e-2 or score_diff > 2e-3:
                    raise AssertionError(
                        f"{family}/{case.name} mismatch: output={output_diff} score={score_diff}"
                    )
                for _ in range(args.warmup):
                    baseline()
                    candidate()
                torch.cuda.synchronize(device)
                samples = {"baseline": [], "candidate": []}
                for repeat in range(args.repeats):
                    order = (
                        (("baseline", baseline), ("candidate", candidate))
                        if repeat % 2 == 0
                        else (("candidate", candidate), ("baseline", baseline))
                    )
                    for implementation, function in order:
                        latency_ms = _time_ms(function, args.iterations)
                        samples[implementation].append(latency_ms)
                        raw_file.write(
                            json.dumps(
                                {
                                    "family": family,
                                    "case": case.name,
                                    "repeat": repeat,
                                    "implementation": implementation,
                                    "latency_ms": latency_ms,
                                }
                            )
                            + "\n"
                        )
                baseline_ms = statistics.median(samples["baseline"])
                candidate_ms = statistics.median(samples["candidate"])
                rows.append(
                    {
                        "family": family,
                        **asdict(case),
                        "case": case.name,
                        "baseline_median_ms": baseline_ms,
                        "candidate_median_ms": candidate_ms,
                        "speedup": baseline_ms / candidate_ms,
                        "output_max_abs_diff": output_diff,
                        "score_max_abs_diff": score_diff,
                        "baseline_samples_ms": samples["baseline"],
                        "candidate_samples_ms": samples["candidate"],
                    }
                )
                print(json.dumps(rows[-1], sort_keys=True), flush=True)
                del data
                torch.cuda.empty_cache()

    summary = {
        "status": "success",
        "oracle": oracle,
        "rows": rows,
        "geomean_speedup": statistics.geometric_mean(
            row["speedup"] for row in rows
        ),
        "cases_at_least_1_5x": sum(row["speedup"] >= 1.5 for row in rows),
        "case_count": len(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
