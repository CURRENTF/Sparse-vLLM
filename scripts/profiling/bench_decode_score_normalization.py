"""Benchmark production decode-score normalization against its Torch oracle."""

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

import sparsevllm.kernels.triton.decode_score as decode_score


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    width: int

    @property
    def name(self) -> str:
        return f"b{self.batch}_h{self.heads}_l{self.width}"


CASES = (
    Case(1, 28, 1024),
    Case(1, 28, 4096),
    Case(1, 28, 16384),
    Case(8, 28, 4096),
    Case(32, 28, 1024),
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


def _reference(
    scores: torch.Tensor,
    candidate_lens: torch.Tensor,
    candidate_start: int,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    candidate_scores = scores[:, :, candidate_start:]
    lengths = candidate_lens.long().clamp(
        min=0,
        max=candidate_scores.shape[-1],
    )
    positions = torch.arange(candidate_scores.shape[-1], device=scores.device)
    mask = positions[None, :] < lengths[:, None]
    logits = candidate_scores.float() * scale
    logits = logits.masked_fill(
        ~mask[:, None, :], torch.finfo(logits.dtype).min
    )
    reduced = torch.softmax(logits, dim=-1).amax(dim=1).to(output_dtype)
    reduced = reduced.masked_fill(~mask, torch.finfo(output_dtype).min)
    output = torch.full(
        (scores.shape[0], scores.shape[-1]),
        torch.finfo(output_dtype).min,
        dtype=output_dtype,
        device=scores.device,
    )
    output[:, candidate_start:] = reduced
    return output


def _preheat_gpu(device: torch.device, seconds: float = 0.5) -> None:
    left = torch.randn((2048, 2048), dtype=torch.bfloat16, device=device)
    right = torch.randn_like(left)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            torch.mm(left, right)
        torch.cuda.synchronize(device)


def _time_ms(function, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.repeats < 3 or args.iterations < 1:
        raise ValueError("warmup, repeats, and iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda")
    candidate_start = 4
    scale = 128**-0.5
    output_dtype = torch.bfloat16
    torch.manual_seed(20260830)
    _preheat_gpu(device)

    implementation_source = Path(decode_score.__file__).resolve()
    repo = implementation_source.parents[4]
    script_source = Path(__file__).resolve()
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_output(repo, "rev-parse", "HEAD"),
        "git_status": _git_output(repo, "status", "--short"),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": 20260830,
        "cases": [asdict(case) for case in CASES],
        "score_dtype": "float32",
        "output_dtype": "bfloat16",
        "candidate_start": candidate_start,
        "softmax_scale": scale,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "baseline": "production Torch masked-softmax, head-max, cast, and fill",
        "candidate": (
            "two source-level Triton kernels with reusable FP32 LSE and output "
            "workspaces"
        ),
        "measurement": "matched steady-state production normalization latency",
        "source_sha256": {
            str(path): _sha256(path)
            for path in (implementation_source, script_source)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    with (args.output_dir / "raw_samples.jsonl").open("w") as raw_file:
        for case_index, case in enumerate(CASES):
            torch.manual_seed(20260830 + case_index)
            scores = torch.randn(
                case.batch,
                case.heads,
                case.width,
                dtype=torch.float32,
                device=device,
            )
            candidate_lens = torch.full(
                (case.batch,),
                case.width - candidate_start - 32,
                dtype=torch.int32,
                device=device,
            )
            if case.batch > 1:
                candidate_lens[-1] -= 17
            lse_workspace = torch.empty(
                case.batch, case.heads, dtype=torch.float32, device=device
            )
            output_workspace = torch.empty(
                case.batch,
                case.width,
                dtype=output_dtype,
                device=device,
            )

            def baseline() -> torch.Tensor:
                return _reference(
                    scores,
                    candidate_lens,
                    candidate_start,
                    scale,
                    output_dtype,
                )

            def candidate() -> torch.Tensor:
                return decode_score.decode_softmax_token_scores(
                    scores,
                    candidate_lens,
                    candidate_start=candidate_start,
                    softmax_scale=scale,
                    output_dtype=output_dtype,
                    lse_workspace=lse_workspace,
                    output=output_workspace,
                )

            expected = baseline()
            actual = candidate()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
            max_abs_diff = float((actual.float() - expected.float()).abs().max())
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
                    **asdict(case),
                    "case": case.name,
                    "baseline_median_ms": baseline_ms,
                    "candidate_median_ms": candidate_ms,
                    "speedup": baseline_ms / candidate_ms,
                    "max_abs_diff": max_abs_diff,
                    "baseline_samples_ms": samples["baseline"],
                    "candidate_samples_ms": samples["candidate"],
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    summary = {
        "status": "success",
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
