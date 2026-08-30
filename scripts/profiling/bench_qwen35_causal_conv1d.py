import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from sparsevllm.kernels.triton.qwen3_5.causal_conv1d import causal_conv1d_fn


@dataclass(frozen=True)
class Case:
    name: str
    dim: int
    sequence_lengths: tuple[int, ...]


QUICK_CASES = (
    Case("tp2_t32", 4096, (32,)),
    Case("tp1_t128", 8192, (128,)),
    Case("tp2_varlen", 4096, (1, 7, 31, 89)),
)

DEFAULT_CASES = (
    Case("tp2_t1", 4096, (1,)),
    Case("tp2_t32", 4096, (32,)),
    Case("tp2_t128", 4096, (128,)),
    Case("tp2_t512", 4096, (512,)),
    Case("tp2_t2048", 4096, (2048,)),
    Case("tp1_t32", 8192, (32,)),
    Case("tp1_t128", 8192, (128,)),
    Case("tp1_t512", 8192, (512,)),
    Case("qwen36_dim5120", 5120, (128,)),
    Case("qwen36_dim10240", 10240, (128,)),
    Case("tp2_varlen", 4096, (1, 7, 31, 89)),
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _make_case(case: Case, device: torch.device) -> dict[str, torch.Tensor]:
    total_tokens = sum(case.sequence_lengths)
    x = torch.randn(
        (total_tokens, case.dim), dtype=torch.bfloat16, device=device
    ).T
    weight = torch.randn(
        (case.dim, 4), dtype=torch.bfloat16, device=device
    ) * 0.1
    starts = torch.tensor(
        [0, *torch.tensor(case.sequence_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    batch = len(case.sequence_lengths)
    indices = torch.arange(batch, dtype=torch.int32, device=device)
    has_initial = torch.zeros(batch, dtype=torch.bool, device=device)
    states = torch.zeros(
        (batch, case.dim, 3), dtype=torch.bfloat16, device=device
    )
    return {
        "x": x,
        "weight": weight,
        "starts": starts,
        "indices": indices,
        "has_initial": has_initial,
        "states": states,
    }


def _invoke(data: dict[str, torch.Tensor], emulate_clone: bool) -> torch.Tensor:
    if emulate_clone:
        data["x"].clone()
    return causal_conv1d_fn(
        data["x"],
        data["weight"],
        bias=None,
        query_start_loc=data["starts"],
        cache_indices=data["indices"],
        has_initial_state=data["has_initial"],
        conv_states=data["states"],
        activation="silu",
    )


def _time_ms(fn: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _correctness(device: torch.device) -> dict[str, float | bool]:
    case = Case("small_correctness", 127, (1, 2, 5, 33))
    data = _make_case(case, device)
    x = data["x"]
    states_before = data["states"].clone()
    expected = torch.empty_like(x)
    starts = data["starts"].cpu().tolist()
    for sequence_id, (start, end) in enumerate(zip(starts, starts[1:])):
        conv_input = F.pad(x[:, start:end].unsqueeze(0), (3, 0))
        result = F.conv1d(
            conv_input,
            data["weight"].unsqueeze(1),
            None,
            groups=case.dim,
        )
        expected[:, start:end] = F.silu(result).squeeze(0).to(expected.dtype)
    actual = _invoke(data, emulate_clone=False)
    max_abs_diff = float((actual - expected).abs().max().item())

    pad_data = _make_case(Case("pad", 128, (3, 5)), device)
    pad_data["indices"][0] = -1
    pad_input = pad_data["x"].clone()
    pad_states = pad_data["states"].clone()
    pad_output = _invoke(pad_data, emulate_clone=False)
    pad_unchanged = bool(
        torch.equal(pad_output[:, :3], pad_input[:, :3])
        and torch.equal(pad_data["states"][0], pad_states[0])
    )
    return {
        "max_abs_diff": max_abs_diff,
        "state_changed": not torch.equal(data["states"], states_before),
        "padded_sequence_unchanged": pad_unchanged,
        "passed": max_abs_diff <= 5e-2 and pad_unchanged,
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
        "baseline": "candidate launch preceded by x.clone(), matching the removed GPU operation",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    correctness = _correctness(device)
    if not correctness["passed"]:
        raise AssertionError(f"correctness failed: {correctness}")

    cases = QUICK_CASES if args.quick else DEFAULT_CASES
    rows = []
    raw_path = args.output_dir / "raw_samples.jsonl"
    with raw_path.open("w") as raw_file:
        for case in cases:
            data = _make_case(case, device)
            for emulate_clone in (True, False):
                for _ in range(args.warmup):
                    _invoke(data, emulate_clone)
            torch.cuda.synchronize()
            samples = {"clone_baseline": [], "empty_candidate": []}
            for repeat in range(args.repeats):
                order = (True, False) if repeat % 2 == 0 else (False, True)
                for emulate_clone in order:
                    name = "clone_baseline" if emulate_clone else "empty_candidate"
                    latency = _time_ms(
                        lambda emulate_clone=emulate_clone: _invoke(
                            data, emulate_clone
                        ),
                        args.iterations,
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
            baseline_ms = statistics.median(samples["clone_baseline"])
            candidate_ms = statistics.median(samples["empty_candidate"])
            row = {
                **asdict(case),
                "total_tokens": sum(case.sequence_lengths),
                "input_mib": case.dim
                * sum(case.sequence_lengths)
                * 2
                / (1024 * 1024),
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "baseline_samples_ms": samples["clone_baseline"],
                "candidate_samples_ms": samples["empty_candidate"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del data
            torch.cuda.empty_cache()

    summary = {
        "manifest": manifest,
        "correctness": correctness,
        "rows": rows,
        "all_candidate_faster": all(row["speedup"] > 1.0 for row in rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
