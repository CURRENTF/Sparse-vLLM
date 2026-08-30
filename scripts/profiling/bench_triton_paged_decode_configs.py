import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sparsevllm.kernels.triton.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.kernels.triton.gqa_flash_decoding_stage1 import (
    flash_decode_stage1,
)
from sparsevllm.operators.decode_attention import (
    DecodeAttentionLaunchSpec,
    prepare_decode_attention_launch_op,
)


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    context: int
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256


QUICK_CASES = (
    Case("b1_l4k", 1, 4096),
    Case("b8_l16k", 8, 16384),
    Case("b32_l1k", 32, 1024),
)

DEFAULT_CASES = (
    Case("b1_l128", 1, 128),
    Case("b1_l1k", 1, 1024),
    Case("b1_l4k", 1, 4096),
    Case("b1_l16k", 1, 16384),
    Case("b8_l1k", 8, 1024),
    Case("b8_l4k", 8, 4096),
    Case("b8_l16k", 8, 16384),
    Case("b32_l1k", 32, 1024),
    Case("b32_l4k", 32, 4096),
)

# (BLOCK_SEQ, BLOCK_N, num_warps). The baseline is the current portable route.
CONFIGS = (
    (256, 16, 2),
    (256, 32, 4),
    (512, 16, 2),
    (512, 32, 4),
    (1024, 16, 2),
    (1024, 32, 4),
)
BASELINE = CONFIGS[0]


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _config_name(config: tuple[int, int, int]) -> str:
    return f"s{config[0]}_n{config[1]}_w{config[2]}"


def _make_case(case: Case, device: torch.device) -> dict[str, object]:
    tokens = case.batch * case.context
    q = torch.randn(
        case.batch,
        case.query_heads,
        case.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        tokens,
        case.kv_heads,
        case.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    slots = torch.arange(tokens, device=device, dtype=torch.int32).view(
        case.batch, case.context
    )
    req_indices = torch.arange(case.batch, device=device, dtype=torch.int32)
    context_lens = torch.full(
        (case.batch,), case.context, device=device, dtype=torch.int32
    )
    workspaces = {}
    for config in CONFIGS:
        num_blocks = (case.context + config[0] - 1) // config[0]
        workspaces[config] = (
            torch.empty(
                case.batch,
                case.query_heads,
                num_blocks,
                case.head_dim,
                device=device,
                dtype=torch.float32,
            ),
            torch.empty(
                case.batch,
                case.query_heads,
                num_blocks,
                device=device,
                dtype=torch.float32,
            ),
            torch.empty_like(q),
        )
    return {
        "q": q,
        "k": k,
        "v": v,
        "slots": slots,
        "req_indices": req_indices,
        "context_lens": context_lens,
        "workspaces": workspaces,
    }


def _invoke(
    case: Case,
    data: dict[str, object],
    config: tuple[int, int, int],
) -> torch.Tensor:
    block_seq, block_n, num_warps = config
    mid_o, mid_lse, output = data["workspaces"][config]
    flash_decode_stage1(
        data["q"],
        data["k"],
        data["v"],
        data["slots"],
        data["req_indices"],
        data["context_lens"],
        case.context,
        mid_o,
        mid_lse,
        block_seq,
        block_n,
        num_warps,
    )
    flash_decode_stage2(
        mid_o, mid_lse, data["context_lens"], output, block_seq
    )
    return output


def _time_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _correctness(device: torch.device) -> dict[str, object]:
    case = Case("oracle", 2, 73)
    data = _make_case(case, device)
    group_size = case.query_heads // case.kv_heads
    expected = torch.empty_like(data["q"])
    scale = case.head_dim**-0.5
    for batch in range(case.batch):
        for head in range(case.query_heads):
            kv_head = head // group_size
            logits = torch.matmul(
                data["k"][
                    batch * case.context : (batch + 1) * case.context,
                    kv_head,
                ].float(),
                data["q"][batch, head].float(),
            ) * scale
            expected[batch, head] = torch.matmul(
                torch.softmax(logits, dim=0),
                data["v"][
                    batch * case.context : (batch + 1) * case.context,
                    kv_head,
                ].float(),
            ).to(expected.dtype)
    diffs = {}
    for config in CONFIGS:
        actual = _invoke(case, data, config)
        diffs[_config_name(config)] = float(
            (actual.float() - expected.float()).abs().max().item()
        )
    return {
        "max_abs_diff_by_config": diffs,
        "passed": all(diff <= 5e-2 for diff in diffs.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(20260824)
    launch_op = prepare_decode_attention_launch_op(
        DecodeAttentionLaunchSpec(8, 2, 256, torch.bfloat16),
        device_index=torch.cuda.current_device(),
    )
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
        "baseline": _config_name(BASELINE),
        "production_launch_provider": launch_op.name,
        "shape": "Qwen3.5 local Hq=8 Hkv=2 D=256 BF16 page_size=1",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    correctness = _correctness(device)
    if not correctness["passed"]:
        raise AssertionError(f"correctness failed: {correctness}")
    rows = []
    raw_path = args.output_dir / "raw_samples.jsonl"
    cases = QUICK_CASES if args.quick else DEFAULT_CASES
    with raw_path.open("w") as raw_file:
        for case in cases:
            data = _make_case(case, device)
            samples = {_config_name(config): [] for config in CONFIGS}
            for config in CONFIGS:
                for _ in range(args.warmup):
                    _invoke(case, data, config)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                order = CONFIGS if repeat % 2 == 0 else tuple(reversed(CONFIGS))
                for config in order:
                    key = _config_name(config)
                    latency = _time_ms(
                        lambda config=config: _invoke(case, data, config),
                        args.iterations,
                    )
                    samples[key].append(latency)
                    raw_file.write(
                        json.dumps(
                            {
                                "case": case.name,
                                "config": key,
                                "repeat": repeat,
                                "latency_ms": latency,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    raw_file.flush()
            medians = {
                key: statistics.median(values)
                for key, values in samples.items()
            }
            baseline_ms = medians[_config_name(BASELINE)]
            best_key = min(medians, key=medians.get)
            production_key = _config_name(
                launch_op.launch_config(
                    block_seq=BASELINE[0],
                    batch_size=case.batch,
                    max_context_len=case.context,
                    requires_attention_scores=False,
                )
            )
            row = {
                **asdict(case),
                "medians_ms": medians,
                "samples_ms": samples,
                "best_config": best_key,
                "best_speedup": baseline_ms / medians[best_key],
                "production_config": production_key,
                "production_speedup": baseline_ms / medians[production_key],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del data
            torch.cuda.empty_cache()
    summary = {"manifest": manifest, "correctness": correctness, "rows": rows}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
