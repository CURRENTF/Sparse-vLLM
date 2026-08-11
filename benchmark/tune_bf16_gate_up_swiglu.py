import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from sparsevllm.layers.activation import SiluAndMul
from sparsevllm.triton_kernel.gate_up_swiglu import gate_up_swiglu
from sparsevllm.triton_kernel.moe_config import MoeGemmConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Tune BF16 gate/up GEMM + SwiGLU.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=300)
    return parser.parse_args()


def candidates():
    for block_n in (16, 32, 64, 128):
        for block_k, warps, stages in (
            (32, 4, 3),
            (32, 4, 4),
            (64, 4, 2),
            (64, 4, 3),
            (64, 4, 4),
            (64, 8, 3),
        ):
            yield MoeGemmConfig(16, block_n, block_k, 8, warps, stages)


def main():
    args = parse_args()
    if min(args.tokens, args.intermediate_size, args.warmup, args.rep) <= 0:
        raise ValueError("All numeric arguments must be positive.")
    torch.manual_seed(0)
    inputs = torch.randn(args.tokens, 2048, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(
        2 * args.intermediate_size,
        2048,
        dtype=torch.bfloat16,
        device="cuda",
    )
    reference_projection = F.linear(inputs, weight)
    gate, up = reference_projection.chunk(2, dim=-1)
    reference = F.silu(gate.float()).mul(up.float())
    activation = SiluAndMul()
    baseline = lambda: activation(F.linear(inputs, weight))
    baseline()
    baseline_us = 1000 * triton.testing.do_bench(
        baseline, warmup=args.warmup, rep=args.rep, return_mode="median"
    )

    records = []
    for config in candidates():
        output = torch.empty_like(reference_projection[:, : args.intermediate_size])
        launch = lambda: gate_up_swiglu(inputs, weight, config, output)
        record = {**config.__dict__}
        try:
            actual = launch().float()
            torch.cuda.synchronize()
            record.update(
                status="success",
                max_abs_error=float((actual - reference).abs().max()),
                relative_l2_error=float(
                    torch.linalg.vector_norm(actual - reference)
                    / torch.linalg.vector_norm(reference)
                ),
                latency_us=1000
                * triton.testing.do_bench(
                    launch,
                    warmup=args.warmup,
                    rep=args.rep,
                    return_mode="median",
                ),
            )
        except Exception as error:
            record.update(status="invalid_config", error=f"{type(error).__name__}: {error}")
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)

    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "tokens": args.tokens,
        "intermediate_size": args.intermediate_size,
        "warmup": args.warmup,
        "rep": args.rep,
        "baseline_us": baseline_us,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
