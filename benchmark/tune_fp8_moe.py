import argparse
import json
from pathlib import Path

import torch
import triton

from sparsevllm.triton_kernel.moe import (
    _prepare_expert_assignment,
    _routed_fp8_gemm,
)
from sparsevllm.triton_kernel.moe_config import MoeGemmConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Qwen3.6 EP2 FP8 routed GEMMs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-assignments", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=300)
    return parser.parse_args()


def candidates():
    for block_n, swap_ab, stages in (
        (64, True, range(2, 6)),
        (64, False, range(2, 5)),
        (128, True, range(2, 5)),
        (128, False, range(2, 5)),
        (32, True, range(3, 5)),
    ):
        yield from (
            MoeGemmConfig(16, block_n, 128, 1, 4, stage, swap_ab)
            for stage in stages
        )


def run_stage(stage, inputs, weights, scales, topk_weights, alignment, args):
    multiply_routing_weight = stage == "w2"
    input_top_k = 1 if multiply_routing_weight else 8
    reference_config = MoeGemmConfig(16, 128, 128, 1, 4, 3)
    reference = torch.empty(8, weights.shape[1], device="cuda", dtype=torch.bfloat16)
    _routed_fp8_gemm(
        inputs,
        weights,
        scales,
        reference,
        topk_weights,
        alignment,
        input_top_k=input_top_k,
        multiply_routing_weight=multiply_routing_weight,
        config=reference_config,
    )
    torch.cuda.synchronize()
    records = []
    for config in candidates():
        output = torch.empty_like(reference)

        def launch():
            _routed_fp8_gemm(
                inputs,
                weights,
                scales,
                output,
                topk_weights,
                alignment,
                input_top_k=input_top_k,
                multiply_routing_weight=multiply_routing_weight,
                config=config,
            )

        record = {
            "stage": stage,
            "local_assignments": args.local_assignments,
            **config.__dict__,
        }
        try:
            launch()
            torch.cuda.synchronize()
            count = args.local_assignments
            actual, expected = output[:count].float(), reference[:count].float()
            record.update(
                status="success",
                max_abs_error=float((actual - expected).abs().max()),
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
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return records


def main():
    args = parse_args()
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("--warmup and --rep must be positive.")
    if not 1 <= args.local_assignments <= 8:
        raise ValueError("--local-assignments must be in [1, 8].")
    torch.manual_seed(0)
    hidden = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    activated = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    expert_ids = list(range(args.local_assignments)) + list(
        range(128, 136 - args.local_assignments)
    )
    topk_ids = torch.tensor([expert_ids], device="cuda", dtype=torch.int32)
    topk_weights = torch.full((1, 8), 0.125, device="cuda", dtype=torch.bfloat16)
    alignment = _prepare_expert_assignment(
        topk_ids,
        block_size=16,
        num_experts=256,
        local_expert_start=0,
        local_expert_end=128,
    )
    shapes = {"w13": (hidden, 1024, 2048), "w2": (activated, 2048, 512)}
    records = []
    for stage, (inputs, output_size, input_size) in shapes.items():
        weights = torch.randn(
            8, output_size, input_size, device="cuda", dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scales = torch.ones(
            8, output_size // 128, input_size // 128, device="cuda", dtype=torch.bfloat16
        )
        records.extend(
            run_stage(
                stage,
                inputs,
                weights,
                scales,
                topk_weights,
                alignment,
                args,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
