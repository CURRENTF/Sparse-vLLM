import argparse
import json
from pathlib import Path

import torch
import triton

from sparsevllm.triton_kernel.moe import (
    _prepare_expert_assignment,
    _routed_gate_up_swiglu,
    _routed_gemm,
)
from sparsevllm.triton_kernel.moe_config import MoeGemmConfig
from sparsevllm.triton_kernel.silu_and_mul import silu_and_mul_fwd


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Qwen3.6 BF16 routed GEMMs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=300)
    return parser.parse_args()


def candidates():
    for block_n in (32, 64, 128):
        for block_k, warps, stages in (
            (32, 4, 3),
            (32, 4, 4),
            (64, 4, 2),
            (64, 4, 3),
            (64, 4, 4),
            (64, 8, 3),
        ):
            yield MoeGemmConfig(16, block_n, block_k, 8, warps, stages)


def benchmark(stage, launch, output, reference, config, args):
    record = {"stage": stage, **config.__dict__}
    try:
        launch()
        torch.cuda.synchronize()
        actual, expected = output.float(), reference.float()
        record.update(
            status="success",
            max_abs_error=float((actual - expected).abs().max()),
            relative_l2_error=float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected)
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
    return record


def main():
    args = parse_args()
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("--warmup and --rep must be positive.")
    torch.manual_seed(0)
    hidden = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(8, 1024, 2048, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(8, 2048, 512, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.arange(8, device="cuda", dtype=torch.int32).view(1, 8)
    topk_weights = torch.full((1, 8), 0.125, device="cuda", dtype=torch.bfloat16)
    alignment = _prepare_expert_assignment(
        topk_ids,
        block_size=16,
        num_experts=256,
        local_expert_start=0,
        local_expert_end=256,
    )
    reference_config = MoeGemmConfig(16, 128, 32, 8, 4, 4).as_triton_kwargs()
    w13_reference = torch.empty(8, 1024, device="cuda", dtype=torch.bfloat16)
    _routed_gemm(
        hidden,
        w13,
        w13_reference,
        topk_weights,
        alignment,
        input_top_k=8,
        multiply_routing_weight=False,
        launch_config=reference_config,
    )
    activated = silu_and_mul_fwd(w13_reference.clone())
    w2_reference = torch.empty(8, 2048, device="cuda", dtype=torch.bfloat16)
    _routed_gemm(
        activated,
        w2,
        w2_reference,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
        launch_config=reference_config,
    )
    torch.cuda.synchronize()

    records = []
    for config in candidates():
        launch_config = config.as_triton_kwargs()
        w13_output = torch.empty_like(w13_reference)
        records.append(
            benchmark(
                "w13",
                lambda: _routed_gemm(
                    hidden,
                    w13,
                    w13_output,
                    topk_weights,
                    alignment,
                    input_top_k=8,
                    multiply_routing_weight=False,
                    launch_config=launch_config,
                ),
                w13_output,
                w13_reference,
                config,
                args,
            )
        )
        fused_output = torch.empty_like(activated)
        records.append(
            benchmark(
                "gate_up_swiglu",
                lambda: _routed_gate_up_swiglu(
                    hidden,
                    w13,
                    fused_output,
                    alignment,
                    input_top_k=8,
                    launch_config=launch_config,
                ),
                fused_output,
                activated,
                config,
                args,
            )
        )
        w2_output = torch.empty_like(w2_reference)
        records.append(
            benchmark(
                "w2",
                lambda: _routed_gemm(
                    activated,
                    w2,
                    w2_output,
                    topk_weights,
                    alignment,
                    input_top_k=1,
                    multiply_routing_weight=True,
                    launch_config=launch_config,
                ),
                w2_output,
                w2_reference,
                config,
                args,
            )
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "warmup": args.warmup,
        "rep": args.rep,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
