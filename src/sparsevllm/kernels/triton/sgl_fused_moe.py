# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2026 SGLang Team
"""Unquantized fused MoE kernels adapted from SGLang.

Source: sgl-project/sglang@24d625698d44c78f6e8ab8b7c19f96f45bbaa90a
``python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py`` and
``python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py``.

This local port keeps the BF16/FP16 routed-GEMM path and integrates it with
Sparse-vLLM's provider-owned alignment and reduction contracts. Quantized,
LoRA, TMA, and fused-collective branches remain owned by their existing
providers.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from sparsevllm.kernels.moe import MoeAlignment
from sparsevllm.kernels.triton.moe import (
    _validate_fused_moe_inputs,
    moe_sum,
)
from sparsevllm.kernels.triton.silu_and_mul import silu_and_mul_fwd


@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def _sgl_fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    TOP_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    assignment_ids = tl.load(sorted_token_ids_ptr + token_offsets).to(tl.int64)
    assignment_mask = assignment_ids < num_valid_tokens
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id == -1:
        return

    n_offsets = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr
        + (assignment_ids // TOP_K)[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_be
        + k_offsets[:, None] * stride_bk
        + n_offsets[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=assignment_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            remaining = K - k_start
            a = tl.load(
                a_ptrs,
                mask=assignment_mask[:, None]
                & (k_offsets[None, :] < remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_offsets[:, None] < remaining,
                other=0.0,
            )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + assignment_ids,
            mask=assignment_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator *= routed_weight[:, None]

    output_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        c_ptr
        + assignment_ids[:, None] * stride_cm
        + output_offsets[None, :] * stride_cn
    )
    tl.store(
        output_ptrs,
        accumulator.to(b_ptr.dtype.element_ty),
        mask=assignment_mask[:, None] & (output_offsets[None, :] < N),
    )


def _config(
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    num_warps: int,
    num_stages: int,
) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _table(
    rows: tuple[tuple[int, tuple[int, int, int, int, int, int]], ...],
) -> dict[int, dict[str, int]]:
    return {tokens: _config(*values) for tokens, values in rows}


# These are the upstream H100 BF16/FP16 profiles for Qwen3-MoE's TP1/TP2/TP4
# expert widths. SGLang falls back to the same table across Triton versions
# when an exact-version profile is unavailable.
_SGL_H100_QWEN3_CONFIGS = {
    192: _table(
        (
            (1, (16, 64, 64, 1, 4, 5)),
            (2, (16, 64, 64, 64, 4, 5)),
            (4, (16, 64, 64, 32, 4, 5)),
            (8, (16, 128, 64, 16, 4, 5)),
            (16, (16, 128, 64, 1, 8, 3)),
            (24, (16, 128, 64, 1, 8, 3)),
            (32, (16, 128, 128, 16, 4, 3)),
            (48, (16, 128, 64, 64, 4, 4)),
            (64, (16, 64, 64, 32, 4, 3)),
            (96, (16, 128, 128, 16, 4, 2)),
            (128, (16, 128, 128, 32, 4, 3)),
            (256, (32, 128, 128, 32, 4, 2)),
            (512, (64, 128, 64, 16, 4, 3)),
            (1024, (64, 128, 64, 32, 4, 3)),
            (1536, (128, 256, 64, 16, 8, 4)),
            (2048, (128, 256, 64, 16, 8, 4)),
            (3072, (128, 128, 64, 16, 8, 3)),
            (4096, (128, 128, 64, 16, 4, 3)),
        )
    ),
    384: _table(
        (
            (1, (16, 64, 64, 1, 4, 5)),
            (2, (16, 64, 64, 1, 4, 5)),
            (4, (16, 64, 128, 1, 4, 2)),
            (8, (16, 64, 128, 64, 4, 3)),
            (16, (16, 32, 128, 64, 4, 2)),
            (24, (16, 128, 128, 16, 8, 3)),
            (32, (16, 128, 128, 16, 8, 5)),
            (48, (16, 128, 128, 16, 4, 3)),
            (64, (16, 128, 128, 64, 8, 2)),
            (96, (16, 128, 128, 16, 4, 2)),
            (128, (16, 128, 128, 16, 4, 2)),
            (256, (32, 128, 128, 32, 4, 2)),
            (512, (64, 64, 64, 16, 4, 3)),
            (1024, (64, 128, 64, 32, 4, 3)),
            (1536, (128, 128, 64, 16, 8, 3)),
            (2048, (128, 256, 64, 16, 8, 4)),
            (3072, (128, 256, 64, 32, 8, 4)),
            (4096, (128, 256, 64, 16, 8, 4)),
        )
    ),
    768: _table(
        (
            (1, (16, 64, 128, 1, 4, 3)),
            (2, (16, 64, 64, 1, 4, 3)),
            (4, (16, 64, 64, 64, 4, 5)),
            (8, (16, 128, 128, 1, 8, 5)),
            (16, (16, 128, 128, 1, 8, 5)),
            (24, (16, 64, 256, 16, 4, 3)),
            (32, (16, 128, 256, 1, 8, 3)),
            (48, (16, 128, 256, 1, 8, 3)),
            (64, (16, 256, 128, 1, 8, 3)),
            (96, (16, 128, 128, 1, 4, 2)),
            (128, (16, 128, 128, 16, 4, 2)),
            (256, (32, 256, 128, 1, 4, 2)),
            (512, (64, 128, 128, 1, 4, 2)),
            (1024, (64, 128, 64, 1, 4, 3)),
            (1536, (128, 256, 64, 1, 8, 4)),
            (2048, (128, 256, 64, 1, 8, 4)),
            (3072, (128, 256, 64, 1, 8, 4)),
            (4096, (128, 256, 64, 1, 8, 4)),
        )
    ),
}


def resolve_sgl_moe_config(
    *,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    intermediate_size: int,
    device_name: str,
) -> dict[str, int]:
    """Resolve an offline SGL profile or its generic unquantized heuristic."""

    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    table = None
    if device_name == "NVIDIA H100 80GB HBM3" and num_local_experts == 128:
        table = _SGL_H100_QWEN3_CONFIGS.get(int(intermediate_size))
    if table:
        bucket = min(table, key=lambda value: abs(value - num_tokens))
        return dict(table[bucket])
    if num_tokens * int(top_k) <= 32:
        return _config(16, 128, 32, 8, 4, 4)
    return _config(16, 64, 64, 8, 4, 3)


def _run_sgl_routed_gemm(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    alignment: MoeAlignment,
    *,
    input_top_k: int,
    multiply_routing_weight: bool,
    config: dict[str, int],
) -> None:
    if alignment.naive or alignment.sorted_token_ids is None:
        raise ValueError("The SGL fused MoE kernel requires grouped alignment metadata.")
    block_m = int(config["BLOCK_SIZE_M"])
    block_n = int(config["BLOCK_SIZE_N"])
    em = int(alignment.sorted_token_ids.numel())
    grid = (
        triton.cdiv(em, block_m) * triton.cdiv(int(weights.shape[1]), block_n),
    )
    _sgl_fused_moe_kernel[grid](
        inputs,
        weights,
        output,
        topk_weights,
        alignment.sorted_token_ids,
        alignment.expert_ids,
        alignment.num_tokens_post_padded,
        N=int(weights.shape[1]),
        K=int(weights.shape[2]),
        EM=em,
        num_valid_tokens=int(topk_weights.numel()),
        stride_am=inputs.stride(0),
        stride_ak=inputs.stride(1),
        stride_be=weights.stride(0),
        stride_bk=weights.stride(2),
        stride_bn=weights.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        TOP_K=int(input_top_k),
        MUL_ROUTED_WEIGHT=bool(multiply_routing_weight),
        EVEN_K=int(weights.shape[2]) % int(config["BLOCK_SIZE_K"]) == 0,
        **config,
    )


def sgl_fused_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    alignment_impl: Callable[..., MoeAlignment],
) -> torch.Tensor:
    """Run SGLang's unquantized routed-GEMM pipeline."""

    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    _validate_fused_moe_inputs(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts,
        local_expert_start,
    )
    num_tokens = int(hidden_states.shape[0])
    top_k = int(topk_ids.shape[1])
    num_assignments = int(topk_ids.numel())
    num_local_experts = int(w13_weight.shape[0])
    intermediate_size = int(w13_weight.shape[1]) // 2
    hidden_size = int(hidden_states.shape[1])
    config = resolve_sgl_moe_config(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        intermediate_size=intermediate_size,
        device_name=torch.cuda.get_device_name(hidden_states.device),
    )
    alignment = alignment_impl(
        topk_ids,
        block_size=config["BLOCK_SIZE_M"],
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_start + num_local_experts,
    )

    gate_up = torch.empty(
        (num_assignments, 2 * intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _run_sgl_routed_gemm(
        hidden_states,
        w13_weight,
        gate_up,
        topk_weights,
        alignment,
        input_top_k=top_k,
        multiply_routing_weight=False,
        config=config,
    )
    activated = silu_and_mul_fwd(gate_up)
    routed_output = torch.empty(
        (num_assignments, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _run_sgl_routed_gemm(
        activated,
        w2_weight,
        routed_output,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
        config=config,
    )
    return moe_sum(
        routed_output.view(num_tokens, top_k, hidden_size),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_start + num_local_experts,
    )


__all__ = ["resolve_sgl_moe_config", "sgl_fused_moe"]
