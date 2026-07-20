from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sparsevllm.quantization.fp8 import load_finegrained_fp8_kernel


_SUPPORTED_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_experts",
        "local_expert_start",
        "local_expert_end",
    ]
)
def _expert_order_moe_sum_kernel(
    inputs_ptr,
    topk_ids_ptr,
    output_ptr,
    num_tokens,
    num_experts,
    local_expert_start,
    local_expert_end,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    stride_im,
    stride_ik,
    stride_in,
    stride_om,
    stride_on,
    OUTPUT_BF16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    topk_slots = tl.arange(0, top_k)
    valid_tokens = token_offsets < num_tokens
    output_mask = valid_tokens[:, None] & (
        hidden_offsets[None, :] < hidden_size
    )

    expert_ids = tl.load(
        topk_ids_ptr
        + token_offsets[:, None] * top_k
        + topk_slots[None, :],
        mask=valid_tokens[:, None],
        other=num_experts,
    ).to(tl.int32)
    packed = expert_ids * (top_k + 1) + topk_slots[None, :]
    packed = tl.sort(packed, dim=1)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for expert_order in tl.static_range(top_k):
        selected = tl.sum(
            tl.where(topk_slots[None, :] == expert_order, packed, 0),
            axis=1,
        )
        expert_id = selected // (top_k + 1)
        topk_slot = selected % (top_k + 1)
        is_local = valid_tokens & (expert_id >= local_expert_start) & (
            expert_id < local_expert_end
        )
        values = tl.load(
            inputs_ptr
            + token_offsets[:, None] * stride_im
            + topk_slot[:, None] * stride_ik
            + hidden_offsets[None, :] * stride_in,
            mask=output_mask & is_local[:, None],
            other=0.0,
        ).to(tl.float32)
        rounded = accumulator + values
        if OUTPUT_BF16:
            rounded = rounded.to(tl.bfloat16).to(tl.float32)
        else:
            rounded = rounded.to(tl.float16).to(tl.float32)
        accumulator = tl.where(is_local[:, None], rounded, accumulator)

    tl.store(
        output_ptr
        + token_offsets[:, None] * stride_om
        + hidden_offsets[None, :] * stride_on,
        accumulator,
        mask=output_mask,
    )


def _expert_order_moe_sum(
    inputs: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> torch.Tensor:
    num_tokens, top_k, hidden_size = (int(dim) for dim in inputs.shape)
    if top_k <= 0 or top_k & (top_k - 1):
        raise ValueError(
            "MiniMax expert-order sum requires power-of-two top_k, "
            f"got {top_k}."
        )
    block_m = 1 if num_tokens <= 4 else 8
    block_n = 128
    output = torch.empty_like(inputs[:, 0])
    grid = (
        triton.cdiv(num_tokens, block_m),
        triton.cdiv(hidden_size, block_n),
    )
    _expert_order_moe_sum_kernel[grid](
        inputs,
        topk_ids,
        output,
        num_tokens,
        int(num_experts),
        int(local_expert_start),
        int(local_expert_end),
        hidden_size=hidden_size,
        top_k=top_k,
        stride_im=inputs.stride(0),
        stride_ik=inputs.stride(1),
        stride_in=inputs.stride(2),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        OUTPUT_BF16=inputs.dtype == torch.bfloat16,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        num_warps=4 if block_m <= 4 else 8,
        num_stages=2,
    )
    return output


def _validate_fused_moe_fp8_inputs(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
) -> None:
    tensors = {
        "hidden_states": hidden_states,
        "w13_weight": w13_weight,
        "w13_scale_inv": w13_scale_inv,
        "w2_weight": w2_weight,
        "w2_scale_inv": w2_scale_inv,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"MiniMax routed FP8 MoE requires CUDA {name}.")
        if tensor.device != hidden_states.device:
            raise ValueError(
                f"MiniMax routed FP8 tensors must share one device; {name} is "
                f"on {tensor.device}, hidden_states is on {hidden_states.device}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"MiniMax routed FP8 MoE requires contiguous {name}.")

    if hidden_states.ndim != 2 or int(hidden_states.shape[0]) <= 0:
        raise ValueError(
            "hidden_states must have non-empty [tokens, hidden] shape, got "
            f"{tuple(hidden_states.shape)}."
        )
    if hidden_states.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "MiniMax routed FP8 MoE activations must be BF16 or FP16, got "
            f"{hidden_states.dtype}."
        )
    if w13_weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"w13_weight must be FP8 E4M3, got {w13_weight.dtype}.")
    if w2_weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"w2_weight must be FP8 E4M3, got {w2_weight.dtype}.")
    if w13_scale_inv.dtype != torch.float32 or w2_scale_inv.dtype != torch.float32:
        raise TypeError("MiniMax routed FP8 expert scales must be FP32.")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must use int32 or int64, got {topk_ids.dtype}.")
    if topk_weights.dtype not in (*_SUPPORTED_ACTIVATION_DTYPES, torch.float32):
        raise TypeError(
            f"topk_weights must use BF16, FP16, or FP32, got {topk_weights.dtype}."
        )
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_ids and topk_weights must share [tokens, top_k] shape, got "
            f"ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}."
        )
    if int(topk_ids.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError("Router token count does not match hidden_states.")

    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError("MiniMax expert weights must be rank-3 packed tensors.")
    num_local_experts = int(w13_weight.shape[0])
    hidden_size = int(hidden_states.shape[1])
    if num_local_experts <= 0 or int(w2_weight.shape[0]) != num_local_experts:
        raise ValueError("w13_weight and w2_weight must share local expert count.")
    if int(w13_weight.shape[1]) % 2:
        raise ValueError("w13_weight output dimension must be even.")
    intermediate_size = int(w13_weight.shape[1]) // 2
    if int(w13_weight.shape[2]) != hidden_size:
        raise ValueError("w13_weight input dimension does not match hidden_states.")
    if tuple(w2_weight.shape[1:]) != (hidden_size, intermediate_size):
        raise ValueError(
            "w2_weight shape mismatch: expected "
            f"({num_local_experts}, {hidden_size}, {intermediate_size}), got "
            f"{tuple(w2_weight.shape)}."
        )
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "MiniMax routed FP8 kernel requires hidden/intermediate dimensions "
            f"aligned to 128, got {hidden_size}/{intermediate_size}."
        )
    expected_w13_scale = (
        num_local_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
    )
    expected_w2_scale = (
        num_local_experts,
        hidden_size // 128,
        intermediate_size // 128,
    )
    if tuple(w13_scale_inv.shape) != expected_w13_scale:
        raise ValueError(
            f"w13_scale_inv shape mismatch: expected={expected_w13_scale}, "
            f"got={tuple(w13_scale_inv.shape)}."
        )
    if tuple(w2_scale_inv.shape) != expected_w2_scale:
        raise ValueError(
            f"w2_scale_inv shape mismatch: expected={expected_w2_scale}, "
            f"got={tuple(w2_scale_inv.shape)}."
        )

    local_expert_end = int(local_expert_start) + num_local_experts
    if not 0 <= int(local_expert_start) < local_expert_end <= int(num_experts):
        raise ValueError(
            f"Invalid local expert range [{local_expert_start}, {local_expert_end}) "
            f"for num_experts={num_experts}."
        )


def fused_moe_fp8(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
) -> torch.Tensor:
    """Graph-stable routed W8A8 expert pipeline for MiniMax M2.7."""

    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    _validate_fused_moe_fp8_inputs(
        hidden_states,
        w13_weight,
        w13_scale_inv,
        w2_weight,
        w2_scale_inv,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
    )

    num_tokens = int(hidden_states.shape[0])
    top_k = int(topk_ids.shape[1])
    num_local_experts = int(w13_weight.shape[0])
    local_expert_end = local_expert_start + num_local_experts
    flat_global_ids = topk_ids.reshape(-1)
    is_local = (flat_global_ids >= local_expert_start) & (
        flat_global_ids < local_expert_end
    )
    local_ids = flat_global_ids - local_expert_start
    expert_ids = torch.where(
        is_local,
        local_ids,
        torch.full_like(local_ids, num_local_experts),
    ).contiguous()
    selected_hidden_states = hidden_states.repeat_interleave(top_k, dim=0)

    kernel = load_finegrained_fp8_kernel()
    gate_up = kernel.matmul_batched(
        selected_hidden_states,
        w13_weight,
        w13_scale_inv,
        expert_ids,
        [128, 128],
    )
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    down = kernel.matmul_batched(
        activated,
        w2_weight,
        w2_scale_inv,
        expert_ids,
        [128, 128],
    )
    weighted = down * topk_weights.reshape(-1, 1).to(down.dtype)
    return _expert_order_moe_sum(
        weighted.view(num_tokens, top_k, hidden_states.shape[1]),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
