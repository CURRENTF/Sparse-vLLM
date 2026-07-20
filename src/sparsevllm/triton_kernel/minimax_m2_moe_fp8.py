from __future__ import annotations

import torch
import torch.nn.functional as F

from sparsevllm.quantization.fp8 import load_finegrained_fp8_kernel
from sparsevllm.triton_kernel.moe import moe_sum


_SUPPORTED_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)


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
    return moe_sum(
        weighted.view(num_tokens, top_k, hidden_states.shape[1]),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
        output_dtype=hidden_states.dtype,
    )
