from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_NUM_EXPERTS = 256
_TOP_K = 8


@triton.jit
def _topk_biased_sigmoid_kernel(
    routing_weights_ptr,
    correction_bias_ptr,
    ids_ptr,
    stride_routing_weights_m,
    stride_ids_m,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, 256)
    routing_weights = tl.load(
        routing_weights_ptr + row * stride_routing_weights_m + offsets
    )
    scores = routing_weights + tl.load(correction_bias_ptr + offsets)

    # CUDA topk(sorted=False) writes values strictly above the kth threshold
    # first, then fills the remaining slots with first-seen threshold ties.
    selection_values = tl.where(scores == scores, scores, float("inf"))
    threshold = tl.min(tl.topk(selection_values, 8), axis=0)
    greater_mask = selection_values > threshold
    equal_mask = selection_values == threshold
    greater_rank = tl.cumsum(greater_mask.to(tl.int32), axis=0) - 1
    equal_rank = tl.cumsum(equal_mask.to(tl.int32), axis=0) - 1
    num_greater = tl.sum(greater_mask.to(tl.int32), axis=0)
    selected_equal = equal_mask & (equal_rank < 8 - num_greater)
    selected = greater_mask | selected_equal
    output_slot = tl.where(greater_mask, greater_rank, num_greater + equal_rank)

    ids_base = ids_ptr + row * stride_ids_m
    tl.store(ids_base + output_slot, offsets, mask=selected)


def _validate_router_inputs(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
) -> None:
    if not router_logits.is_cuda or not correction_bias.is_cuda:
        raise ValueError("MiniMax M2.7 router requires CUDA tensors.")
    if router_logits.device != correction_bias.device:
        raise ValueError("router_logits and correction_bias must be on one device.")
    if router_logits.ndim != 2:
        raise ValueError(
            "router_logits must have shape [tokens, 256], got "
            f"{tuple(router_logits.shape)}."
        )
    if tuple(correction_bias.shape) != (_NUM_EXPERTS,):
        raise ValueError(
            "correction_bias must have shape [256], got "
            f"{tuple(correction_bias.shape)}."
        )
    if router_logits.dtype != torch.float32 or correction_bias.dtype != torch.float32:
        raise TypeError(
            "MiniMax M2.7 router requires FP32 logits and correction_bias, got "
            f"{router_logits.dtype} and {correction_bias.dtype}."
        )
    if not router_logits.is_contiguous() or not correction_bias.is_contiguous():
        raise ValueError("MiniMax M2.7 router inputs must be contiguous.")
    if int(router_logits.shape[0]) <= 0:
        raise ValueError("MiniMax M2.7 router requires at least one token.")
    if int(router_logits.shape[1]) != _NUM_EXPERTS or int(top_k) != _TOP_K:
        raise ValueError(
            "MiniMax M2.7 router requires num_experts=256 and top_k=8, got "
            f"num_experts={router_logits.shape[1]} and top_k={top_k}."
        )


def topk_biased_sigmoid(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int = _TOP_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route MiniMax M2.7 logits with its biased-sigmoid top-k rule."""
    _validate_router_inputs(router_logits, correction_bias, top_k)
    num_tokens = int(router_logits.shape[0])
    routing_weights = torch.sigmoid(router_logits)
    ids = torch.empty(
        (num_tokens, _TOP_K), dtype=torch.int64, device=router_logits.device
    )
    _topk_biased_sigmoid_kernel[(num_tokens,)](
        routing_weights,
        correction_bias,
        ids,
        routing_weights.stride(0),
        ids.stride(0),
        num_warps=2 if num_tokens <= 256 else 1,
    )
    weights = routing_weights.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids


def minimax_m2_router(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int = _TOP_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the FP32 MiniMax M2.7 gate and specialized routing kernel."""
    if hidden_states.ndim != 2 or gate_weight.ndim != 2:
        raise ValueError("hidden_states and gate_weight must both be two-dimensional.")
    if hidden_states.dtype != torch.float32 or gate_weight.dtype != torch.float32:
        raise TypeError("MiniMax M2.7 gate requires FP32 hidden states and weights.")
    if int(gate_weight.shape[0]) != _NUM_EXPERTS:
        raise ValueError(
            f"gate_weight must have 256 rows, got {gate_weight.shape[0]}."
        )
    if int(hidden_states.shape[1]) != int(gate_weight.shape[1]):
        raise ValueError(
            "hidden_states and gate_weight have incompatible hidden sizes: "
            f"{hidden_states.shape[1]} and {gate_weight.shape[1]}."
        )
    router_logits = F.linear(hidden_states, gate_weight)
    topk_weights, topk_ids = topk_biased_sigmoid(
        router_logits,
        correction_bias,
        top_k=top_k,
    )
    return router_logits, topk_weights, topk_ids
