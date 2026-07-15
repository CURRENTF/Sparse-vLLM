from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits_m,
    stride_weights_m,
    stride_ids_m,
    NORM_TOPK_PROB: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(logits_ptr + row * stride_logits_m + offsets).to(tl.float32)
    selection_logits = tl.where(logits == logits, logits, -float("inf"))
    row_max = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - row_max)
    softmax_denominator = tl.sum(exp_logits, axis=0)

    selected = tl.zeros((NUM_EXPERTS,), dtype=tl.int1)
    value0 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id0 = tl.min(
        tl.where((~selected) & (selection_logits == value0), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id0
    value1 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id1 = tl.min(
        tl.where((~selected) & (selection_logits == value1), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id1
    value2 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id2 = tl.min(
        tl.where((~selected) & (selection_logits == value2), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id2
    value3 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id3 = tl.min(
        tl.where((~selected) & (selection_logits == value3), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id3
    value4 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id4 = tl.min(
        tl.where((~selected) & (selection_logits == value4), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id4
    value5 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id5 = tl.min(
        tl.where((~selected) & (selection_logits == value5), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id5
    value6 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id6 = tl.min(
        tl.where((~selected) & (selection_logits == value6), offsets, NUM_EXPERTS),
        axis=0,
    )
    selected |= offsets == id6
    value7 = tl.max(tl.where(selected, -float("inf"), selection_logits), axis=0)
    id7 = tl.min(
        tl.where((~selected) & (selection_logits == value7), offsets, NUM_EXPERTS),
        axis=0,
    )

    weight0 = tl.exp(value0 - row_max)
    weight1 = tl.exp(value1 - row_max)
    weight2 = tl.exp(value2 - row_max)
    weight3 = tl.exp(value3 - row_max)
    weight4 = tl.exp(value4 - row_max)
    weight5 = tl.exp(value5 - row_max)
    weight6 = tl.exp(value6 - row_max)
    weight7 = tl.exp(value7 - row_max)
    if NORM_TOPK_PROB:
        denominator = (
            weight0
            + weight1
            + weight2
            + weight3
            + weight4
            + weight5
            + weight6
            + weight7
        )
    else:
        denominator = softmax_denominator

    weights_base = weights_ptr + row * stride_weights_m
    ids_base = ids_ptr + row * stride_ids_m
    tl.store(weights_base + 0, weight0 / denominator)
    tl.store(weights_base + 1, weight1 / denominator)
    tl.store(weights_base + 2, weight2 / denominator)
    tl.store(weights_base + 3, weight3 / denominator)
    tl.store(weights_base + 4, weight4 / denominator)
    tl.store(weights_base + 5, weight5 / denominator)
    tl.store(weights_base + 6, weight6 / denominator)
    tl.store(weights_base + 7, weight7 / denominator)
    tl.store(ids_base + 0, id0)
    tl.store(ids_base + 1, id1)
    tl.store(ids_base + 2, id2)
    tl.store(ids_base + 3, id3)
    tl.store(ids_base + 4, id4)
    tl.store(ids_base + 5, id5)
    tl.store(ids_base + 6, id6)
    tl.store(ids_base + 7, id7)


def topk_softmax(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not router_logits.is_cuda:
        raise ValueError("Triton topk_softmax requires CUDA router_logits.")
    if router_logits.ndim != 2:
        raise ValueError(
            "router_logits must have shape [tokens, experts], got "
            f"{tuple(router_logits.shape)}."
        )
    if router_logits.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "Triton topk_softmax supports BF16 and FP16 logits, got "
            f"{router_logits.dtype}."
        )
    if not router_logits.is_contiguous():
        raise ValueError("Triton topk_softmax requires contiguous router_logits.")
    if int(router_logits.shape[0]) <= 0:
        raise ValueError("Triton topk_softmax requires at least one token.")
    if int(router_logits.shape[1]) != 128 or int(top_k) != 8:
        raise ValueError(
            "Triton topk_softmax currently requires num_experts=128 and top_k=8, "
            f"got num_experts={router_logits.shape[1]}, top_k={top_k}."
        )

    num_tokens = int(router_logits.shape[0])
    weights = torch.empty(
        (num_tokens, 8),
        dtype=router_logits.dtype,
        device=router_logits.device,
    )
    ids = torch.empty((num_tokens, 8), dtype=torch.int32, device=router_logits.device)
    _topk_softmax_kernel[(num_tokens,)](
        router_logits,
        weights,
        ids,
        router_logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        NORM_TOPK_PROB=bool(norm_topk_prob),
        NUM_EXPERTS=128,
        num_warps=4,
    )
    return weights, ids
