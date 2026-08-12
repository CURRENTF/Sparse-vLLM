from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_single_block_decode_kernel(
    q,
    k,
    v,
    active_slots,
    req_indices,
    context_lens,
    output,
    stride_qb,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_sb,
    stride_ss,
    stride_ob,
    stride_oh,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    groups = tl.arange(0, GROUP_SIZE)
    dims = tl.arange(0, HEAD_DIM)
    sequence_len = tl.load(context_lens + batch)
    start = tl.maximum(0, sequence_len - WINDOW) if WINDOW > 0 else 0
    query_head = kv_head * GROUP_SIZE + groups
    query = tl.load(
        q + batch * stride_qb + query_head[:, None] * stride_qh + dims[None, :]
    )
    request = tl.load(req_indices + batch)
    max_logit = tl.full((GROUP_SIZE,), -float("inf"), tl.float32)
    denominator = tl.zeros((GROUP_SIZE,), tl.float32)
    accumulator = tl.zeros((GROUP_SIZE, HEAD_DIM), tl.float32)
    for offset in range(0, BLOCK_SEQ, BLOCK_N):
        positions = offset + tl.arange(0, BLOCK_N)
        visible = (positions >= start) & (positions < sequence_len)
        slots = tl.load(
            active_slots + request * stride_sb + positions * stride_ss,
            mask=visible,
            other=0,
        )
        key = tl.load(
            k + slots[None, :] * stride_kt + kv_head * stride_kh + dims[:, None],
            mask=visible[None, :],
            other=0.0,
        )
        logits = tl.dot(query, key) * 1.4426950408889634
        logits = tl.where(visible[None, :], logits, -float("inf"))
        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_logit, block_max)
        probabilities = tl.exp2(logits - new_max[:, None])
        correction = tl.exp2(max_logit - new_max)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        value = tl.load(
            v + slots[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
            mask=visible[:, None],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(value.dtype), value)
        max_logit = new_max
    offsets = batch * stride_ob + query_head[:, None] * stride_oh + dims[None, :]
    tl.store(output + offsets, accumulator / denominator[:, None])


def gemma4_single_block_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    block_seq: int,
    sliding_window: int | None,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or output.shape != q.shape:
        raise ValueError(
            "Gemma 4 single-block decode requires matching rank-3 Q/K/V/output."
        )
    if not all(t.is_cuda for t in (q, k, v, output)):
        raise TypeError("Gemma 4 single-block decode requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        t.dtype != q.dtype for t in (k, v, output)
    ):
        raise TypeError(
            "Gemma 4 single-block decode requires matching FP16 or BF16 tensors."
        )
    if any(t.stride(-1) != 1 for t in (q, k, v, output)):
        raise ValueError(
            "Gemma 4 single-block decode requires contiguous head dimensions."
        )
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError(
            "Gemma 4 single-block decode requires divisible Q and KV heads."
        )
    group_size = int(q.shape[1]) // int(k.shape[1])
    if group_size not in {2, 4, 8}:
        raise ValueError(
            f"Gemma 4 single-block decode requires GQA group 2, 4, or 8, got {group_size}."
        )
    head_dim = int(q.shape[-1])
    if head_dim not in {256, 512} or int(k.shape[-1]) != head_dim:
        raise ValueError(
            f"Gemma 4 single-block decode requires head_dim 256 or 512, got {head_dim}."
        )
    if int(block_seq) <= 0:
        raise ValueError(
            f"Gemma 4 single-block decode requires block_seq > 0, got {block_seq}."
        )
    block_n = 32 if head_dim == 256 else 16
    _gemma4_single_block_decode_kernel[(int(q.shape[0]), int(k.shape[1]))](
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        output,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        active_slots.stride(0),
        active_slots.stride(1),
        output.stride(0),
        output.stride(1),
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_SEQ=int(block_seq),
        BLOCK_N=block_n,
        WINDOW=int(sliding_window or 0),
        num_warps=8,
        num_stages=1,
    )


__all__ = ["gemma4_single_block_decode"]
