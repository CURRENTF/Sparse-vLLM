from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_global_decode_stage1_kernel(
    q,
    k,
    v,
    active_slots,
    req_indices,
    context_lens,
    mid_output,
    mid_lse,
    stride_qb,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_sb,
    stride_ss,
    stride_mob,
    stride_moh,
    stride_mos,
    stride_mlb,
    stride_mlh,
    stride_mls,
    GROUP_SIZE: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    head_group = tl.program_id(1)
    sequence_block = tl.program_id(2)
    heads = head_group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    kv_head = head_group * HEADS_PER_PROGRAM // GROUP_SIZE
    dims = tl.arange(0, HEAD_DIM)
    sequence_len = tl.load(context_lens + batch)
    block_start = sequence_block * BLOCK_SEQ
    mid_offset = (
        batch * stride_mob
        + heads[:, None] * stride_moh
        + sequence_block * stride_mos
    )
    lse_offset = (
        batch * stride_mlb + heads * stride_mlh + sequence_block * stride_mls
    )
    if block_start >= sequence_len:
        tl.store(mid_output + mid_offset + dims[None, :], 0.0)
        tl.store(mid_lse + lse_offset, -float("inf"))
        return

    query = tl.load(q + batch * stride_qb + heads[:, None] * stride_qh + dims)
    max_logit = tl.full((HEADS_PER_PROGRAM,), -float("inf"), tl.float32)
    denominator = tl.zeros((HEADS_PER_PROGRAM,), tl.float32)
    accumulator = tl.zeros((HEADS_PER_PROGRAM, HEAD_DIM), tl.float32)
    block_end = tl.minimum(sequence_len, block_start + BLOCK_SEQ)
    request = tl.load(req_indices + batch)
    for offset in range(0, BLOCK_SEQ, BLOCK_N):
        positions = block_start + offset + tl.arange(0, BLOCK_N)
        visible = positions < block_end
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
    tl.store(mid_output + mid_offset + dims[None, :], accumulator / denominator[:, None])
    tl.store(
        mid_lse + lse_offset,
        max_logit * 0.6931471805599453 + tl.log(denominator),
    )


def gemma4_global_decode_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_output: torch.Tensor,
    mid_lse: torch.Tensor,
    *,
    block_seq: int,
    heads_per_program: int = 4,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Gemma 4 global decode requires matching rank-3 Q/K/V.")
    if not all(t.is_cuda for t in (q, k, v, mid_output, mid_lse)):
        raise TypeError("Gemma 4 global decode requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        tensor.dtype != q.dtype for tensor in (k, v)
    ):
        raise TypeError("Gemma 4 global decode requires matching FP16 or BF16 Q/K/V.")
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("Gemma 4 global decode requires divisible Q and KV heads.")
    head_dim = int(q.shape[-1])
    group_size = int(q.shape[1]) // int(k.shape[1])
    heads_per_program = int(heads_per_program)
    if (
        head_dim != 512
        or int(k.shape[-1]) != head_dim
        or group_size % heads_per_program
        or heads_per_program not in {2, 4}
    ):
        raise ValueError(
            "Gemma 4 global decode requires head_dim=512 and GQA groups divisible "
            f"by 2 or 4, got head_dim={head_dim}, group_size={group_size}, "
            f"heads_per_program={heads_per_program}."
        )
    if mid_output.dtype != torch.float32 or mid_lse.dtype != torch.float32:
        raise TypeError("Gemma 4 global decode workspace must use FP32 tensors.")
    expected_mid = (q.shape[0], q.shape[1], mid_output.shape[2], head_dim)
    expected_lse = expected_mid[:-1]
    if mid_output.shape != expected_mid or mid_lse.shape != expected_lse:
        raise ValueError(
            f"Gemma 4 global decode workspace must have shapes {expected_mid} and "
            f"{expected_lse}, got {tuple(mid_output.shape)} and {tuple(mid_lse.shape)}."
        )
    _gemma4_global_decode_stage1_kernel[
        (
            int(q.shape[0]),
            int(q.shape[1]) // heads_per_program,
            int(mid_output.shape[2]),
        )
    ](
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_output,
        mid_lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        active_slots.stride(0),
        active_slots.stride(1),
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        GROUP_SIZE=group_size,
        HEADS_PER_PROGRAM=heads_per_program,
        HEAD_DIM=head_dim,
        BLOCK_SEQ=int(block_seq),
        BLOCK_N=16,
        num_warps=8,
        num_stages=1,
    )


__all__ = ["gemma4_global_decode_stage1"]
