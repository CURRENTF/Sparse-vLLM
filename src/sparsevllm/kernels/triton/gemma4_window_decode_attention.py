from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_window_decode_stage1_kernel(
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
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    sequence_block = tl.program_id(2)
    groups = tl.arange(0, GROUP_SIZE)
    dims = tl.arange(0, HEAD_DIM)
    sequence_len = tl.load(context_lens + batch)
    window_start = tl.maximum(0, sequence_len - WINDOW)
    block_start = window_start + sequence_block * BLOCK_SEQ
    query_head = kv_head * GROUP_SIZE + groups
    mid_offset = (
        batch * stride_mob
        + query_head[:, None] * stride_moh
        + sequence_block * stride_mos
    )
    lse_offset = (
        batch * stride_mlb
        + query_head * stride_mlh
        + sequence_block * stride_mls
    )
    if block_start >= sequence_len:
        tl.store(mid_output + mid_offset + dims[None, :], 0.0)
        tl.store(mid_lse + lse_offset, -float("inf"))
        return

    block_end = tl.minimum(sequence_len, block_start + BLOCK_SEQ)
    query = tl.load(
        q + batch * stride_qb + query_head[:, None] * stride_qh + dims[None, :]
    )
    max_logit = tl.full((GROUP_SIZE,), -float("inf"), tl.float32)
    denominator = tl.zeros((GROUP_SIZE,), tl.float32)
    accumulator = tl.zeros((GROUP_SIZE, HEAD_DIM), tl.float32)
    for offset in range(0, BLOCK_SEQ, BLOCK_N):
        positions = block_start + offset + tl.arange(0, BLOCK_N)
        visible = positions < block_end
        request = tl.load(req_indices + batch)
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


@triton.jit
def _gemma4_window_decode_stage2_kernel(
    context_lens,
    mid_output,
    mid_lse,
    output,
    stride_mob,
    stride_moh,
    stride_mos,
    stride_mlb,
    stride_mlh,
    stride_mls,
    stride_ob,
    stride_oh,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    WINDOW: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    groups = tl.arange(0, GROUP_SIZE)
    dims = tl.arange(0, HEAD_DIM)
    query_head = kv_head * GROUP_SIZE + groups
    sequence_len = tl.load(context_lens + batch)
    block_count = (tl.minimum(sequence_len, WINDOW) + BLOCK_SEQ - 1) // BLOCK_SEQ
    max_lse = tl.full((GROUP_SIZE,), -float("inf"), tl.float32)
    denominator = tl.zeros((GROUP_SIZE,), tl.float32)
    accumulator = tl.zeros((GROUP_SIZE, HEAD_DIM), tl.float32)
    for block in range(0, NUM_BLOCKS):
        valid = block < block_count
        lse = tl.load(
            mid_lse
            + batch * stride_mlb
            + query_head * stride_mlh
            + block * stride_mls
        )
        lse = tl.where(valid, lse, -float("inf"))
        value = tl.load(
            mid_output
            + batch * stride_mob
            + query_head[:, None] * stride_moh
            + block * stride_mos
            + dims[None, :]
        )
        new_max = tl.maximum(max_lse, lse)
        old_scale = tl.exp(max_lse - new_max)
        new_scale = tl.exp(lse - new_max)
        accumulator = accumulator * old_scale[:, None] + value * new_scale[:, None]
        denominator = denominator * old_scale + new_scale
        max_lse = new_max
    tl.store(
        output
        + batch * stride_ob
        + query_head[:, None] * stride_oh
        + dims[None, :],
        accumulator / denominator[:, None],
    )


def gemma4_window_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_output: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    *,
    block_seq: int,
    sliding_window: int,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or output.shape != q.shape:
        raise ValueError("Gemma 4 window decode requires matching rank-3 Q/K/V/output.")
    if not all(t.is_cuda for t in (q, k, v, mid_output, mid_lse, output)):
        raise TypeError("Gemma 4 window decode requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        tensor.dtype != q.dtype for tensor in (k, v, output)
    ):
        raise TypeError("Gemma 4 window decode requires matching FP16 or BF16 Q/K/V.")
    if mid_output.dtype != torch.float32 or mid_lse.dtype != torch.float32:
        raise TypeError("Gemma 4 window decode workspace must use FP32 tensors.")
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("Gemma 4 window decode requires divisible Q and KV heads.")
    group_size = int(q.shape[1]) // int(k.shape[1])
    head_dim = int(q.shape[-1])
    if group_size not in {2, 4} or head_dim != 256 or int(k.shape[-1]) != head_dim:
        raise ValueError(
            "Gemma 4 window decode requires head_dim=256 and GQA group 2 or 4, "
            f"got head_dim={head_dim}, group_size={group_size}."
        )
    block_seq, sliding_window = int(block_seq), int(sliding_window)
    if block_seq <= 0 or sliding_window <= 0:
        raise ValueError("Gemma 4 window decode requires positive block and window sizes.")
    num_blocks = triton.cdiv(sliding_window, block_seq)
    if mid_output.shape[2] < num_blocks or mid_lse.shape[2] < num_blocks:
        raise ValueError(
            f"Gemma 4 window workspace needs {num_blocks} blocks, got "
            f"{mid_output.shape[2]}/{mid_lse.shape[2]}."
        )
    mid_output = mid_output[:, :, :num_blocks]
    mid_lse = mid_lse[:, :, :num_blocks]
    _gemma4_window_decode_stage1_kernel[
        (int(q.shape[0]), int(k.shape[1]), num_blocks)
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
        HEAD_DIM=head_dim,
        BLOCK_SEQ=block_seq,
        BLOCK_N=32,
        WINDOW=sliding_window,
        num_warps=8,
        num_stages=1,
    )
    _gemma4_window_decode_stage2_kernel[(int(q.shape[0]), int(k.shape[1]))](
        context_lens,
        mid_output,
        mid_lse,
        output,
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        output.stride(0),
        output.stride(1),
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_SEQ=block_seq,
        NUM_BLOCKS=num_blocks,
        WINDOW=sliding_window,
        num_warps=8,
        num_stages=2,
    )


__all__ = ["gemma4_window_decode"]
