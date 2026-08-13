from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_decode_stage1_kernel(
    q,
    k,
    v,
    active_slots,
    req_indices,
    context_lens,
    mid_output,
    mid_lse,
    attn_score,
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
    stride_asb,
    stride_ash,
    stride_asl,
    group_size,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
    SCORE_MODE: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    sequence_block = tl.program_id(2)
    kv_head = query_head // group_size
    dims = tl.arange(0, HEAD_DIM)
    sequence_len = tl.load(context_lens + batch)
    request = tl.load(req_indices + batch)
    block_start = sequence_block * BLOCK_SEQ
    mid_offset = (
        batch * stride_mob + query_head * stride_moh + sequence_block * stride_mos
    )
    if block_start >= sequence_len:
        tl.store(mid_output + mid_offset + dims, 0.0)
        tl.store(
            mid_lse
            + batch * stride_mlb
            + query_head * stride_mlh
            + sequence_block * stride_mls,
            -float("inf"),
        )
        return
    if WINDOW > 0:
        block_start = tl.maximum(block_start, sequence_len - WINDOW)
    block_end = tl.minimum(sequence_len, (sequence_block + 1) * BLOCK_SEQ)
    query = tl.load(q + batch * stride_qb + query_head * stride_qh + dims)
    max_logit = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    for offset in range(0, BLOCK_SEQ, BLOCK_N):
        positions = sequence_block * BLOCK_SEQ + offset + tl.arange(0, BLOCK_N)
        visible = (positions >= block_start) & (positions < block_end)
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
        logits = (
            tl.reshape(tl.dot(query[None, :], key), (BLOCK_N,)) * 1.4426950408889634
        )
        if SCORE_MODE == 3:
            tl.store(
                attn_score
                + batch * stride_asb
                + query_head * stride_ash
                + positions * stride_asl,
                logits * 0.6931471805599453,
                mask=visible,
            )
        elif SCORE_MODE == 2:
            tl.atomic_max(
                attn_score + batch * stride_asb + positions * stride_asl,
                logits * 0.6931471805599453,
                mask=visible,
            )
        logits = tl.where(visible, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(max_logit, block_max)
        probabilities = tl.exp2(logits - new_max)
        correction = tl.exp2(max_logit - new_max)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        accumulator *= correction
        value = tl.load(
            v + slots[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
            mask=visible[:, None],
            other=0.0,
        )
        accumulator += tl.reshape(
            tl.dot(probabilities[None, :].to(value.dtype), value), (HEAD_DIM,)
        )
        max_logit = new_max
    valid_block = block_end > block_start
    tl.store(
        mid_output + mid_offset + dims,
        tl.where(valid_block, accumulator / denominator, 0.0),
    )
    tl.store(
        mid_lse
        + batch * stride_mlb
        + query_head * stride_mlh
        + sequence_block * stride_mls,
        tl.where(
            valid_block,
            max_logit * 0.6931471805599453 + tl.log(denominator),
            -float("inf"),
        ),
    )


@torch.no_grad()
def gemma4_decode_stage1(
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
    sliding_window: int | None,
    attn_score: torch.Tensor | None = None,
) -> None:
    head_dim = int(q.shape[-1])
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Gemma 4 decode requires matching rank-3 Q/K/V.")
    if head_dim not in {256, 512} or int(k.shape[-1]) != head_dim:
        raise ValueError(
            f"Gemma 4 decode requires head_dim 256 or 512, got {head_dim}."
        )
    if not all(t.is_cuda for t in (q, k, v, mid_output, mid_lse)):
        raise TypeError("Gemma 4 decode requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        t.dtype != q.dtype for t in (k, v)
    ):
        raise TypeError("Gemma 4 decode requires matching FP16 or BF16 Q/K/V.")
    if mid_output.dtype != torch.float32 or mid_lse.dtype != torch.float32:
        raise TypeError(
            "Gemma 4 decode workspace must use FP32 output and LSE tensors."
        )
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("Gemma 4 decode requires divisible Q and KV heads.")
    if int(block_seq) <= 0:
        raise ValueError(f"Gemma 4 decode requires block_seq > 0, got {block_seq}.")
    block_n = 32 if head_dim == 256 else 16
    if attn_score is not None and attn_score.dim() not in {2, 3}:
        raise ValueError(
            "Gemma 4 decode attention scores must be [B, L] or [B, H, L], "
            f"got {tuple(attn_score.shape)}."
        )
    score = mid_lse if attn_score is None else attn_score
    score_head_stride = score.stride(1) if score.dim() == 3 else 0
    score_length_stride = score.stride(-1)
    _gemma4_decode_stage1_kernel[
        (int(q.shape[0]), int(q.shape[1]), int(mid_output.shape[2]))
    ](
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_output,
        mid_lse,
        score,
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
        score.stride(0),
        score_head_stride,
        score_length_stride,
        int(q.shape[1]) // int(k.shape[1]),
        HEAD_DIM=head_dim,
        BLOCK_SEQ=int(block_seq),
        BLOCK_N=block_n,
        WINDOW=int(sliding_window or 0),
        SCORE_MODE=0 if attn_score is None else attn_score.dim(),
        num_warps=8,
        num_stages=1,
    )


@triton.jit
def _gemma4_decode_stage2_kernel(
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
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    WINDOW: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    sequence_len = tl.load(context_lens + batch)
    first_block = 0
    if WINDOW > 0:
        first_block = tl.maximum(0, sequence_len - WINDOW) // BLOCK_SEQ
    block_count = (sequence_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    max_lse = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    for block in range(first_block, block_count):
        lse = tl.load(
            mid_lse + batch * stride_mlb + head * stride_mlh + block * stride_mls
        )
        value = tl.load(
            mid_output
            + batch * stride_mob
            + head * stride_moh
            + block * stride_mos
            + dims
        )
        new_max = tl.maximum(max_lse, lse)
        old_scale = tl.exp(max_lse - new_max)
        new_scale = tl.exp(lse - new_max)
        accumulator = accumulator * old_scale + value * new_scale
        denominator = denominator * old_scale + new_scale
        max_lse = new_max
    tl.store(
        output + batch * stride_ob + head * stride_oh + dims, accumulator / denominator
    )


@torch.no_grad()
def gemma4_decode_stage2(
    mid_output: torch.Tensor,
    mid_lse: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    block_seq: int,
    sliding_window: int | None,
) -> None:
    head_dim = int(mid_output.shape[-1])
    if head_dim not in {256, 512}:
        raise ValueError(
            f"Gemma 4 decode stage 2 requires head_dim 256 or 512, got {head_dim}."
        )
    if not all(t.is_cuda for t in (mid_output, mid_lse, output)):
        raise TypeError("Gemma 4 decode stage 2 requires CUDA tensors.")
    if mid_output.dtype != torch.float32 or mid_lse.dtype != torch.float32:
        raise TypeError("Gemma 4 decode stage 2 workspace must use FP32 tensors.")
    if output.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("Gemma 4 decode stage 2 output must use FP16 or BF16.")
    if output.shape[:2] != mid_output.shape[:2] or output.shape[-1] != head_dim:
        raise ValueError(
            "Gemma 4 decode stage 2 requires matching batch/head/output shape."
        )
    if int(block_seq) <= 0:
        raise ValueError(
            f"Gemma 4 decode stage 2 requires block_seq > 0, got {block_seq}."
        )
    _gemma4_decode_stage2_kernel[(int(output.shape[0]), int(output.shape[1]))](
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
        HEAD_DIM=head_dim,
        BLOCK_SEQ=int(block_seq),
        WINDOW=int(sliding_window or 0),
        num_warps=8,
        num_stages=2,
    )
