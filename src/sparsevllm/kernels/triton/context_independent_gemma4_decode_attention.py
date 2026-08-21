"""Experimental fixed-split Gemma 4 decode attention.

This is intentionally separate from both the stable Gemma 4 kernels and the
ordinary MHA/GQA experiment.  Gemma 4 keeps its unscaled QK logits, optional
sliding-window coordinates, and reduced-score atomic-max contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_context_independent_stage1(
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
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
    SCORE_MODE: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = query_head // GROUP_SIZE
    sequence_len = tl.load(context_lens + batch)
    visible_start = 0
    if WINDOW > 0:
        visible_start = tl.maximum(0, sequence_len - WINDOW)
    visible_len = sequence_len - visible_start
    num_splits = tl.maximum(
        1,
        tl.minimum(
            tl.cdiv(visible_len, TARGET_TOKENS_PER_SPLIT),
            MAX_KV_SPLITS,
        ),
    )
    split_tokens = tl.cdiv(tl.cdiv(visible_len, num_splits), BLOCK_N) * BLOCK_N
    split_start = visible_start + split * split_tokens
    split_end = tl.minimum(split_start + split_tokens, sequence_len)
    if (split >= num_splits) | (split_start >= split_end):
        return

    dims = tl.arange(0, HEAD_DIM)
    request = tl.load(req_indices + batch)
    query = tl.load(q + batch * stride_qb + query_head * stride_qh + dims)
    max_logit = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    block_count = tl.cdiv(split_end - split_start, BLOCK_N)
    for block in range(0, block_count):
        positions = split_start + block * BLOCK_N + tl.arange(0, BLOCK_N)
        visible = positions < split_end
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
        logits = tl.reshape(
            tl.dot(query[None, :], key),
            (BLOCK_N,),
        ) * 1.4426950408889634
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
        next_max = tl.maximum(max_logit, block_max)
        old_scale = tl.exp2(max_logit - next_max)
        probabilities = tl.exp2(logits - next_max)
        denominator = denominator * old_scale + tl.sum(probabilities, axis=0)
        accumulator *= old_scale
        value = tl.load(
            v + slots[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
            mask=visible[:, None],
            other=0.0,
        )
        accumulator += tl.reshape(
            tl.dot(probabilities[None, :].to(value.dtype), value),
            (HEAD_DIM,),
        )
        max_logit = next_max

    mid_offset = batch * stride_mob + query_head * stride_moh + split * stride_mos
    tl.store(mid_output + mid_offset + dims, accumulator / denominator)
    tl.store(
        mid_lse + batch * stride_mlb + query_head * stride_mlh + split * stride_mls,
        max_logit * 0.6931471805599453 + tl.log(denominator),
    )


@triton.jit
def _gemma4_context_independent_grouped_stage1(
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
    BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    batch = tl.program_id(0)
    head_group = tl.program_id(1)
    split = tl.program_id(2)
    heads = head_group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    kv_head = head_group * HEADS_PER_PROGRAM // GROUP_SIZE
    sequence_len = tl.load(context_lens + batch)
    visible_start = 0
    if WINDOW > 0:
        visible_start = tl.maximum(0, sequence_len - WINDOW)
    visible_len = sequence_len - visible_start
    num_splits = tl.maximum(
        1,
        tl.minimum(
            tl.cdiv(visible_len, TARGET_TOKENS_PER_SPLIT),
            MAX_KV_SPLITS,
        ),
    )
    split_tokens = tl.cdiv(tl.cdiv(visible_len, num_splits), BLOCK_N) * BLOCK_N
    split_start = visible_start + split * split_tokens
    split_end = tl.minimum(split_start + split_tokens, sequence_len)
    if (split >= num_splits) | (split_start >= split_end):
        return

    dims = tl.arange(0, HEAD_DIM)
    query = tl.load(q + batch * stride_qb + heads[:, None] * stride_qh + dims)
    request = tl.load(req_indices + batch)
    max_logit = tl.full((HEADS_PER_PROGRAM,), -float("inf"), tl.float32)
    denominator = tl.zeros((HEADS_PER_PROGRAM,), tl.float32)
    accumulator = tl.zeros((HEADS_PER_PROGRAM, HEAD_DIM), tl.float32)
    block_count = tl.cdiv(split_end - split_start, BLOCK_N)
    for block in range(0, block_count):
        positions = split_start + block * BLOCK_N + tl.arange(0, BLOCK_N)
        visible = positions < split_end
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
        next_max = tl.maximum(max_logit, block_max)
        old_scale = tl.exp2(max_logit - next_max)
        probabilities = tl.exp2(logits - next_max[:, None])
        denominator = denominator * old_scale + tl.sum(probabilities, axis=1)
        accumulator *= old_scale[:, None]
        value = tl.load(
            v + slots[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
            mask=visible[:, None],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(value.dtype), value)
        max_logit = next_max

    mid_offsets = (
        batch * stride_mob
        + heads[:, None] * stride_moh
        + split * stride_mos
        + dims[None, :]
    )
    lse_offsets = batch * stride_mlb + heads * stride_mlh + split * stride_mls
    tl.store(mid_output + mid_offsets, accumulator / denominator[:, None])
    tl.store(
        mid_lse + lse_offsets,
        max_logit * 0.6931471805599453 + tl.log(denominator),
    )


@triton.jit
def _gemma4_context_independent_stage2(
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
    WINDOW: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    sequence_len = tl.load(context_lens + batch)
    visible_len = sequence_len
    if WINDOW > 0:
        visible_len = tl.minimum(sequence_len, WINDOW)
    num_splits = tl.maximum(
        1,
        tl.minimum(
            tl.cdiv(visible_len, TARGET_TOKENS_PER_SPLIT),
            MAX_KV_SPLITS,
        ),
    )
    dims = tl.arange(0, HEAD_DIM)
    max_lse = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    for split in range(0, num_splits):
        lse = tl.load(
            mid_lse + batch * stride_mlb + head * stride_mlh + split * stride_mls
        )
        value = tl.load(
            mid_output
            + batch * stride_mob
            + head * stride_moh
            + split * stride_mos
            + dims
        )
        next_max = tl.maximum(max_lse, lse)
        old_scale = tl.exp(max_lse - next_max)
        split_scale = tl.exp(lse - next_max)
        accumulator = accumulator * old_scale + value * split_scale
        denominator = denominator * old_scale + split_scale
        max_lse = next_max
    tl.store(
        output + batch * stride_ob + head * stride_oh + dims,
        accumulator / denominator,
    )


@triton.jit
def _gemma4_context_independent_grouped_stage2(
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
    HEADS_PER_PROGRAM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    batch = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    sequence_len = tl.load(context_lens + batch)
    visible_len = sequence_len
    if WINDOW > 0:
        visible_len = tl.minimum(sequence_len, WINDOW)
    num_splits = tl.maximum(
        1,
        tl.minimum(
            tl.cdiv(visible_len, TARGET_TOKENS_PER_SPLIT),
            MAX_KV_SPLITS,
        ),
    )
    dims = tl.arange(0, HEAD_DIM)
    max_lse = tl.full((HEADS_PER_PROGRAM,), -float("inf"), tl.float32)
    denominator = tl.zeros((HEADS_PER_PROGRAM,), tl.float32)
    accumulator = tl.zeros((HEADS_PER_PROGRAM, HEAD_DIM), tl.float32)
    for split in range(0, num_splits):
        lse = tl.load(
            mid_lse
            + batch * stride_mlb
            + heads * stride_mlh
            + split * stride_mls
        )
        value = tl.load(
            mid_output
            + batch * stride_mob
            + heads[:, None] * stride_moh
            + split * stride_mos
            + dims[None, :]
        )
        next_max = tl.maximum(max_lse, lse)
        old_scale = tl.exp(max_lse - next_max)
        split_scale = tl.exp(lse - next_max)
        accumulator = accumulator * old_scale[:, None] + value * split_scale[:, None]
        denominator = denominator * old_scale + split_scale
        max_lse = next_max
    tl.store(
        output
        + batch * stride_ob
        + heads[:, None] * stride_oh
        + dims[None, :],
        accumulator / denominator[:, None],
    )


@torch.no_grad()
def context_independent_gemma4_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_output: torch.Tensor,
    mid_lse: torch.Tensor,
    *,
    sliding_window: int | None,
    attn_score: torch.Tensor | None = None,
    target_tokens_per_split: int = 1024,
) -> torch.Tensor:
    head_dim = int(q.shape[-1])
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Gemma 4 decode requires matching rank-3 Q/K/V")
    if head_dim not in {256, 512} or int(k.shape[-1]) != head_dim:
        raise ValueError(f"Gemma 4 decode requires head_dim 256 or 512, got {head_dim}")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        tensor.dtype != q.dtype for tensor in (k, v)
    ):
        raise TypeError("Gemma 4 decode requires matching FP16 or BF16 Q/K/V")
    if mid_output.dtype != torch.float32 or mid_lse.dtype != torch.float32:
        raise TypeError("Gemma 4 decode workspace must use FP32 tensors")
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("Gemma 4 decode requires divisible Q and KV heads")
    if attn_score is not None and attn_score.dim() not in {2, 3}:
        raise ValueError("Gemma 4 attention scores must be 2D or 3D")
    max_splits = int(mid_output.shape[2])
    if max_splits <= 0 or int(mid_lse.shape[2]) != max_splits:
        raise ValueError("Gemma 4 workspace split dimensions must match and be positive")

    score = mid_lse if attn_score is None else attn_score
    score_head_stride = score.stride(1) if score.dim() == 3 else 0
    block_n = 32 if head_dim == 256 else 16
    batch_size, num_heads = int(q.shape[0]), int(q.shape[1])
    group_size = num_heads // int(k.shape[1])
    heads_per_program = (
        group_size
        if head_dim == 256 and group_size in {2, 4}
        else (4 if group_size % 4 == 0 else 2)
    )
    use_grouped = attn_score is None and group_size % heads_per_program == 0
    if use_grouped:
        _gemma4_context_independent_grouped_stage1[
            (batch_size, num_heads // heads_per_program, max_splits)
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
            BLOCK_N=block_n,
            WINDOW=int(sliding_window or 0),
            MAX_KV_SPLITS=max_splits,
            TARGET_TOKENS_PER_SPLIT=int(target_tokens_per_split),
            num_warps=8,
            num_stages=1,
        )
    else:
        _gemma4_context_independent_stage1[
            (batch_size, num_heads, max_splits)
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
            score.stride(-1),
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            BLOCK_N=block_n,
            WINDOW=int(sliding_window or 0),
            MAX_KV_SPLITS=max_splits,
            TARGET_TOKENS_PER_SPLIT=int(target_tokens_per_split),
            SCORE_MODE=0 if attn_score is None else attn_score.dim(),
            num_warps=8,
            num_stages=1,
        )
    output = torch.empty_like(q)
    stage2_args = (
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
    )
    stage2_meta = dict(
        HEAD_DIM=head_dim,
        WINDOW=int(sliding_window or 0),
        MAX_KV_SPLITS=max_splits,
        TARGET_TOKENS_PER_SPLIT=int(target_tokens_per_split),
        num_warps=8,
        num_stages=2,
    )
    if use_grouped:
        _gemma4_context_independent_grouped_stage2[
            (batch_size, num_heads // heads_per_program)
        ](
            *stage2_args,
            HEADS_PER_PROGRAM=heads_per_program,
            **stage2_meta,
        )
    else:
        _gemma4_context_independent_stage2[(batch_size, num_heads)](
            *stage2_args,
            **stage2_meta,
        )
    return output
