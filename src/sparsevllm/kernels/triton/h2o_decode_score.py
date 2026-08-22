from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _h2o_probability_from_lse_kernel(
    q,
    k,
    attention_lse,
    page_table,
    request_indices,
    context_lens,
    score,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_lseh,
    stride_lseb,
    stride_ptb,
    stride_pts,
    stride_sb,
    stride_ss,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    token_offsets = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, HEAD_DIM)
    context_len = tl.load(context_lens + batch)
    request = tl.load(request_indices + batch)
    token_valid = token_offsets < context_len
    slots = tl.load(
        page_table + request * stride_ptb + token_offsets * stride_pts,
        mask=token_valid,
        other=0,
    )
    query_heads = kv_head * GQA_GROUP + head_offsets
    head_valid = head_offsets < GQA_GROUP
    query = tl.load(
        q
        + batch * stride_qb
        + query_heads[:, None] * stride_qh
        + dim_offsets[None, :] * stride_qd,
        mask=head_valid[:, None],
        other=0.0,
    )
    keys = tl.load(
        k
        + slots[None, :] * stride_ks
        + kv_head * stride_kh
        + dim_offsets[:, None] * stride_kd,
        mask=token_valid[None, :],
        other=0.0,
    )
    row_lse = tl.load(
        attention_lse + query_heads * stride_lseh + batch * stride_lseb,
        mask=head_valid,
        other=0.0,
    )
    logits = tl.dot(query, keys) * SM_SCALE
    probabilities = tl.where(
        head_valid[:, None] & token_valid[None, :],
        tl.exp(logits - row_lse[:, None]),
        0.0,
    )
    token_score = tl.sum(probabilities, axis=0)
    tl.atomic_add(
        score + batch * stride_sb + token_offsets * stride_ss,
        token_score,
        mask=token_valid,
    )


@torch.no_grad()
def h2o_probability_from_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_lse: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    score: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Reduce one layer's exact decode probabilities into [batch, context]."""

    if q.ndim != 3 or k.ndim != 3 or score.ndim != 2:
        raise ValueError(
            "H2O decode probability scoring expects Q/K/score ranks 3/3/2, got "
            f"{q.ndim}/{k.ndim}/{score.ndim}."
        )
    batch, query_heads, head_dim = map(int, q.shape)
    kv_heads = int(k.shape[1])
    if query_heads % kv_heads:
        raise ValueError(
            f"H2O decode probability scoring requires GQA divisibility: "
            f"{query_heads}/{kv_heads}."
        )
    if tuple(attention_lse.shape) != (query_heads, batch):
        raise ValueError(
            "H2O decode FA3 LSE must be [query_heads, batch], got "
            f"{tuple(attention_lse.shape)}."
        )
    if attention_lse.dtype != torch.float32 or attention_lse.device != q.device:
        raise TypeError("H2O decode FA3 LSE must be FP32 on the query device.")
    if tuple(score.shape[:1]) != (batch,) or score.dtype != torch.float32:
        raise ValueError(
            "H2O decode probability scoring requires FP32 [batch, width] output, "
            f"got shape={tuple(score.shape)} dtype={score.dtype}."
        )
    if q.dtype != k.dtype or q.stride(-1) != 1 or k.stride(-1) != 1:
        raise TypeError("H2O decode probability scoring requires matching contiguous Q/K.")
    if page_table.dtype != torch.int32 or request_indices.dtype != torch.int32:
        raise TypeError("H2O decode probability scoring requires int32 page metadata.")
    if context_lens.dtype != torch.int32:
        raise TypeError("H2O decode probability scoring requires int32 context lengths.")
    if softmax_scale <= 0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}.")

    score.zero_()
    group = query_heads // kv_heads
    block_n = 64
    _h2o_probability_from_lse_kernel[
        (batch, kv_heads, triton.cdiv(int(score.shape[1]), block_n))
    ](
        q,
        k,
        attention_lse,
        page_table,
        request_indices,
        context_lens,
        score,
        *q.stride(),
        *k.stride(),
        *attention_lse.stride(),
        *page_table.stride(),
        *score.stride(),
        GQA_GROUP=group,
        HEAD_DIM=head_dim,
        BLOCK_H=max(16, triton.next_power_of_2(group)),
        BLOCK_N=block_n,
        SM_SCALE=float(softmax_scale),
        num_warps=4,
        num_stages=2,
    )
