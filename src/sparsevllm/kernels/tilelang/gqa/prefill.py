# SPDX-License-Identifier: Apache-2.0
"""TileLang GQA Prefill & Attention Score Extraction Kernels for SM90."""

from typing import Any

import torch

_KERNEL_CACHE: dict[tuple, Any] = {}
_SCORE_KERNEL_CACHE: dict[tuple, Any] = {}


def _get_heads_per_group(gqa_ratio: int) -> int:
    """Find the largest divisor of gqa_ratio <= 8."""
    for h in (8, 7, 6, 5, 4, 3, 2, 1):
        if gqa_ratio % h == 0:
            return h
    return 1


import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def _build_fused_gqa_kernel(
    h_q: int,
    h_kv: int,
    head_dim: int,
    block_M: int = 64,
    block_N: int = 64,
    softmax_scale: float | None = None,
    need_score: bool = True,
):
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32
    gqa_ratio = h_q // h_kv

    total_q = T.symbolic("total_q")
    cache_slots = T.symbolic("cache_slots")
    slot_rows = T.symbolic("slot_rows")
    slot_cols = T.symbolic("slot_cols")
    B = T.symbolic("B")
    B_plus_1 = T.symbolic("B_plus_1")
    score_len = T.symbolic("score_len")

    @T.prim_func
    def main_gqa(
        Q: T.Tensor([total_q, h_q, head_dim], dtype),
        K: T.Tensor([cache_slots, h_kv, head_dim], dtype),
        V: T.Tensor([cache_slots, h_kv, head_dim], dtype),
        active_slots: T.Tensor([slot_rows, slot_cols], T.int32),
        request_indices: T.Tensor([B], T.int32),
        context_lens: T.Tensor([B], T.int32),
        prompt_cache_lens: T.Tensor([B], T.int32),
        cu_seqlens_q: T.Tensor([B_plus_1], T.int32),
        Output: T.Tensor([total_q, h_q, head_dim], dtype),
        AttnScore: T.Tensor([B, score_len], accum_dtype),
        batch_size: T.int32,
        max_context_len: T.int32,
    ):
        with T.Kernel(batch_size, h_q, T.ceildiv(max_context_len, block_M), threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, head_dim], dtype)
            K_shared = T.alloc_shared([block_N, head_dim], dtype)
            V_shared = T.alloc_shared([block_N, head_dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            token_scores = T.alloc_fragment([block_N], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            cur_kv_head = by // gqa_ratio
            request_row = T.max(request_indices[bx], 0)
            q_start = cu_seqlens_q[bx]
            q_end = cu_seqlens_q[bx + 1]
            q_len = q_end - q_start
            ctx_len = context_lens[bx]
            p_len = prompt_cache_lens[bx]

            q_tile_start = bz * block_M
            if q_tile_start < q_len:
                # Load Q tile
                for i, d in T.Parallel(block_M, head_dim):
                    q_idx = q_start + q_tile_start + i
                    Q_shared[i, d] = T.if_then_else(
                        q_tile_start + i < q_len,
                        Q[q_idx, by, d],
                        0.0,
                    )

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                num_kv_blocks = T.ceildiv(ctx_len, block_N)
                for k in T.Pipelined(num_kv_blocks, num_stages=2):
                    for j, d in T.Parallel(block_N, head_dim):
                        kv_idx = k * block_N + j
                        slot = T.if_then_else(
                            kv_idx < ctx_len,
                            active_slots[request_row, kv_idx],
                            0,
                        )
                        K_shared[j, d] = K[slot, cur_kv_head, d]
                        V_shared[j, d] = V[slot, cur_kv_head, d]

                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    # Apply causal mask
                    for i, j in T.Parallel(block_M, block_N):
                        q_abs = p_len + q_tile_start + i
                        kv_abs = k * block_N + j
                        is_valid = (q_tile_start + i < q_len) and (kv_abs < ctx_len) and (q_abs >= kv_abs)
                        acc_s[i, j] = T.if_then_else(is_valid, acc_s[i, j], -T.infinity(accum_dtype))

                    # Extract Attention Scores into AttnScore via atomic_max
                    if need_score:
                        T.reduce_max(acc_s, token_scores, dim=0)
                        for j in T.Parallel(block_N):
                            kv_abs = k * block_N + j
                            if kv_abs < ctx_len:
                                T.atomic_max(AttnScore[bx, kv_abs], token_scores[j])

                    # Online softmax
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, acc_s_cast)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    for i, d in T.Parallel(block_M, head_dim):
                        acc_o[i, d] *= scores_scale[i]
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                # Normalize output and write back
                for i, d in T.Parallel(block_M, head_dim):
                    q_idx = q_start + q_tile_start + i
                    if q_tile_start + i < q_len:
                        Output[q_idx, by, d] = T.if_then_else(
                            logsum[i] > 0,
                            acc_o[i, d] / logsum[i],
                            0.0,
                        )

    return main_gqa


def _get_fused_prefill_kernel(
    h_q: int,
    h_kv: int,
    head_dim: int,
    block_M: int = 64,
    block_N: int = 64,
    softmax_scale: float | None = None,
    need_score: bool = True,
):
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)

    key = (
        h_q,
        h_kv,
        head_dim,
        block_M,
        block_N,
        round(scale, 6),
        need_score,
    )
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    compiled = _build_fused_gqa_kernel(
        h_q=h_q,
        h_kv=h_kv,
        head_dim=head_dim,
        block_M=block_M,
        block_N=block_N,
        softmax_scale=softmax_scale,
        need_score=need_score,
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _get_kv_score_kernel(
    h_q: int,
    h_kv: int,
    head_dim: int,
    block_M_per_head: int = 32,
    heads_per_group: int = 8,
    block_N: int = 128,
    threads: int = 128,
):
    key = (
        h_q,
        h_kv,
        head_dim,
        block_M_per_head,
        heads_per_group,
        block_N,
        threads,
    )
    if key in _SCORE_KERNEL_CACHE:
        return _SCORE_KERNEL_CACHE[key]

    dtype = T.bfloat16
    accum_dtype = T.float32
    gqa_ratio = h_q // h_kv
    if gqa_ratio % heads_per_group != 0:
        raise ValueError("gqa_ratio must be divisible by heads_per_group")
    num_head_groups = gqa_ratio // heads_per_group
    total_block_M = heads_per_group * block_M_per_head

    total_q = T.symbolic("total_q")
    num_slots = T.symbolic("num_slots")
    slot_rows = T.symbolic("slot_rows")
    slot_cols = T.symbolic("slot_cols")
    score_len = T.symbolic("score_len")
    B = T.symbolic("B")

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _jit_kernel():
        @T.prim_func
        def main_kv_score_pruned(
            Q: T.Tensor([total_q, h_q, head_dim], dtype),
            K: T.Tensor([num_slots, h_kv, head_dim], dtype),
            active_slots: T.Tensor([slot_rows, slot_cols], T.int32),
            request_indices: T.Tensor([B], T.int32),
            context_lens: T.Tensor([B], T.int32),
            prompt_cache_lens: T.Tensor([B], T.int32),
            b_start_loc: T.Tensor([B], T.int32),
            score_q_starts: T.Tensor([B], T.int32),
            score_q_ends: T.Tensor([B], T.int32),
            AttnScore: T.Tensor([B, score_len], accum_dtype),
            batch_size: T.int32,
            max_context_len: T.int32,
            candidate_start: T.int32,
            num_recent_tokens: T.int32,
        ):
            num_kv_head_splits = h_kv * num_head_groups
            with T.Kernel(
                batch_size,
                num_kv_head_splits,
                T.ceildiv(max_context_len, block_N),
                threads=threads,
            ) as (bx, by, bz):
                K_shared = T.alloc_shared([block_N, head_dim], dtype)
                Q_shared = T.alloc_shared([total_block_M, head_dim], dtype)
                acc_s = T.alloc_fragment([total_block_M, block_N], accum_dtype)
                score_max = T.alloc_fragment([block_N], accum_dtype)
                q_token_max = T.alloc_fragment([block_N], accum_dtype)

                cur_kv_head = by // num_head_groups
                head_group_idx = by % num_head_groups
                head_base = (
                    cur_kv_head * gqa_ratio
                    + head_group_idx * heads_per_group
                )

                request_row = T.max(request_indices[bx], 0)
                q_start = b_start_loc[bx]
                ctx_len = context_lens[bx]
                p_len = prompt_cache_lens[bx]
                q_chunk_len = ctx_len - p_len

                win_start = score_q_starts[bx]
                win_end = score_q_ends[bx]
                win_len = T.max(0, win_end - win_start)

                kv_tile_start = bz * block_N
                if (kv_tile_start < ctx_len) and (win_len > 0):
                    for j, d in T.Parallel(block_N, head_dim):
                        kv_idx = kv_tile_start + j
                        slot = T.if_then_else(
                            kv_idx < ctx_len,
                            active_slots[request_row, kv_idx],
                            0,
                        )
                        K_shared[j, d] = K[slot, cur_kv_head, d]

                    T.fill(score_max, -T.infinity(accum_dtype))
                    total_q_blks = T.ceildiv(win_len, block_M_per_head)

                    for q_blk in T.Pipelined(total_q_blks, num_stages=2):
                        q_tile_start = q_blk * block_M_per_head

                        for row_idx, d in T.Parallel(total_block_M, head_dim):
                            g = row_idx // block_M_per_head
                            t = row_idx % block_M_per_head
                            cur_q_head = head_base + g
                            q_abs = win_start + q_tile_start + t
                            q_rel = q_abs - p_len
                            is_valid_q = (
                                (q_tile_start + t < win_len)
                                and (q_rel >= 0)
                                and (q_rel < q_chunk_len)
                            )
                            Q_shared[row_idx, d] = T.if_then_else(
                                is_valid_q,
                                Q[q_start + q_rel, cur_q_head, d],
                                0.0,
                            )

                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                        for row_idx, j in T.Parallel(total_block_M, block_N):
                            t = row_idx % block_M_per_head
                            q_abs = win_start + q_tile_start + t
                            q_rel = q_abs - p_len
                            kv_abs = kv_tile_start + j
                            is_valid = (
                                (q_tile_start + t < win_len)
                                and (kv_abs < ctx_len)
                                and (q_rel >= 0)
                                and (q_rel < q_chunk_len)
                                and (kv_abs >= candidate_start)
                                and (kv_abs < ctx_len - num_recent_tokens)
                                and (q_abs >= kv_abs)
                            )
                            acc_s[row_idx, j] = T.if_then_else(
                                is_valid,
                                acc_s[row_idx, j],
                                -T.infinity(accum_dtype),
                            )

                        T.reduce_max(acc_s, q_token_max, dim=0)
                        for j in T.Parallel(block_N):
                            score_max[j] = T.max(score_max[j], q_token_max[j])

                    for j in T.Parallel(block_N):
                        kv_abs = kv_tile_start + j
                        is_kv_candidate = (
                            (kv_abs >= candidate_start)
                            and (kv_abs < ctx_len - num_recent_tokens)
                            and (kv_abs < ctx_len)
                            and (kv_abs < score_len)
                        )
                        if is_kv_candidate:
                            T.atomic_max(AttnScore[bx, kv_abs], score_max[j])

        return main_kv_score_pruned

    compiled = _jit_kernel()
    _SCORE_KERNEL_CACHE[key] = compiled
    return compiled


@torch.no_grad()
def gqa_paged_prefill_attention_tilelang(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    prompt_cache_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    output: torch.Tensor,
    attn_score: torch.Tensor | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Execute paged GQA prefill forward pass via TileLang."""
    total_q_tokens, h_q, head_dim = q.shape
    cache_slots, h_kv, _ = k_cache.shape
    batch_size = int(context_lens.numel())
    slot_rows, max_context_len = int(active_slots.shape[0]), int(active_slots.shape[1])

    need_score = attn_score is not None
    if attn_score is None:
        target_score = torch.empty((batch_size, max_context_len), dtype=torch.float32, device=q.device)
    else:
        expected_shape = (batch_size, max_context_len)
        if attn_score.ndim != 2 or tuple(attn_score.shape) != expected_shape:
            raise ValueError(
                "TileLang GQA prefill only supports reduced 2D scores with shape "
                f"{expected_shape}, got {tuple(attn_score.shape)}."
            )
        if attn_score.dtype != torch.float32:
            raise TypeError(
                f"TileLang GQA prefill requires FP32 score output, got {attn_score.dtype}."
            )
        if attn_score.device != q.device:
            raise ValueError(
                "TileLang GQA prefill requires score output on the Q device, got "
                f"{attn_score.device} and {q.device}."
            )
        target_score = attn_score
        target_score.fill_(-torch.inf)

    kernel = _get_fused_prefill_kernel(
        h_q=h_q,
        h_kv=h_kv,
        head_dim=head_dim,
        block_M=64,
        block_N=64,
        softmax_scale=sm_scale,
        need_score=need_score,
    )
    kernel(
        q,
        k_cache,
        v_cache,
        active_slots,
        req_indices,
        context_lens,
        prompt_cache_lens,
        cu_seqlens_q,
        output,
        target_score,
        batch_size,
        max_context_len,
    )
    return output


@torch.no_grad()
def gqa_prefill_score_tilelang(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    attn_score: torch.Tensor,
    req_indices: torch.Tensor,
    b_start_loc: torch.Tensor,
    context_lens: torch.Tensor,
    prompt_cache_lens: torch.Tensor,
    max_query_len: int,
    active_slots: torch.Tensor,
    score_q_start: torch.Tensor,
    score_q_end: torch.Tensor,
    *,
    candidate_start: int = 0,
    num_recent_tokens: int = 0,
) -> None:
    """Compute max raw QK logits over query positions and query heads."""
    del max_query_len
    if q.ndim != 3 or k_cache.ndim != 3:
        raise ValueError(
            f"TileLang raw-QK scoring expects 3D Q/K, got {q.shape}/{k_cache.shape}."
        )
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        raise TypeError(
            "TileLang raw-QK scoring requires BF16 Q/K, got "
            f"{q.dtype}/{k_cache.dtype}."
        )
    if q.shape[-1] != 128 or k_cache.shape[-1] != 128:
        raise ValueError(
            "TileLang raw-QK scoring requires head_dim=128, got "
            f"{q.shape[-1]}/{k_cache.shape[-1]}."
        )
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError(
            "TileLang raw-QK scoring requires query heads divisible by KV heads, "
            f"got {q.shape[1]}/{k_cache.shape[1]}."
        )
    batch_size = int(context_lens.numel())
    if attn_score.ndim != 2 or tuple(attn_score.shape[:1]) != (batch_size,):
        raise ValueError(
            "TileLang raw-QK scoring requires a 2D [batch, score_len] output, "
            f"got {tuple(attn_score.shape)} for batch={batch_size}."
        )
    if attn_score.dtype != torch.float32:
        raise TypeError(
            f"TileLang raw-QK scoring requires FP32 output, got {attn_score.dtype}."
        )
    tensors = {
        "q": q,
        "k_cache": k_cache,
        "attn_score": attn_score,
        "req_indices": req_indices,
        "b_start_loc": b_start_loc,
        "context_lens": context_lens,
        "prompt_cache_lens": prompt_cache_lens,
        "active_slots": active_slots,
        "score_q_start": score_q_start,
        "score_q_end": score_q_end,
    }
    if any(tensor.device != q.device for tensor in tensors.values()):
        devices = {name: str(tensor.device) for name, tensor in tensors.items()}
        raise ValueError(f"TileLang raw-QK scoring requires one device, got {devices}.")
    for name in (
        "req_indices",
        "b_start_loc",
        "context_lens",
        "prompt_cache_lens",
        "active_slots",
        "score_q_start",
        "score_q_end",
    ):
        if tensors[name].dtype != torch.int32:
            raise TypeError(
                f"TileLang raw-QK scoring requires int32 {name}, got {tensors[name].dtype}."
            )

    total_q_tokens, h_q, head_dim = q.shape
    cache_slots, h_kv, _ = k_cache.shape
    slot_rows, slot_cols = int(active_slots.shape[0]), int(active_slots.shape[1])
    score_len = int(attn_score.shape[1])
    max_context_len = int(context_lens.max().item())
    heads_per_group = _get_heads_per_group(h_q // h_kv)

    attn_score.fill_(-torch.inf)
    kernel = _get_kv_score_kernel(
        h_q=h_q,
        h_kv=h_kv,
        head_dim=head_dim,
        block_M_per_head=32,
        heads_per_group=heads_per_group,
        block_N=128,
        threads=128,
    )
    kernel(
        q,
        k_cache,
        active_slots,
        req_indices,
        context_lens,
        prompt_cache_lens,
        b_start_loc,
        score_q_start,
        score_q_end,
        attn_score,
        batch_size,
        max_context_len,
        int(candidate_start),
        int(num_recent_tokens),
    )
