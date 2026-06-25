import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_score_partial_stats_kernel(
    Q,
    K,
    Partial_M,
    Partial_L,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    candidate_start: tl.constexpr,
    num_recent_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_n_block = tl.program_id(1)
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
    context_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = score_q_start + (offs_rows % BLOCK_M)
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    start_n = cur_n_block * BLOCK_N
    kv_pos = start_n + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - num_recent_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    qk = tl.dot(q, k) * sm_scale
    causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
    valid = q_row_valid[:, None] & kv_in_candidate[None, :] & causal_mask
    qk = tl.where(valid, qk, -1.0e20)
    m_i = tl.max(qk, axis=1)
    p = tl.exp(qk - m_i[:, None])
    p = tl.where(valid, p, 0.0)
    l_i = tl.sum(p, axis=1)

    stats_offs = (cur_group * NUM_BLOCKS + cur_n_block) * BLOCK_ROWS + offs_rows
    tl.store(Partial_M + stats_offs, m_i)
    tl.store(Partial_L + stats_offs, l_i)


@triton.jit
def _prefill_score_reduce_stats_kernel(
    Partial_M,
    Partial_L,
    Global_M,
    Global_L,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    REDUCE_BLOCKS: tl.constexpr,
    REDUCE_ROWS: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_row_block = tl.program_id(1)
    offs_rows = cur_row_block * REDUCE_ROWS + tl.arange(0, REDUCE_ROWS)
    offs_blocks = tl.arange(0, REDUCE_BLOCKS)

    stats_offs = (
        cur_group * NUM_BLOCKS * BLOCK_ROWS
        + offs_blocks[None, :] * BLOCK_ROWS
        + offs_rows[:, None]
    )
    mask = (offs_rows[:, None] < BLOCK_ROWS) & (offs_blocks[None, :] < NUM_BLOCKS)
    partial_m = tl.load(Partial_M + stats_offs, mask=mask, other=-1.0e20)
    partial_l = tl.load(Partial_L + stats_offs, mask=mask, other=0.0)
    m_i = tl.max(partial_m, axis=1)
    l_i = tl.sum(partial_l * tl.exp(partial_m - m_i[:, None]), axis=1)

    out_offs = cur_group * BLOCK_ROWS + offs_rows
    tl.store(Global_M + out_offs, m_i, mask=offs_rows < BLOCK_ROWS)
    tl.store(Global_L + out_offs, l_i, mask=offs_rows < BLOCK_ROWS)


@triton.jit
def _prefill_score_final_kernel(
    Q,
    K,
    Attn_Score,
    Global_M,
    Global_L,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_asb,
    stride_asl,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    candidate_start: tl.constexpr,
    num_recent_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_n_block = tl.program_id(1)
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
    context_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)
    score_q_len = tl.maximum(score_q_end - score_q_start, 1)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = score_q_start + (offs_rows % BLOCK_M)
    q_rel_pos = q_abs_pos - prompt_cache_len
    row_head_in_block = offs_rows // BLOCK_M
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    start_n = cur_n_block * BLOCK_N
    kv_pos = start_n + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - num_recent_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    qk = tl.dot(q, k) * sm_scale
    causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
    valid = q_row_valid[:, None] & kv_in_candidate[None, :] & causal_mask
    qk = tl.where(valid, qk, -1.0e20)

    stats_offs = cur_group * BLOCK_ROWS + offs_rows
    m_i = tl.load(Global_M + stats_offs)
    l_i = tl.load(Global_L + stats_offs)
    safe_l_i = tl.where(l_i > 0.0, l_i, 1.0)
    probs = tl.exp(qk - m_i[:, None]) / safe_l_i[:, None]
    probs = tl.where(valid, probs, 0.0)

    token_score = tl.zeros([BLOCK_N], dtype=tl.float32)
    for head_idx in tl.static_range(0, BLOCK_H):
        head_rows = row_head_in_block == head_idx
        head_score = tl.sum(tl.where(head_rows[:, None], probs, 0.0), axis=0) / (score_q_len * 1.0)
        token_score = tl.maximum(token_score, head_score)

    tl.atomic_max(
        Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
        token_score,
        mask=kv_in_candidate,
    )


@torch.no_grad()
def prefill_score_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_score: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_query_len: int,
    req_to_token_indexs: torch.Tensor,
    score_q_start: torch.Tensor,
    score_q_end: torch.Tensor,
    *,
    candidate_start: int = 0,
    num_recent_tokens: int = 0,
):
    head_dim = q.shape[-1]
    assert k.shape[-1] == head_dim
    assert q.dtype == k.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert attn_score.dim() == 2
    assert head_dim in {16, 32, 64, 128, 256}
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_head = k.shape[1]
    kv_group_num = head // kv_head
    if kv_group_num <= 0 or head % kv_head != 0:
        raise ValueError(f"num query heads must be divisible by num kv heads: q={head} k={kv_head}")
    max_score_len = int((score_q_end - score_q_start).max().item())
    if max_score_len <= 0:
        return
    block_m = max(16, triton.next_power_of_2(max_score_len))
    if block_m > 128:
        raise ValueError(f"prefill score query range is too large for this kernel: {max_score_len} > 128")

    candidate_ends = torch.clamp(b_seq_len - int(num_recent_tokens), min=int(candidate_start))
    max_candidate_end = int(candidate_ends.max().item()) if batch > 0 else 0
    if max_candidate_end <= 0:
        return

    block_n = 64 if head_dim >= 128 else 128
    candidate_blocks = triton.cdiv(max_candidate_end, block_n)
    if candidate_blocks <= 0:
        return

    # Keep the dot tile bounded. Common GQA (7 heads per KV, W=32) fits in one
    # head block; larger query windows or MQA split heads across multiple blocks.
    max_rows = 256
    block_h_limit = max(1, min(8, max_rows // block_m))
    block_h = min(triton.next_power_of_2(kv_group_num), block_h_limit)
    head_blocks = triton.cdiv(kv_group_num, block_h)
    block_rows = block_h * block_m
    group_count = batch * kv_head * head_blocks
    if group_count <= 0:
        return

    reduce_blocks = triton.next_power_of_2(candidate_blocks)
    reduce_rows = 16
    while reduce_rows > 1 and reduce_rows * reduce_blocks > 32768:
        reduce_rows //= 2

    partial_m = torch.empty((group_count, candidate_blocks, block_rows), device=q.device, dtype=torch.float32)
    partial_l = torch.empty_like(partial_m)
    global_m = torch.empty((group_count, block_rows), device=q.device, dtype=torch.float32)
    global_l = torch.empty_like(global_m)

    dot_warps = 8 if block_rows >= 128 or block_n >= 128 else 4
    _prefill_score_partial_stats_kernel[(group_count, candidate_blocks)](
        q,
        k,
        partial_m,
        partial_l,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        score_q_start,
        score_q_end,
        b_start_loc,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        H_PER_KV=kv_group_num,
        H_KV=kv_head,
        HEAD_BLOCKS=head_blocks,
        candidate_start=int(candidate_start),
        num_recent_tokens=int(num_recent_tokens),
        sm_scale=float(head_dim) ** -0.5,
        NUM_BLOCKS=candidate_blocks,
        BLOCK_H=block_h,
        BLOCK_ROWS=block_rows,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=dot_warps,
        num_stages=3,
    )
    reduce_grid = (group_count, triton.cdiv(block_rows, reduce_rows))
    _prefill_score_reduce_stats_kernel[reduce_grid](
        partial_m,
        partial_l,
        global_m,
        global_l,
        NUM_BLOCKS=candidate_blocks,
        BLOCK_ROWS=block_rows,
        REDUCE_BLOCKS=reduce_blocks,
        REDUCE_ROWS=reduce_rows,
        num_warps=8 if reduce_blocks >= 1024 else 4,
        num_stages=4,
    )
    _prefill_score_final_kernel[(group_count, candidate_blocks)](
        q,
        k,
        attn_score,
        global_m,
        global_l,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        score_q_start,
        score_q_end,
        b_start_loc,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        attn_score.stride(0),
        attn_score.stride(1),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        H_PER_KV=kv_group_num,
        H_KV=kv_head,
        HEAD_BLOCKS=head_blocks,
        candidate_start=int(candidate_start),
        num_recent_tokens=int(num_recent_tokens),
        sm_scale=float(head_dim) ** -0.5,
        NUM_BLOCKS=candidate_blocks,
        BLOCK_H=block_h,
        BLOCK_ROWS=block_rows,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=dot_warps,
        num_stages=3,
    )
