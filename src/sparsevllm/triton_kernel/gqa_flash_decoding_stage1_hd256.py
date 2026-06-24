import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_gqa_flash_decode_stage1_hd256(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Attn_Score,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_asbs,
    stride_ash,
    stride_asl,
    gqa_group_size,
    STORE_SCORE_3D: tl.constexpr,
    STORE_SCORE_2D: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    q = tl.load(Q + cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd)
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )

        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))

        if STORE_SCORE_3D:
            tl.store(
                Attn_Score
                + cur_batch * stride_asbs
                + cur_head * stride_ash
                + offs_n_new * stride_asl,
                att_value,
                mask=offs_n_new < cur_batch_end_index,
            )
        if STORE_SCORE_2D:
            tl.atomic_max(
                Attn_Score + cur_batch * stride_asbs + offs_n_new * stride_asl,
                att_value,
                mask=offs_n_new < cur_batch_end_index,
            )

        att_value *= sm_scale
        off_v = k_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v.to(tl.float32), axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d * stride_mid_od
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb
            + cur_head * stride_mid_o_eh
            + seq_start_block * stride_mid_o_es
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))


def _validate_inputs(q, k, v, req_to_tokens, b_req_idx, b_seqlen, max_len_in_batch):
    lq, lk, lv = q.shape[-1], k.shape[-1], v.shape[-1]
    if not (lq == lk == lv == 256):
        raise ValueError(f"Qwen3.5 hd256 decode expects q/k/v head_dim=256, got {lq}/{lk}/{lv}.")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"GQA head count mismatch: q_heads={q.shape[1]} kv_heads={k.shape[1]}.")
    if q.shape[1] <= k.shape[1]:
        raise ValueError(f"hd256 kernel expects GQA, got q_heads={q.shape[1]} kv_heads={k.shape[1]}.")
    if req_to_tokens.dim() != 2:
        raise ValueError(f"Req_to_tokens must be rank-2, got shape={tuple(req_to_tokens.shape)}.")
    if b_req_idx.dim() != 1 or b_seqlen.dim() != 1:
        raise ValueError("B_req_idx and B_Seqlen must be rank-1 tensors.")
    if int(max_len_in_batch) <= 0:
        raise ValueError(f"max_len_in_batch must be positive, got {max_len_in_batch}.")


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
):
    _validate_inputs(q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch)
    block_n = 16
    if int(block_seq) % block_n != 0:
        raise ValueError(f"block_seq must be divisible by {block_n}, got {block_seq}.")

    grid = (B_req_idx.shape[0], q.shape[1], triton.cdiv(max_len_in_batch, block_seq))
    _fwd_kernel_gqa_flash_decode_stage1_hd256[grid](
        q,
        k,
        v,
        1.0 / (q.shape[-1] ** 0.5),
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        mid_out,
        Req_to_tokens.stride(0),
        Req_to_tokens.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        0,
        0,
        0,
        q.shape[1] // k.shape[1],
        STORE_SCORE_3D=False,
        STORE_SCORE_2D=False,
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=256,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )


@torch.no_grad()
def flash_decode_stage1_with_score(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    attn_score,
    block_seq,
):
    _validate_inputs(q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch)
    block_n = 16
    if int(block_seq) % block_n != 0:
        raise ValueError(f"block_seq must be divisible by {block_n}, got {block_seq}.")
    if attn_score.dim() not in (2, 3):
        raise ValueError(f"attn_score must be rank-2 or rank-3, got shape={tuple(attn_score.shape)}.")

    store_score_3d = attn_score.dim() == 3
    store_score_2d = attn_score.dim() == 2
    stride_ash = attn_score.stride(1) if store_score_3d else 0
    stride_asl = attn_score.stride(2) if store_score_3d else attn_score.stride(1)

    grid = (B_req_idx.shape[0], q.shape[1], triton.cdiv(max_len_in_batch, block_seq))
    _fwd_kernel_gqa_flash_decode_stage1_hd256[grid](
        q,
        k,
        v,
        1.0 / (q.shape[-1] ** 0.5),
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        attn_score,
        Req_to_tokens.stride(0),
        Req_to_tokens.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        attn_score.stride(0),
        stride_ash,
        stride_asl,
        q.shape[1] // k.shape[1],
        STORE_SCORE_3D=store_score_3d,
        STORE_SCORE_2D=store_score_2d,
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=256,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )
