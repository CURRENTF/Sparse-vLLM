import torch
import triton
import triton.language as tl


def _is_tesla() -> bool:
    return torch.cuda.is_available() and "Tesla" in torch.cuda.get_device_name(0)


@triton.jit
def _fwd_kernel_with_v(
    Q_NOPE,
    Q_ROPE,
    K_NOPE,
    K_ROPE,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Kv_Start_Loc,
    B_Seqlen,
    b_prompt_cache_len,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_q_rope_d,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_k_rope_bs,
    stride_k_rope_h,
    stride_k_rope_d,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num
    cur_batch_in_q_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_kv_start_index = tl.load(B_Kv_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    off_q = (
        (cur_batch_in_q_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    off_q_rope = (
        (cur_batch_in_q_start_index + offs_m[:, None]) * stride_q_rope_bs
        + cur_head * stride_q_rope_h
        + offs_rope_d[None, :] * stride_q_rope_d
    )
    off_k = offs_n[None, :] * stride_k_bs + cur_kv_head * stride_k_h + offs_d[:, None] * stride_k_d
    off_k_rope = (
        offs_n[None, :] * stride_k_rope_bs
        + 0 * stride_k_rope_h
        + offs_rope_d[:, None] * stride_k_rope_d
    )
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    q = tl.load(Q_NOPE + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    q_rope = tl.load(Q_ROPE + off_q_rope, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K_NOPE + off_k
    k_rope_ptrs = K_ROPE + off_k_rope
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )
        k_rope = tl.load(
            k_rope_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_k_rope_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk += tl.dot(q_rope, k_rope)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float("-100000000.0"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        v = tl.load(
            v_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i_new
        m_i = m_i_new

    off_o = (
        (cur_batch_in_q_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd_with_v(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_kv_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_input_len: int,
    softmax_scale: float,
):
    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]
    assert q_nope_dim == k_nope.shape[-1]
    assert q_rope_dim == k_rope.shape[-1]
    assert q_nope_dim == v.shape[-1]
    assert q_rope_dim in {16, 32, 64, 128, 256}
    assert q_nope_dim in {16, 32, 64, 128, 256, 512}

    if q_nope_dim >= 512:
        block = 32 if _is_tesla() or torch.cuda.get_device_capability()[0] >= 9 else 64
    else:
        block = 128 if not _is_tesla() else 64

    if q_nope.dtype == torch.float32:
        block = block // 4

    batch = b_seq_len.shape[0]
    head = q_nope.shape[1]
    kv_group_num = q_nope.shape[1] // k_nope.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, block))
    num_warps = 4 if q_nope_dim <= 64 else 8

    _fwd_kernel_with_v[grid](
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        v,
        softmax_scale,
        o,
        b_start_loc,
        b_kv_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_rope.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(1),
        k_rope.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_M=block,
        BLOCK_DMODEL=q_nope_dim,
        BLOCK_ROPE_DMODEL=q_rope_dim,
        BLOCK_N=block,
        num_warps=num_warps,
        num_stages=1,
    )
