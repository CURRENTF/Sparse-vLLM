from functools import lru_cache
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sparsevllm.platforms import device_runtime

TESLA = "Tesla" in device_runtime.optional_device_name(0)
_HD256_BLOCK128_REQUIRED_SHARED_MEMORY = 163_840

_GROUPED_SCORE_PROFILED_DEVICES = frozenset({
    "NVIDIA H100 80GB HBM3",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4080 SUPER",
})
_GROUPED_SCORE_PROFILED_TOOLCHAIN = ("2.11.0+cu130", "13.0", "3.6.0")
_GROUPED_SCORE_COMMON_CASES = frozenset({
    (4096, 1),
    (4096, 8),
    (4096, 16),
    (4096, 32),
    (16384, 32),
})
_GROUPED_SCORE_PROFILED_CASES = {
    (28, 4, 64): _GROUPED_SCORE_COMMON_CASES
    | {(4096, 128), (4096, 256)},
    (28, 4, 128): _GROUPED_SCORE_COMMON_CASES | {(4096, 128)},
    (28, 4, 256): _GROUPED_SCORE_COMMON_CASES,
    (32, 8, 128): _GROUPED_SCORE_COMMON_CASES | {(4096, 128)},
    (32, 4, 128): _GROUPED_SCORE_COMMON_CASES | {(4096, 128)},
}


def _matches_grouped_score_performance_profile(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    batch_size: int,
    max_query_len: int,
    max_context_len: int | None,
) -> bool:
    """Match only the exact B1 shapes recorded by the grouped-score benchmark."""

    if (
        q.device.type != "cuda"
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or int(batch_size) != 1
        or max_context_len is None
    ):
        return False
    toolchain = (str(torch.__version__), str(torch.version.cuda), str(triton.__version__))
    if toolchain != _GROUPED_SCORE_PROFILED_TOOLCHAIN:
        return False
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if torch.cuda.get_device_name(device_index) not in _GROUPED_SCORE_PROFILED_DEVICES:
        return False
    shape = (int(q.shape[1]), int(k.shape[1]), int(q.shape[2]))
    runtime_case = (int(max_context_len), int(max_query_len))
    return runtime_case in _GROUPED_SCORE_PROFILED_CASES.get(shape, ())


def select_context_attention_launch_config(
    head_dim: int,
    *,
    max_shared_memory: int,
    is_tesla: bool = False,
) -> tuple[int, int, int, int]:
    """Return a resource-safe static launch specialization.

    The head-dim-256 BLOCK_M/BLOCK_N=128 specialization uses 163,840 bytes of
    shared memory with Triton 3.6. SM120 exposes only 101,376 bytes per block,
    while H100 can retain the larger tile.
    """

    if head_dim not in {16, 32, 64, 128, 256}:
        raise ValueError(f"Unsupported context-attention head_dim={head_dim}.")
    if max_shared_memory <= 0:
        raise ValueError(
            "Context-attention max shared memory must be positive, got "
            f"{max_shared_memory}."
        )
    resource_limited_hd256 = (
        head_dim == 256
        and max_shared_memory < _HD256_BLOCK128_REQUIRED_SHARED_MEMORY
    )
    block_m = 64 if is_tesla or resource_limited_hd256 else 128
    block_n = block_m
    num_warps = 4 if head_dim <= 64 else 8
    return block_m, block_n, num_warps, 1


@lru_cache(maxsize=None)
def _device_max_shared_memory(device_index: int) -> int:
    properties = triton.runtime.driver.active.utils.get_device_properties(
        int(device_index)
    )
    try:
        max_shared_memory = int(properties["max_shared_mem"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Triton device properties do not expose a valid max_shared_mem: "
            f"{properties!r}."
        ) from error
    return max_shared_memory


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out, Softmax_Lse,
    B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_lseh, stride_lset,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, STORE_LSE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    if STORE_LSE:
        lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
        tl.store(
            Softmax_Lse
            + cur_head * stride_lseh
            + (cur_batch_in_all_start_index + offs_m) * stride_lset,
            lse,
            mask=offs_m < cur_batch_seq_len,
        )


@triton.jit
def _fwd_kernel_with_score(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_asl,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        
        # 收集评分：使用原始点积 (Raw Logits)，且掩码位置设为 0 以便后续计算 Mean
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        score_to_collect = tl.where(mask, qk, 0.0)
        block_sum = tl.sum(score_to_collect, 0)
        tl.atomic_add(Attn_Score + cur_batch * stride_asb + cur_head * stride_ash + (start_n + offs_n) * stride_asl, 
                      block_sum, mask=(start_n + offs_n) < block_end_loc)

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_kernel_with_score_2d(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_asl,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        
        # Max across the complete query chunk and all heads. Query tiles and
        # heads race through atomic_max into one [batch, context] raw-QK row.
        block_max = tl.max(tl.where(mask, qk, -float("inf")), axis=0)
        tl.atomic_max(Attn_Score + cur_batch * stride_asb + (start_n + offs_n) * stride_asl, 
                      block_max, mask=(start_n + offs_n) < block_end_loc)

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_grouped_kernel_with_score_2d(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Attn_Score,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_asb,
    stride_asl,
    b_prompt_cache_len,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_kv_head_block = tl.program_id(1)
    head_block = batch_kv_head_block % HEAD_BLOCKS
    batch_kv_head = batch_kv_head_block // HEAD_BLOCKS
    kv_head = batch_kv_head % H_KV
    batch = batch_kv_head // H_KV

    q_start = tl.load(B_Start_Loc + batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + batch)
    context_len = tl.load(B_Seqlen + batch)
    query_len = context_len - prompt_cache_len
    request = tl.load(B_req_idx + batch)

    rows = tl.arange(0, BLOCK_ROWS)
    dims = tl.arange(0, BLOCK_DMODEL)
    positions = tl.arange(0, BLOCK_N)
    local_head = head_block * BLOCK_H + rows // BLOCK_M
    query_head = kv_head * H_PER_KV + local_head
    query_rel = query_block * BLOCK_M + rows % BLOCK_M
    query_abs = prompt_cache_len + query_rel
    query_valid = (local_head < H_PER_KV) & (query_rel < query_len)
    q_offsets = (
        (q_start + query_rel[:, None]) * stride_qbs
        + query_head[:, None] * stride_qh
        + dims[None, :] * stride_qd
    )
    q = tl.load(Q + q_offsets, mask=query_valid[:, None], other=0.0)

    max_logit = tl.full([BLOCK_ROWS], -float("inf"), tl.float32)
    exp_sum = tl.zeros([BLOCK_ROWS], tl.float32)
    accumulator = tl.zeros([BLOCK_ROWS, BLOCK_DMODEL], tl.float32)
    block_end = tl.minimum(
        context_len,
        prompt_cache_len + (query_block + 1) * BLOCK_M,
    )
    block_count = tl.where(
        query_block * BLOCK_M < query_len,
        tl.cdiv(block_end, BLOCK_N),
        0,
    )

    for key_block in range(0, block_count):
        key_positions = key_block * BLOCK_N + positions
        key_valid = key_positions < block_end
        slots = tl.load(
            Req_to_tokens
            + request * stride_req_to_tokens_b
            + key_positions * stride_req_to_tokens_s,
            mask=key_valid,
            other=0,
        )
        keys = tl.load(
            K
            + slots[None, :] * stride_kbs
            + kv_head * stride_kh
            + dims[:, None] * stride_kd,
            mask=key_valid[None, :],
            other=0.0,
        )
        raw_logits = tl.dot(q, keys)
        causal = query_abs[:, None] >= key_positions[None, :]
        valid = query_valid[:, None] & key_valid[None, :] & causal
        reduced_logits = tl.max(
            tl.where(valid, raw_logits, -float("inf")), axis=0
        )
        tl.atomic_max(
            Attn_Score
            + batch * stride_asb
            + key_positions * stride_asl,
            reduced_logits,
            mask=key_valid,
        )

        logits = tl.where(valid, raw_logits * sm_scale, -1.0e8)
        next_max = tl.maximum(max_logit, tl.max(logits, axis=1))
        old_scale = tl.math.exp2(max_logit - next_max)
        probabilities = tl.math.exp2(logits - next_max[:, None])
        exp_sum = exp_sum * old_scale + tl.sum(probabilities, axis=1)
        accumulator *= old_scale[:, None]
        values = tl.load(
            V
            + slots[:, None] * stride_vbs
            + kv_head * stride_vh
            + dims[None, :] * stride_vd,
            mask=key_valid[:, None],
            other=0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator
        )
        max_logit = next_max

    output_offsets = (
        (q_start + query_rel[:, None]) * stride_obs
        + query_head[:, None] * stride_oh
        + dims[None, :] * stride_od
    )
    safe_sum = tl.where(query_valid, exp_sum, 1.0)
    tl.store(
        Out + output_offsets,
        accumulator / safe_sum[:, None],
        mask=query_valid[:, None],
    )


@torch.no_grad()
def context_attention_fwd(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs,
    attn_score=None, softmax_lse=None, *, max_context_len=None,
    _force_grouped_score=None,
):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    
    # 补齐断言：安全防护
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1 and o.stride(-1) == 1
    if softmax_lse is not None:
        if attn_score is not None:
            raise ValueError("Triton prefill cannot return scores and softmax LSE together.")
        if tuple(softmax_lse.shape) != (int(q.shape[1]), int(q.shape[0])):
            raise ValueError(
                "Triton prefill softmax LSE must have shape "
                f"[query_heads, total_query_tokens], got {tuple(softmax_lse.shape)}."
            )
        if softmax_lse.dtype != torch.float32 or softmax_lse.device != q.device:
            raise TypeError(
                "Triton prefill softmax LSE must be FP32 on the Q device, got "
                f"{softmax_lse.dtype} on {softmax_lse.device}."
            )

    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    BLOCK_M, BLOCK_N, num_warps, num_stages = (
        select_context_attention_launch_config(
            Lk,
            max_shared_memory=_device_max_shared_memory(device_index),
            is_tesla=TESLA,
        )
    )
    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    if attn_score is None:
        lse_output = o if softmax_lse is None else softmax_lse
        _fwd_kernel[grid](
            q, k, v, sm_scale, o, lse_output,
            b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            0 if softmax_lse is None else softmax_lse.stride(0),
            0 if softmax_lse is None else softmax_lse.stride(1),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, STORE_LSE=softmax_lse is not None,
            BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    elif attn_score.dim() == 3:
        _fwd_kernel_with_score[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), attn_score.stride(1), attn_score.stride(2),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    else: # 2D version
        use_grouped_score = kv_group_num > 1 and (
            bool(_force_grouped_score)
            if _force_grouped_score is not None
            else _matches_grouped_score_performance_profile(
                q,
                k,
                batch_size=batch,
                max_query_len=max_input_len,
                max_context_len=max_context_len,
            )
        )
        if use_grouped_score:
            score_block_m = {64: 16, 128: 8, 256: 4}[Lk]
            max_block_h = 4 if Lk == 256 else 8
            score_block_h = min(
                max_block_h,
                triton.next_power_of_2(kv_group_num),
            )
            score_block_n = 64 if Lk == 256 else 128
            score_head_blocks = triton.cdiv(kv_group_num, score_block_h)
            score_block_rows = score_block_h * score_block_m
            score_grid = (
                triton.cdiv(max_input_len, score_block_m),
                batch * k.shape[1] * score_head_blocks,
            )
            _fwd_grouped_kernel_with_score_2d[score_grid](
                q,
                k,
                v,
                sm_scale,
                o,
                b_start_loc,
                b_seq_len,
                req_to_token_indexs,
                b_req_idx,
                attn_score,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                req_to_token_indexs.stride(0),
                req_to_token_indexs.stride(1),
                attn_score.stride(0),
                attn_score.stride(1),
                b_prompt_cache_len,
                H_PER_KV=kv_group_num,
                H_KV=k.shape[1],
                HEAD_BLOCKS=score_head_blocks,
                BLOCK_H=score_block_h,
                BLOCK_ROWS=score_block_rows,
                BLOCK_DMODEL=Lk,
                BLOCK_M=score_block_m,
                BLOCK_N=score_block_n,
                num_warps=4,
                num_stages=2,
            )
            return
        _fwd_kernel_with_score_2d[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), 0, attn_score.stride(1), # ash=0
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
