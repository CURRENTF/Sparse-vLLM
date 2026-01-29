"""
DeltaKV 专用 Triton 内核

包含以下优化操作:
1. batch_l2_distance_topk: 批量 L2 距离计算 + TopK 选择
2. batch_gather_mean: 批量 gather + mean 操作
3. batch_reconstruct: 批量重建操作
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_l2_distance_kernel(
    A,  # (B, N, D) - 待计算的 tokens
    B,  # (B, M, D) - 参考 centers
    Out,  # (B, N, M) - 输出距离矩阵
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stride_ab, stride_an, stride_ad,
    stride_bb, stride_bm, stride_bd,
    stride_ob, stride_on, stride_om,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    计算批量 L2 距离: Out[b, n, m] = ||A[b, n] - B[b, m]||^2
    使用分块计算: a_norm + b_norm - 2 * dot(a, b)
    """
    batch_id = tl.program_id(0)
    block_n = tl.program_id(1)
    block_m = tl.program_id(2)

    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_m = offs_m < M

    # 加载 A 块: (BLOCK_N, D)
    a_ptrs = A + batch_id * stride_ab + offs_n[:, None] * stride_an + offs_d[None, :]
    a = tl.load(a_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)

    # 加载 B 块: (BLOCK_M, D)
    b_ptrs = B + batch_id * stride_bb + offs_m[:, None] * stride_bm + offs_d[None, :]
    b = tl.load(b_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)

    # 计算 a_norm: (BLOCK_N,)
    a_norm = tl.sum(a * a, axis=1)

    # 计算 b_norm: (BLOCK_M,)
    b_norm = tl.sum(b * b, axis=1)

    # 计算 dot product: (BLOCK_N, BLOCK_M)
    dot = tl.dot(a, tl.trans(b))

    # L2 距离: a_norm + b_norm - 2 * dot
    dist = a_norm[:, None] + b_norm[None, :] - 2.0 * dot

    # 存储结果
    out_ptrs = Out + batch_id * stride_ob + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om
    tl.store(out_ptrs, dist, mask=mask_n[:, None] & mask_m[None, :])


@triton.jit
def _batch_gather_mean_kernel(
    Src,  # (num_centers, D) - 源数据
    Indices,  # (B, N, K) - 索引
    Out,  # (B, N, D) - 输出
    B_size: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_sb, stride_sd,
    stride_ib, stride_in, stride_ik,
    stride_ob, stride_on, stride_od,
    BLOCK_D: tl.constexpr,
):
    """
    批量 gather + mean: Out[b, n] = mean(Src[Indices[b, n, k]] for k in range(K))
    """
    batch_id = tl.program_id(0)
    n_id = tl.program_id(1)
    block_d = tl.program_id(2)

    offs_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # 累加 K 个 neighbors 的值
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for k in range(K):
        idx = tl.load(Indices + batch_id * stride_ib + n_id * stride_in + k * stride_ik)
        src_ptrs = Src + idx * stride_sb + offs_d * stride_sd
        val = tl.load(src_ptrs, mask=mask_d, other=0.0)
        acc += val

    # 计算均值
    mean_val = acc / K

    # 存储结果
    out_ptrs = Out + batch_id * stride_ob + n_id * stride_on + offs_d * stride_od
    tl.store(out_ptrs, mean_val, mask=mask_d)


@triton.jit
def _batch_indexed_add_kernel(
    Latent,  # (num_slots, latent_dim) - 压缩的隐变量
    RefKV,  # (num_centers, kv_dim) - 参考 KV
    FatherIndices,  # (num_slots, K) - 每个 slot 的 K 个父索引
    OutKV,  # (num_slots, kv_dim) - 输出重建的 KV
    UpWeight,  # (latent_dim, kv_dim) - 解压权重
    UpBias,  # (kv_dim,) - 解压偏置
    num_slots,
    latent_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    K: tl.constexpr,
    stride_lb, stride_ld,
    stride_rb, stride_rd,
    stride_fb, stride_fk,
    stride_ob, stride_od,
    stride_wl, stride_wd,
    BLOCK_D: tl.constexpr,
):
    """
    融合的重建操作:
    1. 从 FatherIndices gather RefKV 并求均值
    2. 解压 Latent
    3. 相加得到最终 KV
    """
    slot_id = tl.program_id(0)
    block_d = tl.program_id(1)

    if slot_id >= num_slots:
        return

    offs_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < kv_dim

    # Step 1: 计算父 KV 均值
    father_mean = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(K):
        father_idx = tl.load(FatherIndices + slot_id * stride_fb + k * stride_fk)
        ref_ptrs = RefKV + father_idx * stride_rb + offs_d * stride_rd
        ref_val = tl.load(ref_ptrs, mask=mask_d, other=0.0)
        father_mean += ref_val
    father_mean = father_mean / K

    # Step 2: 简化版线性解压 (完整 MLP 需要分开处理)
    # 这里只做 latent -> output 的线性变换部分
    # 完整的非线性解压仍需在 PyTorch 侧完成
    # TODO: 如果需要完整 fuse MLP，需要更复杂的实现

    # Step 3: 加载 Up(latent) 结果并相加
    # 注意: 这个 kernel 假设 Up(latent) 已经预计算
    latent_ptrs = Latent + slot_id * stride_lb + offs_d * stride_ld
    latent_val = tl.load(latent_ptrs, mask=mask_d, other=0.0)

    out_val = latent_val + father_mean

    # 存储结果
    out_ptrs = OutKV + slot_id * stride_ob + offs_d * stride_od
    tl.store(out_ptrs, out_val, mask=mask_d)


@torch.no_grad()
def batch_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算批量 L2 距离
    
    Args:
        a: (B, N, D) 待计算 tokens
        b: (B, M, D) 参考 centers
    
    Returns:
        dist: (B, N, M) L2 距离矩阵
    """
    B, N, D = a.shape
    _, M, _ = b.shape
    
    out = torch.empty((B, N, M), dtype=a.dtype, device=a.device)
    
    BLOCK_N = min(32, triton.next_power_of_2(N))
    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (B, triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    
    _batch_l2_distance_kernel[grid](
        a, b, out,
        N, M, D,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )
    
    return out


@torch.no_grad()
def batch_gather_mean(
    src: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    批量 gather + mean 操作
    
    Args:
        src: (num_centers, D) 源数据
        indices: (B, N, K) 索引
    
    Returns:
        out: (B, N, D) 输出
    """
    B, N, K = indices.shape
    D = src.shape[1]
    
    out = torch.empty((B, N, D), dtype=src.dtype, device=src.device)
    
    BLOCK_D = min(128, triton.next_power_of_2(D))
    
    grid = (B, N, triton.cdiv(D, BLOCK_D))
    
    _batch_gather_mean_kernel[grid](
        src, indices, out,
        B, N, K, D,
        src.stride(0), src.stride(1),
        indices.stride(0), indices.stride(1), indices.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=BLOCK_D,
    )
    
    return out


@torch.no_grad()
def batch_reconstruct_kv(
    latent_up: torch.Tensor,  # (num_slots, kv_dim) - 已解压的隐变量
    ref_kv: torch.Tensor,  # (num_centers, kv_dim) - 参考 KV
    father_indices: torch.Tensor,  # (num_slots, K) - 父索引
) -> torch.Tensor:
    """
    批量重建 KV: out = latent_up + mean(ref_kv[father_indices])
    
    Args:
        latent_up: 已解压的隐变量
        ref_kv: 参考 KV cache
        father_indices: 每个 slot 的 K 个父索引
    
    Returns:
        out: (num_slots, kv_dim) 重建的 KV
    """
    num_slots, kv_dim = latent_up.shape
    K = father_indices.shape[1]
    
    out = torch.empty_like(latent_up)
    
    BLOCK_D = min(128, triton.next_power_of_2(kv_dim))
    
    grid = (num_slots, triton.cdiv(kv_dim, BLOCK_D))
    
    _batch_indexed_add_kernel[grid](
        latent_up, ref_kv, father_indices, out,
        None, None,  # UpWeight, UpBias - 不使用
        num_slots,
        kv_dim, kv_dim, K,
        latent_up.stride(0), latent_up.stride(1),
        ref_kv.stride(0), ref_kv.stride(1),
        father_indices.stride(0), father_indices.stride(1),
        out.stride(0), out.stride(1),
        0, 0,  # weight strides - 不使用
        BLOCK_D=BLOCK_D,
    )
    
    return out


@triton.jit
def _deltakv_gather_kv_unrope_kernel(
    slots_ptr,  # (N,) int32
    pos_ptr,  # (N,) int32
    cos_sin_ptr,  # (max_pos, head_dim) where [0:HD2]=cos, [HD2:]=sin
    k_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    out_ptr,  # (N, 2*D) where D=num_kv_heads*head_dim; [0:D]=K_unrope_flat, [D:]=V_flat
    stride_cos_p,
    stride_cos_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_out_n,
    stride_out_d,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    slot = tl.load(slots_ptr + pid_n).to(tl.int32)
    pos = tl.load(pos_ptr + pid_n).to(tl.int32)

    offs = tl.arange(0, HD2)

    cos = tl.load(cos_sin_ptr + pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

    k_base = slot * stride_k_s + pid_h * stride_k_h
    y1 = tl.load(k_cache_ptr + k_base + offs * stride_k_d).to(tl.float32)
    y2 = tl.load(k_cache_ptr + k_base + (offs + HD2) * stride_k_d).to(tl.float32)
    x1 = y1 * cos + y2 * sin
    x2 = y2 * cos - y1 * sin

    v_base = slot * stride_v_s + pid_h * stride_v_h
    v1 = tl.load(v_cache_ptr + v_base + offs * stride_v_d).to(tl.float32)
    v2 = tl.load(v_cache_ptr + v_base + (offs + HD2) * stride_v_d).to(tl.float32)

    out_row = pid_n * stride_out_n
    out_k_base = out_row + (pid_h * HEAD_DIM) * stride_out_d
    tl.store(out_ptr + out_k_base + offs * stride_out_d, x1)
    tl.store(out_ptr + out_k_base + (offs + HD2) * stride_out_d, x2)

    out_v_base = out_row + (D + pid_h * HEAD_DIM) * stride_out_d
    tl.store(out_ptr + out_v_base + offs * stride_out_d, v1)
    tl.store(out_ptr + out_v_base + (offs + HD2) * stride_out_d, v2)


@torch.no_grad()
def deltakv_gather_kv_unrope(
    *,
    slots: torch.Tensor,  # (N,) int32
    pos: torch.Tensor,  # (N,) int32
    cos_sin: torch.Tensor,  # (max_pos, head_dim) float/bf16
    k_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
) -> torch.Tensor:
    assert slots.is_cuda and pos.is_cuda and cos_sin.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda
    assert slots.dim() == 1 and pos.dim() == 1
    assert int(slots.numel()) == int(pos.numel())

    num_slots = int(slots.numel())
    num_kv_heads = int(k_cache.shape[1])
    head_dim = int(k_cache.shape[2])
    assert head_dim % 2 == 0
    d = num_kv_heads * head_dim

    out = torch.empty((num_slots, 2 * d), device=k_cache.device, dtype=k_cache.dtype)
    if num_slots == 0:
        return out

    grid = (num_slots, num_kv_heads)
    _deltakv_gather_kv_unrope_kernel[grid](
        slots,
        pos,
        cos_sin,
        k_cache,
        v_cache,
        out,
        cos_sin.stride(0),
        cos_sin.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        D=d,
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        num_warps=4,
    )
    return out


@triton.jit
def _deltakv_gather_kv_unrope_grouped_heads_kernel(
    slots_ptr,  # (N,) int32
    pos_ptr,  # (N,) int32
    cos_sin_ptr,  # (max_pos, head_dim) where [0:HD2]=cos, [HD2:]=sin
    k_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    out_ptr,  # (N, 2*D) where D=num_kv_heads*head_dim; [0:D]=K_unrope_flat, [D:]=V_flat
    stride_cos_p,
    stride_cos_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_out_n,
    stride_out_d,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hb = tl.program_id(1)

    slot = tl.load(slots_ptr + pid_n).to(tl.int32)
    pos = tl.load(pos_ptr + pid_n).to(tl.int32)

    offs = tl.arange(0, HD2)
    cos = tl.load(cos_sin_ptr + pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

    head_ids = pid_hb * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
    head_mask = head_ids < NUM_KV_HEADS

    # Load K (roped) and de-RoPE it.
    k_base = slot * stride_k_s + head_ids[:, None] * stride_k_h + offs[None, :] * stride_k_d
    y1 = tl.load(k_cache_ptr + k_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    y2 = tl.load(k_cache_ptr + k_base + (HD2 * stride_k_d), mask=head_mask[:, None], other=0.0).to(tl.float32)
    x1 = y1 * cos[None, :] + y2 * sin[None, :]
    x2 = y2 * cos[None, :] - y1 * sin[None, :]

    # Load V.
    v_base = slot * stride_v_s + head_ids[:, None] * stride_v_h + offs[None, :] * stride_v_d
    v1 = tl.load(v_cache_ptr + v_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    v2 = tl.load(v_cache_ptr + v_base + (HD2 * stride_v_d), mask=head_mask[:, None], other=0.0).to(tl.float32)

    # Store flatten [K_unrope (D)] + [V (D)].
    out_row = pid_n * stride_out_n
    out_k_base = out_row + (head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_out_d
    tl.store(out_ptr + out_k_base, x1, mask=head_mask[:, None])
    tl.store(out_ptr + out_k_base + (HD2 * stride_out_d), x2, mask=head_mask[:, None])

    out_v_base = out_row + (D + head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_out_d
    tl.store(out_ptr + out_v_base, v1, mask=head_mask[:, None])
    tl.store(out_ptr + out_v_base + (HD2 * stride_out_d), v2, mask=head_mask[:, None])


@torch.no_grad()
def deltakv_gather_kv_unrope_grouped_heads(
    *,
    slots: torch.Tensor,  # (N,) int32
    pos: torch.Tensor,  # (N,) int32
    cos_sin: torch.Tensor,  # (max_pos, head_dim) float/bf16
    k_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    heads_per_program: int = 4,
) -> torch.Tensor:
    assert slots.is_cuda and pos.is_cuda and cos_sin.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda
    assert slots.dim() == 1 and pos.dim() == 1
    assert int(slots.numel()) == int(pos.numel())

    num_slots = int(slots.numel())
    num_kv_heads = int(k_cache.shape[1])
    head_dim = int(k_cache.shape[2])
    assert head_dim % 2 == 0
    d = num_kv_heads * head_dim

    out = torch.empty((num_slots, 2 * d), device=k_cache.device, dtype=k_cache.dtype)
    if num_slots == 0:
        return out

    heads_per_program = int(heads_per_program)
    if heads_per_program <= 0:
        raise ValueError("heads_per_program must be a positive integer.")
    if heads_per_program == 1:
        return deltakv_gather_kv_unrope(slots=slots, pos=pos, cos_sin=cos_sin, k_cache=k_cache, v_cache=v_cache)

    grid = (num_slots, triton.cdiv(num_kv_heads, heads_per_program))
    _deltakv_gather_kv_unrope_grouped_heads_kernel[grid](
        slots,
        pos,
        cos_sin,
        k_cache,
        v_cache,
        out,
        cos_sin.stride(0),
        cos_sin.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        D=d,
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        NUM_KV_HEADS=num_kv_heads,
        HEADS_PER_PROG=heads_per_program,
        num_warps=4,
    )
    return out


@triton.jit
def _deltakv_reconstruct_writeback_kernel(
    kv_delta_ptr,  # (N, 2*D) where D = num_kv_heads*head_dim, in de-RoPE space for K
    father_slots_ptr,  # (N, K)
    slot_to_pos_ptr,  # (num_slots,)
    out_slots_ptr,  # (N,)
    out_pos_ptr,  # (N,)
    cos_sin_ptr,  # (max_pos, head_dim)
    k_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    stride_delta_n, stride_delta_d,
    stride_father_n, stride_father_k,
    stride_cos_p, stride_cos_d,
    stride_k_s, stride_k_h, stride_k_d,
    stride_v_s, stride_v_h, stride_v_d,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # token id
    pid_h = tl.program_id(1)  # kv head id

    out_slot = tl.load(out_slots_ptr + pid_n).to(tl.int32)
    out_pos = tl.load(out_pos_ptr + pid_n).to(tl.int32)

    offs = tl.arange(0, HD2)

    # cos/sin for output position
    cos_sin_out = tl.load(cos_sin_ptr + out_pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
    cos_out = cos_sin_out
    sin_out = tl.load(cos_sin_ptr + out_pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

    # Accumulate mean of fathers in de-RoPE space.
    acc_k1 = tl.zeros([HD2], dtype=tl.float32)
    acc_k2 = tl.zeros([HD2], dtype=tl.float32)
    acc_v1 = tl.zeros([HD2], dtype=tl.float32)
    acc_v2 = tl.zeros([HD2], dtype=tl.float32)

    for kk in tl.static_range(K):
        father_slot = tl.load(
            father_slots_ptr + pid_n * stride_father_n + kk * stride_father_k
        ).to(tl.int32)
        father_pos = tl.load(slot_to_pos_ptr + father_slot).to(tl.int32)

        cos_sin_f = tl.load(cos_sin_ptr + father_pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
        cos_f = cos_sin_f
        sin_f = tl.load(cos_sin_ptr + father_pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

        # Load father K (roped) and de-RoPE it.
        k_base = father_slot * stride_k_s + pid_h * stride_k_h
        y1 = tl.load(k_cache_ptr + k_base + offs * stride_k_d).to(tl.float32)
        y2 = tl.load(k_cache_ptr + k_base + (offs + HD2) * stride_k_d).to(tl.float32)
        x1 = y1 * cos_f + y2 * sin_f
        x2 = y2 * cos_f - y1 * sin_f
        acc_k1 += x1
        acc_k2 += x2

        # Load father V.
        v_base = father_slot * stride_v_s + pid_h * stride_v_h
        v1 = tl.load(v_cache_ptr + v_base + offs * stride_v_d).to(tl.float32)
        v2 = tl.load(v_cache_ptr + v_base + (offs + HD2) * stride_v_d).to(tl.float32)
        acc_v1 += v1
        acc_v2 += v2

    inv_k = 1.0 / K
    mean_k1 = acc_k1 * inv_k
    mean_k2 = acc_k2 * inv_k
    mean_v1 = acc_v1 * inv_k
    mean_v2 = acc_v2 * inv_k

    # Load delta (de-RoPE space) for this head.
    # Layout: [K_unrope_flat (D)] + [V_flat (D)].
    delta_k_base = pid_n * stride_delta_n + (pid_h * HEAD_DIM) * stride_delta_d
    delta_k1 = tl.load(kv_delta_ptr + delta_k_base + offs * stride_delta_d).to(tl.float32)
    delta_k2 = tl.load(kv_delta_ptr + delta_k_base + (offs + HD2) * stride_delta_d).to(tl.float32)

    delta_v_base = pid_n * stride_delta_n + (D + pid_h * HEAD_DIM) * stride_delta_d
    delta_v1 = tl.load(kv_delta_ptr + delta_v_base + offs * stride_delta_d).to(tl.float32)
    delta_v2 = tl.load(kv_delta_ptr + delta_v_base + (offs + HD2) * stride_delta_d).to(tl.float32)

    k1 = delta_k1 + mean_k1
    k2 = delta_k2 + mean_k2
    v1 = delta_v1 + mean_v1
    v2 = delta_v2 + mean_v2

    # Re-RoPE K to its position.
    out_y1 = k1 * cos_out - k2 * sin_out
    out_y2 = k2 * cos_out + k1 * sin_out

    # Write back into cache at out_slot.
    out_k_base = out_slot * stride_k_s + pid_h * stride_k_h
    tl.store(k_cache_ptr + out_k_base + offs * stride_k_d, out_y1)
    tl.store(k_cache_ptr + out_k_base + (offs + HD2) * stride_k_d, out_y2)

    out_v_base = out_slot * stride_v_s + pid_h * stride_v_h
    tl.store(v_cache_ptr + out_v_base + offs * stride_v_d, v1)
    tl.store(v_cache_ptr + out_v_base + (offs + HD2) * stride_v_d, v2)


@torch.no_grad()
def deltakv_reconstruct_writeback(
    kv_delta: torch.Tensor,  # (N, 2*D) in de-RoPE space for K
    father_slots: torch.Tensor,  # (N, K) int32
    slot_to_pos: torch.Tensor,  # (num_slots,) int32
    out_slots: torch.Tensor,  # (N,) int32
    out_pos: torch.Tensor,  # (N,) int32
    cos_sin: torch.Tensor,  # (max_pos, head_dim) float/bf16
    k_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
):
    assert kv_delta.is_cuda and father_slots.is_cuda and slot_to_pos.is_cuda and out_slots.is_cuda and out_pos.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda and cos_sin.is_cuda
    assert father_slots.dim() == 2
    assert kv_delta.dim() == 2

    N = kv_delta.shape[0]
    K = father_slots.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2]
    assert head_dim % 2 == 0
    D = num_kv_heads * head_dim
    assert kv_delta.shape[1] == 2 * D

    # Use a 2D grid: (token, head).
    grid = (N, num_kv_heads)
    _deltakv_reconstruct_writeback_kernel[grid](
        kv_delta,
        father_slots,
        slot_to_pos,
        out_slots,
        out_pos,
        cos_sin,
        k_cache,
        v_cache,
        kv_delta.stride(0), kv_delta.stride(1),
        father_slots.stride(0), father_slots.stride(1),
        cos_sin.stride(0), cos_sin.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        D=D,
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        NUM_KV_HEADS=num_kv_heads,
        K=K,
        num_warps=4,
    )


@triton.jit
def _deltakv_reconstruct_writeback_grouped_heads_kernel(
    kv_delta_ptr,  # (N, 2*D) where D = num_kv_heads*head_dim, in de-RoPE space for K
    father_slots_ptr,  # (N, K)
    slot_to_pos_ptr,  # (num_slots,)
    out_slots_ptr,  # (N,)
    out_pos_ptr,  # (N,)
    cos_sin_ptr,  # (max_pos, head_dim)
    k_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    v_cache_ptr,  # (num_slots, num_kv_heads, head_dim)
    stride_delta_n, stride_delta_d,
    stride_father_n, stride_father_k,
    stride_cos_p, stride_cos_d,
    stride_k_s, stride_k_h, stride_k_d,
    stride_v_s, stride_v_h, stride_v_d,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HD2: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    K: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
):
    pid_n = tl.program_id(0)  # token id
    pid_hb = tl.program_id(1)  # head block id

    out_slot = tl.load(out_slots_ptr + pid_n).to(tl.int32)
    out_pos = tl.load(out_pos_ptr + pid_n).to(tl.int32)

    head_ids = pid_hb * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
    head_mask = head_ids < NUM_KV_HEADS

    offs = tl.arange(0, HD2)

    # cos/sin for output position
    cos_out = tl.load(cos_sin_ptr + out_pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
    sin_out = tl.load(cos_sin_ptr + out_pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

    # Accumulate mean of fathers in de-RoPE space.
    acc_k1 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)
    acc_k2 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)
    acc_v1 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)
    acc_v2 = tl.zeros([HEADS_PER_PROG, HD2], dtype=tl.float32)

    for kk in tl.static_range(K):
        father_slot = tl.load(father_slots_ptr + pid_n * stride_father_n + kk * stride_father_k).to(tl.int32)
        father_pos = tl.load(slot_to_pos_ptr + father_slot).to(tl.int32)

        cos_f = tl.load(cos_sin_ptr + father_pos * stride_cos_p + offs * stride_cos_d).to(tl.float32)
        sin_f = tl.load(cos_sin_ptr + father_pos * stride_cos_p + (offs + HD2) * stride_cos_d).to(tl.float32)

        # Load father K (roped) and de-RoPE it.
        k_base = father_slot * stride_k_s + head_ids[:, None] * stride_k_h + offs[None, :] * stride_k_d
        y1 = tl.load(k_cache_ptr + k_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
        y2 = tl.load(k_cache_ptr + k_base + (HD2 * stride_k_d), mask=head_mask[:, None], other=0.0).to(tl.float32)
        x1 = y1 * cos_f[None, :] + y2 * sin_f[None, :]
        x2 = y2 * cos_f[None, :] - y1 * sin_f[None, :]
        acc_k1 += x1
        acc_k2 += x2

        # Load father V.
        v_base = father_slot * stride_v_s + head_ids[:, None] * stride_v_h + offs[None, :] * stride_v_d
        fv1 = tl.load(v_cache_ptr + v_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
        fv2 = tl.load(v_cache_ptr + v_base + (HD2 * stride_v_d), mask=head_mask[:, None], other=0.0).to(tl.float32)
        acc_v1 += fv1
        acc_v2 += fv2

    inv_k = 1.0 / K
    mean_k1 = acc_k1 * inv_k
    mean_k2 = acc_k2 * inv_k
    mean_v1 = acc_v1 * inv_k
    mean_v2 = acc_v2 * inv_k

    # Load delta (de-RoPE space) for this head block.
    delta_k_base = pid_n * stride_delta_n + (head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_delta_d
    delta_k1 = tl.load(kv_delta_ptr + delta_k_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    delta_k2 = tl.load(kv_delta_ptr + delta_k_base + (HD2 * stride_delta_d), mask=head_mask[:, None], other=0.0).to(tl.float32)

    delta_v_base = pid_n * stride_delta_n + (D + head_ids[:, None] * HEAD_DIM + offs[None, :]) * stride_delta_d
    delta_v1 = tl.load(kv_delta_ptr + delta_v_base, mask=head_mask[:, None], other=0.0).to(tl.float32)
    delta_v2 = tl.load(kv_delta_ptr + delta_v_base + (HD2 * stride_delta_d), mask=head_mask[:, None], other=0.0).to(tl.float32)

    k1 = delta_k1 + mean_k1
    k2 = delta_k2 + mean_k2
    v1 = delta_v1 + mean_v1
    v2 = delta_v2 + mean_v2

    # Re-RoPE K to its position.
    out_y1 = k1 * cos_out[None, :] - k2 * sin_out[None, :]
    out_y2 = k2 * cos_out[None, :] + k1 * sin_out[None, :]

    # Write back into cache at out_slot.
    out_k_base = out_slot * stride_k_s + head_ids[:, None] * stride_k_h + offs[None, :] * stride_k_d
    tl.store(k_cache_ptr + out_k_base, out_y1, mask=head_mask[:, None])
    tl.store(k_cache_ptr + out_k_base + (HD2 * stride_k_d), out_y2, mask=head_mask[:, None])

    out_v_base = out_slot * stride_v_s + head_ids[:, None] * stride_v_h + offs[None, :] * stride_v_d
    tl.store(v_cache_ptr + out_v_base, v1, mask=head_mask[:, None])
    tl.store(v_cache_ptr + out_v_base + (HD2 * stride_v_d), v2, mask=head_mask[:, None])


@torch.no_grad()
def deltakv_reconstruct_writeback_grouped_heads(
    kv_delta: torch.Tensor,  # (N, 2*D) in de-RoPE space for K
    father_slots: torch.Tensor,  # (N, K) int32
    slot_to_pos: torch.Tensor,  # (num_slots,) int32
    out_slots: torch.Tensor,  # (N,) int32
    out_pos: torch.Tensor,  # (N,) int32
    cos_sin: torch.Tensor,  # (max_pos, head_dim) float/bf16
    k_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_slots, num_kv_heads, head_dim)
    *,
    heads_per_program: int = 4,
):
    assert kv_delta.is_cuda and father_slots.is_cuda and slot_to_pos.is_cuda and out_slots.is_cuda and out_pos.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda and cos_sin.is_cuda
    assert father_slots.dim() == 2
    assert kv_delta.dim() == 2

    N = kv_delta.shape[0]
    K = father_slots.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2]
    assert head_dim % 2 == 0
    D = num_kv_heads * head_dim
    assert kv_delta.shape[1] == 2 * D

    heads_per_program = int(heads_per_program)
    if heads_per_program <= 0:
        raise ValueError("heads_per_program must be a positive integer.")
    if heads_per_program == 1:
        return deltakv_reconstruct_writeback(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    grid = (N, triton.cdiv(num_kv_heads, heads_per_program))
    _deltakv_reconstruct_writeback_grouped_heads_kernel[grid](
        kv_delta,
        father_slots,
        slot_to_pos,
        out_slots,
        out_pos,
        cos_sin,
        k_cache,
        v_cache,
        kv_delta.stride(0), kv_delta.stride(1),
        father_slots.stride(0), father_slots.stride(1),
        cos_sin.stride(0), cos_sin.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        D=D,
        HEAD_DIM=head_dim,
        HD2=head_dim // 2,
        NUM_KV_HEADS=num_kv_heads,
        K=K,
        HEADS_PER_PROG=heads_per_program,
        num_warps=4,
    )


@triton.jit
def _deltakv_l2_topk_block_kernel(
    A_ptr,  # (N, D) tokens
    B_ptr,  # (M, D) centers
    out_scores_ptr,  # (NB, MB, BN, K) low-precision scores
    out_indices_ptr,  # (NB, MB, BN, K) int32 global center indices
    N,
    M,
    m0,  # number of existing centers
    cluster_step,  # int32
    stride_an,
    stride_ad,
    stride_bm,
    stride_bd,
    stride_s_nb,
    stride_s_mb,
    stride_s_bn,
    stride_s_k,
    stride_i_nb,
    stride_i_mb,
    stride_i_bn,
    stride_i_k,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)  # token block
    pid_m = tl.program_id(1)  # center block

    n_local = tl.arange(0, BLOCK_N)
    n_offs = pid_n * BLOCK_N + n_local
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_mask = n_offs < N
    m_mask = m_offs < M

    # GEMM tile: (BLOCK_N, BLOCK_M) = (BLOCK_N, D) x (BLOCK_M, D)^T
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    b_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for d_start in tl.static_range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        a_ptrs = A_ptr + n_offs[:, None] * stride_an + d_offs[None, :] * stride_ad
        b_ptrs = B_ptr + m_offs[:, None] * stride_bm + d_offs[None, :] * stride_bd

        a = tl.load(a_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        acc += tl.dot(a, tl.trans(b))
        bf = b.to(tl.float32)
        b_norm += tl.sum(bf * bf, axis=1)

    scores = acc * 2.0 - b_norm[None, :]

    # Causal mask for new centers: (m - m0) * cluster_step <= n
    # Existing centers (m < m0) are always visible.
    g = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
    # Keep scalars in int32 for indexing/masking math.
    m0_i32 = m0.to(tl.int32)
    cs_i32 = cluster_step.to(tl.int32)
    new_pos = (g - m0_i32) * cs_i32  # negative for existing centers
    allow = new_pos[None, :] <= n_offs[:, None].to(tl.int32)

    valid = n_mask[:, None] & m_mask[None, :] & allow
    scores = tl.where(valid, scores, -float("inf"))

    # Top-k within this center block (over BLOCK_M), store directly to output.
    idxs = tl.arange(0, BLOCK_M)[None, :].to(tl.int32)
    s_base = (
        out_scores_ptr
        + pid_n * stride_s_nb
        + pid_m * stride_s_mb
        + n_local * stride_s_bn
    )
    i_base = (
        out_indices_ptr
        + pid_n * stride_i_nb
        + pid_m * stride_i_mb
        + n_local * stride_i_bn
    )
    for kk in tl.static_range(K):
        maxv = tl.max(scores, axis=1)
        is_max = scores == maxv[:, None]
        arg = tl.min(tl.where(is_max, idxs, BLOCK_M), axis=1).to(tl.int32)
        tl.store(s_base + kk * stride_s_k, maxv, mask=n_mask)
        tl.store(i_base + kk * stride_i_k, pid_m * BLOCK_M + arg, mask=n_mask)
        scores = tl.where(idxs == arg[:, None], -float("inf"), scores)


@torch.no_grad()
def deltakv_l2_topk_blockwise(
    *,
    tokens: torch.Tensor,  # (N, D) bf16/fp16
    centers: torch.Tensor,  # (M, D) bf16/fp16
    m0: int,
    cluster_step: int,
    k: int,
    block_n: int = 16,
    block_m: int = 64,
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k L2-equivalent scores in a blockwise fused way.

    Returns:
      partial_scores: (NB, MB, BN, K) low-precision
      partial_indices: (NB, MB, BN, K) int32 global center indices
    """
    assert tokens.is_cuda and centers.is_cuda
    assert tokens.dim() == 2 and centers.dim() == 2
    N, D = tokens.shape
    M, D2 = centers.shape
    assert D == D2
    assert k >= 1
    assert block_m >= k

    NB = triton.cdiv(N, block_n)
    MB = triton.cdiv(M, block_m)

    partial_scores = torch.empty((NB, MB, block_n, k), device=tokens.device, dtype=tokens.dtype)
    partial_indices = torch.empty((NB, MB, block_n, k), device=tokens.device, dtype=torch.int32)

    grid = (NB, MB)
    _deltakv_l2_topk_block_kernel[grid](
        tokens,
        centers,
        partial_scores,
        partial_indices,
        N,
        M,
        m0,
        cluster_step,
        tokens.stride(0),
        tokens.stride(1),
        centers.stride(0),
        centers.stride(1),
        partial_scores.stride(0),
        partial_scores.stride(1),
        partial_scores.stride(2),
        partial_scores.stride(3),
        partial_indices.stride(0),
        partial_indices.stride(1),
        partial_indices.stride(2),
        partial_indices.stride(3),
        D=D,
        K=k,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return partial_scores, partial_indices
