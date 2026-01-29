import triton
import triton.language as tl
import torch
import numpy as np

# --- 下面抄自 baselines/kivi/quant/new_pack.py 的 Triton Kernel ---

@triton.jit
def _pack_along_last_dim(
    bits: tl.constexpr,
    intensor_ptr,
    code_ptr,
    N,
    num_feats: tl.constexpr,
    feat_per_int: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N<N, other=0.)
        element = element << (i * bits)
        # Combine the value using bitwise OR
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)

@triton.jit
def _minmax_along_last_dim(
    x_ptr,
    mn_ptr, mx_ptr,
    total_elements: tl.constexpr,
    N: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    mask = offsets < total_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
    tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)

# --- 下面是封装逻辑，也保持与 KIVI 一致 ---

def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
    assert len(data.shape) == 4
    shape = data.shape
    B, nh, D, T = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # Quantize
    data = data.reshape(new_shape)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
    with torch.cuda.device(data.device):
        _minmax_along_last_dim[grid](data, mn, mx,
                             data.numel(), data.shape[0], num_groups, group_size,
                             BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
    scale = (mx - mn) / (2 ** bit - 1)
    data = data - mn.unsqueeze(-1)
    data.div_(scale.unsqueeze(-1) + 1e-6)
    data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
    data = data.view(-1, T)
    feat_per_int = 32 // bit
    packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
    code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
    with torch.cuda.device(data.device):
        _pack_along_last_dim[grid](bit, data, code, data.shape[0],
                                data.shape[1], feat_per_int,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                num_warps=8)
    return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

def unpack_tensor(v_code: torch.Tensor, bits: int, pack_dim: int):
    """
    KIVI 原版解包逻辑 (基于 PyTorch 向量化索引)
    """
    assert bits in [2,4,8]
    shape = v_code.shape
    feat_per_int = 32 // bits
    new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
    unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    num = 0xFF >> (8 - bits)
    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    if pack_dim == 2:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
    elif pack_dim == 3:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
    else:
        raise NotImplementedError
    return unpacked_v_code

def unpack_4bit_to_16bit(packed, scale, mn, group_size):
    """
    适配 ClusterCompressedKVCache 的包装函数
    """
    # 这里的 packed 形状通常是 (B, nh, num_imp, D//8)
    # 我们调用 KIVI 的 unpack_tensor，它在 dim=3 上打包
    unpacked_int4 = unpack_tensor(packed, bits=4, pack_dim=3)
    
    # 反量化
    # scale/mn 形状是 (B, nh, num_imp, 1) 或者 (B, nh, num_imp, num_groups)
    # 如果是 Per-token 量化，num_groups = 1
    if scale.shape[-1] == 1:
        return unpacked_int4.to(scale.dtype) * scale + mn
    else:
        # Group-wise 逻辑
        B, nh, num_imp, D = unpacked_int4.shape
        num_groups = D // group_size
        res = unpacked_int4.view(B, nh, num_imp, num_groups, group_size)
        res = res * scale.unsqueeze(-1) + mn.unsqueeze(-1)
        return res.view(B, nh, num_imp, D)