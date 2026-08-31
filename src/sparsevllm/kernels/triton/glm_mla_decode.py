from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _glm_mla_decode_rope_kernel(
    q_nope,
    q_rope,
    k_rope,
    positions,
    cos_sin_cache,
    output_q,
    output_k_rope,
    q_nope_stride_row,
    q_nope_stride_head,
    q_rope_stride_row,
    q_rope_stride_head,
    k_rope_stride_row,
    cache_stride_position,
    output_q_stride_row,
    output_q_stride_head,
    output_k_stride_row,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    nope_mask = offsets < NOPE_DIM
    rope_output_offsets = offsets - NOPE_DIM
    rope_mask = (offsets >= NOPE_DIM) & (rope_output_offsets < ROPE_DIM)
    half_rope: tl.constexpr = ROPE_DIM // 2
    pair_offsets = rope_output_offsets % half_rope
    even_offsets = 2 * pair_offsets
    odd_offsets = even_offsets + 1

    position = tl.load(positions + row).to(tl.int64)
    cache_base = position * cache_stride_position
    cos = tl.load(
        cos_sin_cache + cache_base + pair_offsets,
        mask=rope_mask,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache + cache_base + half_rope + pair_offsets,
        mask=rope_mask,
        other=0.0,
    )
    q_rope_base = row * q_rope_stride_row + head * q_rope_stride_head
    q_even = tl.load(
        q_rope + q_rope_base + even_offsets,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    q_odd = tl.load(
        q_rope + q_rope_base + odd_offsets,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    q_rotated = tl.where(
        rope_output_offsets < half_rope,
        q_even * cos - q_odd * sin,
        q_odd * cos + q_even * sin,
    )
    q_nope_values = tl.load(
        q_nope
        + row * q_nope_stride_row
        + head * q_nope_stride_head
        + offsets,
        mask=nope_mask,
        other=0.0,
    )
    tl.store(
        output_q
        + row * output_q_stride_row
        + head * output_q_stride_head
        + offsets,
        tl.where(nope_mask, q_nope_values, q_rotated),
        mask=nope_mask | rope_mask,
    )

    if head == 0:
        k_base = row * k_rope_stride_row
        k_even = tl.load(
            k_rope + k_base + even_offsets,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        k_odd = tl.load(
            k_rope + k_base + odd_offsets,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        k_rotated = tl.where(
            rope_output_offsets < half_rope,
            k_even * cos - k_odd * sin,
            k_odd * cos + k_even * sin,
        )
        tl.store(
            output_k_rope + row * output_k_stride_row + rope_output_offsets,
            k_rotated,
            mask=rope_mask,
        )


def fuse_glm_mla_decode_rope(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate GLM MLA Q/K RoPE and compose the contiguous decode query."""

    if q_nope.ndim != 3 or q_rope.ndim != 3 or k_rope.ndim != 2:
        raise ValueError("GLM MLA decode RoPE expects rank-3 Q and rank-2 K tensors.")
    batch_size, num_heads, nope_dim = map(int, q_nope.shape)
    rope_dim = int(q_rope.shape[-1])
    if tuple(q_rope.shape[:2]) != (batch_size, num_heads):
        raise ValueError("GLM MLA decode Q tensors must share row/head axes.")
    if tuple(k_rope.shape) != (batch_size, rope_dim):
        raise ValueError("GLM MLA decode K RoPE must be batch aligned with Q.")
    if positions.shape != (batch_size,):
        raise ValueError("GLM MLA decode positions must contain one value per row.")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(
            "GLM MLA decode RoPE dimension must be positive and even, "
            f"got {rope_dim}."
        )
    if cos_sin_cache.ndim != 3 or tuple(cos_sin_cache.shape[1:]) != (1, rope_dim):
        raise ValueError("GLM MLA cos/sin cache must have shape [positions, 1, rope_dim].")
    tensors = (q_rope, k_rope, positions, cos_sin_cache)
    if any(tensor.device != q_nope.device for tensor in tensors):
        raise ValueError("GLM MLA decode RoPE inputs must share one device.")
    if (
        q_nope.dtype != torch.bfloat16
        or q_rope.dtype != q_nope.dtype
        or k_rope.dtype != q_nope.dtype
    ):
        raise TypeError("GLM MLA decode RoPE requires BF16 Q/K tensors.")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError("GLM MLA cos/sin cache must use FP32.")
    if positions.dtype not in {torch.int32, torch.int64}:
        raise TypeError("GLM MLA decode positions must use int32 or int64.")
    if not q_nope.is_cuda:
        raise ValueError("GLM MLA decode RoPE requires CUDA tensors.")
    if any(
        tensor.stride(-1) != 1
        for tensor in (q_nope, q_rope, k_rope, cos_sin_cache)
    ):
        raise ValueError(
            "GLM MLA decode RoPE inputs must be contiguous in the last dimension."
        )

    output_q = torch.empty(
        (batch_size, num_heads, nope_dim + rope_dim),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    output_k_rope = torch.empty_like(k_rope)
    _glm_mla_decode_rope_kernel[(batch_size, num_heads)](
        q_nope,
        q_rope,
        k_rope,
        positions,
        cos_sin_cache,
        output_q,
        output_k_rope,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        k_rope.stride(0),
        cos_sin_cache.stride(0),
        output_q.stride(0),
        output_q.stride(1),
        output_k_rope.stride(0),
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_D=triton.next_power_of_2(nope_dim + rope_dim),
        num_warps=4,
        num_stages=1,
    )
    return output_q, output_k_rope


__all__ = ["fuse_glm_mla_decode_rope"]
