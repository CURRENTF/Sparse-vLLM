# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch

import triton
import triton.language as tl
from sparsevllm.kernels.triton.qwen3_5.autotuner import autotune

BT_LIST = [8, 16, 32, 64, 128]

USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit(do_not_specialize=["NB"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


@triton.jit
def fused_qk_l2norm_fwd_kernel(
    Q,
    K,
    Q_Out,
    K_Out,
    eps,
    NUM_ROWS,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dims = tl.arange(0, BLOCK_DIM)
    tokens = rows // NUM_HEADS
    heads = rows % NUM_HEADS
    mask = (rows[:, None] < NUM_ROWS) & (dims[None, :] < HEAD_DIM)
    output_offsets = rows[:, None] * HEAD_DIM + dims[None, :]

    q = tl.load(
        Q
        + tokens[:, None] * stride_qt
        + heads[:, None] * stride_qh
        + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q_rstd = tl.rsqrt(tl.sum(q * q, axis=1) + eps)
    tl.store(Q_Out + output_offsets, q * q_rstd[:, None], mask=mask)

    k = tl.load(
        K
        + tokens[:, None] * stride_kt
        + heads[:, None] * stride_kh
        + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=1) + eps)
    tl.store(K_Out + output_offsets, k * k_rstd[:, None], mask=mask)


def _get_l2norm_kernel1_configs():
    return [{"num_warps": num_warps} for num_warps in [1, 2, 4, 8, 16, 32]]


def _get_l2norm_kernel1_static_key(x):
    D = x.shape[-1]
    return {"D": D}


def _get_l2norm_kernel1_run_key(x):
    return x.shape[0]  # T


@autotune(
    kernel_name="l2norm_fwd_kernel1",
    configs_gen_func=_get_l2norm_kernel1_configs,
    static_key_func=_get_l2norm_kernel1_static_key,
    run_key_func=_get_l2norm_kernel1_run_key,
)
def _l2norm_fwd_kernel1_wrapper(x, y, eps, D, BD, run_config=None):
    if run_config is None:
        run_config = {"num_warps": 4}

    num_warps = run_config.get("num_warps", 4)
    T = x.shape[0]

    l2norm_fwd_kernel1[(T,)](x, y, eps=eps, D=D, BD=BD, num_warps=num_warps)


def _get_l2norm_kernel_configs():
    return [{"BT": BT, "num_warps": num_warps} for num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST]


def _get_l2norm_kernel_static_key(x):
    D = x.shape[-1]
    return {"D": D}


def _get_l2norm_kernel_run_key(x):
    return x.shape[0]  # T


@autotune(
    kernel_name="l2norm_fwd_kernel",
    configs_gen_func=_get_l2norm_kernel_configs,
    static_key_func=_get_l2norm_kernel_static_key,
    run_key_func=_get_l2norm_kernel_run_key,
)
def _l2norm_fwd_kernel_wrapper(x, y, eps, T, D, BD, NB, run_config=None):
    if run_config is None:
        run_config = {"BT": 32, "num_warps": 4}

    BT = run_config.get("BT", 32)
    num_warps = run_config.get("num_warps", 4)

    grid = (triton.cdiv(T, BT),)
    l2norm_fwd_kernel[grid](x, y, eps, NB=NB, T=T, D=D, BT=BT, BD=BD, num_warps=num_warps)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if not USE_DEFAULT_FLA_NORM:
        MBLOCK = 32
        # M, N = x.shape
        l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
            x,
            y,
            eps,
            T,
            D,
            MBLOCK,
        )
    else:
        if D <= 512:
            NB = triton.cdiv(T, 2048)
            _l2norm_fwd_kernel_wrapper(x, y, eps, T, D, BD, NB)
        else:
            _l2norm_fwd_kernel1_wrapper(x, y, eps, D, BD)

    return y.view(x_shape_og)


def fused_qk_l2norm_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape:
        raise ValueError(
            "fused Q/K L2Norm expects matching [1, tokens, heads, dim] tensors, "
            f"got {tuple(q.shape)} and {tuple(k.shape)}."
        )
    if q.dtype != k.dtype or q.device != k.device:
        raise TypeError(
            "fused Q/K L2Norm requires matching dtypes and devices, got "
            f"{q.dtype}/{k.dtype} on {q.device}/{k.device}."
        )
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("fused Q/K L2Norm requires a contiguous head dimension.")
    tokens, heads, head_dim = map(int, q.shape[1:])
    if tokens == 0:
        empty = torch.empty(
            (tokens, heads, head_dim), dtype=q.dtype, device=q.device
        )
        return empty, torch.empty_like(empty)
    block_dim = triton.next_power_of_2(head_dim)
    max_fused_size = 65536 // q.element_size()
    if head_dim > min(max_fused_size, block_dim):
        raise RuntimeError("fused Q/K L2Norm does not support a head >= 64KB.")
    block_rows = 16 if block_dim >= 128 else 32
    q_out = torch.empty(
        (tokens, heads, head_dim), dtype=q.dtype, device=q.device
    )
    k_out = torch.empty_like(q_out)
    num_rows = tokens * heads
    fused_qk_l2norm_fwd_kernel[(triton.cdiv(num_rows, block_rows),)](
        q,
        k,
        q_out,
        k_out,
        float(eps),
        num_rows,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_ROWS=block_rows,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )
    return q_out, k_out
