from __future__ import annotations

import torch
import triton
import triton.language as tl

from sparsevllm.triton_kernel.moe_config import MoeGemmConfig


_H20_DECODE_CONFIGS = {
    (1, 2048, 512): MoeGemmConfig(16, 32, 64, 8, 4, 4),
    (1, 2048, 256): MoeGemmConfig(16, 32, 64, 8, 4, 4),
}


def resolve_h20_gate_up_swiglu_config(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> MoeGemmConfig:
    shape = (int(num_tokens), int(hidden_size), int(intermediate_size))
    try:
        return _H20_DECODE_CONFIGS[shape]
    except KeyError as error:
        raise ValueError(
            f"No H20 gate/up SwiGLU config for shape {shape}."
        ) from error


@triton.jit
def _gate_up_swiglu_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    input_ptrs = (
        input_ptr
        + m_offsets[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    gate_ptrs = (
        weight_ptr
        + n_offsets[None, :] * stride_bn
        + k_offsets[:, None] * stride_bk
    )
    up_ptrs = gate_ptrs + N * stride_bn
    gate_accumulator = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
    )
    up_accumulator = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
    )
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining_k = K - k_start * BLOCK_SIZE_K
        input_values = tl.load(
            input_ptrs,
            mask=(m_offsets[:, None] < M)
            & (k_offsets[None, :] < remaining_k),
            other=0.0,
        )
        weight_mask = (k_offsets[:, None] < remaining_k) & (
            n_offsets[None, :] < N
        )
        gate_accumulator += tl.dot(
            input_values,
            tl.load(gate_ptrs, mask=weight_mask, other=0.0),
        )
        up_accumulator += tl.dot(
            input_values,
            tl.load(up_ptrs, mask=weight_mask, other=0.0),
        )
        input_ptrs += BLOCK_SIZE_K * stride_ak
        gate_ptrs += BLOCK_SIZE_K * stride_bk
        up_ptrs += BLOCK_SIZE_K * stride_bk

    element_dtype = weight_ptr.dtype.element_ty
    gate = gate_accumulator.to(element_dtype).to(tl.float32)
    up = up_accumulator.to(element_dtype)
    gate = (gate / (1.0 + tl.exp(-gate))).to(element_dtype)
    tl.store(
        output_ptr
        + m_offsets[:, None] * stride_cm
        + n_offsets[None, :] * stride_cn,
        gate * up,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
    )


def gate_up_swiglu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    config: MoeGemmConfig,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        inputs.ndim != 2
        or weight.ndim != 2
        or weight.shape[1] != inputs.shape[1]
    ):
        raise ValueError(
            "gate_up_swiglu expects input [M, K] and weight [2N, K], "
            f"got {tuple(inputs.shape)} and {tuple(weight.shape)}."
        )
    if weight.shape[0] % 2:
        raise ValueError(
            f"gate_up_swiglu weight rows must be even, got {weight.shape[0]}."
        )
    if inputs.dtype != torch.bfloat16 or weight.dtype != inputs.dtype:
        raise TypeError("gate_up_swiglu requires matching BF16 inputs and weights.")
    if not inputs.is_cuda or weight.device != inputs.device:
        raise ValueError("gate_up_swiglu requires CUDA tensors on one device.")
    if not inputs.is_contiguous() or not weight.is_contiguous():
        raise ValueError("gate_up_swiglu requires contiguous inputs and weights.")

    m, k = inputs.shape
    n = weight.shape[0] // 2
    output = (
        torch.empty((m, n), dtype=inputs.dtype, device=inputs.device)
        if output is None
        else output
    )
    if (
        output.shape != (m, n)
        or output.dtype != inputs.dtype
        or output.device != inputs.device
    ):
        raise ValueError(
            f"gate_up_swiglu output must be {(m, n)} {inputs.dtype} on {inputs.device}."
        )
    launch = config.as_triton_kwargs()
    grid = (
        triton.cdiv(m, launch["BLOCK_SIZE_M"])
        * triton.cdiv(n, launch["BLOCK_SIZE_N"]),
    )
    _gate_up_swiglu_kernel[grid](
        inputs,
        weight,
        output,
        M=m,
        N=n,
        K=k,
        stride_am=inputs.stride(0),
        stride_ak=inputs.stride(1),
        stride_bn=weight.stride(0),
        stride_bk=weight.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        **launch,
    )
    return output
