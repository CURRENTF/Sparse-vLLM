from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sparsevllm.triton_kernel.silu_and_mul import silu_and_mul_fwd
from sparsevllm.triton_kernel.moe_config import (
    device_info,
    resolve_moe_gemm_config,
)


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class MoeAlignment:
    sorted_token_ids: torch.Tensor | None
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    block_size: int
    naive: bool


@triton.jit(
    do_not_specialize=[
        "num_assignments",
        "local_expert_start",
        "local_expert_end",
    ]
)
def _count_local_assignments_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_assignments,
    local_expert_start,
    local_expert_end,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_assignments
    global_expert_ids = tl.load(topk_ids_ptr + offsets, mask=valid, other=-1)
    is_local = (
        valid
        & (global_expert_ids >= local_expert_start)
        & (global_expert_ids < local_expert_end)
    )
    local_expert_ids = global_expert_ids - local_expert_start
    tl.atomic_add(counts_ptr + local_expert_ids, 1, mask=is_local)


@triton.jit(
    do_not_specialize=[
        "num_assignments",
        "local_expert_start",
        "local_expert_end",
    ]
)
def _fill_local_assignments_kernel(
    topk_ids_ptr,
    write_positions_ptr,
    sorted_token_ids_ptr,
    num_assignments,
    local_expert_start,
    local_expert_end,
    BLOCK_SIZE: tl.constexpr,
):
    assignment_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = assignment_ids < num_assignments
    global_expert_ids = tl.load(
        topk_ids_ptr + assignment_ids,
        mask=valid,
        other=-1,
    )
    is_local = (
        valid
        & (global_expert_ids >= local_expert_start)
        & (global_expert_ids < local_expert_end)
    )
    local_expert_ids = global_expert_ids - local_expert_start
    positions = tl.atomic_add(
        write_positions_ptr + local_expert_ids,
        1,
        mask=is_local,
    )
    tl.store(sorted_token_ids_ptr + positions, assignment_ids, mask=is_local)


def _validate_alignment_inputs(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> None:
    if not topk_ids.is_cuda:
        raise ValueError("moe_align_block_size requires CUDA topk_ids.")
    if topk_ids.ndim != 2:
        raise ValueError(
            f"topk_ids must have shape [tokens, top_k], got {tuple(topk_ids.shape)}."
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"topk_ids must use int32 or int64, got dtype={topk_ids.dtype}."
        )
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous.")
    if topk_ids.numel() == 0:
        raise ValueError("topk_ids must contain at least one assignment.")
    if int(block_size) <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if int(num_experts) <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}.")
    if not 0 <= int(local_expert_start) < int(local_expert_end) <= int(num_experts):
        raise ValueError(
            "Invalid local expert range: "
            f"[{local_expert_start}, {local_expert_end}) for num_experts={num_experts}."
        )
    if topk_ids.numel() >= torch.iinfo(torch.int32).max:
        raise ValueError(
            "The Triton MoE assignment representation requires fewer than "
            f"{torch.iinfo(torch.int32).max} assignments, got {topk_ids.numel()}."
        )


@triton.jit
def _alignment_prefix_kernel(
    counts_ptr,
    expert_offsets_ptr,
    write_positions_ptr,
    num_tokens_post_padded_ptr,
    block_size: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    counts = tl.load(counts_ptr + offsets, mask=offsets < NUM_LOCAL_EXPERTS, other=0)
    padded_counts = tl.cdiv(counts, block_size) * block_size
    padded_ends = tl.cumsum(padded_counts, axis=0)
    tl.store(
        expert_offsets_ptr + offsets + 1,
        padded_ends,
        mask=offsets < NUM_LOCAL_EXPERTS,
    )
    tl.store(
        write_positions_ptr + offsets,
        padded_ends - padded_counts,
        mask=offsets < NUM_LOCAL_EXPERTS,
    )
    tl.store(expert_offsets_ptr, 0)
    tl.store(num_tokens_post_padded_ptr, tl.sum(padded_counts, axis=0))


@triton.jit
def _fill_expert_blocks_kernel(
    expert_offsets_ptr,
    expert_ids_ptr,
    block_size: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    expert_id = tl.program_id(0)
    block_offsets = tl.arange(0, MAX_BLOCKS_PER_EXPERT)
    first_block = tl.load(expert_offsets_ptr + expert_id) // block_size
    end_block = tl.load(expert_offsets_ptr + expert_id + 1) // block_size
    blocks = first_block + block_offsets
    tl.store(expert_ids_ptr + blocks, expert_id, mask=blocks < end_block)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    local_expert_start: int = 0,
    local_expert_end: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_size = int(block_size)
    num_experts = int(num_experts)
    if local_expert_end is None:
        local_expert_end = num_experts
    local_expert_start = int(local_expert_start)
    local_expert_end = int(local_expert_end)
    _validate_alignment_inputs(
        topk_ids,
        block_size,
        num_experts,
        local_expert_start,
        local_expert_end,
    )
    num_assignments = int(topk_ids.numel())
    num_local_experts = local_expert_end - local_expert_start
    max_num_tokens_padded = triton.cdiv(
        num_assignments + num_local_experts * (block_size - 1),
        block_size,
    ) * block_size
    max_num_blocks = max_num_tokens_padded // block_size
    counts = torch.zeros(num_local_experts, dtype=torch.int32, device=topk_ids.device)
    expert_offsets = torch.empty(
        num_local_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    write_positions = torch.empty(
        num_local_experts, dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), num_assignments, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )

    assignment_block = 256
    _count_local_assignments_kernel[(triton.cdiv(num_assignments, assignment_block),)](
        topk_ids,
        counts,
        num_assignments,
        local_expert_start,
        local_expert_end,
        BLOCK_SIZE=assignment_block,
    )
    _alignment_prefix_kernel[(1,)](
        counts,
        expert_offsets,
        write_positions,
        num_tokens_post_padded,
        block_size=block_size,
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=triton.next_power_of_2(num_local_experts),
        num_warps=4,
    )
    _fill_expert_blocks_kernel[(num_local_experts,)](
        expert_offsets,
        expert_ids,
        block_size=block_size,
        MAX_BLOCKS_PER_EXPERT=triton.next_power_of_2(
            max(1, triton.cdiv(num_assignments, block_size))
        ),
        num_warps=4,
    )
    _fill_local_assignments_kernel[(triton.cdiv(num_assignments, assignment_block),)](
        topk_ids,
        write_positions,
        sorted_token_ids,
        num_assignments,
        local_expert_start,
        local_expert_end,
        BLOCK_SIZE=assignment_block,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> MoeAlignment:
    num_assignments = int(topk_ids.numel())
    if num_assignments * 4 <= int(num_experts):
        flat_ids = topk_ids.view(-1)
        is_local = (flat_ids >= local_expert_start) & (
            flat_ids < local_expert_end
        )
        expert_ids = torch.where(
            is_local,
            flat_ids - local_expert_start,
            torch.full_like(flat_ids, -1),
        ).to(torch.int32)
        return MoeAlignment(
            sorted_token_ids=None,
            expert_ids=expert_ids.contiguous(),
            num_tokens_post_padded=torch.full(
                (1,),
                num_assignments * block_size,
                dtype=torch.int32,
                device=topk_ids.device,
            ),
            block_size=block_size,
            naive=True,
        )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
    return MoeAlignment(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size=block_size,
        naive=False,
    )


@triton.jit(do_not_specialize=["EM", "num_assignments"])
def _routed_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    routing_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_assignments,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    INPUT_TOP_K: tl.constexpr,
    MUL_ROUTING_WEIGHT: tl.constexpr,
    NAIVE_ASSIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    if NAIVE_ASSIGNMENT:
        assignment_ids = tl.where(
            row_offsets == 0,
            pid_m,
            num_assignments,
        )
    else:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        assignment_offsets = pid_m * BLOCK_SIZE_M + row_offsets
        assignment_ids = tl.load(sorted_token_ids_ptr + assignment_offsets)
    assignment_ids = assignment_ids.to(tl.int64)
    assignment_mask = assignment_ids < num_assignments

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id < 0:
        return

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    input_rows = assignment_ids // INPUT_TOP_K
    a_ptrs = (
        a_ptr
        + input_rows[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_be
        + n_offsets[None, :] * stride_bn
        + k_offsets[:, None] * stride_bk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining_k = K - k_start * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=assignment_mask[:, None] & (k_offsets[None, :] < remaining_k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < remaining_k)
            & (n_offsets[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTING_WEIGHT:
        routing_weights = tl.load(
            routing_weights_ptr + assignment_ids,
            mask=assignment_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator *= routing_weights[:, None]

    c_ptrs = (
        c_ptr
        + assignment_ids[:, None] * stride_cm
        + n_offsets[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        accumulator,
        mask=assignment_mask[:, None] & (n_offsets[None, :] < N),
    )


def _routed_gemm(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    alignment: MoeAlignment,
    *,
    input_top_k: int,
    multiply_routing_weight: bool,
    launch_config: dict[str, int],
) -> None:
    num_assignments = int(topk_weights.numel())
    if alignment.naive:
        em = num_assignments * alignment.block_size
        sorted_token_ids = topk_weights
    else:
        if alignment.sorted_token_ids is None:
            raise RuntimeError("Aligned routed GEMM is missing sorted_token_ids.")
        em = int(alignment.sorted_token_ids.numel())
        sorted_token_ids = alignment.sorted_token_ids

    block_m = launch_config["BLOCK_SIZE_M"]
    block_n = launch_config["BLOCK_SIZE_N"]
    grid = (
        triton.cdiv(em, block_m) * triton.cdiv(int(weights.shape[1]), block_n),
    )
    _routed_gemm_kernel[grid](
        inputs,
        weights,
        output,
        topk_weights,
        sorted_token_ids,
        alignment.expert_ids,
        alignment.num_tokens_post_padded,
        N=int(weights.shape[1]),
        K=int(weights.shape[2]),
        EM=em,
        num_assignments=num_assignments,
        stride_am=inputs.stride(0),
        stride_ak=inputs.stride(1),
        stride_be=weights.stride(0),
        stride_bn=weights.stride(1),
        stride_bk=weights.stride(2),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        INPUT_TOP_K=int(input_top_k),
        MUL_ROUTING_WEIGHT=bool(multiply_routing_weight),
        NAIVE_ASSIGNMENT=alignment.naive,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=launch_config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=launch_config["GROUP_SIZE_M"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "local_expert_start",
        "local_expert_end",
    ]
)
def _moe_sum_kernel(
    inputs_ptr,
    topk_ids_ptr,
    output_ptr,
    num_tokens,
    local_expert_start,
    local_expert_end,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    stride_im,
    stride_ik,
    stride_in,
    stride_om,
    stride_on,
    FILTER_REMOTE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (token_offsets[:, None] < num_tokens) & (
        hidden_offsets[None, :] < hidden_size
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for topk_slot in tl.static_range(top_k):
        slot_mask = mask
        if FILTER_REMOTE:
            expert_ids = tl.load(
                topk_ids_ptr + token_offsets * top_k + topk_slot,
                mask=token_offsets < num_tokens,
                other=-1,
            )
            is_local = (expert_ids >= local_expert_start) & (
                expert_ids < local_expert_end
            )
            slot_mask = mask & is_local[:, None]
        values = tl.load(
            inputs_ptr
            + token_offsets[:, None] * stride_im
            + topk_slot * stride_ik
            + hidden_offsets[None, :] * stride_in,
            mask=slot_mask,
            other=0.0,
        )
        values = values.to(tl.float32)
        accumulator += values
    tl.store(
        output_ptr
        + token_offsets[:, None] * stride_om
        + hidden_offsets[None, :] * stride_on,
        accumulator,
        mask=mask,
    )


def _moe_sum(
    inputs: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    num_tokens, top_k, hidden_size = (int(dim) for dim in inputs.shape)
    block_m = 1 if num_tokens <= 4 else 8
    block_n = 256 if hidden_size == 2048 and top_k == 8 else 128
    if output_dtype is None:
        output_dtype = inputs.dtype
    if output_dtype not in (*_SUPPORTED_DTYPES, torch.float32):
        raise TypeError(
            "Triton MoE sum output must use BF16, FP16, or FP32, got "
            f"dtype={output_dtype}."
        )
    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=output_dtype,
        device=inputs.device,
    )
    grid = (
        triton.cdiv(num_tokens, block_m),
        triton.cdiv(hidden_size, block_n),
    )
    _moe_sum_kernel[grid](
        inputs,
        topk_ids,
        output,
        num_tokens,
        local_expert_start,
        local_expert_end,
        hidden_size=hidden_size,
        top_k=top_k,
        stride_im=inputs.stride(0),
        stride_ik=inputs.stride(1),
        stride_in=inputs.stride(2),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        FILTER_REMOTE=(
            local_expert_start != 0 or local_expert_end != int(num_experts)
        ),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        num_warps=4 if block_m <= 4 else 8,
        num_stages=2,
    )
    return output


def _validate_fused_moe_inputs(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    local_expert_start: int,
) -> None:
    tensors = {
        "hidden_states": hidden_states,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"Triton MoE requires CUDA {name}.")
        if tensor.device != hidden_states.device:
            raise ValueError(
                f"All Triton MoE tensors must share one device; {name} is on "
                f"{tensor.device}, hidden_states is on {hidden_states.device}."
            )
    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 0:
        raise ValueError(
            "hidden_states must have non-empty shape [tokens, hidden], got "
            f"{tuple(hidden_states.shape)}."
        )
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError(
            "Expert weights must have shapes [local_experts, 2*intermediate, hidden] "
            "and [local_experts, hidden, intermediate]."
        )
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_ids and topk_weights must have the same [tokens, top_k] shape, "
            f"got ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}."
        )
    if int(topk_ids.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError(
            "Router token count does not match hidden_states: "
            f"{topk_ids.shape[0]} != {hidden_states.shape[0]}."
        )
    if int(topk_ids.shape[1]) <= 0 or int(topk_ids.shape[1]) > int(num_experts):
        raise ValueError(
            f"top_k must be in [1, {num_experts}], got {topk_ids.shape[1]}."
        )
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "Triton MoE supports BF16 and FP16 activations, got "
            f"dtype={hidden_states.dtype}."
        )
    if w13_weight.dtype != hidden_states.dtype or w2_weight.dtype != hidden_states.dtype:
        raise TypeError(
            "Triton MoE activations and expert weights must share BF16/FP16 dtype, got "
            f"hidden={hidden_states.dtype}, w13={w13_weight.dtype}, w2={w2_weight.dtype}."
        )
    if topk_weights.dtype not in (*_SUPPORTED_DTYPES, torch.float32):
        raise TypeError(
            f"topk_weights must use BF16, FP16, or FP32, got {topk_weights.dtype}."
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must use int32 or int64, got {topk_ids.dtype}.")
    for name, tensor in tensors.items():
        if not tensor.is_contiguous():
            raise ValueError(f"Triton MoE requires contiguous {name}.")

    num_local_experts = int(w13_weight.shape[0])
    hidden_size = int(hidden_states.shape[1])
    if num_local_experts <= 0:
        raise ValueError("Triton MoE requires at least one local expert.")
    if int(w2_weight.shape[0]) != num_local_experts:
        raise ValueError("w13_weight and w2_weight local expert counts differ.")
    if int(w13_weight.shape[1]) % 2 != 0:
        raise ValueError("w13_weight output size must be even for SiLU-and-mul.")
    intermediate_size = int(w13_weight.shape[1]) // 2
    if int(w13_weight.shape[2]) != hidden_size:
        raise ValueError("w13_weight input size does not match hidden_states.")
    if tuple(w2_weight.shape[1:]) != (hidden_size, intermediate_size):
        raise ValueError(
            "w2_weight shape does not match hidden/intermediate sizes: expected "
            f"({num_local_experts}, {hidden_size}, {intermediate_size}), got "
            f"{tuple(w2_weight.shape)}."
        )
    local_expert_end = int(local_expert_start) + num_local_experts
    if not 0 <= int(local_expert_start) < local_expert_end <= int(num_experts):
        raise ValueError(
            f"Invalid local expert range [{local_expert_start}, {local_expert_end}) "
            f"for num_experts={num_experts}."
        )


def fused_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run unquantized routed experts with a generic Triton MoE pipeline.

    The default output dtype matches ``hidden_states`` and is the production EP
    path. Explicit FP32 output is retained for numerical diagnostics.
    """

    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    _validate_fused_moe_inputs(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts,
        local_expert_start,
    )

    num_tokens = int(hidden_states.shape[0])
    top_k = int(topk_ids.shape[1])
    num_assignments = num_tokens * top_k
    intermediate_size = int(w13_weight.shape[1]) // 2
    hidden_size = int(hidden_states.shape[1])
    local_expert_end = local_expert_start + int(w13_weight.shape[0])

    num_local_experts = int(w13_weight.shape[0])
    device_name, capability = device_info(
        hidden_states.device.type,
        int(hidden_states.device.index),
    )
    w13_config = resolve_moe_gemm_config(
        dtype=hidden_states.dtype,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        stage="w13",
        device_name=device_name,
        device_capability=capability,
    ).as_triton_kwargs()
    alignment = _prepare_expert_assignment(
        topk_ids,
        block_size=w13_config["BLOCK_SIZE_M"],
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )

    w13_output = torch.empty(
        (num_assignments, 2 * intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _routed_gemm(
        hidden_states,
        w13_weight,
        w13_output,
        topk_weights,
        alignment,
        input_top_k=top_k,
        multiply_routing_weight=False,
        launch_config=w13_config,
    )
    activated = silu_and_mul_fwd(w13_output)

    w2_config = resolve_moe_gemm_config(
        dtype=hidden_states.dtype,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        stage="w2",
        device_name=device_name,
        device_capability=capability,
    ).as_triton_kwargs()
    w2_config["BLOCK_SIZE_M"] = alignment.block_size
    w2_output = torch.empty(
        (num_assignments, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _routed_gemm(
        activated,
        w2_weight,
        w2_output,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
        launch_config=w2_config,
    )
    return _moe_sum(
        w2_output.view(num_tokens, top_k, hidden_size),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
        output_dtype=output_dtype,
    )
