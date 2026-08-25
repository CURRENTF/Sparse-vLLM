from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_page_indices_kernel(
    active_slots,
    request_indices,
    context_lens,
    packed_indices,
    active_slots_stride_0: tl.constexpr,
    active_slots_stride_1: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    CONTEXT_CAPACITY: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_block_idx = tl.program_id(1)

    batch_offsets = tl.arange(0, BATCH_BLOCK)
    lengths = tl.load(context_lens + batch_offsets, mask=batch_offsets < BATCH_SIZE)
    packed_start = tl.sum(tl.where(batch_offsets < batch_idx, lengths, 0))

    token_offsets = token_block_idx * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    context_len = tl.load(context_lens + batch_idx)
    request_idx = tl.load(request_indices + batch_idx)
    valid = (token_offsets < context_len) & (token_offsets < CONTEXT_CAPACITY)
    slots = tl.load(
        active_slots
        + request_idx * active_slots_stride_0
        + token_offsets * active_slots_stride_1,
        mask=valid,
    )
    tl.store(packed_indices + packed_start + token_offsets, slots, mask=valid)


def pack_flashinfer_page_indices(
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    packed_indices: torch.Tensor,
    *,
    context_capacity: int,
) -> None:
    """Pack a layer's page-size-one slot table into graph-stable storage."""

    if active_slots.ndim != 2 or active_slots.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires a rank-2 int32 slot table.")
    if request_indices.ndim != 1 or request_indices.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires int32 request indices.")
    if context_lens.ndim != 1 or context_lens.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires int32 context lengths.")
    if request_indices.shape != context_lens.shape:
        raise ValueError("FlashInfer graph request indices and context lengths must match.")

    batch_size = int(context_lens.numel())
    context_capacity = int(context_capacity)
    if context_capacity <= 0 or context_capacity > int(active_slots.shape[1]):
        raise ValueError(
            "FlashInfer graph context capacity is outside the slot table: "
            f"capacity={context_capacity} width={int(active_slots.shape[1])}."
        )
    if packed_indices.ndim != 1 or packed_indices.dtype != torch.int32:
        raise TypeError("FlashInfer graph packed indices must be a 1D int32 tensor.")
    required = batch_size * context_capacity
    if int(packed_indices.numel()) < required:
        raise ValueError(
            "FlashInfer graph packed-index buffer is too small: "
            f"required={required} actual={int(packed_indices.numel())}."
        )

    token_block = 128
    _pack_page_indices_kernel[
        (batch_size, triton.cdiv(context_capacity, token_block))
    ](
        active_slots,
        request_indices,
        context_lens,
        packed_indices,
        active_slots.stride(0),
        active_slots.stride(1),
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=triton.next_power_of_2(batch_size),
        CONTEXT_CAPACITY=context_capacity,
        TOKEN_BLOCK=token_block,
    )


__all__ = ["pack_flashinfer_page_indices"]
