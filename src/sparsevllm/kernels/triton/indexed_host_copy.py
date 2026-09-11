"""GPU-indexed copies between CUDA and UVA-mapped pinned host pools."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(
    SRC,
    DST_PTR,
    SLOTS,
    SLOT_MAP,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    COMPONENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    if SLOT_MAP is not None and COMPONENT == 0:  # noqa: SIM102 (constexpr guard)
        if tl.program_id(1) == 0:
            tl.store(SLOT_MAP + slot, slot, slot >= 0)
    dst = tl.load(DST_PTR + COMPONENT).to(tl.pointer_type(SRC.dtype.element_ty))
    x = tl.load(SRC + row * SRC_STRIDE + d, (slot >= 0) & (d < WIDTH), 0)
    tl.store(dst + slot.to(tl.int64) * WIDTH + d, x, (slot >= 0) & (d < WIDTH))


@triton.jit
def _gather(
    SRC_PTR,
    DST,
    TABLE,
    ROWS,
    LENGTHS,
    SLOT_MAP,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMPONENT: tl.constexpr,
    SKIP_LAST: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    row = tl.load(ROWS + batch)
    length = tl.load(LENGTHS + batch)
    src = tl.load(SRC_PTR + COMPONENT).to(tl.pointer_type(DST.dtype.element_ty))
    for tile in range(
        tl.program_id(1), tl.cdiv(CAPACITY * WIDTH, BLOCK), tl.num_programs(1)
    ):
        offset = tile * BLOCK + tl.arange(0, BLOCK)
        token = offset // WIDTH
        d = offset % WIDTH
        valid = (token < length - SKIP_LAST) & (token < CAPACITY)
        slot = tl.load(TABLE + row * TABLE_STRIDE + token, valid, 0)
        source_slot = slot
        if SLOT_MAP is not None:
            source_slot = tl.load(SLOT_MAP + slot, valid, 0)
        x = tl.load(src + source_slot.to(tl.int64) * WIDTH + d, valid, 0)
        target = slot if SCATTER else batch * CAPACITY + token
        tl.store(DST + target.to(tl.int64) * WIDTH + d, x, valid)


@triton.jit
def _append(
    SRC,
    DST,
    LENGTHS,
    WRITE_SLOTS,
    WIDTH: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row)
    slot = tl.load(WRITE_SLOTS + row)
    valid = (slot >= 0) & (length > 0) & (length <= CAPACITY) & (d < WIDTH)
    x = tl.load(SRC + row * SRC_STRIDE + d, valid, 0)
    tl.store(DST + (row * CAPACITY + length - 1).to(tl.int64) * WIDTH + d, x, valid)


def store_rows(
    source: torch.Tensor,
    destination_ptrs: torch.Tensor,
    slots: torch.Tensor,
    component: int,
    slot_map=None,
) -> None:
    width = source.shape[-2] * source.shape[-1]
    _copy_rows[(source.shape[0], triton.cdiv(width, 256))](
        source,
        destination_ptrs,
        slots,
        slot_map,
        source.shape[0],
        width,
        source.stride(0),
        component,
        256,
    )


def gather_rows(
    source_ptrs: torch.Tensor,
    destination: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    lengths: torch.Tensor,
    *,
    capacity: int,
    component: int,
    skip_last: bool = False,
    scatter: bool = False,
    slot_map=None,
    max_blocks: int = 0,
) -> None:
    width = destination.shape[-2] * destination.shape[-1]
    blocks = triton.cdiv(capacity * width, 4096)
    if max_blocks:
        blocks = min(blocks, max_blocks)
    _gather[(rows.numel(), blocks)](
        source_ptrs,
        destination,
        table,
        rows,
        lengths,
        slot_map,
        table.stride(0),
        width,
        capacity,
        component,
        skip_last,
        scatter,
        4096,
    )


def append_rows(
    source: torch.Tensor,
    destination: torch.Tensor,
    lengths: torch.Tensor,
    write_slots: torch.Tensor,
    capacity: int,
) -> None:
    width = source.shape[-2] * source.shape[-1]
    _append[(source.shape[0], triton.cdiv(width, 256))](
        source,
        destination,
        lengths,
        write_slots,
        width,
        source.stride(0),
        capacity,
        256,
    )


@triton.jit
def _transfer(
    SRC_PTR,
    DST_PTR,
    SOURCE,
    TARGET,
    SLOT_MAP,
    WIDTH: tl.constexpr,
    COMPONENT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(SOURCE + row)
    target = tl.load(TARGET + row)
    if SLOT_MAP is not None:
        source = tl.load(SLOT_MAP + source)
    src = tl.load(SRC_PTR + COMPONENT).to(tl.pointer_type(DTYPE))
    dst = tl.load(DST_PTR + COMPONENT).to(tl.pointer_type(DTYPE))
    x = tl.load(src + source.to(tl.int64) * WIDTH + d, d < WIDTH, 0)
    tl.store(dst + target.to(tl.int64) * WIDTH + d, x, d < WIDTH)


def transfer_rows(
    source_ptrs,
    destination_ptrs,
    source_slots,
    destination_slots,
    *,
    width,
    dtype,
    component,
    slot_map=None,
):
    if source_slots.numel():
        _transfer[(source_slots.numel(), triton.cdiv(width, 256))](
            source_ptrs,
            destination_ptrs,
            source_slots,
            destination_slots,
            slot_map,
            width,
            component,
            tl.bfloat16 if dtype == torch.bfloat16 else tl.float16,
            256,
        )
