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
    EXCLUDE_SLOTS,
    CACHE,
    PLAN,
    DIRECT: tl.constexpr,
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
        active = True
        if EXCLUDE_SLOTS is not None:
            current_slot = tl.load(EXCLUDE_SLOTS + batch)
            active = current_slot >= 0
            valid = valid & active
        slot = tl.load(TABLE + row * TABLE_STRIDE + token, valid, 0)
        if EXCLUDE_SLOTS is not None:
            valid = valid & (slot != current_slot)
        source_slot = slot
        if SLOT_MAP is not None:
            source_slot = tl.load(SLOT_MAP + slot, valid, 0)
        host_read = valid
        if PLAN is not None:
            entry = tl.load(PLAN + batch * CAPACITY + token, valid, 0)
            cache_slot = tl.where(entry < 0, -entry - 1, entry).to(tl.int64)
            host_read = valid & (entry < 0)
        x = tl.load(src + source_slot.to(tl.int64) * WIDTH + d, host_read, 0)
        if PLAN is not None:
            tl.store(CACHE + cache_slot * WIDTH + d, x, host_read)
            if not DIRECT:
                cached = tl.load(CACHE + cache_slot * WIDTH + d, valid & ~host_read, 0)
                x = tl.where(host_read, x, cached)
        target = slot if SCATTER else batch * CAPACITY + token
        # Padded rows have replicated lengths but no write slot. Do not read
        # their host KV, and leave finite zeros for their private attention view.
        store_mask = valid | ((not active) & (token < CAPACITY))
        if not DIRECT:
            tl.store(DST + target.to(tl.int64) * WIDTH + d, x, store_mask)


@triton.jit
def _append(
    SRC,
    DST,
    LENGTHS,
    WRITE_SLOTS,
    TABLE,
    ROWS,
    CACHE,
    PLAN,
    DIRECT: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SEEK_BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row)
    slot = tl.load(WRITE_SLOTS + row)
    if TABLE is not None:
        token = tl.arange(0, SEEK_BLOCK)
        table_row = tl.load(ROWS + row)
        selected = tl.load(
            TABLE + table_row * TABLE_STRIDE + token,
            (token < length) & (token < CAPACITY),
            -1,
        )
        position = tl.max(tl.where((selected == slot) & (token < length), token, -1), 0)
        length = position + 1
    valid = (slot >= 0) & (length > 0) & (length <= CAPACITY) & (d < WIDTH)
    x = tl.load(SRC + row * SRC_STRIDE + d, valid, 0)
    if not DIRECT:
        tl.store(DST + (row * CAPACITY + length - 1).to(tl.int64) * WIDTH + d, x, valid)
    if PLAN is not None:
        entry = tl.load(
            PLAN + row * CAPACITY + length - 1,
            (slot >= 0) & (length > 0) & (length <= CAPACITY),
            0,
        )
        cache_slot = tl.where(entry < 0, -entry - 1, entry).to(tl.int64)
        tl.store(CACHE + cache_slot * WIDTH + d, x, valid)


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
    exclude_slots=None,
    cache=None,
    plan=None,
    direct=False,
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
        exclude_slots,
        cache,
        plan,
        direct,
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
    *,
    table=None,
    rows=None,
    cache=None,
    plan=None,
    direct=False,
) -> None:
    width = source.shape[-2] * source.shape[-1]
    _append[(source.shape[0], triton.cdiv(width, 256))](
        source,
        destination,
        lengths,
        write_slots,
        table,
        rows,
        cache,
        plan,
        direct,
        0 if table is None else table.stride(0),
        triton.next_power_of_2(capacity),
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
