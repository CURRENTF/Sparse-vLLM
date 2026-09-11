"""GPU-indexed copies between CUDA and UVA-mapped pinned host pools."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(SRC, DST_PTR, SLOTS, N: tl.constexpr, WIDTH: tl.constexpr,
               SRC_STRIDE: tl.constexpr, COMPONENT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    dst = tl.load(DST_PTR + COMPONENT).to(tl.pointer_type(SRC.dtype.element_ty))
    x = tl.load(SRC + row * SRC_STRIDE + d, (slot >= 0) & (d < WIDTH), 0)
    tl.store(dst + slot.to(tl.int64) * WIDTH + d, x, (slot >= 0) & (d < WIDTH))


@triton.jit
def _gather(SRC_PTR, DST, TABLE, ROWS, LENGTHS,
            TABLE_STRIDE: tl.constexpr, WIDTH: tl.constexpr,
            CAPACITY: tl.constexpr, COMPONENT: tl.constexpr,
            SKIP_LAST: tl.constexpr, SCATTER: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    d = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(ROWS + batch)
    length = tl.load(LENGTHS + batch)
    valid = token < length - SKIP_LAST
    slot = tl.load(TABLE + row * TABLE_STRIDE + token, valid, 0)
    src = tl.load(SRC_PTR + COMPONENT).to(tl.pointer_type(DST.dtype.element_ty))
    x = tl.load(src + slot.to(tl.int64) * WIDTH + d, valid & (d < WIDTH), 0)
    target = slot if SCATTER else batch * CAPACITY + token
    tl.store(DST + target.to(tl.int64) * WIDTH + d, x, valid & (d < WIDTH))


@triton.jit
def _append(SRC, DST, LENGTHS, WRITE_SLOTS, WIDTH: tl.constexpr,
            SRC_STRIDE: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row)
    slot = tl.load(WRITE_SLOTS + row)
    valid = (slot >= 0) & (length > 0) & (length <= CAPACITY) & (d < WIDTH)
    x = tl.load(SRC + row * SRC_STRIDE + d, valid, 0)
    tl.store(DST + (row * CAPACITY + length - 1).to(tl.int64) * WIDTH + d, x, valid)


def store_rows(source: torch.Tensor, destination_ptrs: torch.Tensor,
               slots: torch.Tensor, component: int) -> None:
    width = source.shape[-2] * source.shape[-1]
    _copy_rows[(source.shape[0], triton.cdiv(width, 256))](
        source, destination_ptrs, slots, source.shape[0], width,
        source.stride(0), component, 256,
    )


def gather_rows(source_ptrs: torch.Tensor, destination: torch.Tensor,
                table: torch.Tensor, rows: torch.Tensor, lengths: torch.Tensor,
                *, capacity: int, component: int, skip_last: bool = False,
                scatter: bool = False) -> None:
    width = destination.shape[-2] * destination.shape[-1]
    _gather[(rows.numel(), capacity, triton.cdiv(width, 256))](
        source_ptrs, destination, table, rows, lengths, table.stride(0), width,
        capacity, component, skip_last, scatter, 256,
    )


def append_rows(source: torch.Tensor, destination: torch.Tensor,
                lengths: torch.Tensor, write_slots: torch.Tensor, capacity: int) -> None:
    width = source.shape[-2] * source.shape[-1]
    _append[(source.shape[0], triton.cdiv(width, 256))](
        source, destination, lengths, write_slots, width, source.stride(0), capacity, 256,
    )
