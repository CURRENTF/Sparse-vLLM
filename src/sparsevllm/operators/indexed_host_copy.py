"""Launch GPU-indexed physical KV transfers and cached miss gathers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sparsevllm.kernels.triton.indexed_host_copy import (
    _append,
    _copy_rows,
    _gather,
    _gather_cached,
    _gather_prefill,
    _transfer,
)


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
    slot_map=None,
    block_budget: int = 0,
    exclude_slots=None,
    cache=None,
    plan=None,
    miss_tokens=None,
    miss_counts=None,
) -> None:
    """Gather a complete view or fill cache misses within a fixed launch budget.

    Zero leaves the grid unbounded. Plain gathers retain at least one block
    per request; cached gathers share the budget across all request misses.
    """
    width = destination.shape[-2] * destination.shape[-1]
    blocks = triton.cdiv(capacity * width, 4096)
    batch = rows.numel()
    if plan is not None:
        _gather_cached[(block_budget or batch * blocks,)](
            source_ptrs,
            cache,
            table,
            rows,
            lengths,
            slot_map,
            exclude_slots,
            plan,
            miss_tokens,
            miss_counts,
            table.stride(0),
            width,
            capacity,
            component,
            skip_last,
            batch,
            triton.next_power_of_2(batch),
            4096,
        )
        return
    if block_budget:
        blocks = min(blocks, max(1, block_budget // batch))
    _gather[(rows.numel(), blocks)](
        source_ptrs,
        destination,
        table,
        rows,
        lengths,
        slot_map,
        exclude_slots,
        table.stride(0),
        width,
        capacity,
        component,
        skip_last,
        4096,
    )


def gather_prefill_rows(
    source_ptrs,
    current,
    destination,
    table,
    rows,
    lengths,
    cu_query,
    slot_map,
    *,
    capacity,
    component,
):
    """Restore full history while reading the current chunk directly from GPU."""
    width = destination.shape[-2] * destination.shape[-1]
    _gather_prefill[(rows.numel(), triton.cdiv(capacity * width, 4096))](
        source_ptrs, current, destination, table, rows, lengths, cu_query, slot_map,
        table.stride(0), current.stride(0), width, capacity, component, 4096,
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
        0 if table is None else table.stride(0),
        triton.next_power_of_2(capacity),
        width,
        source.stride(0),
        capacity,
        256,
    )


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
