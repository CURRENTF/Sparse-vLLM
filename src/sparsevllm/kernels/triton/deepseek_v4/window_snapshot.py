"""Copy a causal prefix window before the active prefill ring is overwritten."""

import triton as tr
import triton.language as tl


@tr.jit
def _snapshot_window(
    Cache, Rows, Cu, Starts, SourceRequests, DestinationRows, Ends,
    WINDOW: tl.constexpr, DIM: tl.constexpr, TEMP_OFFSET: tl.constexpr,
    DECODE: tl.constexpr, BLOCK: tl.constexpr,
):
    snapshot = tl.program_id(0)
    destination = tl.load(DestinationRows + snapshot)
    if destination >= 0:
        batch = tl.load(SourceRequests + snapshot)
        row = tl.load(Rows + batch)
        start = tl.load(Starts + batch)
        end = tl.load(Ends + snapshot)
        packed_start = tl.load(Cu + batch)
        index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ring_slot, dim = index // DIM, index % DIM
        distance = ((end - 1) % WINDOW + WINDOW - ring_slot) % WINDOW
        position = end - 1 - distance
        old_slot = row * WINDOW + ring_slot
        if DECODE:
            source_slot = old_slot
        else:
            source_slot = tl.where(position >= start, TEMP_OFFSET + packed_start + position - start, old_slot)
        value = tl.load(Cache + source_slot * DIM + dim, (index < WINDOW * DIM) & (position >= 0), other=0)
        target_slot = destination * WINDOW + ring_slot
        tl.store(Cache + target_slot * DIM + dim, value, index < WINDOW * DIM)


def snapshot_window(cache, request_rows, cu_seqlens, starts, snapshots, *, window_size, temporary_offset, decode):
    if snapshots.destination_rows.numel():
        dim = cache.shape[-1]
        _snapshot_window[(snapshots.destination_rows.numel(), tr.cdiv(window_size * dim, 256))](
            cache, request_rows, cu_seqlens, starts, snapshots.source_requests,
            snapshots.destination_rows, snapshots.ends, window_size, dim,
            temporary_offset, decode, 256,
        )
