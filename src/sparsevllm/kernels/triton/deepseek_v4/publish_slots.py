"""Publish reserved compressed slots from address-stable decode staging."""

import triton as tr
import triton.language as tl


@tr.jit
def _publish(Table, Rows, Positions, Slots, SnapshotRows, CAPACITY: tl.constexpr, BATCH: tl.constexpr,
             RATIO: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Rows + token, token < BATCH, other=-1)
    position = tl.load(Positions + token, token < BATCH, other=-1)
    slot = tl.load(Slots + token, token < BATCH, other=-1)
    snapshot = tl.load(SnapshotRows + token, token < BATCH, other=-1)
    column = (position + 1) // RATIO - 1
    valid = (token < BATCH) & (row >= 0) & (slot >= 0)
    tl.store(Table + row * CAPACITY + column, slot, valid)
    tl.store(Table + snapshot * CAPACITY + column, slot, valid & (snapshot >= 0))


def publish_compressed_slots(table, rows, positions, slots, snapshot_rows, *, ratio):
    if rows.numel():
        _publish[(tr.cdiv(rows.numel(), 128),)](table, rows, positions, slots, snapshot_rows,
                                             table.shape[1], rows.numel(), ratio, 128)
