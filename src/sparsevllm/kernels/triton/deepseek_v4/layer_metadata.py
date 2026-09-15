"""Device-side causal lengths and compression-event write slots."""

import triton
import triton.language as tl


@triton.jit
def _lengths(Rows, Positions, Lengths, N: tl.constexpr, RATIO: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Rows + i, i < N, -1)
    pos = tl.load(Positions + i, i < N, -1)
    tl.store(Lengths + i, tl.where((row >= 0) & (pos >= 0), (pos + 1) // RATIO, 0), i < N)


@triton.jit
def _writes(Rows, Starts, Requests, Ends, Table, MainSlots, IndexSlots,
            N: tl.constexpr, CAPACITY: tl.constexpr, OFFSET: tl.constexpr,
            RATIO: tl.constexpr, DECODE: tl.constexpr, HAS_INDEX: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if DECODE:
        request = i
        end = tl.load(Starts + i, i < N, -1) + 1
    else:
        request = tl.load(Requests + i, i < N, 0)
        end = tl.load(Ends + i, i < N, 0)
    row = tl.load(Rows + request, i < N, -1)
    block = end // RATIO - 1
    valid = (i < N) & (row >= 0) & (end > 0) & (end % RATIO == 0) & (block < CAPACITY)
    slot = tl.load(Table + row * CAPACITY + block, valid, -1)
    tl.store(MainSlots + i, tl.where(slot >= 0, OFFSET + slot, -1), i < N)
    if HAS_INDEX:
        tl.store(IndexSlots + i, slot, i < N)


def prepare_layer_metadata(window, compression, table, lengths, main_slots, index_slots, *, ratio, compressed_offset):
    if len(window.positions):
        _lengths[(triton.cdiv(len(window.positions), 256),)](
            window.request_rows, window.positions, lengths, len(window.positions), ratio, 256,
        )
    if len(main_slots):
        _writes[(triton.cdiv(len(main_slots), 256),)](
            compression.request_rows, compression.start_positions, compression.boundary_requests,
            compression.boundary_ends, table, main_slots, index_slots, len(main_slots), table.shape[1],
            compressed_offset, ratio, compression.decode, index_slots is not None, 256,
        )
