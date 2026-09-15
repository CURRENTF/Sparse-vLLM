"""Physical ring/temporary selection and post-attention window commit mapping."""

import triton
import triton.language as tl


@triton.jit
def _indices(Rows, Positions, Starts, Packed, Compressed, Lengths, Out,
             WINDOW: tl.constexpr, WINDOW_OFFSET: tl.constexpr, TEMP_OFFSET: tl.constexpr,
             COMP_OFFSET: tl.constexpr, COMP_CAPACITY: tl.constexpr, PER_REQUEST: tl.constexpr,
             PREFILL: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    row, position = tl.load(Rows + token), tl.load(Positions + token)
    col = tl.arange(0, BLOCK)
    active = (row >= 0) & (position >= 0)
    window_position = tl.maximum(0, position - WINDOW + 1) + col
    window_valid = active & (col < WINDOW) & (window_position <= position)
    slot = WINDOW_OFFSET + row * WINDOW + window_position % WINDOW
    if PREFILL:
        start, packed_start = tl.load(Starts + token), tl.load(Packed + token)
        slot = tl.where(window_position >= start, TEMP_OFFSET + packed_start + window_position - start, slot)
    slot = tl.where(window_valid, slot, -1)
    if COMP_CAPACITY:
        comp_col = col - WINDOW
        if PER_REQUEST:
            length = tl.load(Lengths + token)
            comp_row = row
        else:
            length = COMP_CAPACITY
            comp_row = token
        valid = active & (comp_col >= 0) & (comp_col < tl.minimum(length, COMP_CAPACITY))
        compressed = tl.load(Compressed + comp_row * COMP_CAPACITY + comp_col, valid, -1)
        slot = tl.where(col >= WINDOW, tl.where(valid & (compressed >= 0), COMP_OFFSET + compressed, -1), slot)
    tl.store(Out + token * (WINDOW + COMP_CAPACITY) + col, slot, col < WINDOW + COMP_CAPACITY)


@triton.jit
def _commit_slots(Rows, Positions, Ends, Out, N: tl.constexpr,
                  WINDOW: tl.constexpr, WINDOW_OFFSET: tl.constexpr, PREFILL: tl.constexpr,
                  BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Rows + token, token < N, -1)
    position = tl.load(Positions + token, token < N, -1)
    valid = (token < N) & (row >= 0) & (position >= 0)
    if PREFILL:
        end = tl.load(Ends + token, token < N, 0)
        valid &= position >= end - WINDOW
    slot = WINDOW_OFFSET + row * WINDOW + position % WINDOW
    tl.store(Out + token, tl.where(valid, slot, -1), token < N)


def shared_kv_indices(batch, compressed, lengths, out, *, window_size, window_offset,
                      temporary_offset, compressed_offset, compressed_per_request):
    if len(batch.positions):
        capacity = 0 if compressed is None else compressed.shape[1]
        _indices[(len(batch.positions),)](
            batch.request_rows, batch.positions, batch.chunk_starts, batch.packed_starts,
            compressed, lengths, out, window_size, window_offset, temporary_offset,
            compressed_offset, capacity, compressed_per_request, batch.chunk_starts is not None,
            triton.next_power_of_2(window_size + capacity),
        )


def window_commit_slots(batch, out, *, window_size, window_offset):
    if len(batch.positions):
        _commit_slots[(triton.cdiv(len(batch.positions), 256),)](
            batch.request_rows, batch.positions, batch.chunk_ends, out, len(batch.positions),
            window_size, window_offset, batch.chunk_starts is not None, 256,
        )
