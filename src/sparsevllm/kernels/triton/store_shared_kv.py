"""Write native KV vectors to owner-selected slots, masking graph padding."""

import triton
import triton.language as tl


@triton.jit
def _store(Values, Slots, Positions, Cache, D: tl.constexpr, CAP: tl.constexpr,
           HAS_POSITIONS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    active = (slot >= 0) & (slot < CAP)
    if HAS_POSITIONS:
        active &= tl.load(Positions + row) >= 0
    col = tl.arange(0, BLOCK)
    if active:
        value = tl.load(Values + row * D + col, col < D, 0)
        tl.store(Cache + slot * D + col, value, col < D)


def store_shared_kv(values, slots, cache, positions=None):
    if values.shape[0]:
        _store[(values.shape[0],)](values, slots, positions, cache, values.shape[-1], cache.shape[0],
                                  positions is not None, triton.next_power_of_2(values.shape[-1]))
