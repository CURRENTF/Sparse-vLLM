"""Exact per-request LRU planning shared by an OmniKV observation group."""

import triton
import triton.language as tl


@triton.jit
def _lookup(
    DIRECTORY,
    AGES,
    CLOCK,
    PLAN,
    COUNTERS,
    TABLE,
    ROWS,
    OWNERS,
    LENGTHS,
    WRITES,
    SLOTS: tl.constexpr,
    CACHE: tl.constexpr,
    CAP: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    current = tl.load(WRITES + batch)
    if current >= 0:
        owner = tl.load(OWNERS + batch)
        row = tl.load(ROWS + batch)
        length = tl.load(LENGTHS + batch)
        token = tl.arange(0, BLOCK)
        valid = (token < length) & (token < CAP)
        slot = tl.load(TABLE + row * STRIDE + token, valid, 0)
        position = tl.load(DIRECTORY + owner * SLOTS + slot, valid, -1)
        # The current projection has not been produced at prefetch time.
        hit = valid & (position >= 0)
        clock = tl.load(CLOCK + owner) + 1
        tl.store(CLOCK + owner, clock)
        tl.store(AGES + owner * CACHE + position, clock, hit)
        tl.store(PLAN + batch * CAP + token, tl.where(hit, position, -1), token < CAP)
        hits = tl.sum((hit & (slot != current)).to(tl.int64), 0)
        misses = tl.sum((valid & ~hit & (slot != current)).to(tl.int64), 0)
        tl.atomic_add(COUNTERS, hits, sem="relaxed")
        tl.atomic_add(COUNTERS + 1, misses, sem="relaxed")


@triton.jit
def _admit(
    DIRECTORY,
    KEYS,
    AGES,
    CLOCK,
    PLAN,
    TABLE,
    ROWS,
    OWNERS,
    LENGTHS,
    WRITES,
    SLOTS: tl.constexpr,
    CACHE: tl.constexpr,
    CAP: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    if tl.load(WRITES + batch) >= 0:
        owner = tl.load(OWNERS + batch)
        row = tl.load(ROWS + batch)
        clock = tl.load(CLOCK + owner)
        i = tl.arange(0, BLOCK)
        age = tl.load(AGES + owner * CACHE + i, i < CACHE, 0)
        # Hits were touched first, so no victim can be a selected hit. Ties
        # within one decode step are broken by pool position deterministically.
        order = tl.sort(
            tl.where(i < CACHE, age * BLOCK + i, 0x7FFFFFFFFFFFFFFF), descending=False
        )
        victims = (order % BLOCK).to(tl.int32)
        length = tl.load(LENGTHS + batch)
        valid = (i < length) & (i < CAP)
        old_plan = tl.load(PLAN + batch * CAP + i, i < CAP, -1)
        missing = valid & (old_plan < 0)
        rank = tl.cumsum(missing.to(tl.int32), 0) - 1
        victim = tl.gather(victims, tl.maximum(rank, 0), 0)
        previous = tl.load(KEYS + owner * CACHE + victim, missing, -1)
        tl.store(DIRECTORY + owner * SLOTS + previous, -1, missing & (previous >= 0))
        slot = tl.load(TABLE + row * STRIDE + i, valid, 0)
        tl.store(KEYS + owner * CACHE + victim, slot, missing)
        tl.store(AGES + owner * CACHE + victim, clock, missing)
        tl.store(DIRECTORY + owner * SLOTS + slot, victim, missing)
        position = owner * CACHE + tl.where(missing, victim, old_plan)
        # Negative entries encode a host miss; positive entries are GPU hits.
        tl.store(
            PLAN + batch * CAP + i, tl.where(missing, -position - 1, position), valid
        )


def plan_lru(
    directory, keys, ages, clock, plan, counters, table, rows, owners, lengths, writes
):
    capacity = plan.shape[1]
    cache = keys.shape[1]
    args = (
        directory,
        ages,
        clock,
        plan,
        counters,
        table,
        rows,
        owners,
        lengths,
        writes,
        directory.shape[1],
        cache,
        capacity,
        table.stride(0),
    )
    _lookup[(rows.numel(),)](*args, triton.next_power_of_2(capacity))
    _admit[(rows.numel(),)](
        directory,
        keys,
        ages,
        clock,
        plan,
        table,
        rows,
        owners,
        lengths,
        writes,
        directory.shape[1],
        cache,
        capacity,
        table.stride(0),
        triton.next_power_of_2(cache),
        num_warps=8,
    )
