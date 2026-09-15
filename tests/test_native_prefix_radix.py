"""Native radix tests protect pinning/publication and physical reference lifetime."""

import pytest
import torch

from sparsevllm.engine.cache_manager.methods.deepseek_v4_prefix import NativePrefixCache
from sparsevllm.engine.cache_manager.storage.native_pool import NativeCachePool
from sparsevllm.engine.sequence import Sequence


def make_cache(*, snapshots=8):
    pool = NativeCachePool(ratios=(0, 4, 128), live_rows=3, snapshot_rows=snapshots,
                           max_model_len=512, prefill_capacity=128, compressed_capacities={4: 40, 128: 8},
                           device="cpu", head_dim=8, index_dim=4, window_size=8)
    return pool, NativePrefixCache(pool, block_size=16, fingerprint=b"native-test")


def publish(pool, cache, seq, start, end):
    requests = ((seq.seq_id, start, end),)
    plan = cache.plan([seq], requests)
    step = pool.reserve_prefill(requests, snapshot_ends=plan.snapshot_ends)
    pool.commit(step)
    cache.publish(plan, step)
    return step


def test_lookup_pin_survives_pressure_and_child_survives_radix_eviction():
    pool, cache = make_cache()
    parent = Sequence(list(range(80)))
    publish(pool, cache, parent, 0, 64)
    pool.release_sequence(parent.seq_id)
    cache.release_sequence(parent.seq_id)
    child = Sequence(list(range(64)) + [999])
    cache.refresh_hit(child)
    assert child.prefix_cache_hit_len == 64
    # A pending hit protects its leaf and ancestors before it has a live row.
    assert not cache.evict_for_capacity({}, snapshot_rows=8)
    assert len(cache.index) == 4
    cache.attach_hit(child)
    assert pool.live[child.seq_id].length == 64
    assert cache.evict_for_capacity({}, snapshot_rows=8)
    assert not cache.index and not pool.snapshots
    # The child retains compressed slots after every indexed snapshot is gone.
    assert pool.slots[4].free_count == 40 - 64 // 4
    step = publish(pool, cache, child, 64, 65)
    assert not step.snapshots
    pool.release_sequence(child.seq_id)
    cache.release_sequence(child.seq_id)
    assert pool.slots[4].free_count == 40


def test_inflight_plan_pins_parent_and_failed_execution_can_release_everything():
    pool, cache = make_cache(snapshots=3)
    seq = Sequence(list(range(80)))
    publish(pool, cache, seq, 0, 16)
    plan = cache.plan([seq], ((seq.seq_id, 16, 48),))
    step = pool.reserve_prefill(((seq.seq_id, 16, 48),), snapshot_ends=plan.snapshot_ends)
    with pytest.raises(RuntimeError, match="committed physical snapshot"):
        cache.publish(plan, step)
    assert not cache.evict_for_capacity({}, snapshot_rows=3)
    pool.abort(step)
    cache.finish_plan(plan)
    cache.release_sequence(seq.seq_id)
    assert cache.evict_for_capacity({}, snapshot_rows=3)
    assert not cache.index and not pool.snapshots and not pool.live
    assert pool.slots[4].free_count == 40


def test_duplicate_batch_prefix_uses_one_snapshot_chain_and_keeps_logits_work():
    pool, cache = make_cache()
    seqs = [Sequence(list(range(64))) for _ in range(2)]
    requests = tuple((seq.seq_id, 0, 64) for seq in seqs)
    plan = cache.plan(seqs, requests)
    step = pool.reserve_prefill(requests, snapshot_ends=plan.snapshot_ends)
    pool.commit(step)
    cache.publish(plan, step)
    assert len(cache.index) == 4 and len(pool.snapshots) == 4
    probe = Sequence(list(range(64)))
    cache.refresh_hit(probe)
    assert probe.prefix_cache_hit_len == 48
    assert probe.num_prompt_tokens - probe.prefix_cache_hit_len > 0
    cache.clear_hit(probe)
    for seq in seqs:
        pool.release_sequence(seq.seq_id)
        cache.release_sequence(seq.seq_id)
    assert cache.evict_for_capacity({}, snapshot_rows=8)
    assert pool.slots[4].free_count == 40


def test_scheduler_eviction_releases_storage_created_in_inference_mode():
    # Startup builds inference tensors, but scheduler admission can reclaim
    # their prefix rows outside a model forward when compressed slots run out.
    with torch.inference_mode():
        pool, cache = make_cache()
        seq = Sequence(list(range(80)))
        publish(pool, cache, seq, 0, 64)
        pool.release_sequence(seq.seq_id)
        cache.release_sequence(seq.seq_id)
    assert all(table.is_inference() for table in pool.tables.values())
    assert cache.evict_for_capacity({4: len(pool.slots[4].refs)})
    assert not pool.snapshots and not cache.index
    for ratio, allocator in pool.slots.items():
        assert allocator.free_count == len(allocator.refs)
        assert not allocator.refs.any()
        assert torch.all(pool.tables[ratio] == -1)
