"""CPU-owned allocation contracts; CUDA snapshot numerics are tested separately."""

import numpy as np
import pytest
import torch

from sparsevllm.engine.cache_manager.storage.native_pool import NativeCachePool
from sparsevllm.engine.cache_manager.storage.native_capacity import NativeCacheGeometry


def _pool(*, compressed_capacity=130):
    return NativeCachePool(ratios=(0, 4, 4, 128), live_rows=2, snapshot_rows=2,
                           max_model_len=512, prefill_capacity=512,
                           compressed_capacities={4: compressed_capacity, 128: 6},
                           device="cpu", head_dim=8, index_dim=4, window_size=8)


def test_prefix_slots_survive_parent_release_and_are_reclaimed_after_last_owner():
    # A saved prefix can outlive its originating request, then outlive radix
    # eviction through an attached child. Only the child's new suffix is private.
    pool = _pool()
    step = pool.reserve_prefill(((1, 0, 260),), snapshot_ends={1: (127, 129)})
    early, later = step.snapshots
    with pytest.raises(ValueError, match="unpublished"):
        pool.attach(3, later)
    pool.commit(step)
    with pytest.raises(RuntimeError, match="pending request reservation"):
        pool.commit(step)
    shared = {ratio: pool.host_tables[ratio][later.row, :later.length // ratio].copy() for ratio in pool.slots}
    # Same-ratio layers share step metadata but retain their own physical carry.
    assert step.batches[1].attention_indices is step.batches[2].attention_indices
    assert step.batches[1].compression.state_kv is not step.batches[2].compression.state_kv
    pool.release_sequence(1)
    pool.release_snapshot(early)
    for ratio, allocator in pool.slots.items():
        assert np.count_nonzero(allocator.refs) == len(shared[ratio])
    pool.attach(3, later)
    child = pool.live[3]
    for ratio in pool.slots:
        np.testing.assert_array_equal(pool.host_tables[ratio][child.row, :child.length // ratio], shared[ratio])
    pool.release_snapshot(later)
    with pytest.raises(ValueError, match="stale"):
        pool.attach(4, later)
    suffix = pool.reserve_prefill(((3, 129, 260),))
    for ratio, allocator in pool.slots.items():
        row = pool.host_tables[ratio][child.row]
        new = row[129 // ratio:260 // ratio]
        assert set(new).isdisjoint(shared[ratio])
        assert np.all(allocator.refs[row[:260 // ratio]] == 1)
    pool.abort(suffix)
    assert not pool.live and not pool.snapshots
    for allocator in pool.slots.values():
        assert allocator.free_count == len(allocator.refs)
        assert not allocator.refs.any()
        assert len(np.unique(allocator.free)) == len(allocator.free)


def test_failed_batch_reservation_does_not_mutate_other_requests_or_pools():
    # Admission cannot partially allocate the first member of a mixed batch
    # when another member exhausts a compressed pool or uses stale coordinates.
    pool = _pool(compressed_capacity=2)
    initial = pool.reserve_prefill(((1, 0, 4),))
    pool.commit(initial)
    before = tuple(t.clone() for t in pool.accounting_tensors())
    refs = {ratio: allocator.refs.copy() for ratio, allocator in pool.slots.items()}
    free = {ratio: allocator.free_count for ratio, allocator in pool.slots.items()}
    for requests, error in ((((2, 0, 1), (1, 4, 20)), RuntimeError),
                            (((2, 0, 1), (1, 5, 6)), ValueError)):
        with pytest.raises(error):
            pool.reserve_prefill(requests)
        assert set(pool.live) == {1}
        assert pool.live[1].length == 4 and pool.live[1].pending_end is None
        assert len(pool.free_live_rows) == 1 and len(pool.free_snapshot_rows) == 2
        for actual, previous in zip(pool.accounting_tensors(), before):
            torch.testing.assert_close(actual, previous, rtol=0, atol=0, equal_nan=True)
        for ratio, allocator in pool.slots.items():
            np.testing.assert_array_equal(allocator.refs, refs[ratio])
            assert allocator.free_count == free[ratio]
    failed = pool.reserve_prefill(((1, 4, 8),), snapshot_ends={1: (7,)})
    pool.abort(failed)
    retry = pool.reserve_prefill(((1, 0, 8),))
    with pytest.raises(RuntimeError, match="pending request reservation"):
        pool.commit(failed)
    pool.commit(retry)
    pool.release_sequence(1)
    assert all(allocator.free_count == len(allocator.refs) for allocator in pool.slots.values())


def test_decode_capacity_failure_preserves_staging_and_request_state():
    # Two requests crossing a compression boundary compete for the final slot;
    # neither the first reservation nor the next graph's staging may leak out.
    pool = _pool(compressed_capacity=1)
    initial = pool.reserve_prefill(((1, 0, 3), (2, 0, 3)))
    pool.commit(initial)
    state = pool.make_decode_state(2)
    before_host, before_device = state.host.clone(), state.device.clone()
    with pytest.raises(RuntimeError, match="complete decode step"):
        pool.reserve_decode(((1, 3), (2, 3)), state)
    torch.testing.assert_close(state.host, before_host, rtol=0, atol=0)
    torch.testing.assert_close(state.device, before_device, rtol=0, atol=0)
    assert all(row.length == 3 and row.pending_end is None for row in pool.live.values())
    assert pool.slots[4].free_count == 1
    single = pool.reserve_decode(((2, 3),), state)
    assert state.host[0, 1] == -1 and state.host[1, 1] == -1
    assert pool.slots[4].free_count == 0
    pool.abort(single)
    pool.release_sequence(1)
    assert pool.slots[4].free_count == 1


@pytest.mark.parametrize("ratios", [(0, 4, 4, 128), (128,)])
def test_native_byte_budget_matches_allocated_storage(ratios):
    # An omitted carry/index tensor or duplicated shared table in the planner
    # would make the actual allocation violate its advertised byte budget.
    geometry = NativeCacheGeometry(ratios, 512, 32, head_dim=8, index_dim=4, window_size=8)
    budget = geometry.row_bytes * 5 + geometry.temporary_bytes
    plan = geometry.capacity_for_budget(budget, live_rows=2, snapshot_rows=2)
    pool = NativeCachePool(ratios=ratios, live_rows=2, snapshot_rows=2, max_model_len=512,
                           prefill_capacity=32, compressed_capacities=plan.compressed_capacities,
                           device="cpu", head_dim=8, index_dim=4, window_size=8)
    assert pool.allocated_bytes() == plan.allocated_bytes <= budget
    assert plan.allocated_bytes + plan.unused_bytes == budget
    with pytest.raises(MemoryError):
        geometry.capacity_for_budget(geometry.row_bytes * 4 + geometry.temporary_bytes,
                                     live_rows=2, snapshot_rows=2)
