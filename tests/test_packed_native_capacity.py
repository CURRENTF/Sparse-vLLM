"""Packed admission includes rounded pages and shared materialization storage."""

import pytest
import torch

from sparsevllm.engine.cache_manager.storage.native_capacity import NativeCacheGeometry
from sparsevllm.engine.cache_manager.storage.native_pool import NativeCachePool
from sparsevllm.engine.cache_manager.storage.packed_gather import PackedSharedKVGather
from sparsevllm.engine.cache_manager.storage.packed_shared_kv import PackedSharedKVStorage
from sparsevllm.engine.cache_manager.storage.shared_kv import CompressionCarryStorage, SharedKVStorage


def _physical_bytes(ratios, rows, prefill, capacities, max_model_len):
    tensors = []
    page_counts = {}
    for ratio in ratios:
        storage = PackedSharedKVStorage()
        storage.allocate(num_layers=1, num_slots=rows * 128 + prefill + capacities.get(ratio, 0), device="cpu")
        tensors.extend(storage.accounting_tensors())
        page_counts[ratio] = storage.cache.shape[1]
        if ratio:
            carry = CompressionCarryStorage(num_rows=rows, ratio=ratio, head_dim=512, device="cpu")
            tensors.extend(carry.accounting_tensors())
        if ratio == 4:
            carry = CompressionCarryStorage(num_rows=rows, ratio=4, head_dim=128, device="cpu")
            tensors.extend(carry.accounting_tensors())
            index = SharedKVStorage(head_dim=128)
            index.allocate(num_layers=1, num_slots=capacities[4], device="cpu")
            tensors.extend(index.accounting_tensors())
    for ratio in capacities:
        tensors.append(torch.empty(rows, max(1, max_model_len // ratio), dtype=torch.int32))
    # Allocate one materialization buffer across layers, plus per-ratio physical
    # page metadata. Counting these as per-layer or omitting them breaks admission.
    tensors.extend(PackedSharedKVGather(page_counts, device="cpu").accounting_tensors())
    return sum(t.numel() * t.element_size() for t in tensors)


@pytest.mark.parametrize("ratios,capacities", [((0, 4, 4, 128), {4: 65, 128: 7}), ((0,), {})])
def test_packed_capacity_matches_physical_allocations(ratios, capacities):
    geometry = NativeCacheGeometry(ratios, 512, 33, packed_kv=True)
    actual = _physical_bytes(ratios, 5, 33, capacities, 512)
    assert geometry.allocation_bytes(3, 2, capacities) == actual
    pool = NativeCachePool(ratios=ratios, live_rows=3, snapshot_rows=2, max_model_len=512,
                           prefill_capacity=33, compressed_capacities=capacities, device="cpu", packed_kv=True)
    assert pool.allocated_bytes() == actual


def test_packed_admission_finds_largest_capacity_across_page_rounding():
    # A linear bytes-per-token estimate misses page growth and the corresponding
    # growth of the shared BF16 gather buffer.
    geometry = NativeCacheGeometry((4, 4), 512, 33, packed_kv=True)
    budget = _physical_bytes((4, 4), 2, 33, {4: 64}, 512) - 1
    plan = geometry.capacity_for_budget(budget, live_rows=1, snapshot_rows=1)
    selected = plan.compressed_capacities[4]
    actual = _physical_bytes((4, 4), 2, 33, {4: selected}, 512)
    assert actual == plan.allocated_bytes <= budget
    assert actual + plan.unused_bytes == budget
    assert _physical_bytes((4, 4), 2, 33, {4: selected + 1}, 512) > budget
    with pytest.raises(MemoryError):
        geometry.capacity_for_budget(_physical_bytes((4, 4), 2, 33, {4: 1}, 512) - 1,
                                     live_rows=1, snapshot_rows=1)
