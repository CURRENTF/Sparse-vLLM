"""Physical writes, carry forking and recycling for native attention caches."""

import pytest
import torch

from sparsevllm.engine.cache_manager.base import SharedKVWrite
from sparsevllm.engine.cache_manager.storage.shared_kv import CompressionCarryStorage, SharedKVStorage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_kv_graph_write_masks_inactive_events_and_updates_slots():
    storage = SharedKVStorage(head_dim=512)
    storage.allocate(num_layers=2, num_slots=13, device="cuda")
    storage.cache.fill_(71.)
    values = torch.randn((4, 1, 512), device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([2, 5, -1, 8], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, -1, 12, 16], device="cuda", dtype=torch.int32)
    payload = SharedKVWrite(values, positions)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        storage.store(0, slots, payload)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        storage.store(0, slots, payload)
    storage.cache.fill_(71.)
    expected = storage.cache.clone()
    for _ in range(2):
        for index, (slot, position) in enumerate(zip(slots.cpu().tolist(), positions.cpu().tolist())):
            if slot >= 0 and position >= 0:
                expected[0, slot].copy_(values[index])
        graph.replay()
        torch.testing.assert_close(storage.cache, expected, rtol=0, atol=0)
        slots.copy_(torch.tensor([3, 5, 1, -1], device="cuda", dtype=torch.int32))
        positions.copy_(torch.tensor([-1, 20, 24, 28], device="cuda", dtype=torch.int32))
        values.mul_(.5)
    # Prefix copies may overlap source and destination lists.
    src = torch.tensor([2, 3, 5], device="cuda")
    dst = torch.tensor([5, 2, 3], device="cuda")
    snapshot = storage.cache[0, src].clone()
    storage.copy_slots(0, src, dst)
    torch.testing.assert_close(storage.cache[0, dst], snapshot, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [4, 128])
def test_compression_carry_fork_and_row_reuse_preserve_independent_requests(ratio):
    storage = CompressionCarryStorage(num_rows=4, ratio=ratio, head_dim=128, device="cuda")
    for tensor in storage.accounting_tensors():
        tensor.normal_()
    original = tuple(t.clone() for t in storage.accounting_tensors())
    storage.copy_rows(torch.tensor([0, 0], device="cuda", dtype=torch.int32),
                      torch.tensor([1, 2], device="cuda", dtype=torch.int32))
    storage.clear_rows(torch.tensor([0], device="cuda", dtype=torch.int32))
    for tensor, before in zip(storage.accounting_tensors(), original):
        assert torch.count_nonzero(tensor[0]) == 0
        torch.testing.assert_close(tensor[1], before[0], rtol=0, atol=0)
        torch.testing.assert_close(tensor[2], before[0], rtol=0, atol=0)
        torch.testing.assert_close(tensor[3], before[3], rtol=0, atol=0)
        tensor[1].add_(1.)
        torch.testing.assert_close(tensor[2], before[0], rtol=0, atol=0)
