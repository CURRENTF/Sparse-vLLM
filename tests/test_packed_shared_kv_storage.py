"""Physical packed-page ownership, byte accounting and external numerical writes."""

import pytest
import torch
import numpy as np

from sparsevllm.engine.cache_manager.base import SharedKVWrite
from sparsevllm.engine.cache_manager.storage.packed_shared_kv import (
    PackedSharedKVStorage, packed_shared_kv_bytes,
)
from sparsevllm.engine.cache_manager.storage.packed_gather import PackedSharedKVGather
from sparsevllm.kernels.external.vllm_cache import shared_kv_cache_ops


def test_overlapping_slot_copy_preserves_neighbor_bytes_and_other_layers():
    # A whole-page copy would overwrite independently owned neighbors; copying
    # data without its scale region silently changes restored prefix values.
    storage = PackedSharedKVStorage()
    storage.allocate(num_layers=2, num_slots=130, device="cpu")
    storage.cache.random_(0, 256)
    before = storage.cache.clone()
    expected = before.clone()
    sources, destinations = [0, 63, 64, 129], [64, 129, 63, 0]
    for source, destination in zip(sources, destinations):
        sp, so = divmod(source, 64)
        dp, do = divmod(destination, 64)
        expected[0, dp, do * 576:(do + 1) * 576] = before[0, sp, so * 576:(so + 1) * 576]
        expected[0, dp, 64 * 576 + do * 8:64 * 576 + (do + 1) * 8] = before[0, sp, 64 * 576 + so * 8:64 * 576 + (so + 1) * 8]
    storage.copy_slots(0, torch.tensor(sources), torch.tensor(destinations))
    torch.testing.assert_close(storage.cache, expected, rtol=0, atol=0)
    payload = storage.layer_payload(0)
    assert payload.slot_capacity == 130
    assert payload.cache.data_ptr() == storage.cache[0].data_ptr()
    assert storage.cache.numel() == 2 * packed_shared_kv_bytes(130)
    assert sum(t.numel() * t.element_size() for t in storage.accounting_tensors()) > storage.cache.numel()
    component, = storage.component_tensors(0)
    spec, = storage.component_specs(0)
    assert component.data_ptr() == storage.cache[0].data_ptr()
    assert component.numel() == len(component) * spec.row_bytes


def test_partial_page_padding_is_not_allocatable_capacity():
    storage = PackedSharedKVStorage()
    previous_dtype = torch.get_default_dtype()
    try:
        # Model initialization may change the default; the upstream writer's
        # rotary table still requires FP32.
        torch.set_default_dtype(torch.bfloat16)
        storage.allocate(num_layers=1, num_slots=65, device="cpu")
    finally:
        torch.set_default_dtype(previous_dtype)
    assert storage.accounting_tensors()[1].dtype == torch.float32
    storage.validate_slot_mapping(torch.tensor([-1, 0, 64], dtype=torch.int32))
    with pytest.raises(ValueError, match="out of bounds"):
        storage.validate_slot_mapping(torch.tensor([65], dtype=torch.int32))
    before = storage.cache.clone()
    for source, destination in ((-1, 0), (0, 65)):
        with pytest.raises(ValueError, match="logical capacity"):
            storage.copy_slots(0, torch.tensor([source]), torch.tensor([destination]))
        torch.testing.assert_close(storage.cache, before, rtol=0, atol=0)


def test_active_page_plan_deduplicates_and_rejects_invalid_updates_atomically():
    gather = PackedSharedKVGather({4: 7, 128: 3}, device="cpu")
    plan = gather.prepare(4, np.array([6, 0, 6, 3]))
    torch.testing.assert_close(plan.block_table, torch.tensor([[0, 3, 6]], dtype=torch.int32))
    torch.testing.assert_close(plan.page_map, torch.tensor([0, -1, -1, 1, -1, -1, 2], dtype=torch.int32))
    before = plan.metadata.clone()
    with pytest.raises(ValueError, match="physical domain"):
        gather.prepare(4, np.array([0, 7]))
    torch.testing.assert_close(plan.metadata, before)
    assert plan.active_pages == 3
    gather.prepare(4, np.array([1]))
    assert plan.page_map[0] == -1 and plan.page_map[1] == 0
    assert plan.length.item() == 64
    # Preparing another ratio cannot invalidate a preceding layer's page plan.
    saved = plan.metadata.clone()
    gather.prepare(128, np.array([2]))
    torch.testing.assert_close(plan.metadata, saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_storage_masks_writes_and_restores_values_across_graph_replay():
    if shared_kv_cache_ops() is None:
        pytest.skip("optional vLLM cache kernels are unavailable")
    storage = PackedSharedKVStorage()
    storage.allocate(num_layers=1, num_slots=130, device="cuda")
    storage.cache.zero_()
    torch.manual_seed(731)
    values = torch.randn(7, 1, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([0, 63, 64, 127, 128, 129, -1], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, -1, 4, 5, 6], device="cuda", dtype=torch.int32)
    out = torch.empty(1, 192, 512, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(3, device="cuda", dtype=torch.int32)[None]
    lengths = torch.tensor([192], device="cuda", dtype=torch.int32)

    def run():
        storage.store(0, slots, SharedKVWrite(values, positions))
        storage.gather(0, out, table, lengths)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(2):
        storage.cache.zero_()
        if iteration:
            values.mul_(.5)
            slots.copy_(slots.roll(2))
            positions.copy_(positions.roll(1))
        before_values = values.clone()
        graph.replay()
        torch.testing.assert_close(values, before_values, rtol=0, atol=0)
        valid = (slots >= 0) & (positions >= 0)
        x = values[valid, 0].float()
        groups = x[:, :448].reshape(-1, 7, 64)
        scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)))
        rounded = (groups / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
        expected = torch.zeros_like(out)
        expected[0, slots[valid].long()] = torch.cat([rounded.flatten(1), x[:, 448:]], -1).bfloat16()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        gather = PackedSharedKVGather({4: 3}, device="cuda")
        plan = gather.prepare(4, np.array([2, 0, 2]))
        compact, page_map = gather.materialize(storage, 4)
        physical = torch.cat((torch.arange(64, device="cuda"), torch.arange(128, 192, device="cuda")))
        torch.testing.assert_close(compact[:, 0], expected[0, physical], rtol=0, atol=0)
        torch.testing.assert_close(page_map, torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32))
        assert compact.data_ptr() == gather.values.data_ptr()
        gather.prepare(4, np.empty(0, dtype=np.int32))
        empty, _ = gather.materialize(storage, 4)
        assert empty.shape == (0, 1, 512) and plan.length.item() == 0
        source = torch.tensor([0, 64], device="cuda")
        destination = source.flip(0)
        storage.copy_slots(0, source, destination)
        storage.gather(0, out, table, lengths)
        snapshot = expected.clone()
        expected[0, destination] = snapshot[0, source]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph.reset()
