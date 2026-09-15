"""Numerical and capture coverage of the installed standalone SGL adapter."""

from dataclasses import replace

import pytest
import torch

from sparsevllm.engine.cache_manager.native_attention import IndexedSharedKVView
from sparsevllm.operators.indexed_shared_kv_attention import (
    INDEXED_SHARED_KV_REGISTRY,
    IndexedSharedKVAttentionSpec,
    prepare_indexed_shared_kv_attention,
)
from sparsevllm.operators.registry import NoProviderError, OpResolver
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


def test_unsupported_local_heads_rejected_before_loading_device_kernel():
    # FlashMLA's binary rejects TP-sharded 32-head inputs. Never silently pad,
    # choose dense attention, or defer this known error to first execution.
    caps = DeviceCaps(PlatformEnum.CUDA, "cuda", 0, "contract test", (9, 0),
                      supports_graph_capture=True)
    spec = IndexedSharedKVAttentionSpec(32, 512, 640, 512**-.5, max_query_tokens=5)
    with pytest.raises(NoProviderError, match="64 or 128"):
        OpResolver(INDEXED_SHARED_KV_REGISTRY).resolve(spec, caps)
    with pytest.raises(NoProviderError, match="SM90 or SM100"):
        OpResolver(INDEXED_SHARED_KV_REGISTRY).resolve(
            replace(spec, num_heads=64), replace(caps, compute_capability=(8, 0)))
    with pytest.raises(NoProviderError, match="physical cache storage"):
        OpResolver(INDEXED_SHARED_KV_REGISTRY).resolve(replace(spec, num_heads=64, cache_dtype=torch.float16), caps)


def _reference(q, kv, indices, sink, scale):
    results = []
    for query, selected in zip(q.float(), indices[:, 0]):
        keys = kv[selected[selected >= 0].long(), 0].float()
        scores = query @ keys.T * scale
        probabilities = torch.cat((scores, sink[:, None]), dim=1).softmax(dim=1)
        results.append(probabilities[:, :-1] @ keys)
    return torch.stack(results).bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("capacity", [128, 160, 640])
def test_sparse_sink_attention_replays_dynamic_indices(capacity):
    # The same captured callable must read a newly appended physical slot and
    # newly padded rows. A sink affects normalization even though its value is 0.
    torch.manual_seed(731)
    spec = IndexedSharedKVAttentionSpec(64, 512, capacity, 512**-.5, max_query_tokens=5)
    provider = prepare_indexed_shared_kv_attention(spec, device_index=0)
    query = torch.randn((5, 64, 512), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((3*capacity, 1, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.full((5, 1, capacity), -1, device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda")
    view = IndexedSharedKVView(cache, indices)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        provider.run(query, view, sink)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = provider.run(query, view, sink)
    address = captured.data_ptr()
    workspace_address = provider._padded_indices.data_ptr() if provider._padded_indices is not None else None
    for lengths in ((1, 4, 127, 128, 0), (capacity, 7, 0, 1, 32)):
        indices.fill_(-1)
        for row, length in enumerate(lengths):
            offset = (row % 3) * capacity
            indices[row, 0, :length] = torch.randperm(capacity, device="cuda")[:length] + offset
        query.mul_(0.75)
        sink.add_(1.)
        eager = provider.run(query, view, sink)
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.testing.assert_close(captured, _reference(query, cache, indices, sink, spec.softmax_scale),
                                   rtol=2e-2, atol=2e-2)
        assert captured.data_ptr() == address
        if workspace_address is not None:
            assert provider._padded_indices.data_ptr() == workspace_address
            assert torch.all(provider._padded_indices[:, :, capacity:] == -1)
    before = captured.clone()
    cache[capacity:2*capacity].mul_(2.)
    graph.replay()
    # Queries 0/2/3 cannot observe writes into request 1's physical region.
    torch.testing.assert_close(captured[[0, 2, 3]], before[[0, 2, 3]], rtol=0, atol=0)
    with pytest.raises(ValueError, match="selection capacity"):
        provider.run(query, IndexedSharedKVView(cache, indices[:, :, :-1]), sink)
    with pytest.raises(ValueError, match="workspace capacity"):
        provider.run(query.repeat(2, 1, 1), view, sink)
    with pytest.raises(ValueError, match="prepared device"):
        provider.run(query.cpu(), view, sink)
    empty = provider.run(query[:0], IndexedSharedKVView(cache, indices[:0]), sink)
    assert empty.shape == query[:0].shape
