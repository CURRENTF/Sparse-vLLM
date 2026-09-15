"""Native pool transitions through packed prefill and captured sparse decode."""

from dataclasses import replace

import pytest
import torch

from sparsevllm.engine.cache_manager.storage.native_pool import NativeCachePool
from sparsevllm.engine.cache_manager.native_attention import IndexedPackedSharedKVView
from sparsevllm.kernels.external.vllm_cache import shared_kv_cache_ops
from sparsevllm.operators.indexed_shared_kv_attention import (
    IndexedSharedKVAttentionSpec, prepare_indexed_shared_kv_attention,
)


def _quantized(values):
    result = values.clone()
    groups = values[..., :448].float().reshape(-1, 7, 64)
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)))
    rounded = (groups / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
    result[..., :448] = rounded.reshape(result[..., :448].shape).bfloat16()
    return result


def _reference(query, view, sink):
    indices = view.indices[:, 0]
    keys = view.kv[indices.clamp_min(0).long(), 0].float()
    scores = torch.einsum("qhd,qkd->qhk", query.float(), keys) * 512**-.5
    scores.masked_fill_((indices < 0)[:, None], -torch.inf)
    scores = torch.cat((scores, sink[None, :, None].expand(len(query), -1, 1)), -1)
    probabilities = scores.softmax(-1)[..., :-1]
    return torch.einsum("qhk,qkd->qhd", probabilities, keys).bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_pool_prefill_prefix_fork_and_decode_graph():
    if shared_kv_cache_ops() is None:
        pytest.skip("optional vLLM cache kernels are unavailable")
    torch.manual_seed(731)
    pools = [NativeCachePool(ratios=(4,), live_rows=2, snapshot_rows=1, max_model_len=256,
                             prefill_capacity=64, compressed_capacities={4: 65}, device="cuda", packed_kv=packed)
             for packed in (False, True)]
    for pool in pools:
        pool.layers[0].kv.cache.zero_()
        assert pool.allocated_bytes() == pool.geometry.allocation_bytes(2, 1, {4: 65})
    provider = prepare_indexed_shared_kv_attention(
        IndexedSharedKVAttentionSpec(64, 512, 640, 512**-.5, 64, cache_dtype=torch.uint8), device_index=0)
    sink = torch.randn(64, device="cuda", dtype=torch.float32)

    def forward(pool, batch, keys, compressed, *, decode=False):
        layer = pool.layers[0]
        layer.prepare(batch)
        layer.store_window(batch, keys)
        positions = (torch.where((batch.window.positions + 1) % 4 == 0, batch.window.positions, -1)
                     if decode else batch.compression.boundary_ends - 1)
        layer.store_compressed(batch, compressed, positions)
        # Select every causally visible compressed token in this short trace.
        table = layer.compressed_slots[batch.window.request_rows.clamp_min(0).long()]
        visible = torch.arange(table.shape[1], device="cuda")[None] < batch.compressed_lengths[:, None]
        batch.selected_compressed.fill_(-1)
        batch.selected_compressed[:, :table.shape[1]].copy_(torch.where(visible, table, -1))
        view = layer.attention_view(batch, batch.selected_compressed)
        return view

    steps = [pool.reserve_prefill(((1, 0, 17), (2, 0, 5)), snapshot_ends={1: (16,)}) for pool in pools]
    query = torch.randn(22, 64, 512, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(22, 1, 512, device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    expected_view = forward(pools[0], steps[0].batches[0], _quantized(keys), _quantized(compressed))
    packed_view = forward(pools[1], steps[1].batches[0], keys, compressed)
    actual = provider.run(query, packed_view, sink)
    torch.testing.assert_close(actual, _reference(query, expected_view, sink), rtol=2e-2, atol=2e-2)
    plan = pools[1].gather.plans[4]
    assert plan.active_pages < plan.page_capacity
    for pool, step in zip(pools, steps):
        pool.layers[0].finish_window(step.batches[0], _quantized(keys) if pool is pools[0] else keys)
        pool.commit(step)
        pool.release_sequence(2)
        pool.attach(3, step.snapshots[0])

    # A second prefill must gather restored windows and historical compressed
    # pages as well as freshly reserved slots, without reading inactive rows.
    steps = [pool.reserve_prefill(((1, 17, 21), (3, 16, 25))) for pool in pools]
    query = torch.randn(13, 64, 512, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(13, 1, 512, device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    expected_view = forward(pools[0], steps[0].batches[0], _quantized(keys), _quantized(compressed))
    packed_view = forward(pools[1], steps[1].batches[0], keys, compressed)
    torch.testing.assert_close(provider.run(query, packed_view, sink), _reference(query, expected_view, sink),
                               rtol=2e-2, atol=2e-2)
    for pool, step in zip(pools, steps):
        pool.layers[0].finish_window(step.batches[0], _quantized(keys) if pool is pools[0] else keys)
        pool.commit(step)

    states = [pool.make_decode_state(3) for pool in pools]
    query = torch.randn(3, 64, 512, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(3, 1, 512, device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)

    def decode():
        states[1].publish()
        view = forward(pools[1], states[1].batches[0], keys, compressed, decode=True)
        return provider.run(query, view, sink)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode()
    try:
        for iteration in range(4):
            requests = ((1, 21 + iteration), (3, 25 + iteration))
            steps = [pool.reserve_decode(requests, state) for pool, state in zip(pools, states)]
            query.mul_(.9)
            keys.add_(.125)
            compressed.mul_(.75)
            sink.add_(.25)
            states[0].publish()
            expected_view = forward(pools[0], states[0].batches[0], _quantized(keys),
                                    _quantized(compressed), decode=True)
            graph.replay()
            torch.testing.assert_close(captured, _reference(query, expected_view, sink), rtol=2e-2, atol=2e-2)
            assert torch.count_nonzero(captured[2]) == 0
            if iteration == 1:
                # Preparing an eager batch of another size must not replace the
                # scheduler tensors retained by the three-row decode graph.
                layer = pools[1].layers[0]
                smaller = IndexedPackedSharedKVView(layer.kv.layer_payload(0), states[1].batches[0].attention_indices[:1])
                torch.testing.assert_close(provider.run(query[:1], smaller, sink), captured[:1], rtol=2e-2, atol=2e-2)
                with pytest.raises(ValueError, match="DSv4 FP8 pages"):
                    provider.run(query[:1], replace(smaller, payload=replace(
                        smaller.payload, cache=smaller.payload.cache.contiguous())), sink)
            for pool, step in zip(pools, steps):
                pool.commit(step)
    finally:
        graph.reset()
    for pool in pools:
        pool.release_sequence(1)
        pool.release_sequence(3)
        for snapshot in list(pool.snapshots.values()):
            pool.release_snapshot(snapshot)
        assert pool.slots[4].free_count == len(pool.slots[4].refs)
