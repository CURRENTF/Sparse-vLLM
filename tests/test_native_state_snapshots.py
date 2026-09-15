"""Exact prefix state from the middle of a chunk, independent of its final carry."""

import pytest
import torch

from sparsevllm.engine.cache_manager.methods.deepseek_v4 import CompressionChunk, StateSnapshot
from sparsevllm.engine.cache_manager.storage.native_layer import NativeAttentionLayerStorage
from sparsevllm.engine.cache_manager.storage.packed_gather import PackedSharedKVGather
from sparsevllm.engine.cache_manager.storage.shared_kv_views import SharedKVRegions
from sparsevllm.operators.compression import CompressionOpSpec, prepare_compression
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("packed_kv", [False, True])
def test_native_prefix_snapshot_restore_and_graph(ratio, packed_kv):
    # Early snapshots need old carry; later snapshots need an intermediate
    # chunk tail that the ordinary commit would overwrite. Both active
    # requests advance beyond every saved boundary before either is restored.
    torch.manual_seed(731)
    rows, capacity, dim, total = 12, 535, 512, 400
    columns = total // ratio if ratio else 0
    slots = torch.arange(rows * columns, device="cuda", dtype=torch.int32).view(rows, columns) if ratio else None
    regions = SharedKVRegions(rows, 128, rows * columns, capacity)
    gather = None
    if packed_kv:
        from sparsevllm.kernels.external.vllm_cache import shared_kv_cache_ops
        if shared_kv_cache_ops() is None:
            pytest.skip("optional vLLM cache kernels are unavailable")
        gather = PackedSharedKVGather({ratio: (regions.num_slots + 63) // 64}, device="cuda")
    cache = NativeAttentionLayerStorage(regions=regions, head_dim=dim, ratio=ratio,
                                       compressed_slots=slots, device="cuda", packed_kv=packed_kv, gather=gather)
    cache.kv.cache.fill_(71.)
    histories = {row: torch.randn((total, 1, dim), device="cuda", dtype=torch.bfloat16) for row in (0, 1)}
    projections, providers, apes = {}, {}, {}
    for name, carry in (("compression", cache.carry), ("index_compression", cache.index_carry)):
        if carry is not None:
            head_dim = carry.shape.head_dim
            width = carry.shape.row_shape[1]
            providers[name] = prepare_compression(CompressionOpSpec(ratio, head_dim, capacity, 2), device_index=0)
            apes[name] = torch.randn((ratio, width), device="cuda", dtype=torch.float32)
            projections[name] = {row: (torch.randn((total, width), device="cuda", dtype=torch.float32),
                                       torch.randn((total, width), device="cuda", dtype=torch.float32)) for row in (0, 1)}
    lock_workspace_manager()

    def run(batch, keys, projected):
        cache.prepare(batch)
        cache.store_window(batch, keys)
        for name, op in providers.items():
            op.run(*projected[name], apes[name], getattr(batch, name))
        cache.finish_window(batch, keys)

    def packed(chunks):
        keys = torch.cat([histories[c.request_row][c.start:c.start+c.length] for c in chunks])
        projected = {name: tuple(torch.cat([values[c.request_row][kind][c.start:c.start+c.length]
                                          for c in chunks]) for kind in (0, 1))
                     for name, values in projections.items()}
        return keys, projected

    def expected_ring(values, end, window):
        result = torch.zeros((window, *values.shape[1:]), device="cuda", dtype=values.dtype)
        positions = torch.arange(max(0, end-window), end, device="cuda")
        result[positions % window] = values[positions]
        return result

    def assert_snapshot(source, end, destination):
        expected = expected_ring(histories[source], end, 128)
        if packed_kv:
            window = torch.empty(1, 128, dim, dtype=torch.bfloat16, device="cuda")
            table = torch.tensor([[destination * 2, destination * 2 + 1]], device="cuda", dtype=torch.int32)
            cache.kv.gather(0, window, table, torch.tensor([128], device="cuda", dtype=torch.int32))
            window = window[0, :, None]
            groups = expected[:, 0, :448].float().reshape(128, 7, 64)
            scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)))
            rounded = (groups / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
            expected[:, 0, :448] = rounded.flatten(1).bfloat16()
        else:
            window = cache.kv.cache[0, destination*128:(destination+1)*128]
        torch.testing.assert_close(window, expected, rtol=0, atol=0)
        for name, values in projections.items():
            view = getattr(cache, "carry" if name == "compression" else "index_carry")
            for tensor, history in zip(view.accounting_tensors(), values[source]):
                torch.testing.assert_close(tensor[destination], expected_ring(history, end, len(tensor[destination])),
                                           rtol=0, atol=0)

    try:
        chunks = (CompressionChunk(0, 0, 133), CompressionChunk(1, 0, 3))
        run(cache.make_prefill_batch(chunks), *packed(chunks))
        chunks = (CompressionChunk(0, 133, 266), CompressionChunk(1, 3, 269))
        snapshots = (StateSnapshot(0, 134, 2), StateSnapshot(0, 257, 3), StateSnapshot(0, 399, 4),
                     StateSnapshot(1, 4, 5), StateSnapshot(1, 128, 6), StateSnapshot(1, 271, 7))
        with pytest.raises(ValueError, match="outside the active batch"):
            cache.make_prefill_batch(chunks, snapshots=(StateSnapshot(0, 134, 1),))
        run(cache.make_prefill_batch(chunks, snapshots=snapshots), *packed(chunks))
        for snapshot in snapshots:
            assert_snapshot(snapshot.request_row, snapshot.end, snapshot.snapshot_row)
        source_rows = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
        destination_rows = torch.tensor([8, 9], device="cuda", dtype=torch.int32)
        cache.restore_state_rows(source_rows, destination_rows)
        assert_snapshot(0, 134, 8)
        assert_snapshot(1, 4, 9)
        # Forks get distinct continuations. Original snapshot rows stay immutable.
        for destination, source, end in ((8, 0, 134), (9, 1, 4)):
            histories[destination] = histories[source].clone()
            histories[destination][end:end+5].add_(1.)
            for values in projections.values():
                values[destination] = tuple(v.clone() for v in values[source])
                for v in values[destination]:
                    v[end:end+5].add_(1.)
        chunks = (CompressionChunk(8, 134, 5), CompressionChunk(9, 4, 5))
        run(cache.make_prefill_batch(chunks), *packed(chunks))
        assert_snapshot(8, 139, 8)
        assert_snapshot(9, 9, 9)
        for snapshot in snapshots:
            assert_snapshot(snapshot.request_row, snapshot.end, snapshot.snapshot_row)
        batch = cache.make_decode_batch(2, capture_snapshots=True)
        keys = torch.zeros((2, 1, dim), device="cuda", dtype=torch.bfloat16)
        projected = {name: tuple(torch.zeros((2, ape.shape[1]), device="cuda", dtype=torch.float32) for _ in range(2))
                     for name, ape in apes.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(batch, keys, projected)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(batch, keys, projected)
        try:
            for token, source, position, destination in ((0, 8, 139, 10), (1, 9, 9, 11)):
                batch.window.request_rows.fill_(-1)
                batch.window.positions.fill_(-1)
                batch.state_snapshots.destination_rows.fill_(-1)
                batch.window.request_rows[token] = source
                batch.window.positions[token] = position
                batch.state_snapshots.destination_rows[token] = destination
                keys[token].copy_(histories[source][position])
                for name, values in projected.items():
                    for value, history in zip(values, projections[name][source]):
                        value[token].copy_(history[position])
                graph.replay()
                assert_snapshot(source, position+1, destination)
                assert_snapshot(source, position+1, source)
            for snapshot in snapshots:
                assert_snapshot(snapshot.request_row, snapshot.end, snapshot.snapshot_row)
        finally:
            graph.reset()
    finally:
        close_workspace_manager()
