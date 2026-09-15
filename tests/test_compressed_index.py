"""Native score rounding, causal top-k and changing physical maps in graphs."""

import pytest
import torch
from types import SimpleNamespace

from sparsevllm.engine.cache_manager.native_attention import CompressedIndexView
from sparsevllm.operators.compressed_index import CompressedIndexSpec, prepare_compressed_index
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager
from sparsevllm.engine.sparse_methods.native_index import NativeIndexSelection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_query_batching_preserves_selection_and_request_isolation():
    # Larger query batches change the persistent scoring grid. Protect causal
    # lengths and physical request maps when reducing the number of launches.
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        rows, capacity, heads, dim = 129, 1024, 16, 128
        selectors = [NativeIndexSelection(num_heads=heads, head_dim=dim,
            max_index_tokens=capacity, query_chunk_size=chunk, top_k=512,
            parallel_context=SimpleNamespace(attn_tp_size=1), device_index=0)
            for chunk in (7, rows)]
        lock_workspace_manager()
        query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(2 * capacity, dim, device="cuda", dtype=torch.bfloat16)
        slots = torch.randperm(2 * capacity, device="cuda", dtype=torch.int32).view(2, capacity)
        request_rows = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
        lengths = torch.linspace(0, capacity, rows, device="cuda").int()
        view = CompressedIndexView(keys, slots, request_rows, lengths)
        outputs = [torch.empty(rows, 512, device="cuda", dtype=torch.int32) for _ in selectors]
        for _ in range(2):
            for selector, output in zip(selectors, outputs):
                selector.select_compressed_index(query, weights, view, out=output)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            slots.copy_(slots.roll(1, 0))
            query.mul_(.5)
    finally:
        close_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("capacity,qat", [(1024, False), (65539, False), (65539, True)])
def test_compressed_index_reference_isolation_and_graph(capacity, qat):
    torch.manual_seed(731)
    rows, heads, dim, top_k = 7, 64, 128, 512
    op = prepare_compressed_index(CompressedIndexSpec(heads, dim, capacity, rows, top_k), device_index=0)
    lock_workspace_manager()
    def values(shape):
        if not qat:
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        # Independently generate values from the native per-32 MXFP4 lattice.
        codebook = torch.tensor([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.],
                                device="cuda", dtype=torch.bfloat16)
        codes = codebook[torch.randint(len(codebook), shape, device="cuda")]
        scales = torch.exp2(torch.randint(-2, 2, (*shape[:-1], dim // 32, 1), device="cuda").float())
        return (codes.unflatten(-1, (-1, 32)) * scales).flatten(-2).bfloat16()

    query = values((rows, heads, dim))
    weights = torch.randn((rows, heads), device="cuda", dtype=torch.bfloat16) * .01
    keys = values((4 * capacity, dim))
    slots = torch.randperm(4 * capacity, device="cuda", dtype=torch.int32).view(4, capacity)
    request_rows = torch.tensor([2, 0, 1, -1, 0, 2, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([513, capacity, 511, 0, 0, 17, 512], device="cuda", dtype=torch.int32)
    view = CompressedIndexView(keys, slots, request_rows, lengths)
    selected = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
    raw = torch.empty_like(selected)
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            scores = op.score(query, weights, view)
            op.select(scores, view, out=selected, raw_indices=raw)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scores = op.score(query, weights, view)
            op.select(scores, view, out=selected, raw_indices=raw)
        for replay in range(3):
            graph.replay()
            for token, (request, length) in enumerate(zip(request_rows.tolist(), lengths.tolist())):
                count = min(length, top_k)
                assert (selected[token, count:] == -1).all()
                assert (raw[token, count:] == -1).all()
                if not count:
                    continue
                gathered = keys[slots[request, :length].long()]
                expected_scores = ((query[token] @ gathered.t()).relu() * weights[token, :, None]).sum(0)
                torch.testing.assert_close(scores[token, :length], expected_scores,
                                           rtol=0 if qat else 1e-2, atol=0 if qat else 4e-3)
                ids = raw[token, :count].long()
                assert ((ids >= 0) & (ids < length)).all()
                assert ids.unique().numel() == count
                torch.testing.assert_close(selected[token, :count], slots[request, ids], rtol=0, atol=0)
                # The reference top-k does not specify an index tie-break. Check
                # the full score multiset, including a replay with all tied scores.
                torch.testing.assert_close(expected_scores[ids].sort().values,
                                           expected_scores.topk(count).values.sort().values, rtol=0, atol=0)
            eager_scores = op.score(query, weights, view)
            eager_selected = torch.empty_like(selected)
            op.select(eager_scores, view, out=eager_selected)
            torch.testing.assert_close(selected, eager_selected, rtol=0, atol=0)
            query.mul_(.5)
            slots.copy_(slots.roll(1, 0))
            lengths.copy_(torch.tensor([capacity, 0, 513, 0, 512, 1, 511], device="cuda", dtype=torch.int32))
            if replay == 1:
                weights.zero_()
        empty = CompressedIndexView(keys, slots, request_rows[:0], lengths[:0])
        assert op.select(op.score(query[:0], weights[:0], empty), empty, out=selected[:0]).shape == (0, top_k)
        graph.reset()
    finally:
        close_workspace_manager()
