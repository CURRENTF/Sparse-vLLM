"""Chunked-prefill history preservation and physical decode view replay."""

import pytest
import torch

from sparsevllm.engine.cache_manager.base import SharedKVWrite
from sparsevllm.engine.cache_manager.native_attention import SharedKVWindowBatch
from sparsevllm.engine.cache_manager.storage.shared_kv import SharedKVStorage
from sparsevllm.engine.cache_manager.storage.shared_kv_views import SharedKVRegions


def _ints(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefill_reads_history_and_temporary_region_before_committing_tail():
    torch.manual_seed(731)
    regions = SharedKVRegions(3, 128, 17, 158)
    storage = SharedKVStorage(head_dim=512)
    storage.allocate(num_layers=1, num_slots=regions.num_slots, device="cuda")
    storage.cache.fill_(71.)
    histories = {row: torch.randn((end, 1, 512), device="cuda", dtype=torch.bfloat16)
                 for row, end in ((0, 270), (2, 145))}
    chunks = [(0, 130, 270, 0), (2, 127, 145, 140)]
    rows, positions, starts, ends, offsets = [], [], [], [], []
    for row, start, end, offset in chunks:
        old = range(max(0, start - 128), start)
        storage.store(0, _ints([row * 128 + p % 128 for p in old]), SharedKVWrite(histories[row][list(old)]))
        length = end - start
        rows.extend([row] * length)
        positions.extend(range(start, end))
        starts.extend([start] * length)
        ends.extend([end] * length)
        offsets.extend([offset] * length)
    batch = SharedKVWindowBatch(*map(_ints, (rows, positions, starts, ends, offsets)))
    values = torch.cat([histories[row][start:end] for row, start, end, _ in chunks])
    old_window = storage.cache[0, :regions.compressed_offset].clone()
    storage.store(0, _ints(range(regions.temporary_offset, regions.num_slots)), SharedKVWrite(values))
    selected = torch.stack([_ints([3, -1, 15]) for _ in rows])
    table = _ints([[2, 5, 9], [6, 10, 11], [1, 4, 16]])
    for kind in ("none", "selected", "all"):
        compressed = None if kind == "none" else selected if kind == "selected" else table
        lengths = _ints([2] * len(rows)) if kind == "all" else None
        capacity = 0 if compressed is None else 3
        indices = torch.empty((len(rows), 1, 128 + capacity), device="cuda", dtype=torch.int32)
        view = regions.attention_view(storage.cache[0], batch, indices, compressed=compressed,
                                      compressed_lengths=lengths, compressed_per_request=kind == "all")
        expected = []
        for row, pos, start, offset in zip(rows, positions, starts, offsets):
            window = [row * 128 + p % 128 if p < start else regions.temporary_offset + offset + p - start
                      for p in range(pos - 127, pos + 1)]
            if kind == "selected":
                window += [regions.compressed_offset + 3, -1, regions.compressed_offset + 15]
            elif kind == "all":
                window += [regions.compressed_offset + int(slot) for slot in table[row, :2].tolist()] + [-1]
            expected.append(window)
        torch.testing.assert_close(view.indices[:, 0], _ints(expected), rtol=0, atol=0)
        torch.testing.assert_close(storage.cache[0, :regions.compressed_offset], old_window, rtol=0, atol=0)
        for token in (0, 127, 139, 140, 157):
            row, pos = rows[token], positions[token]
            torch.testing.assert_close(view.kv[view.indices[token, 0, :128].long()],
                                       histories[row][pos - 127:pos + 1], rtol=0, atol=0)
    # Only the newest token for a ring slot may commit, after all chunk queries.
    commits = torch.empty(len(rows), device="cuda", dtype=torch.int32)
    regions.commit_slots(batch, commits)
    expected = [row * 128 + p % 128 if p >= end - 128 else -1 for row, p, end in zip(rows, positions, ends)]
    torch.testing.assert_close(commits, _ints(expected), rtol=0, atol=0)
    storage.store(0, commits, SharedKVWrite(values))
    for row, _, end, _ in chunks:
        slots = _ints([row * 128 + p % 128 for p in range(end - 128, end)])
        torch.testing.assert_close(storage.cache[0, slots.long()], histories[row][-128:], rtol=0, atol=0)
    assert (storage.cache[0, 128:256] == 71.).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_graph_updates_window_positions_rows_and_compressed_selection():
    regions = SharedKVRegions(3, 128, 17, 3)
    storage = SharedKVStorage(head_dim=512)
    storage.allocate(num_layers=1, num_slots=regions.num_slots, device="cuda")
    storage.cache.fill_(71.)
    batch = SharedKVWindowBatch(_ints([-1, 1, 2]), _ints([-1, 0, 145]))
    compressed = _ints([[-1, -1], [-1, -1], [3, 12]])
    indices = torch.empty((3, 1, 130), device="cuda", dtype=torch.int32)
    commits = torch.empty(3, device="cuda", dtype=torch.int32)
    values = torch.randn((3, 1, 512), device="cuda", dtype=torch.bfloat16)

    def run():
        regions.commit_slots(batch, commits)
        storage.store(0, commits, SharedKVWrite(values))
        regions.attention_view(storage.cache[0], batch, indices, compressed=compressed)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        before = storage.cache.clone()
        graph.replay()
        for token, (row, pos) in enumerate(zip(batch.request_rows.tolist(), batch.positions.tolist())):
            expected = [-1] * 130
            if row >= 0:
                count = min(128, pos + 1)
                expected[:count] = [row * 128 + p % 128 for p in range(max(0, pos - 127), pos + 1)]
                expected[128:] = [regions.compressed_offset + s if s >= 0 else -1 for s in compressed[token].tolist()]
                before[0, row * 128 + pos % 128].copy_(values[token])
            torch.testing.assert_close(indices[token, 0], _ints(expected), rtol=0, atol=0)
        torch.testing.assert_close(storage.cache, before, rtol=0, atol=0)
        batch.request_rows.copy_(_ints([0, -1, 1]))
        batch.positions.copy_(_ints([270, -1, 1]))
        compressed.copy_(_ints([[7, -1], [-1, -1], [-1, -1]]))
        values.mul_(.5)
    graph.reset()
