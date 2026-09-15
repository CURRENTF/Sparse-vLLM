"""CUDA numerics against direct gated pooling of complete request histories.

The oracle follows the supplied DeepSeek Compressor formula, with no ring,
packed indexing or production pooling helpers. CPU/interpreter runs are only
development diagnostics and must not be reported as CUDA validation.
"""

import pytest
import torch

from sparsevllm.engine.cache_manager.methods.deepseek_v4 import (
    CompressionChunk,
    CompressionPlan,
    CompressionStateShape,
)
from sparsevllm.kernels.triton.deepseek_v4.compression import compress_projected


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA device")


def _reference(kv, gate, ape, end, ratio, dim):
    # First-half channels of the preceding block, second-half channels of
    # the current block; the missing preceding block is excluded at position 0.
    if ratio == 4:
        values = kv[end-ratio:end, dim:]
        scores = gate[end-ratio:end, dim:] + ape[:, dim:]
        if end > ratio:
            values = torch.cat((kv[end-2*ratio:end-ratio, :dim], values))
            scores = torch.cat((gate[end-2*ratio:end-ratio, :dim] + ape[:, :dim], scores))
    else:
        values = kv[end-ratio:end]
        scores = gate[end-ratio:end] + ape
    return (values * scores.softmax(dim=0)).sum(dim=0).bfloat16()


def _int(data, device):
    return torch.tensor(data, dtype=torch.int32, device=device)


def _prefill(chunks, histories, ape, state, ratio, dim, device):
    plan = CompressionPlan.prefill(tuple(chunks), ratio)
    values = torch.cat([histories[c.request_row][0][c.start:c.start+c.length] for c in chunks]).to(device)
    gates = torch.cat([histories[c.request_row][1][c.start:c.start+c.length] for c in chunks]).to(device)
    out = torch.empty((len(plan.boundary_ends), dim), dtype=torch.bfloat16, device=device)
    rope = torch.empty(len(plan.boundary_ends), dtype=torch.int32, device=device)
    compress_projected(values, gates, ape.to(device), *state,
                       _int(plan.request_rows, device), _int(plan.cu_seqlens, device),
                       _int(plan.start_positions, device), _int(plan.boundary_requests, device),
                       _int(plan.boundary_ends, device), out, rope, ratio=ratio)
    for i, (batch, end) in enumerate(zip(plan.boundary_requests, plan.boundary_ends)):
        kv, gate = histories[chunks[batch].request_row]
        torch.testing.assert_close(out[i].cpu(), _reference(kv, gate, ape, end, ratio, dim), rtol=8e-3, atol=8e-3)
    assert rope.cpu().tolist() == [end-ratio for end in plan.boundary_ends]
    return out


@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_mixed_chunks_match_whole_history_and_preserve_carry(ratio, dim):
    # Nonaligned chunks and physical row permutations catch batch-global
    # start_pos, overlap omission, ring-overwrite races and wrong APE phase.
    device = "cuda"
    gen = torch.Generator().manual_seed(731)
    window, width = CompressionStateShape(ratio, dim).row_shape
    ape = torch.randn((ratio, width), generator=gen)
    histories = {row: (torch.randn((5*ratio+7, width), generator=gen),
                       torch.randn((5*ratio+7, width), generator=gen) * 8)
                 for row in (2, 0)}
    state = (torch.full((4, window, width), 91., device=device),
             torch.full((4, window, width), -73., device=device))
    starts = {0: 0, 2: 0}
    for lengths in ((1, ratio-1), (ratio+2, 2), (2*ratio+3, 2*ratio+1)):
        chunks = [CompressionChunk(row, starts[row], length) for row, length in zip((2, 0), lengths)]
        _prefill(chunks, histories, ape, state, ratio, dim, device)
        for c in chunks:
            starts[c.request_row] += c.length
        for row, end in starts.items():
            for p in range(max(0, end-window), end):
                for tensor, history in zip(state, histories[row]):
                    torch.testing.assert_close(tensor[row, p % window].cpu(), history[p], rtol=0, atol=0)
            if end < window:
                assert torch.all(state[0][row, end:] == 91.)
                assert torch.all(state[1][row, end:] == -73.)
    assert torch.all(state[0][1] == 91.) and torch.all(state[1][1] == -73.)


@pytest.mark.parametrize("ratio", [4, 128])
def test_prefix_carry_copy_can_fork_at_unaligned_boundary(ratio):
    # A KV-only prefix restore loses the previous overlap or partial block.
    # This tests the required physical carry-copy contract, not radix wiring.
    dim, device = 128, "cuda"
    window, width = CompressionStateShape(ratio, dim).row_shape
    gen = torch.Generator().manual_seed(19)
    prefix, tail = ratio+1, ratio+3
    common = [torch.randn((prefix, width), generator=gen) for _ in range(2)]
    histories = {row: tuple(torch.cat((c, torch.randn((tail, width), generator=gen))) for c in common)
                 for row in (0, 1, 2)}
    ape = torch.randn((ratio, width), generator=gen)
    state = tuple(torch.zeros((3, window, width), device=device) for _ in range(2))
    _prefill([CompressionChunk(0, 0, prefix)], histories, ape, state, ratio, dim, device)
    snapshot = tuple(s[0].clone() for s in state)
    for s, saved in zip(state, snapshot):
        s[1].copy_(saved)
        s[2].copy_(saved)
    _prefill([CompressionChunk(2, prefix, tail), CompressionChunk(1, prefix, tail)],
             histories, ape, state, ratio, dim, device)
    for s, saved in zip(state, snapshot):
        torch.testing.assert_close(s[0], saved, rtol=0, atol=0)


@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_graph_replay_reads_new_boundaries_and_ignores_padding(ratio, dim):
    # Capture away from a compression event, then cross several boundaries,
    # change request ordering and recycle a dirty row for a new request.
    batch, device = 4,  "cuda"
    window, width = CompressionStateShape(ratio, dim).row_shape
    gen = torch.Generator().manual_seed(41)
    ape_cpu = torch.randn((ratio, width), generator=gen)
    ape = ape_cpu.to(device)
    states = [tuple(torch.zeros((5, window, width), device=device) for _ in range(2)) for _ in range(2)]
    values = torch.zeros((batch, width), device=device)
    gates = torch.zeros_like(values)
    rows = _int([-1] * batch, device)
    cu = _int(range(batch+1), device)
    starts = _int([0] * batch, device)
    empty = _int([], device)
    outputs = [torch.empty((batch, dim), device=device, dtype=torch.bfloat16) for _ in range(2)]
    positions = [torch.empty(batch, device=device, dtype=torch.int32) for _ in range(2)]

    def run(index):
        compress_projected(values, gates, ape, *states[index], rows, cu, starts,
                           empty, empty, outputs[index], positions[index], ratio=ratio, decode=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(1)
    histories = {r: ([], []) for r in (0, 2)}
    for step in range(2*ratio+2):
        if step == ratio+1:
            histories[0] = ([], [])
        live = [2, -1, 0, -1] if step % 2 else [0, 2, -1, -1]
        rows.copy_(_int(live, device))
        new_values = torch.randn((batch, width), generator=gen)
        new_gates = torch.randn((batch, width), generator=gen)
        values.copy_(new_values)
        gates.copy_(new_gates)
        start_list = []
        for i, row in enumerate(live):
            start_list.append(len(histories[row][0]) if row >= 0 else 0)
            if row >= 0:
                histories[row][0].append(new_values[i])
                histories[row][1].append(new_gates[i])
        starts.copy_(_int(start_list, device))
        run(0)
        graph.replay()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(positions[0], positions[1], rtol=0, atol=0)
        for a, b in zip(states[0], states[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert torch.count_nonzero(b[4]) == 0
        for i, row in enumerate(live):
            if row >= 0 and len(histories[row][0]) % ratio == 0:
                kv, gate = (torch.stack(h) for h in histories[row])
                ref = _reference(kv, gate, ape_cpu, len(kv), ratio, dim)
                torch.testing.assert_close(outputs[1][i].cpu(), ref, rtol=8e-3, atol=8e-3)
            else:
                assert positions[1][i].item() == -1
                assert torch.count_nonzero(outputs[1][i]) == 0
