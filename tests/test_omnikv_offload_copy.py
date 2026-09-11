"""Pinned-memory copies must follow replay metadata and protect padding rows."""

import pytest
import torch

from sparsevllm.kernels.triton.indexed_host_copy import (
    append_rows,
    gather_rows,
    store_rows,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((4, 128), torch.bfloat16),
        ((4, 128), torch.float16),
        ((1, 512), torch.bfloat16),
        ((1, 64), torch.bfloat16),
    ],
)
def test_indexed_host_copy_replay(shape, dtype):
    torch.manual_seed(17)
    host = torch.randn(23, *shape, dtype=dtype).pin_memory()
    ptr = torch.tensor([host.data_ptr()], dtype=torch.uint64, device="cuda")
    table = torch.tensor(
        [[8, 2, 19, 4], [10, 3, 5, 7]], dtype=torch.int32, device="cuda"
    )
    rows = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([4, 3], dtype=torch.int32, device="cuda")
    slots = torch.tensor([7, -1], dtype=torch.int32, device="cuda")
    source = torch.randn(2, *shape, dtype=dtype, device="cuda")
    output = torch.zeros(8, *shape, dtype=dtype, device="cuda")

    def run():
        store_rows(source, ptr, slots, 0)
        gather_rows(
            ptr,
            output,
            table,
            rows,
            lengths,
            capacity=4,
            component=0,
            skip_last=True,
            max_blocks=2,
            exclude_slots=slots,
        )
        append_rows(source, output, lengths, slots, 4)

    run()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for order in ([1, 0], [0, 1]):
        rows.copy_(torch.tensor(order, device="cuda", dtype=torch.int32))
        source.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        actual = output.cpu().view(2, 4, *shape)
        for batch, row in enumerate(order):
            n = int(lengths[batch]) - 1
            expected = (
                host[table[row, :n].cpu().long()]
                if batch == 0
                else torch.zeros_like(actual[batch, :n])
            )
            torch.testing.assert_close(actual[batch, :n], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[0, 3], source[0].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(host[7], source[0].cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_current_token_follows_selection_when_recent_budget_is_zero():
    # Top-K is unsorted and may exclude the newest token entirely. The old
    # last-position append corrupted the selected history in both situations.
    host = torch.randn(11, 1, 64, dtype=torch.bfloat16).pin_memory()
    pointers = torch.tensor([host.data_ptr()], dtype=torch.uint64, device="cuda")
    table = torch.tensor([[2, 5, 8], [6, 1, 9]], dtype=torch.int32, device="cuda")
    rows = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    write_slots = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    current = torch.randn(2, 1, 64, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(6, 1, 64, dtype=torch.bfloat16, device="cuda")

    def run():
        gather_rows(
            pointers,
            output,
            table,
            rows,
            lengths,
            capacity=3,
            component=0,
            exclude_slots=write_slots,
        )
        append_rows(current, output, lengths, write_slots, 3, table=table, rows=rows)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for selected in ([[2, 5, 8], [6, 1, 9]], [[8, 2, 9], [7, 1, 6]]):
        table.copy_(torch.tensor(selected, dtype=torch.int32, device="cuda"))
        current.add_(1)
        graph.replay()
        actual = output.cpu().view(2, 3, 1, 64)
        for batch, slots in enumerate(selected):
            reference = host[slots].clone()
            for index, slot in enumerate(slots):
                if slot == int(write_slots[batch]):
                    reference[index] = current[batch].cpu()
            torch.testing.assert_close(actual[batch], reference, rtol=0, atol=0)
