"""AG/RS must preserve token ownership, including idle and graph-padded ranks."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparsevllm.distributed import ParallelTopology
from sparsevllm.distributed.moe_communication import (
    AllGatherReduceScatterMoeCommunication,
)
from sparsevllm.distributed.parallel_context import (
    init_parallel_context,
    reset_parallel_context,
)
from sparsevllm.distributed.topology import ParallelMode, parallel_group_ranks


def test_dp_attention_groups_share_experts_without_replicating_attention():
    topology = ParallelTopology(1, 4, 4, ParallelMode.DP_ATTENTION)
    groups = parallel_group_ranks(topology)
    assert topology.world_size == 4
    assert all(len(group) == 1 for group in groups["tensor"])
    assert groups["expert"] == groups["data"]
    assert set(groups["expert"][0]) == set(range(topology.world_size))
    with pytest.raises(ValueError, match="EP=DP"):
        ParallelTopology(1, 2, 4, ParallelMode.DP_ATTENTION)


def _agrs_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    parallel = init_parallel_context(
        topology=ParallelTopology(1, 2, 2, ParallelMode.DP_ATTENTION)
    )
    communication = AllGatherReduceScatterMoeCommunication(parallel)
    try:
        for sizes in ((3, 1), (0, 5), (5, 0)):
            capacity = max(sizes)
            source = (
                torch.arange(sizes[rank] * 32, device="cuda", dtype=torch.float32).view(
                    -1, 32
                )
                / 32
                + rank
            )

            def forward(source=source, capacity=capacity):
                # Independent additive expert contributions; the owner should
                # get (1+2)*x regardless of remote/padded token rows. Chunking
                # must stay inside dispatch/combine so idle ranks cannot skip
                # or reorder a collective.
                return communication.run(
                    source,
                    route=lambda x: (
                        torch.zeros((len(x), 1), dtype=torch.int64, device=x.device),
                        torch.ones((len(x), 1), device=x.device),
                    ),
                    experts=lambda x, ids, weights: x * (rank + 1) * weights,
                    chunk_size=2,
                    capacity=capacity,
                )

            torch.testing.assert_close(forward(), source * 3)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            for _ in range(3):
                source.add_(1)
                graph.replay()
                torch.testing.assert_close(result, source * 3)
            del graph
    finally:
        torch.cuda.synchronize()
        reset_parallel_context()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two idle CUDA devices"
)
def test_agrs_uneven_idle_and_graph_replay(tmp_path: Path):
    mp.spawn(
        _agrs_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )


def _prepared_agrs_worker(
    rank, rendezvous, force_torch, world_size, hidden_size, dtype
):
    from sparsevllm.distributed.collective_runtime import ParallelCollectiveRuntime
    from sparsevllm.operators.agrs import FlashInferMixedAgRsProvider
    from sparsevllm.operators.registry import SupportResult
    from sparsevllm.utils.context import reset_context, set_context

    if force_torch:
        FlashInferMixedAgRsProvider.supports = classmethod(
            lambda cls, spec, caps: SupportResult.unsupported("test selects NCCL")
        )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size
    )
    parallel = init_parallel_context(
        topology=ParallelTopology(1, world_size, world_size, ParallelMode.DP_ATTENTION)
    )
    runtime = ParallelCollectiveRuntime(parallel, cuda_graph=True, device_index=rank)
    collectives = runtime.request_dp_collectives(
        max_rows=1024, hidden_size=hidden_size, dtype=dtype
    )
    runtime.prepare()
    communication = collectives.moe_transport
    set_context(False)
    try:
        for sizes in (
            (3,) + (1,) * (world_size - 1),
            (0,) * (world_size - 1) + (5,),
            (5,) + (0,) * (world_size - 1),
            (513,) * (world_size - 1) + (0,),
        ):
            capacity = max(sizes)
            sources = [
                (
                    torch.arange(rows * hidden_size, device="cuda")
                    .remainder(16)
                    .reshape(rows, hidden_size)
                    / 16
                    + peer
                ).to(dtype)
                for peer, rows in enumerate(sizes)
            ]
            source = sources[rank]
            set_context(rank == 0)
            dispatched = communication.dispatch(source, capacity=capacity)
            expected = torch.zeros(
                capacity * world_size, hidden_size, device="cuda", dtype=dtype
            )
            for peer, value in enumerate(sources):
                expected[peer * capacity : peer * capacity + len(value)].copy_(value)
            # Check AG ordering independently: paired inverse permutations in
            # AG/RS could otherwise pass an end-to-end additive expert oracle.
            torch.testing.assert_close(
                dispatched.hidden_states, expected, atol=0, rtol=0
            )
            partial = (
                torch.arange(capacity * world_size * hidden_size, device="cuda")
                .remainder(16)
                .reshape(capacity * world_size, hidden_size)
                .to(dtype)
            )
            result = communication.combine(partial * (rank + 1), dispatched)
            reference = partial[rank * capacity : rank * capacity + sizes[rank]] * (
                world_size * (world_size + 1) // 2
            )
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
            set_context(False)

            def forward(source=source, capacity=capacity):
                # A nonlinear owner-local branch catches missing/doubled shared
                # outputs and stale graph inputs after capture or replacement.
                return communication.run_with_shared_experts(
                    source,
                    shared_experts=lambda x: x.square(),
                    route=lambda x: (None, None),
                    experts=lambda x, ids, weights: x,
                    chunk_size=256,
                    capacity=capacity,
                )

            torch.testing.assert_close(
                forward(), source * world_size + source.square(), atol=0, rtol=0
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            for _ in range(3):
                source.add_(0.125)
                graph.replay()
                torch.testing.assert_close(
                    result, source * world_size + source.square(), atol=0, rtol=0
                )
            torch.cuda.synchronize()
            del graph
    finally:
        torch.cuda.synchronize()
        runtime.close()
        reset_context()
        reset_parallel_context()
        dist.destroy_process_group()


@pytest.mark.parametrize("force_torch", [False, True], ids=["default", "torch"])
@pytest.mark.parametrize(
    "world_size,hidden_size,dtype",
    [(2, 2048, torch.bfloat16), (4, 4096, torch.float16)],
    ids=["two-rank-bf16", "four-rank-fp16"],
)
def test_prepared_agrs_rank_order_idle_and_replaced_graphs(
    tmp_path, force_torch, world_size, hidden_size, dtype
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} idle CUDA devices")
    mp.spawn(
        _prepared_agrs_worker,
        args=(
            f"file://{tmp_path / 'prepared-rendezvous'}",
            force_torch,
            world_size,
            hidden_size,
            dtype,
        ),
        nprocs=world_size,
        join=True,
    )
