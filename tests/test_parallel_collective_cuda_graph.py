import gc

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparsevllm.distributed import (
    ParallelCollectiveRuntime,
    ParallelTopology,
    init_parallel_context,
    reset_parallel_context,
)
from sparsevllm.utils.device_name import device_name_contains
from sparsevllm.utils.context import reset_context, set_context


def _cuda_graph_collective_worker(
    rank: int,
    world_size: int,
    init_method: str,
    tp_size: int,
    ep_size: int,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    runtime = None
    graph = None
    graphs = None
    outputs = None
    try:
        context = init_parallel_context(
            topology=ParallelTopology(tp_size, ep_size, 1)
        )
        runtime = ParallelCollectiveRuntime(
            context,
            cuda_graph=True,
            device_index=rank,
        )
        collectives = runtime.request_decode_collectives(
            attention_max_rows=2,
            moe_max_rows=2,
            hidden_size=2048,
            dtype=torch.bfloat16,
        )
        runtime.prepare()
        if device_name_contains(torch.cuda.get_device_name(rank), "H100"):
            assert collectives.attention.name == (
                "identity" if tp_size == 1 else "flashinfer_vllm_sm90"
            )
            assert collectives.moe.name == (
                "flashinfer_vllm_sm90"
                if world_size == 2
                else "torch_distributed"
            )

        set_context(False)
        static_inputs = {
            rows: torch.full(
                (rows, 2048),
                float(rank + 1),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for rows in (1, 2)
        }
        for static_input in static_inputs.values():
            collectives.attention.run(static_input.clone())
            collectives.moe.run(static_input.clone())
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

        runtime.begin_cuda_graph_capture()
        graphs = {}
        outputs = {}
        for rows, static_input in static_inputs.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = (
                    collectives.attention.run(static_input),
                    collectives.moe.run(static_input),
                )
            graphs[rows] = graph
            outputs[rows] = output

        runtime.collect_local_cuda_graph_metadata()
        runtime.exchange_cuda_graph_metadata()
        runtime.register_cuda_graph_buffers()
        runtime.mark_cuda_graph_replayable()

        attention_expected = (
            3.0
            if tp_size == 2 and rank < 2
            else 7.0
            if tp_size == 2
            else float(rank + 1)
        )
        for rows, scale in ((1, 1), (2, 2), (1, 3)):
            static_inputs[rows].fill_(scale * (rank + 1))
            graphs[rows].replay()
            torch.cuda.synchronize()
            assert torch.all(outputs[rows][0] == scale * attention_expected)
            assert torch.all(
                outputs[rows][1] == scale * sum(range(1, world_size + 1))
            )
    finally:
        torch.cuda.synchronize()
        outputs = None
        graphs = None
        graph = None
        gc.collect()
        if runtime is not None:
            runtime.close()
        reset_context()
        reset_parallel_context()
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,ep_size", [(2, 1), (1, 2)])
def test_tp2_and_ep2_collectives_capture_register_and_replay(
    tmp_path,
    tp_size,
    ep_size,
):
    if torch.cuda.device_count() < 2:
        pytest.skip("TP2/EP2 CUDA Graph test requires two visible CUDA devices")
    rendezvous = tmp_path / f"collective-{tp_size}-{ep_size}-rendezvous"
    mp.spawn(
        _cuda_graph_collective_worker,
        args=(2, f"file://{rendezvous}", tp_size, ep_size),
        nprocs=2,
        join=True,
    )


def test_tp2_ep2_collectives_capture_register_and_replay(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("TP2+EP2 CUDA Graph test requires four visible CUDA devices")
    rendezvous = tmp_path / "collective-2-2-rendezvous"
    mp.spawn(
        _cuda_graph_collective_worker,
        args=(4, f"file://{rendezvous}", 2, 2),
        nprocs=4,
        join=True,
    )
