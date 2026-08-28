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
            attention_max_rows=1,
            moe_max_rows=1,
            hidden_size=2048,
            dtype=torch.bfloat16,
        )
        runtime.prepare()
        if torch.cuda.get_device_name(rank) == "NVIDIA H100 80GB HBM3":
            assert collectives.attention.name == (
                "identity" if tp_size == 1 else "flashinfer_vllm_sm90"
            )
            assert collectives.moe.name == (
                "flashinfer_vllm_sm90"
                if world_size == 2
                else "torch_distributed"
            )

        set_context(False)
        static_input = torch.full(
            (1, 2048),
            float(rank + 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        collectives.attention.run(static_input.clone())
        collectives.moe.run(static_input.clone())
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

        runtime.begin_cuda_graph_capture()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = (
                collectives.attention.run(static_input),
                collectives.moe.run(static_input),
            )

        runtime.collect_local_cuda_graph_metadata()
        runtime.exchange_cuda_graph_metadata()
        runtime.register_cuda_graph_buffers()
        runtime.mark_cuda_graph_replayable()

        static_input.fill_(rank + 1)
        graph.replay()
        torch.cuda.synchronize()

        attention_expected = (
            3.0
            if tp_size == 2 and rank < 2
            else 7.0
            if tp_size == 2
            else float(rank + 1)
        )
        assert torch.all(outputs[0] == attention_expected)
        assert torch.all(outputs[1] == sum(range(1, world_size + 1)))

        static_input.fill_(2 * (rank + 1))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(outputs[0] == 2 * attention_expected)
        assert torch.all(outputs[1] == 2 * sum(range(1, world_size + 1)))
    finally:
        torch.cuda.synchronize()
        outputs = None
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
