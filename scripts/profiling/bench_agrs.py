"""Two-GPU BF16 AG/compute/RS graph microbenchmark, including callable overhead.

Check idleness and select two GPUs with CUDA_VISIBLE_DEVICES before launch.
Compare NCCL and the public FlashInfer mixed-comm MC/UC modes with fixed warmup,
interleaved order and changed-input replay. This does not measure request TPOT.
The richer independent primitive/ownership oracle lives in test_moe_communication.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashinfer.comm.mixed_comm import (
    MixedCommHandler,
    MixedCommMode,
    MixedCommOp,
    run_mixed_comm,
)


def worker(rank, path, out, rows_list, sample_count, replays):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=path, rank=rank, world_size=2)
    group = dist.new_group(backend="nccl")
    handler = MixedCommHandler(
        rank,
        2,
        rank,
        2,
        0,
        1,
        1,
        2,
        1,
        1,
        torch.bfloat16,
        torch.device("cuda", rank),
        use_autotune=False,
    )
    modes = ["torch", "FUSED_OPT_WAITS_MC", "FUSED_OPT_WAITS_UC"]
    records = []
    for rows in rows_list:
        x = (
            (torch.arange(rows * 2048, device="cuda").remainder(16) / 16 + rank)
            .to(torch.bfloat16)
            .reshape(rows, 2048)
        )

        def operation(mode, rows=rows, x=x):
            gathered = torch.empty((rows * 2, 2048), device="cuda", dtype=x.dtype)
            if mode != "torch":
                run_mixed_comm(
                    MixedCommOp.ALLGATHER, handler, x, gathered, MixedCommMode[mode]
                )
            else:
                dist.all_gather_into_tensor(gathered, x, group=group)
            expert = gathered * (rank + 1)
            output = torch.empty_like(x)
            if mode != "torch":
                run_mixed_comm(
                    MixedCommOp.REDUCESCATTER,
                    handler,
                    expert,
                    output,
                    MixedCommMode[mode],
                )
            else:
                dist.reduce_scatter_tensor(output, expert, group=group)
            return output

        graphs = {}
        outputs = {}
        for mode in modes:
            for _ in range(5):
                operation(mode)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                y = operation(mode)
            graphs[mode] = graph
            outputs[mode] = y
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(y, x * 3, rtol=0, atol=0)
        # Refresh inputs through captured wrappers; do not accept a capture-time output.
        x.add_(0.125)
        for mode, replay_graph in graphs.items():
            replay_graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[mode], x * 3, rtol=0, atol=0)
        samples = {mode: [] for mode in graphs}
        for i in range(sample_count):
            for mode in modes if i % 2 == 0 else modes[::-1]:
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graphs[mode].replay()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) * 1000 / replays)
        records.append(
            {
                "rows": rows,
                "hidden": 2048,
                "dtype": "bfloat16",
                "rank": rank,
                "samples_us": samples,
                "median_us": {k: statistics.median(v) for k, v in samples.items()},
            }
        )
        Path(out, f"rank{rank}.json").write_text(json.dumps(records, indent=2))
        del graph, replay_graph
        graphs.clear()
        outputs.clear()
        torch.cuda.synchronize()
        dist.barrier()
    handler.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 8, 32, 128, 512, 4096])
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--replays", type=int, default=20)
    a = p.parse_args()
    if min(*a.rows, a.samples, a.replays) <= 0:
        p.error("rows, samples and replays must be positive")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_text(Path(__file__).read_text())
    status = {
        "status": "running",
        "command": sys.argv,
        "args": vars(a),
        "flashinfer": version("flashinfer-python"),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (out / "manifest.json").write_text(json.dumps(status, indent=2))
    try:
        mp.spawn(
            worker,
            args=(
                f"file://{out.resolve()}/rendezvous",
                str(out),
                a.rows,
                a.samples,
                a.replays,
            ),
            nprocs=2,
        )
        status["status"] = "success"
    except BaseException as e:
        status.update(status="failed", error=repr(e))
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(status, indent=2))
