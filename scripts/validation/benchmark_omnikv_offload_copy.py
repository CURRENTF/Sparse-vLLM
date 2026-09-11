"""Time exact history gathering versus LRU planning plus direct cache fill.

Check GPU idleness before launch. Timing includes GPU LRU planning and both KV
components, excludes current-token projection/write-through and attention.
Each timed replay starts from the same seeded cache, so misses do not disappear
as the benchmark warms up. Real serving performance belongs in BenchProbe.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

from sparsevllm.engine.cache_manager.omnikv_lru import OmniKVLRU
from sparsevllm.kernels.triton.indexed_host_copy import gather_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--selected", type=int, default=2080)
    parser.add_argument("--cache-tokens", type=int, default=4096)
    parser.add_argument("--layout", choices=("explicit", "mla"), default="explicit")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "command": sys.argv,
        "args": vars(args),
        "seed": 19,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "diff": subprocess.check_output(["git", "diff"], text=True),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "timing": "cuda events around one graph replay; cache reseeded outside timed window",
        "status": "running",
    }
    raw = []
    try:
        torch.manual_seed(19)
        batch, selected = args.batch, args.selected
        history = 4 * selected
        shapes = [(8, 128)] * 2 if args.layout == "explicit" else [(1, 512), (1, 64)]
        dtype = torch.bfloat16
        host = [
            torch.randn(history, *shape, dtype=dtype).pin_memory() for shape in shapes
        ]
        pointers = torch.tensor(
            [x.data_ptr() for x in host], dtype=torch.uint64, device="cuda"
        )
        rows = torch.arange(batch, dtype=torch.int32, device="cuda")
        writes = torch.full((batch,), history - 1, dtype=torch.int32, device="cuda")
        lengths = torch.full((batch,), selected, dtype=torch.int32, device="cuda")
        base = (
            torch.arange(selected, dtype=torch.int32, device="cuda")
            .expand(batch, -1)
            .contiguous()
        )
        table = base.clone()
        view = torch.empty_like(table)
        storage = SimpleNamespace(shapes=shapes, dtype=dtype, num_slots=history)
        lru = OmniKVLRU(
            storage, {1: 0}, batch, args.cache_tokens, selected, "cuda", view=view
        )
        staging = [
            torch.empty(batch * selected, *s, dtype=dtype, device="cuda")
            for s in shapes
        ]

        def run(cached):
            if cached:
                lru.planned.clear()
                plan = lru.prepare(1, table, rows, rows, lengths, writes)
                destinations = lru.parts[1]
            else:
                plan = None
                destinations = staging
            for component, destination in enumerate(destinations):
                gather_rows(
                    pointers,
                    destination,
                    table,
                    rows,
                    lengths,
                    capacity=selected,
                    component=component,
                    exclude_slots=writes,
                    max_blocks=max(1, 32 // batch),
                    plan=plan,
                    cache=destination if cached else None,
                    direct=cached,
                    miss_tokens=lru.misses(1)[0] if cached else None,
                    miss_counts=lru.misses(1)[1] if cached else None,
                )

        summaries = []
        for cached, hit_rate in [
            (False, 0),
            (True, 0),
            (True, 0.75),
            (True, 0.95),
            (True, 0.99),
        ]:
            hits = int(selected * hit_rate)
            target = torch.cat(
                (torch.arange(hits), torch.arange(selected, 2 * selected - hits))
            )
            target = (
                target[torch.randperm(selected)]
                .to(device="cuda", dtype=torch.int32)
                .expand(batch, -1)
                .contiguous()
            )
            table.copy_(target)
            run(cached)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(cached)
            timings = []
            for iteration in range(args.repeats + 5):
                if cached:
                    for row in range(batch):
                        lru.invalidate(row)
                    table.copy_(base)
                    run(True)
                    lru.metadata[0][-1].zero_()
                table.copy_(target)
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
                for component, source in enumerate(host):
                    actual = (
                        lru.parts[1][component][view.reshape(-1).long()]
                        if cached
                        else staging[component]
                    )
                    torch.testing.assert_close(
                        actual.cpu(),
                        source[target.cpu().reshape(-1).long()],
                        atol=0,
                        rtol=0,
                    )
                if cached:
                    assert lru.metadata[0][-1].tolist() == [
                        hits * batch,
                        (selected - hits) * batch,
                    ]
                sample = {
                    "status": "success",
                    "cached": cached,
                    "hit_rate": hit_rate,
                    "iteration": iteration,
                    "warmup": iteration < 5,
                    "elapsed_ms": elapsed,
                }
                raw.append(sample)
                if iteration >= 5:
                    timings.append(elapsed)
            summaries.append(
                {
                    "cached": cached,
                    "hit_rate": hit_rate,
                    "count": len(timings),
                    "median_ms": statistics.median(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                }
            )
        (args.output / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        manifest["status"] = "success"
    except Exception as exc:
        manifest.update(
            status="metric_failed"
            if isinstance(exc, AssertionError)
            else "model_failed",
            error=repr(exc),
        )
        raise
    finally:
        (args.output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
        (args.output / "raw_samples.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in raw)
        )


if __name__ == "__main__":
    main()
