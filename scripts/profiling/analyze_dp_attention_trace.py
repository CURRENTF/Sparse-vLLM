"""Infer graph kernel stages from an NVTX-tagged eager warmup sequence.

Nsight associates replay kernels with cudaGraphLaunch, not the original Python
range. Align each fixed graph sequence against a warmup from the same process,
then retain that node-to-stage mapping for every replay. Unmatched nodes remain
explicit. This is diagnostic inference, not direct NVTX attribution or wall time.
Capture startup as well as the marked window with --cuda-graph-trace=node.
"""

import argparse
import collections
import difflib
import json
import sqlite3
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-match", type=float, default=0.99)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    c = sqlite3.connect(f"file:{a.db.resolve()}?mode=ro", uri=True)
    strings = dict(c.execute("select id,value from StringIds"))
    window = c.execute(
        "select start,end from NVTX_EVENTS where text='svllm_decode_profile'"
    ).fetchall()
    if len(window) != 1:
        raise ValueError("Expected exactly one svllm_decode_profile NVTX window.")
    begin, end = window[0]
    runtime = {
        (tid >> 24, co): (s, e, tid)
        for s, e, tid, co in c.execute(
            "select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    ranges = collections.defaultdict(list)
    labels = {
        "attention_inclusive",
        "attention_all_reduce",
        "router",
        "routed_mlp",
        "shared_mlp",
        "all_gather",
        "reduce_scatter",
        "all2all_dispatch",
        "all2all_combine",
        "moe_all_reduce",
    }
    models = []
    for s, e, tid, t in c.execute(
        "select start,end,globalTid,text from NVTX_EVENTS where text is not null and end is not null"
    ):
        if t in labels:
            ranges[tid].append((s, e, t))
        if t == "model_inclusive":
            models.append((s, e, tid))
    eager = collections.defaultdict(list)
    graphs = collections.defaultdict(list)
    for s, e, pid, co, node, name in c.execute(
        "select start,end,globalPid,correlationId,graphNodeId,shortName from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        row = (s, e, node, strings[name])
        if begin <= s and e <= end and node is not None:
            graphs[(pid >> 24, co)].append(row)
        if node is None and (rt := runtime.get((pid >> 24, co))):
            eager[rt[2]].append((*row, rt[0], rt[1]))
    if not graphs:
        raise ValueError("No CUDA Graph kernels in the measurement window.")
    result = {}
    for pid in sorted({k[0] for k in graphs}):
        replays = [sorted(v) for k, v in graphs.items() if k[0] == pid]
        sequence = replays[0]
        names = [v[3] for v in sequence]
        candidates = []
        for s, e, tid in models:
            if tid >> 24 != pid:
                continue
            warm = sorted([v for v in eager[tid] if s <= v[4] and v[5] <= e])
            if not warm:
                continue
            matcher = difflib.SequenceMatcher(
                None, [v[3] for v in warm], names, autojunk=False
            )
            blocks = matcher.get_matching_blocks()
            matched = sum(x.size for x in blocks)
            candidates.append((matched, warm, blocks, tid, s))
        if not candidates:
            raise ValueError(
                "Trace must include NVTX-annotated eager warmup, not only replay."
            )
        matched, warm, blocks, tid, t = max(candidates, key=lambda v: (v[0], v[4]))
        if matched / len(names) < a.min_match:
            raise ValueError(
                f"Insufficient warmup alignment for PID {pid}: {matched}/{len(names)}"
            )
        mapping = {}
        assignments = []
        for block in blocks:
            for i, j in zip(
                range(block.a, block.a + block.size),
                range(block.b, block.b + block.size),
            ):
                w = warm[i]
                enclosing = [r for r in ranges[tid] if r[0] <= w[4] and w[5] <= r[1]]
                label = (
                    min(enclosing, key=lambda v: v[1] - v[0])[2]
                    if enclosing
                    else "other"
                )
                if label == "attention_inclusive":
                    label = "attention_compute"
                mapping[sequence[j][2]] = label
                assignments.append((sequence[j][2], sequence[j][3], label))
        times = collections.defaultdict(float)
        unmatched = collections.Counter()
        for replay in replays:
            if [(v[2], v[3]) for v in replay] != [(v[2], v[3]) for v in sequence]:
                raise ValueError(
                    "Graph kernel sequence changed inside the measurement window."
                )
            for s, e, node, name in replay:
                label = mapping.get(node, "unmatched")
                times[label] += (e - s) / 1e6 / len(replays)
                if label == "unmatched":
                    unmatched[name] += 1
        result[pid] = {
            "replays": len(replays),
            "kernels_per_replay": len(names),
            "matched": matched,
            "kernel_ms_per_replay": dict(times),
            "warmup_start_ns": t,
            "warmup_kernels": len(warm),
            "unmatched": dict(unmatched),
            "assignments": assignments,
        }
        print(pid, matched, len(names), dict(times), "unmatched:", dict(unmatched))
    c.close()
    with a.output.open("x") as output:
        json.dump(
            {
                "source": str(a.db.resolve()),
                "window_ns": window[0],
                "attribution": "exclusive kernel times inferred by sequence alignment against NVTX-tagged eager warmup; collective time includes peer waits",
                "processes": result,
            },
            output,
            indent=2,
        )


if __name__ == "__main__":
    main()
