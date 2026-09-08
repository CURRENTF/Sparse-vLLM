"""Compare MLA partial-attention wrappers against a fixed Git revision."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from sparsevllm.kernels.triton.mla import prefill
from sparsevllm.kernels.triton.context_flashattention_nopad import context_attention_fwd


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def check_sampled_oracle(q, k, v, cu_q, cu_k, output, lse, causal):
    # Bound oracle memory even for long contexts; cover first/middle/last Q.
    for qa, qb, ka, kb in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
        qn, kn = qb - qa, kb - ka
        rows = torch.tensor(sorted({0, qn // 2, qn - 1}), device=q.device)
        z = torch.einsum("qhd,khd->hqk", q[qa + rows].float(), k[ka:kb].float()) * 0.0625
        if causal:
            z.masked_fill_(torch.arange(kn, device=q.device)[None, None] > (rows + kn - qn)[None, :, None], -torch.inf)
        expected = torch.einsum("hqk,khd->qhd", z.softmax(-1), v[ka:kb].float())
        torch.testing.assert_close(output[qa + rows].float(), expected, atol=0.006, rtol=0.03)
        if lse is not None:
            torch.testing.assert_close(lse[:, qa + rows], z.logsumexp(-1), atol=0.003, rtol=0.001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--heads", type=int, choices=(5, 10, 20), default=20)
    args = parser.parse_args()
    if args.warmup < 1 or args.rounds < 1:
        parser.error("warmup and rounds must be positive")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_manifest.json").exists() or (root / "raw_samples.jsonl").exists():
        raise FileExistsError("Use a new output directory to preserve previous results")
    path = "src/sparsevllm/kernels/triton/mla/prefill.py"
    source = git("show", f"{args.baseline_ref}:{path}") + "\n"
    baseline_path = root / "baseline_prefill.py"
    baseline_path.write_text(source)
    spec = importlib.util.spec_from_file_location("mla_prefill_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    props = torch.cuda.get_device_properties(0)
    manifest = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "repo": git("rev-parse", "--show-toplevel"), "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "git_status": git("status", "--short"),
        "baseline": git("rev-parse", args.baseline_ref),
        "gpu": props.name, "capability": [props.major, props.minor],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "gpu_state": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,power.limit", "--format=csv"], text=True),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "triton", "sglang-kernel")},
        "cuda": torch.version.cuda, "seed": 42, "dtype": "bfloat16", "heads": args.heads,
        "head_dim": 256, "graph": False, "warmup": args.warmup, "rounds": args.rounds,
        "timing": "CUDA events around wrapper including output/LSE allocation; warm caches; ABBA pairs; no clock control",
        "candidate_sha256": hashlib.sha256(Path(prefill.__file__).read_bytes()).hexdigest(),
    }
    (root / "candidate_prefill.py").write_text(Path(prefill.__file__).read_text())
    (root / "benchmark_script.py").write_text(Path(__file__).read_text())
    (root / "changes.patch").write_text(git("diff", "HEAD"))
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = []
    try:
        with (root / "raw_samples.jsonl").open("x") as raw:
            for queries, keys, causal in (
                ((4096,), (4096,), True), ((8192,), (8192,), True),
                ((4096, 7), (4096, 7), True),
                ((32,), (16384,), False), ((128,), (16384,), False),
                ((129,), (16384,), False), ((512,), (16384,), False),
                ((8192,), (16384,), False),
            ):
                torch.manual_seed(42)
                q = torch.randn(sum(queries), args.heads, 256, device="cuda", dtype=torch.bfloat16) * 0.2
                k = torch.randn(sum(keys), args.heads, 256, device="cuda", dtype=torch.bfloat16) * 0.2
                # Match the V view returned by the serving joint KV projection.
                projected = torch.randn(sum(keys), args.heads, 448, device="cuda", dtype=torch.bfloat16) * 0.2
                v = projected[..., 192:]
                cq = [0, *torch.tensor(queries).cumsum(0).tolist()]
                ck = [0, *torch.tensor(keys).cumsum(0).tolist()]
                cu_q, cu_k = [torch.tensor(x, device="cuda", dtype=torch.int32) for x in (cq, ck)]

                def call(module):
                    return module.attention_partial(q, k, v, cu_q, cu_k, max(queries), max(keys), scale=0.0625, causal=causal)

                calls = {"pr_original": lambda: call(baseline), "candidate": lambda: call(prefill)}
                if causal:
                    slots = torch.full((len(keys), max(keys)), -1, device="cuda", dtype=torch.int32)
                    for i, (a, b) in enumerate(zip(ck, ck[1:])):
                        slots[i, :b - a] = torch.arange(a, b, device="cuda", dtype=torch.int32)
                    rows = torch.arange(len(keys), device="cuda", dtype=torch.int32)
                    lengths = cu_k[1:] - cu_k[:-1]
                    cached = lengths - (cu_q[1:] - cu_q[:-1])
                    old_output = torch.empty_like(q)

                    def legacy():
                        context_attention_fwd(q, k, v, old_output, rows, cu_q[:-1], lengths, cached, max(queries), slots)
                        return old_output, None

                    calls["legacy_causal"] = legacy
                cold_ms = {}
                for name, fn in calls.items():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output, lse = fn()
                    torch.cuda.synchronize()
                    cold_ms[name] = (time.perf_counter() - start) * 1000
                    check_sampled_oracle(q, k, v, cq, ck, output, lse, causal)
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                samples = {name: [] for name in calls}
                pairs = [("pr_original", "candidate")]
                if causal:
                    pairs.append(("legacy_causal", "candidate"))
                for pair in pairs:
                    for iteration in range(args.rounds):
                        for name in (*pair, *reversed(pair)):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            calls[name]()
                            end.record()
                            end.synchronize()
                            elapsed = start.elapsed_time(end)
                            samples[name].append(elapsed)
                            raw.write(json.dumps({"queries": queries, "keys": keys, "causal": causal, "pair": pair, "iteration": iteration, "implementation": name, "ms": elapsed}) + "\n")
                row = {"queries": queries, "keys": keys, "causal": causal, "status": "success", "cold_first_call_ms": cold_ms,
                       "strides": {"q": q.stride(), "k": k.stride(), "v": v.stride()},
                       "timings": {name: {"n": len(values), "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)} for name, values in samples.items()}}
                results.append(row)
                print(json.dumps(row), flush=True)
                (root / "summary.json").write_text(json.dumps(results, indent=2))
        manifest["status"] = "success"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
