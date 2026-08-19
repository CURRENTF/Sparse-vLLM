# SPDX-License-Identifier: Apache-2.0
"""Standardized Synthetic Length Sweep & Micro-Efficiency Benchmark for Sparse-vLLM & vLLM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark.efficiency.metrics_calculator import (
    ModelArchitectureSpecs,
    calculate_mbu,
    calculate_mfu,
    detect_gpu_hardware,
)


def _git_metadata() -> dict[str, Any]:
    def _run_git(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip() or None
        except Exception:
            return None

    return {
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("branch", "--show-current"),
        "git_dirty": bool(_run_git("status", "--porcelain")),
    }


def _parse_ints(val: str) -> list[int]:
    return [int(x.strip()) for x in val.split(",") if x.strip()]


def _parse_json_arg(val: str | None) -> dict[str, Any]:
    if not val:
        return {}
    val = val.strip()
    if val.startswith("@"):
        path = Path(val[1:]).expanduser()
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(val)


def _format_markdown_report(
    summary_rows: list[dict[str, Any]],
    hardware_name: str,
    model_name: str,
    tp_size: int,
) -> str:
    lines = [
        "# Standardized LLM Efficiency Benchmark Report",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Hardware**: `{hardware_name}` (TP={tp_size})",
        f"- **Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| System / Method | Prompt Len | Output Len | Batch | TTFT (ms) | TPOT (ms) | Prefill MFU | Decode MBU | Peak VRAM (GB) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in summary_rows:
        sys_label = f"`{r['engine']}-{r.get('sparse_method', 'vanilla')}`"
        lines.append(
            f"| {sys_label} | {r['prompt_len']} | {r['output_len']} | {r['batch_size']} "
            f"| {r.get('ttft_ms_mean', 0.0):.2f} | {r.get('tpot_ms_mean', 0.0):.2f} "
            f"| {r.get('prefill_mfu_pct_mean', 0.0):.1f}% | {r.get('decode_mbu_pct_mean', 0.0):.1f}% "
            f"| {r.get('peak_vram_gb_max', 0.0):.2f} | {r.get('status', 'success')} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_sparsevllm_probe(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
    gpu_profile: Any,
) -> list[dict[str, Any]]:
    import torch
    from sparsevllm import LLM, SamplingParams
    from sparsevllm.utils.profiler import profiler

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples_file = output_dir / "raw_samples.jsonl"

    hyper_params = _parse_json_arg(args.hyper_params)
    hyper_params.setdefault("tensor_parallel_size", args.tensor_parallel_size)
    hyper_params.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
    hyper_params.setdefault("decode_cuda_graph", True)

    sparse_kwargs: dict[str, Any] = {"sparse_method": args.sparse_method}
    if args.sparse_method == "snapkv":
        hyper_params.setdefault("snapkv_window_size", 64)
        hyper_params.setdefault("sink_keep_tokens", 64)
        hyper_params.setdefault("decode_keep_tokens", 2048)
        hyper_params.setdefault("recent_keep_tokens", 64)
        hyper_params.setdefault("pool_kernel_size", 7)
        hyper_params.setdefault("sparse_prefill_score_mode", "tilelang_raw_qk")

    sparse_budget = (
        int(hyper_params.get("sink_keep_tokens", 64))
        + int(hyper_params.get("decode_keep_tokens", 2048))
        + int(hyper_params.get("recent_keep_tokens", 64))
        if args.sparse_method == "snapkv"
        else None
    )

    max_len_needed = max(args.prompt_lens) + max(args.output_lens) + 128
    engine_kwargs = {
        **hyper_params,
        "max_model_len": max_len_needed,
        **sparse_kwargs,
    }

    print(f"[Sparse-vLLM Probe] Initializing LLM with method={args.sparse_method}, max_model_len={max_len_needed}...")
    llm = LLM(args.model_path, **engine_kwargs)

    for p_len in args.prompt_lens:
        for o_len in args.output_lens:
            for bs in args.batch_sizes:
                print(f"\n---> Benchmarking Sweep: PromptLen={p_len}, OutputLen={o_len}, BatchSize={bs} <---")
                prompt_tokens = [[100] * p_len for _ in range(bs)]
                sampling_params = [
                    SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        ignore_eos=True,
                        max_tokens=o_len,
                    )
                    for _ in range(bs)
                ]

                # 1. Warmup
                for w_idx in range(args.num_warmups):
                    print(f"  [Warmup {w_idx + 1}/{args.num_warmups}]...")
                    for b_i in range(bs):
                        llm.add_request(prompt_tokens[b_i], sampling_params[b_i])
                    while not llm.is_finished():
                        llm.step()

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                # 2. Benchmark Iterations
                iter_records = []
                for it in range(args.num_iters):
                    profiler.reset()
                    prefill_times = []
                    decode_times = []
                    ttft_ms = None
                    t_start = time.perf_counter()

                    for b_i in range(bs):
                        llm.add_request(prompt_tokens[b_i], sampling_params[b_i])

                    while not llm.is_finished():
                        step_s = time.perf_counter()
                        finished_outputs, num_tokens = llm.step()
                        torch.cuda.synchronize()
                        step_dt = time.perf_counter() - step_s

                        if num_tokens > 0:
                            prefill_times.append(step_dt)
                            if ttft_ms is None and (
                                llm.scheduler.decoding or finished_outputs
                            ):
                                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                        elif num_tokens < 0:
                            decode_times.append(step_dt)
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t_start) * 1000.0

                    if ttft_ms is None:
                        ttft_ms = sum(prefill_times) * 1000.0
                    tpot_ms = (sum(decode_times) / len(decode_times) * 1000.0) if decode_times else 0.0
                    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    # Calculate FLOPs & MFU
                    p_flops = model_specs.calculate_prefill_flops(p_len, batch_size=bs)
                    prefill_mfu = calculate_mfu(
                        p_flops,
                        ttft_ms / 1000.0,
                        args.tensor_parallel_size,
                        gpu_profile.peak_tflops_bf16,
                    )

                    # Calculate Bytes & MBU
                    d_bytes = model_specs.calculate_decode_step_bytes(
                        context_len=p_len + o_len // 2,
                        sparse_budget=sparse_budget,
                        batch_size=bs,
                    )
                    decode_mbu = calculate_mbu(
                        d_bytes,
                        tpot_ms / 1000.0,
                        args.tensor_parallel_size,
                        gpu_profile.peak_bandwidth_tbs,
                    )

                    profiler_snap = profiler.snapshot()

                    rec = {
                        "engine": "sparsevllm",
                        "sparse_method": args.sparse_method,
                        "prompt_len": p_len,
                        "output_len": o_len,
                        "batch_size": bs,
                        "iteration": it,
                        "ttft_ms": round(ttft_ms, 2),
                        "tpot_ms": round(tpot_ms, 2),
                        "prefill_mfu_pct": round(prefill_mfu, 2),
                        "decode_mbu_pct": round(decode_mbu, 2),
                        "peak_vram_gb": round(peak_mem_gb, 2),
                        "profiler_breakdown": profiler_snap,
                    }
                    iter_records.append(rec)
                    with open(raw_samples_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    print(
                        f"  Iter {it + 1}/{args.num_iters}: TTFT={ttft_ms:.1f}ms (MFU={prefill_mfu:.1f}%) | "
                        f"TPOT={tpot_ms:.2f}ms (MBU={decode_mbu:.1f}%) | Peak VRAM={peak_mem_gb:.2f}GB"
                    )

                # Aggregated summary
                ttft_vals = [r["ttft_ms"] for r in iter_records]
                tpot_vals = [r["tpot_ms"] for r in iter_records]
                mfu_vals = [r["prefill_mfu_pct"] for r in iter_records]
                mbu_vals = [r["decode_mbu_pct"] for r in iter_records]
                vram_vals = [r["peak_vram_gb"] for r in iter_records]

                summary_row = {
                    "engine": "sparsevllm",
                    "sparse_method": args.sparse_method,
                    "prompt_len": p_len,
                    "output_len": o_len,
                    "batch_size": bs,
                    "ttft_ms_mean": round(sum(ttft_vals) / len(ttft_vals), 2),
                    "ttft_ms_min": round(min(ttft_vals), 2),
                    "tpot_ms_mean": round(sum(tpot_vals) / len(tpot_vals), 2),
                    "tpot_ms_min": round(min(tpot_vals), 2),
                    "prefill_mfu_pct_mean": round(sum(mfu_vals) / len(mfu_vals), 2),
                    "decode_mbu_pct_mean": round(sum(mbu_vals) / len(mbu_vals), 2),
                    "peak_vram_gb_max": round(max(vram_vals), 2),
                    "status": "success",
                }
                results.append(summary_row)

    return results


def run_vllm_probe(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
    gpu_profile: Any,
) -> list[dict[str, Any]]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples_file = output_dir / "raw_samples.jsonl"

    max_len_needed = max(args.prompt_lens) + max(args.output_lens) + 128
    print(f"[vLLM Probe] Initializing vLLM (TP={args.tensor_parallel_size}, max_model_len={max_len_needed})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_len_needed,
        enable_prefix_caching=False,
        disable_log_stats=True,
        trust_remote_code=True,
    )

    for p_len in args.prompt_lens:
        for o_len in args.output_lens:
            for bs in args.batch_sizes:
                print(f"\n---> Benchmarking vLLM Sweep: PromptLen={p_len}, OutputLen={o_len}, BatchSize={bs} <---")
                sampling_params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    ignore_eos=True,
                    max_tokens=o_len,
                    detokenize=False,
                )

                # Warmup with unique token IDs
                for w_idx in range(args.num_warmups):
                    print(f"  [Warmup {w_idx + 1}/{args.num_warmups}]...")
                    warmup_prompts = [{"prompt_token_ids": [50 + w_idx] * p_len} for _ in range(bs)]
                    llm.generate(warmup_prompts, sampling_params=sampling_params, use_tqdm=False)

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                iter_records = []
                for it in range(args.num_iters):
                    prompt_token_id = 100 + it * 17
                    prompt_tokens = [{"prompt_token_ids": [prompt_token_id] * p_len} for _ in range(bs)]
                    
                    # 1. Measure TTFT via single-token prefill
                    torch.cuda.synchronize()
                    t_ttft_0 = time.perf_counter()
                    llm.generate(
                        prompt_tokens,
                        sampling_params=SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False),
                        use_tqdm=False,
                    )
                    torch.cuda.synchronize()
                    ttft_ms = (time.perf_counter() - t_ttft_0) * 1000.0

                    # 2. Measure full generation with fresh prompt tokens to avoid any cache
                    gen_prompt_tokens = [{"prompt_token_ids": [prompt_token_id + 1] * p_len} for _ in range(bs)]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    outputs = llm.generate(
                        gen_prompt_tokens,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    total_duration_ms = (t1 - t0) * 1000.0
                    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    decode_time_ms = max(0.001, total_duration_ms - ttft_ms)
                    tpot_ms = decode_time_ms / max(1, (o_len - 1))

                    p_flops = model_specs.calculate_prefill_flops(p_len, batch_size=bs)
                    prefill_mfu = calculate_mfu(
                        p_flops,
                        ttft_ms / 1000.0,
                        args.tensor_parallel_size,
                        gpu_profile.peak_tflops_bf16,
                    )

                    d_bytes = model_specs.calculate_decode_step_bytes(
                        context_len=p_len + o_len // 2,
                        sparse_budget=None,
                        batch_size=bs,
                    )
                    decode_mbu = calculate_mbu(
                        d_bytes,
                        tpot_ms / 1000.0,
                        args.tensor_parallel_size,
                        gpu_profile.peak_bandwidth_tbs,
                    )

                    rec = {
                        "engine": "vllm",
                        "sparse_method": "vanilla",
                        "prompt_len": p_len,
                        "output_len": o_len,
                        "batch_size": bs,
                        "iteration": it,
                        "ttft_ms": round(ttft_ms, 2),
                        "tpot_ms": round(tpot_ms, 2),
                        "prefill_mfu_pct": round(prefill_mfu, 2),
                        "decode_mbu_pct": round(decode_mbu, 2),
                        "peak_vram_gb": round(peak_mem_gb, 2),
                    }
                    iter_records.append(rec)
                    with open(raw_samples_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    print(
                        f"  Iter {it + 1}/{args.num_iters}: TTFT={ttft_ms:.1f}ms (MFU={prefill_mfu:.1f}%) | "
                        f"TPOT={tpot_ms:.2f}ms (MBU={decode_mbu:.1f}%) | Peak VRAM={peak_mem_gb:.2f}GB"
                    )

                ttft_vals = [r["ttft_ms"] for r in iter_records]
                tpot_vals = [r["tpot_ms"] for r in iter_records]
                mfu_vals = [r["prefill_mfu_pct"] for r in iter_records]
                mbu_vals = [r["decode_mbu_pct"] for r in iter_records]
                vram_vals = [r["peak_vram_gb"] for r in iter_records]

                summary_row = {
                    "engine": "vllm",
                    "sparse_method": "vanilla",
                    "prompt_len": p_len,
                    "output_len": o_len,
                    "batch_size": bs,
                    "ttft_ms_mean": round(sum(ttft_vals) / len(ttft_vals), 2),
                    "ttft_ms_min": round(min(ttft_vals), 2),
                    "tpot_ms_mean": round(sum(tpot_vals) / len(tpot_vals), 2),
                    "tpot_ms_min": round(min(tpot_vals), 2),
                    "prefill_mfu_pct_mean": round(sum(mfu_vals) / len(mfu_vals), 2),
                    "decode_mbu_pct_mean": round(sum(mbu_vals) / len(mbu_vals), 2),
                    "peak_vram_gb_max": round(max(vram_vals), 2),
                    "status": "success",
                }
                results.append(summary_row)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Standardized Synthetic Length Sweep & Efficiency Probe.")
    parser.add_argument("--engine", type=str, choices=["sparsevllm", "vllm"], default="sparsevllm")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--sparse-method", type=str, default="vanilla", help="Sparse method name (for sparsevllm)")
    parser.add_argument("--prompt-lens", type=_parse_ints, default="8192,16384,32768")
    parser.add_argument("--output-lens", type=_parse_ints, default="128")
    parser.add_argument("--batch-sizes", type=_parse_ints, default="1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hyper-params", type=str, default="{}")
    parser.add_argument("--hardware", type=str, default="h100_sxm", help="Target hardware key in registry")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_profile = detect_gpu_hardware(args.hardware)
    model_specs = ModelArchitectureSpecs.from_model_path_or_name(args.model_path)

    # 1. Save Run Manifest
    manifest = {
        "manifest_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "git": _git_metadata(),
        "args": vars(args),
        "hardware": {
            "name": gpu_profile.name,
            "peak_tflops_bf16": gpu_profile.peak_tflops_bf16,
            "peak_bandwidth_tbs": gpu_profile.peak_bandwidth_tbs,
            "tdp_watts": gpu_profile.tdp_watts,
        },
        "model_specs": {
            "hidden_size": model_specs.hidden_size,
            "num_hidden_layers": model_specs.num_hidden_layers,
            "is_moe": model_specs.is_moe,
            "num_experts": model_specs.num_experts,
            "num_experts_per_tok": model_specs.num_experts_per_tok,
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 2. Run Benchmarks
    if args.engine == "sparsevllm":
        summary_rows = run_sparsevllm_probe(args, model_specs, gpu_profile)
    else:
        summary_rows = run_vllm_probe(args, model_specs, gpu_profile)

    # 3. Save Summary & Markdown Report
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows}, f, indent=2, ensure_ascii=False)

    md_report = _format_markdown_report(
        summary_rows,
        gpu_profile.name,
        Path(args.model_path).name,
        args.tensor_parallel_size,
    )
    with open(output_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + "=" * 100)
    print(md_report)
    print("=" * 100)
    print(f"\n[Bench Probe] Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
