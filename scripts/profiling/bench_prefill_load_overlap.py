#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median

import torch
from flash_attn import flash_attn_func


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def event_ms(fn, *, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def h2d_time_ms(
    *,
    n_tokens: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    if n_tokens <= 0:
        return 0.0, math.inf
    shape = (n_tokens, kv_heads, head_dim)
    k_cpu = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    v_cpu = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    k_gpu = torch.empty(shape, dtype=dtype, device="cuda")
    v_gpu = torch.empty(shape, dtype=dtype, device="cuda")

    def copy_once() -> None:
        k_gpu.copy_(k_cpu, non_blocking=True)
        v_gpu.copy_(v_cpu, non_blocking=True)

    times = event_ms(copy_once, repeats=repeats, warmup=warmup)
    ms = median(times)
    bytes_moved = 2 * n_tokens * kv_heads * head_dim * torch.tensor([], dtype=dtype).element_size()
    gib_s = (bytes_moved / (1024**3)) / (ms / 1000.0) if ms > 0 else math.inf
    del k_cpu, v_cpu, k_gpu, v_gpu
    torch.cuda.empty_cache()
    return ms, gib_s


def attention_time_ms(
    *,
    n_tokens: int,
    chunk_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
) -> float:
    kv_len = n_tokens + chunk_size
    q = torch.randn((1, chunk_size, q_heads, head_dim), dtype=dtype, device="cuda")
    k = torch.randn((1, kv_len, kv_heads, head_dim), dtype=dtype, device="cuda")
    v = torch.randn((1, kv_len, kv_heads, head_dim), dtype=dtype, device="cuda")

    def attn_once() -> None:
        flash_attn_func(q, k, v, causal=True)

    times = event_ms(attn_once, repeats=repeats, warmup=warmup)
    del q, k, v
    torch.cuda.empty_cache()
    return median(times)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure when one-layer chunk prefill attention becomes longer than "
            "loading one layer of historical KV from pinned CPU memory."
        )
    )
    parser.add_argument("--n-tokens", default="0,16384,32768,65536,131072")
    parser.add_argument("--chunk-sizes", default="8192,16384,32768,65536")
    parser.add_argument("--q-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    n_values = parse_int_list(args.n_tokens)
    chunk_values = parse_int_list(args.chunk_sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    device_name = torch.cuda.get_device_name(0)
    for chunk_size in chunk_values:
        for n_tokens in n_values:
            torch.cuda.empty_cache()
            load_ms, load_gib_s = h2d_time_ms(
                n_tokens=n_tokens,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            torch.cuda.empty_cache()
            attn_ms = attention_time_ms(
                n_tokens=n_tokens,
                chunk_size=chunk_size,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            kv_gib = (
                2
                * n_tokens
                * args.kv_heads
                * args.head_dim
                * torch.tensor([], dtype=dtype).element_size()
                / (1024**3)
            )
            row = {
                "device": device_name,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "n_tokens": n_tokens,
                "chunk_size": chunk_size,
                "kv_gib": kv_gib,
                "load_ms": load_ms,
                "load_gib_s": load_gib_s,
                "attn_ms": attn_ms,
                "attn_over_load": attn_ms / load_ms if load_ms > 0 else math.inf,
                "attn_ge_load": bool(attn_ms >= load_ms),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    csv_path = args.output.with_suffix(".csv")
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
