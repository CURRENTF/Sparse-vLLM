#!/usr/bin/env python3
"""Matched Qwen-style decode attention and routed-MoE microbenchmark.

The benchmark measures serving-relevant steady-state call boundaries and reports
the attention/MoE share of their two-component sum. It intentionally excludes
QKV/O projections, router logits/top-k, collectives, normalization, sampling,
and host scheduling, so the result is not an end-to-end decode-step metric.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass(frozen=True)
class ModelShape:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


@dataclass
class TimedCallable:
    name: str
    run: Callable[[], object]
    output: torch.Tensor
    keepalive: tuple[object, ...]


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("duplicate values are not allowed")
    return values


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p90_ms": _percentile(values, 0.9),
    }


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = actual_f - expected_f
    denominator = expected_f.norm().clamp_min(1.0e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / denominator).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual_f.flatten(), expected_f.flatten(), dim=0
        ).item(),
    }


def _check_error(label: str, error: dict[str, float], *, relative_l2: float) -> None:
    if not math.isfinite(error["relative_l2"]) or error["relative_l2"] > relative_l2:
        raise RuntimeError(
            f"{label} failed correctness: relative_l2={error['relative_l2']:.6f} "
            f"limit={relative_l2:.6f}, cosine={error['cosine']:.6f}, "
            f"max_abs={error['max_abs']:.6f}"
        )


def _capture(callable_: TimedCallable) -> TimedCallable:
    for _ in range(3):
        callable_.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        callable_.run()
    torch.cuda.synchronize()
    return TimedCallable(
        name=callable_.name,
        run=graph.replay,
        output=callable_.output,
        keepalive=(*callable_.keepalive, graph),
    )


def _measure(
    callable_: TimedCallable,
    *,
    warmup: int,
    samples: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        callable_.run()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            callable_.run()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / iterations)
    return values


def _attention_oracle(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch_size: int,
    context_len: int,
) -> torch.Tensor:
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[1])
    groups = num_query_heads // num_kv_heads
    q_grouped = q.float().view(
        batch_size, num_kv_heads, groups, q.shape[-1]
    )
    k = k_cache.view(batch_size, context_len, num_kv_heads, -1)
    v = v_cache.view(batch_size, context_len, num_kv_heads, -1)
    k = k.permute(0, 2, 1, 3).float()
    v = v.permute(0, 2, 1, 3).float()
    scores = torch.einsum("bkgd,bkld->bkgl", q_grouped, k)
    scores.mul_(q.shape[-1] ** -0.5)
    output = torch.einsum(
        "bkgl,bkld->bkgd", torch.softmax(scores, dim=-1), v
    )
    return output.reshape(batch_size, num_query_heads, q.shape[-1]).to(q.dtype)


def _make_attention_callables(
    shape: ModelShape,
    *,
    batch_size: int,
    context_len: int,
    seed: int,
    backends: set[str],
) -> tuple[dict[str, TimedCallable], dict[str, dict[str, float]]]:

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 1009 * batch_size + context_len)
    q = torch.randn(
        batch_size,
        shape.num_query_heads,
        shape.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    cache_shape = (
        batch_size * context_len,
        shape.num_kv_heads,
        shape.head_dim,
    )
    k_cache = torch.randn(
        cache_shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    v_cache = torch.randn(
        cache_shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    active_slots = torch.arange(
        batch_size * context_len, dtype=torch.int32, device="cuda"
    ).view(batch_size, context_len)
    req_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device="cuda"
    )

    shared = (
        q,
        k_cache,
        v_cache,
        active_slots,
        req_indices,
        context_lens,
    )
    callables: dict[str, TimedCallable] = {}
    if "triton" in backends:
        from sparsevllm.kernels.triton.flash_decoding_stage2 import (
            flash_decode_stage2,
        )
        from sparsevllm.kernels.triton.gqa_flash_decoding_stage1 import (
            flash_decode_stage1,
        )

        block_seq = 256
        num_blocks = (context_len + block_seq - 1) // block_seq
        triton_mid_o = torch.empty(
            batch_size,
            shape.num_query_heads,
            num_blocks,
            shape.head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        triton_mid_lse = torch.empty(
            batch_size,
            shape.num_query_heads,
            num_blocks,
            dtype=torch.float32,
            device="cuda",
        )
        triton_output = torch.empty_like(q)

        def run_triton() -> torch.Tensor:
            flash_decode_stage1(
                q,
                k_cache,
                v_cache,
                active_slots,
                req_indices,
                context_lens,
                context_len,
                triton_mid_o,
                triton_mid_lse,
                block_seq,
                16,
                2,
            )
            flash_decode_stage2(
                triton_mid_o,
                triton_mid_lse,
                context_lens,
                triton_output,
                block_seq,
            )
            return triton_output

        callables["triton"] = TimedCallable(
            "triton",
            run_triton,
            triton_output,
            (*shared, triton_mid_o, triton_mid_lse),
        )

    if "flashinfer" in backends:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        flashinfer_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend="auto",
        )
        indptr = torch.arange(
            0,
            (batch_size + 1) * context_len,
            context_len,
            dtype=torch.int32,
            device="cuda",
        )
        indices = torch.arange(
            batch_size * context_len, dtype=torch.int32, device="cuda"
        )
        last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")
        flashinfer_wrapper.plan(
            indptr,
            indices,
            last_page_len,
            shape.num_query_heads,
            shape.num_kv_heads,
            shape.head_dim,
            1,
            sm_scale=shape.head_dim**-0.5,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            non_blocking=False,
        )
        flashinfer_output = torch.empty_like(q)
        paged_cache = (k_cache.unsqueeze(1), v_cache.unsqueeze(1))

        def run_flashinfer() -> torch.Tensor:
            result = flashinfer_wrapper.run(q, paged_cache, out=flashinfer_output)
            if result.data_ptr() != flashinfer_output.data_ptr():
                raise RuntimeError(
                    "FlashInfer decode did not use the supplied output"
                )
            return flashinfer_output

        callables["flashinfer"] = TimedCallable(
            "flashinfer",
            run_flashinfer,
            flashinfer_output,
            (
                *shared,
                workspace,
                flashinfer_wrapper,
                indptr,
                indices,
                last_page_len,
            ),
        )

    for callable_ in callables.values():
        callable_.run()
    torch.cuda.synchronize()
    expected = _attention_oracle(
        q,
        k_cache,
        v_cache,
        batch_size=batch_size,
        context_len=context_len,
    )
    errors = {}
    for name, callable_ in callables.items():
        errors[f"{name}_vs_torch"] = _error(callable_.output, expected)
        _check_error(
            f"{name} attention",
            errors[f"{name}_vs_torch"],
            relative_l2=0.04,
        )
    if set(callables) == {"triton", "flashinfer"}:
        errors["flashinfer_vs_triton"] = _error(
            callables["flashinfer"].output,
            callables["triton"].output,
        )
    return callables, errors


def _dequantize_blocks(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    expanded = scale.float().repeat_interleave(128, dim=-2).repeat_interleave(
        128, dim=-1
    )
    return weight.float() * expanded


def _torch_quantize_group128(inputs: torch.Tensor) -> torch.Tensor:
    groups = inputs.float().view(inputs.shape[0], -1, 128)
    scale = (groups.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1.0e-10)
    quantized = (groups / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).view_as(inputs.float())


def _moe_oracle(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    hidden_qdq = _torch_quantize_group128(hidden_states)
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_tensor in torch.unique(topk_ids):
        expert = int(expert_tensor.item())
        token_indices, route_indices = torch.where(topk_ids == expert)
        w13 = _dequantize_blocks(w13_weight[expert], w13_scale[expert])
        projected = hidden_qdq[token_indices] @ w13.T
        up = projected[:, :intermediate_size]
        gate = projected[:, intermediate_size:]
        activated = torch.nn.functional.silu(gate) * up
        activated_qdq = _torch_quantize_group128(activated)
        w2 = _dequantize_blocks(w2_weight[expert], w2_scale[expert])
        expert_output = activated_qdq @ w2.T
        expert_output.mul_(topk_weights[token_indices, route_indices, None].float())
        output.index_add_(0, token_indices, expert_output)
    return output.to(hidden_states.dtype)


def _make_moe_callables(
    shape: ModelShape,
    *,
    batch_size: int,
    seed: int,
    include_flashinfer: bool,
) -> tuple[
    TimedCallable,
    TimedCallable | None,
    dict[str, dict[str, float]],
]:
    from sparsevllm.kernels.triton.moe import fused_moe_fp8

    if include_flashinfer:
        from flashinfer.fused_moe import (
            Fp8QuantizationType,
            trtllm_fp8_block_scale_routed_moe,
        )
        from flashinfer.tllm_enums import ActivationType, RoutingMethodType
        from sparsevllm.kernels.external.sgl.moe import (
            sgl_per_token_group_quant_8bit,
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 7919 * batch_size)
    hidden_states = torch.randn(
        batch_size,
        shape.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    w13_weight = torch.randn(
        shape.num_experts,
        2 * shape.intermediate_size,
        shape.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).clamp_(-3, 3).to(torch.float8_e4m3fn)
    w2_weight = torch.randn(
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).clamp_(-3, 3).to(torch.float8_e4m3fn)
    w13_scale = torch.rand(
        shape.num_experts,
        2 * shape.intermediate_size // 128,
        shape.hidden_size // 128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).mul_(0.01).add_(0.005)
    w2_scale = torch.rand(
        shape.num_experts,
        shape.hidden_size // 128,
        shape.intermediate_size // 128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).mul_(0.01).add_(0.005)
    token_offsets = torch.arange(batch_size, dtype=torch.int32, device="cuda")[:, None]
    route_offsets = torch.arange(shape.top_k, dtype=torch.int32, device="cuda")[None, :]
    topk_ids = (token_offsets * shape.top_k + route_offsets).remainder(
        shape.num_experts
    ).contiguous()
    topk_weights = torch.rand(
        batch_size,
        shape.top_k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights.div_(topk_weights.float().sum(dim=-1, keepdim=True).to(torch.bfloat16))

    triton_output_holder: dict[str, torch.Tensor] = {}

    def run_triton() -> torch.Tensor:
        output = fused_moe_fp8(
            hidden_states,
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
            num_experts=shape.num_experts,
            local_expert_start=0,
            gate_up_order="up_gate",
        )
        triton_output_holder["output"] = output
        return output

    flashinfer_callable = None
    flashinfer_output = None
    if include_flashinfer:
        hidden_states_q = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
        hidden_scale_token_major = torch.empty(
            batch_size,
            shape.hidden_size // 128,
            dtype=torch.float32,
            device="cuda",
        )
        hidden_scale_k_major = torch.empty(
            shape.hidden_size // 128,
            batch_size,
            dtype=torch.float32,
            device="cuda",
        )
        packed_topk = torch.empty_like(topk_ids, dtype=torch.int32)
        flashinfer_output = torch.empty_like(hidden_states)
        fp8_info = torch.finfo(torch.float8_e4m3fn)

        def run_flashinfer() -> torch.Tensor:
            sgl_per_token_group_quant_8bit(
                hidden_states,
                hidden_states_q,
                hidden_scale_token_major,
                128,
                1.0e-10,
                fp8_info.min,
                fp8_info.max,
                enable_v2=True,
            )
            hidden_scale_k_major.copy_(hidden_scale_token_major.T)
            weight_bits = topk_weights.view(torch.int16).to(torch.int32)
            packed_topk.copy_((topk_ids << 16) | (weight_bits & 0xFFFF))
            trtllm_fp8_block_scale_routed_moe(
                packed_topk,
                None,
                hidden_states_q,
                hidden_scale_k_major,
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                shape.num_experts,
                shape.top_k,
                None,
                None,
                shape.intermediate_size,
                0,
                shape.num_experts,
                None,
                routing_method_type=RoutingMethodType.TopK.value,
                use_shuffled_weight=False,
                weight_layout=0,
                do_finalize=True,
                enable_pdl=None,
                output=flashinfer_output,
                tune_max_num_tokens=max(128, batch_size),
                fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
                activation_type=ActivationType.Swiglu.value,
            )
            return flashinfer_output

        flashinfer_callable = TimedCallable(
            "flashinfer",
            run_flashinfer,
            flashinfer_output,
            (
                hidden_states_q,
                hidden_scale_token_major,
                hidden_scale_k_major,
                packed_topk,
            ),
        )

    triton_output = run_triton()
    if flashinfer_callable is not None:
        flashinfer_callable.run()
    torch.cuda.synchronize()
    expected = _moe_oracle(
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        shape.intermediate_size,
    )
    errors = {"triton_vs_torch": _error(triton_output, expected)}
    _check_error("Triton MoE", errors["triton_vs_torch"], relative_l2=0.12)
    if flashinfer_output is not None:
        errors["flashinfer_vs_torch"] = _error(flashinfer_output, expected)
        errors["flashinfer_vs_triton"] = _error(flashinfer_output, triton_output)
        _check_error(
            "FlashInfer MoE", errors["flashinfer_vs_torch"], relative_l2=0.12
        )

    shared = (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
    )
    return (
        TimedCallable(
            "triton",
            run_triton,
            triton_output,
            (*shared, triton_output_holder),
        ),
        flashinfer_callable,
        errors,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _record_samples(
    path: Path,
    *,
    component: str,
    backend: str,
    batch_size: int,
    context_len: int | None,
    values: list[float],
    graph: bool,
) -> None:
    for index, value in enumerate(values):
        _append_jsonl(
            path,
            {
                "status": "success",
                "component": component,
                "backend": backend,
                "batch_size": batch_size,
                "context_len": context_len,
                "graph": graph,
                "sample_index": index,
                "latency_ms": value,
            },
        )


def _format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Decode component microbenchmark",
        "",
        "| KV len | Batch | Attn Triton ms | Attn FI ms | FI/Tri attn | "
        "MoE Triton ms | MoE FI ms | FI/Tri MoE | Triton attn share | "
        "Triton MoE share |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {context_len} | {batch_size} | {attention_triton_ms} | "
            "{attention_flashinfer_ms} | {attention_ratio} | {moe_triton_ms} | "
            "{moe_flashinfer_ms} | {moe_ratio} | {attention_share:.1%} | "
            "{moe_share:.1%} |".format(
                context_len=row["context_len"],
                batch_size=row["batch_size"],
                attention_triton_ms=_format_metric(row["attention_triton_ms"]),
                attention_flashinfer_ms=_format_metric(
                    row["attention_flashinfer_ms"]
                ),
                attention_ratio=_format_metric(
                    row["attention_flashinfer_over_triton"], 3
                ),
                moe_triton_ms=_format_metric(row["moe_triton_ms"]),
                moe_flashinfer_ms=_format_metric(row["moe_flashinfer_ms"]),
                moe_ratio=_format_metric(row["moe_flashinfer_over_triton"], 3),
                attention_share=row["triton_attention_share"],
                moe_share=row["triton_moe_share"],
            )
        )
    lines.extend(
        [
            "",
            "Ratios above 1 mean FlashInfer is slower. Shares use the Triton "
            "attention + Triton MoE two-component sum and are not end-to-end "
            "decode shares. `n/a` means the provider failed or was skipped by "
            "the eligibility probe; consult `summary.json` and the probe log.",
            "",
        ]
    )
    return "\n".join(lines)


def _model_shape_from_args(args: argparse.Namespace) -> ModelShape:
    return ModelShape(
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )


def _probe_flashinfer_moe(args: argparse.Namespace) -> None:
    shape = _model_shape_from_args(args)
    _, flashinfer_call, _ = _make_moe_callables(
        shape,
        batch_size=args.batch_sizes[0],
        seed=args.seed,
        include_flashinfer=True,
    )
    if flashinfer_call is None:
        raise RuntimeError("FlashInfer MoE probe did not construct a callable")
    if args.graph:
        flashinfer_call = _capture(flashinfer_call)
    flashinfer_call.run()
    torch.cuda.synchronize()


def _run_flashinfer_moe_probe(
    args: argparse.Namespace, run_root: Path
) -> dict[str, object]:
    log_path = run_root / "flashinfer_moe_probe.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-flashinfer-moe",
        "--run-root",
        str(run_root),
        "--batch-sizes",
        str(args.batch_sizes[0]),
        "--seed",
        str(args.seed),
        "--num-query-heads",
        str(args.num_query_heads),
        "--num-kv-heads",
        str(args.num_kv_heads),
        "--head-dim",
        str(args.head_dim),
        "--hidden-size",
        str(args.hidden_size),
        "--intermediate-size",
        str(args.intermediate_size),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--graph" if args.graph else "--no-graph",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        return {"status": "available", "probe_log": str(log_path)}
    signal_number = -result.returncode if result.returncode < 0 else None
    return {
        "status": "unavailable",
        "return_code": result.returncode,
        "signal": signal_number,
        "probe_log": str(log_path),
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            f"this run targets SM120, got {torch.cuda.get_device_capability()}"
        )
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    raw_path = run_root / "raw_samples.jsonl"
    for artifact in (raw_path, run_root / "summary.json", run_root / "failure.json"):
        if artifact.exists():
            raise FileExistsError(
                f"refusing to overwrite benchmark artifact {artifact}"
            )
    raw_path.touch()

    shape = _model_shape_from_args(args)
    if shape.num_query_heads % shape.num_kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    for value, label in (
        (shape.hidden_size, "hidden_size"),
        (shape.intermediate_size, "intermediate_size"),
    ):
        if value % 128:
            raise ValueError(f"{label} must be divisible by 128")

    import flashinfer
    import sgl_kernel
    import triton

    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "flashinfer": flashinfer.__version__,
            "sglang_kernel": getattr(sgl_kernel, "__version__", "unknown"),
        },
        "model_shape": asdict(shape),
        "components": args.components,
        "attention_backends": list(args.attention_backends),
        "context_lengths": args.context_lengths,
        "batch_sizes": args.batch_sizes,
        "timing": {
            "graph": args.graph,
            "warmup": args.warmup,
            "samples": args.samples,
            "iterations_per_sample": args.iterations,
            "synchronization": "CUDA events with end-event synchronization",
        },
        "boundaries": {
            "attention_triton": "GQA stage1 + stage2 with preallocated workspace",
            "attention_flashinfer": "planned BatchDecodeWithPagedKVCacheWrapper.run",
            "moe_triton": (
                "fused_moe_fp8 including activation quantization and routing "
                "alignment"
            ),
            "moe_flashinfer": (
                "activation quantization + K-major scale copy + route packing "
                "+ trtllm_fp8_block_scale_routed_moe"
            ),
        },
    }
    _write_json(run_root / "run_manifest.json", manifest)

    run_attention = args.components in {"attention", "both"}
    run_moe = args.components in {"moe", "both"}
    if not run_moe:
        flashinfer_moe_probe = {
            "status": "skipped_by_policy",
            "reason": "MoE component was not requested",
        }
    elif args.flashinfer_moe == "skip":
        flashinfer_moe_probe = {"status": "skipped_by_policy"}
    else:
        print("[moe] probing FlashInfer in an isolated process", flush=True)
        flashinfer_moe_probe = _run_flashinfer_moe_probe(args, run_root)
        if (
            flashinfer_moe_probe["status"] != "available"
            and args.flashinfer_moe == "required"
        ):
            raise RuntimeError(
                "FlashInfer MoE preflight failed; see "
                f"{flashinfer_moe_probe['probe_log']}"
            )
    manifest["flashinfer_moe_probe"] = flashinfer_moe_probe
    _write_json(run_root / "run_manifest.json", manifest)
    flashinfer_moe_available = flashinfer_moe_probe["status"] == "available"
    if run_moe and not flashinfer_moe_available:
        _append_jsonl(
            raw_path,
            {
                "status": (
                    "skipped_by_policy"
                    if flashinfer_moe_probe["status"] == "skipped_by_policy"
                    else "model_failed"
                ),
                "component": "moe",
                "backend": "flashinfer",
                "batch_size": args.batch_sizes[0],
                "context_len": None,
                "graph": args.graph,
                "probe": flashinfer_moe_probe,
            },
        )

    moe_results: dict[int, dict[str, object]] = {}
    if run_moe:
        for batch_size in args.batch_sizes:
            print(f"[moe] batch={batch_size}", flush=True)
            triton_call, flashinfer_call, errors = _make_moe_callables(
                shape,
                batch_size=batch_size,
                seed=args.seed,
                include_flashinfer=flashinfer_moe_available,
            )
            if args.graph:
                triton_call = _capture(triton_call)
                if flashinfer_call is not None:
                    flashinfer_call = _capture(flashinfer_call)
            values = {}
            order = (triton_call,) + (
                (flashinfer_call,) if flashinfer_call is not None else ()
            )
            for callable_ in order:
                measured = _measure(
                    callable_,
                    warmup=args.warmup,
                    samples=args.samples,
                    iterations=args.iterations,
                )
                values[callable_.name] = measured
                _record_samples(
                    raw_path,
                    component="moe",
                    backend=callable_.name,
                    batch_size=batch_size,
                    context_len=None,
                    values=measured,
                    graph=args.graph,
                )
            moe_results[batch_size] = {
                "triton": _stats(values["triton"]),
                "flashinfer": (
                    _stats(values["flashinfer"])
                    if "flashinfer" in values
                    else flashinfer_moe_probe
                ),
                "correctness": errors,
            }
            del triton_call, flashinfer_call
            torch.cuda.empty_cache()

    rows = []
    attention_results = []
    if run_attention:
        for context_len in args.context_lengths:
            for batch_size in args.batch_sizes:
                print(
                    f"[attention] context={context_len} batch={batch_size}",
                    flush=True,
                )
                attention_callables, errors = _make_attention_callables(
                    shape,
                    batch_size=batch_size,
                    context_len=context_len,
                    seed=args.seed,
                    backends=set(args.attention_backends),
                )
                if args.graph:
                    attention_callables = {
                        name: _capture(callable_)
                        for name, callable_ in attention_callables.items()
                    }
                values = {}
                for callable_ in attention_callables.values():
                    measured = _measure(
                        callable_,
                        warmup=args.warmup,
                        samples=args.samples,
                        iterations=args.iterations,
                    )
                    values[callable_.name] = measured
                    _record_samples(
                        raw_path,
                        component="attention",
                        backend=callable_.name,
                        batch_size=batch_size,
                        context_len=context_len,
                        values=measured,
                        graph=args.graph,
                    )
                attention = {
                    "context_len": context_len,
                    "batch_size": batch_size,
                    "correctness": errors,
                }
                for backend, samples in values.items():
                    attention[backend] = _stats(samples)
                attention_results.append(attention)
                if run_moe and set(values) == {"triton", "flashinfer"}:
                    attention_triton = float(attention["triton"]["median_ms"])
                    attention_flashinfer = float(
                        attention["flashinfer"]["median_ms"]
                    )
                    moe_triton = float(
                        moe_results[batch_size]["triton"]["median_ms"]
                    )
                    moe_flashinfer_stats = moe_results[batch_size]["flashinfer"]
                    moe_flashinfer = (
                        float(moe_flashinfer_stats["median_ms"])
                        if "median_ms" in moe_flashinfer_stats
                        else None
                    )
                    triton_sum = attention_triton + moe_triton
                    rows.append(
                        {
                            "context_len": context_len,
                            "batch_size": batch_size,
                            "attention_triton_ms": attention_triton,
                            "attention_flashinfer_ms": attention_flashinfer,
                            "attention_flashinfer_over_triton": (
                                attention_flashinfer / attention_triton
                            ),
                            "moe_triton_ms": moe_triton,
                            "moe_flashinfer_ms": moe_flashinfer,
                            "moe_flashinfer_over_triton": (
                                moe_flashinfer / moe_triton
                                if moe_flashinfer is not None
                                else None
                            ),
                            "triton_attention_share": (
                                attention_triton / triton_sum
                            ),
                            "triton_moe_share": moe_triton / triton_sum,
                            "component_sum_ms": {
                                "triton_attention_triton_moe": triton_sum,
                                "flashinfer_attention_triton_moe": (
                                    attention_flashinfer + moe_triton
                                ),
                                "triton_attention_flashinfer_moe": (
                                    attention_triton + moe_flashinfer
                                    if moe_flashinfer is not None
                                    else None
                                ),
                                "flashinfer_attention_flashinfer_moe": (
                                    attention_flashinfer + moe_flashinfer
                                    if moe_flashinfer is not None
                                    else None
                                ),
                            },
                        }
                    )
                del attention_callables
                torch.cuda.empty_cache()

    summary = {
        "status": "success",
        "components": args.components,
        "metric_scope": (
            "steady-state decode attention and routed-MoE component callables"
        ),
        "graph": args.graph,
        "model_shape": asdict(shape),
        "flashinfer_moe_probe": flashinfer_moe_probe,
        "moe": moe_results,
        "attention": attention_results,
        "rows": rows,
        "limitations": [
            (
                "Component sums exclude projections, router logits/top-k, "
                "collectives, normalization, sampling, and host scheduling."
            ),
            (
                "The FlashInfer MoE candidate is a raw public-API composition, "
                "not yet a production Sparse-vLLM provider."
            ),
            (
                "Synthetic FP8 expert weights preserve shapes and layouts but "
                "are not checkpoint values."
            ),
        ],
    }
    _write_json(run_root / "summary.json", summary)
    report = _report(rows) if args.components == "both" else ""
    (run_root / "report.md").write_text(report, encoding="utf-8")
    manifest["status"] = "completed"
    _write_json(run_root / "run_manifest.json", manifest)
    if report:
        print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--components",
        choices=("attention", "moe", "both"),
        default="both",
    )
    parser.add_argument(
        "--attention-backends",
        type=lambda value: tuple(
            part.strip() for part in value.split(",") if part.strip()
        ),
        default=("triton", "flashinfer"),
    )
    parser.add_argument(
        "--context-lengths", type=_parse_int_list, default=[1024, 4096, 8192]
    )
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--flashinfer-moe",
        choices=("probe", "required", "skip"),
        default="probe",
        help=(
            "isolate the optional FlashInfer MoE launch and record failure, "
            "fail the benchmark when unavailable, or skip it"
        ),
    )
    parser.add_argument(
        "--probe-flashinfer-moe", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    valid_attention_backends = {"triton", "flashinfer"}
    if not args.attention_backends:
        parser.error("--attention-backends must select at least one backend")
    if len(set(args.attention_backends)) != len(args.attention_backends):
        parser.error("--attention-backends contains duplicates")
    invalid_attention_backends = (
        set(args.attention_backends) - valid_attention_backends
    )
    if invalid_attention_backends:
        parser.error(
            "unsupported --attention-backends values: "
            + ",".join(sorted(invalid_attention_backends))
        )
    for name in ("warmup", "samples", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.probe_flashinfer_moe:
        _probe_flashinfer_moe(args)
        return
    try:
        run(args)
    except Exception as error:
        if args.run_root.exists():
            failure = {
                "status": "model_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(args.run_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
