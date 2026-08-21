# SPDX-License-Identifier: Apache-2.0
"""Standardized Synthetic Length Sweep & Micro-Efficiency Benchmark for Sparse-vLLM & vLLM."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark.efficiency.hardware_monitor import GPUHardwareMonitor
from benchmark.efficiency.metrics_calculator import ModelArchitectureSpecs
from benchmark.efficiency.workload import (
    TRACE_GENERATOR_VERSION,
    build_request_trace,
    derive_trace_seed,
    trace_metadata,
)


class HardwareMetricError(RuntimeError):
    """Raised when directly sampled GPU metrics are unavailable or incomplete."""


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


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}.")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _monitor_gpu_ids(explicit: str | None) -> list[int]:
    value = explicit or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not value:
        raise ValueError(
            "Actual GPU metric collection requires --monitor-gpus or an integer-only "
            "CUDA_VISIBLE_DEVICES list."
        )
    try:
        gpu_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "Actual GPU metric collection currently requires physical integer GPU IDs, "
            f"got {value!r}."
        ) from exc
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU metric IDs must be non-empty and unique, got {gpu_ids}.")
    return gpu_ids


def _case_monitor(
    args: argparse.Namespace,
    case_name: str,
) -> GPUHardwareMonitor:
    output_file = Path(args.output_dir) / "case_hardware" / f"{case_name}.json"
    monitor = GPUHardwareMonitor(
        _monitor_gpu_ids(args.monitor_gpus),
        interval_ms=args.hardware_sampling_interval_ms,
        output_file=output_file,
    )
    monitor.start()
    return monitor


def _actual_hardware_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "success":
        raise HardwareMetricError(f"Actual GPU metric collection failed: {summary}")
    aggregate = summary.get("aggregate")
    gpus = summary.get("gpus")
    if not isinstance(aggregate, dict) or not isinstance(gpus, dict) or not gpus:
        raise HardwareMetricError(f"Actual GPU metric summary is incomplete: {summary}")
    return {
        "metric_source": "nvidia-smi sampled activity",
        "sample_count": int(summary["total_samples"]),
        "sampling_interval_ms": int(summary["sampling_interval_ms"]),
        "gpu_compute_activity_pct_mean": float(aggregate["mean_compute_util_pct"]),
        "gpu_memory_io_activity_pct_mean": float(
            aggregate["mean_memory_io_activity_pct"]
        ),
        "gpu_active_duty_pct_mean": float(
            aggregate["mean_coarse_gpu_active_duty_pct"]
        ),
        "gpu_power_w_mean_total": float(aggregate["avg_total_power_w"]),
        "peak_vram_gb_max": max(float(gpu["peak_vram_gb"]) for gpu in gpus.values()),
        "per_gpu": gpus,
    }


def _trace_for_iteration(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
    *,
    scenario: str,
    phase: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
    iteration: int,
    request_count: int,
    vary_output_lengths: bool,
):
    seed = derive_trace_seed(
        args.seed,
        scenario=scenario,
        phase=phase,
        nominal_prompt_len=prompt_len,
        nominal_output_len=output_len,
        concurrency=concurrency,
        iteration=iteration,
    )
    return build_request_trace(
        seed=seed,
        request_count=request_count,
        nominal_prompt_len=prompt_len,
        nominal_output_len=output_len,
        vocab_size=model_specs.vocab_size,
        prompt_jitter_fraction=args.prompt_length_jitter,
        output_jitter_fraction=args.output_length_jitter,
        vary_output_lengths=vary_output_lengths,
    )


def _attach_churn_comparisons(rows: list[dict[str, Any]]) -> None:
    fixed_by_case: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("prompt_len"),
            row.get("output_len"),
            row.get("concurrency"),
        )
        if row.get("scenario") == "fixed_batch":
            fixed_by_case[key] = row
    for row in rows:
        if row.get("scenario") != "oversubscribed_churn":
            continue
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("prompt_len"),
            row.get("output_len"),
            row.get("concurrency"),
        )
        fixed = fixed_by_case.get(key)
        if fixed is None:
            row["fixed_batch_comparison_status"] = "skipped_by_policy"
            continue
        fixed_output_tps = float(fixed["output_token_throughput_tps"])
        fixed_request_rps = float(fixed["request_throughput_rps"])
        if fixed_output_tps <= 0 or fixed_request_rps <= 0:
            raise ValueError(f"Fixed-batch throughput must be positive for comparison: {fixed}")
        row["fixed_batch_comparison_status"] = "success"
        row["churn_output_tps_ratio_vs_fixed_batch"] = (
            float(row["output_token_throughput_tps"]) / fixed_output_tps
        )
        row["churn_request_rps_ratio_vs_fixed_batch"] = (
            float(row["request_throughput_rps"]) / fixed_request_rps
        )
        row["churn_ttft_p99_delta_ms_vs_fixed_batch"] = (
            float(row["ttft_ms_p99"]) - float(fixed["ttft_ms_p99"])
        )


def _attach_saturation_metrics(rows: list[dict[str, Any]]) -> None:
    """Attach finite-concurrency-sweep scaling metrics without claiming a hard limit."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("scenario"),
            row.get("prompt_len"),
            row.get("output_len"),
        )
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: int(row["concurrency"]))
        concurrencies = [int(row["concurrency"]) for row in ordered]
        if len(concurrencies) != len(set(concurrencies)):
            raise ValueError(
                f"Saturation sweep contains duplicate concurrency values: {concurrencies}."
            )
        rates = [float(row["output_token_throughput_tps"]) for row in ordered]
        if any(rate <= 0 for rate in rates):
            raise ValueError(f"Saturation sweep throughput must be positive: {rates}.")

        observed_peak = max(rates)
        base_concurrency = concurrencies[0]
        base_rate = rates[0]
        saturation_concurrency = next(
            concurrency
            for concurrency, rate in zip(concurrencies, rates)
            if rate >= 0.95 * observed_peak
        )
        analysis_status = "success" if len(ordered) > 1 else "skipped_by_policy"

        for index, (row, concurrency, rate) in enumerate(
            zip(ordered, concurrencies, rates)
        ):
            row["saturation_analysis_status"] = analysis_status
            row["output_tps_pct_of_observed_sweep_peak"] = rate / observed_peak * 100.0
            row["output_tps_scaling_efficiency_pct_vs_min_concurrency"] = (
                (rate / base_rate) / (concurrency / base_concurrency) * 100.0
            )
            row["marginal_output_tps_gain_pct_vs_previous_concurrency"] = (
                None if index == 0 else (rate / rates[index - 1] - 1.0) * 100.0
            )
            row["observed_output_saturation_threshold_pct"] = 95.0
            row["observed_output_saturation_concurrency"] = (
                saturation_concurrency if analysis_status == "success" else None
            )


def _append_request_samples(
    output_dir: Path,
    *,
    engine: str,
    sparse_method: str,
    scenario: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
    iteration: int,
    requests: list[dict[str, Any]],
) -> None:
    path = output_dir / "request_samples.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for request in requests:
            row = {
                "engine": engine,
                "sparse_method": sparse_method,
                "scenario": scenario,
                "nominal_prompt_len": prompt_len,
                "nominal_output_len": output_len,
                "concurrency": concurrency,
                "iteration": iteration,
                **request,
            }
            if row.get("status") != "success":
                raise ValueError(f"Benchmark request sample is not successful: {row}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _vllm_request_phase_seconds(metrics: Any) -> tuple[float, float, str]:
    """Return TTFT, decode duration, and the explicit vLLM timing contract."""
    legacy_first = getattr(metrics, "first_token_time", None)
    legacy_finished = getattr(metrics, "finished_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    if legacy_first is not None and legacy_finished is not None and arrival is not None:
        ttft_s = float(legacy_first) - float(arrival)
        decode_s = float(legacy_finished) - float(legacy_first)
        source = "vllm_legacy_wall_timestamps"
    else:
        first_token_latency = getattr(metrics, "first_token_latency", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        if (
            first_token_latency is None
            or first_token_ts is None
            or last_token_ts is None
            or float(first_token_latency) <= 0
            or float(first_token_ts) <= 0
            or float(last_token_ts) <= 0
        ):
            raise RuntimeError(
                "vLLM request metrics expose neither complete legacy timestamps nor "
                "complete V1 latency/timestamps."
            )
        ttft_s = float(first_token_latency)
        decode_s = float(last_token_ts) - float(first_token_ts)
        source = "vllm_v1_first_token_latency_and_monotonic_decode"
    if ttft_s <= 0 or decode_s < 0:
        raise RuntimeError(
            f"vLLM reported invalid phase timing: ttft_s={ttft_s}, decode_s={decode_s}."
        )
    return ttft_s, decode_s, source


def _vllm_phase_metrics(outputs: list[Any], expected_output_len: int) -> tuple[float, float | None]:
    """Return batch TTFT and mean per-request TPOT from the same requests."""
    if not outputs:
        raise RuntimeError("vLLM returned no request outputs.")
    ttft_values = []
    tpot_values = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            raise RuntimeError(
                "vLLM request metrics are unavailable; TTFT/TPOT require request timing."
            )
        token_count = len(output.outputs[0].token_ids) if output.outputs else 0
        if token_count != expected_output_len:
            raise RuntimeError(
                f"vLLM generated {token_count} tokens, expected output_len={expected_output_len}."
            )
        ttft_s, decode_s, _source = _vllm_request_phase_seconds(metrics)
        ttft_values.append(ttft_s * 1000.0)
        if token_count > 1:
            decode_ms = decode_s * 1000.0
            if decode_ms <= 0:
                raise RuntimeError(f"vLLM reported non-positive decode duration {decode_ms} ms.")
            tpot_values.append(decode_ms / (token_count - 1))
    return max(ttft_values), (sum(tpot_values) / len(tpot_values) if tpot_values else None)


def _record_batch_first_tokens(
    expected_seq_ids: set[int],
    observed_seq_ids: set[int],
    step_outputs: list[tuple[int, list[int]]],
    finished_outputs: list[tuple[int, list[int], Any, Any]],
) -> bool:
    """Record first-token publication and report when the whole batch is covered."""
    for seq_id, token_ids in step_outputs:
        if token_ids and int(seq_id) in expected_seq_ids:
            observed_seq_ids.add(int(seq_id))
    for seq_id, token_ids, _token_logprobs, _top_logprobs in finished_outputs:
        if token_ids and int(seq_id) in expected_seq_ids:
            observed_seq_ids.add(int(seq_id))
    return observed_seq_ids == expected_seq_ids


def _resolve_sparse_probe_protocol(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int | None, dict[str, Any], str]:
    hyper_params = _parse_json_arg(args.hyper_params)
    hyper_params.setdefault("tensor_parallel_size", args.tensor_parallel_size)
    hyper_params.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
    hyper_params.setdefault("decode_cuda_graph", True)
    hyper_params.setdefault("max_num_batched_tokens", args.max_num_batched_tokens)
    hyper_params.setdefault("engine_prefill_chunk_size", args.max_num_batched_tokens)
    if args.sparse_method == "snapkv":
        hyper_params.setdefault("snapkv_window_size", 64)
        hyper_params.setdefault("sink_keep_tokens", 64)
        hyper_params.setdefault("decode_keep_tokens", 2048)
        hyper_params.setdefault("recent_keep_tokens", 64)
        hyper_params.setdefault("pool_kernel_size", 7)
    elif args.sparse_method == "h2o":
        hyper_params.setdefault("h2o_decode_budget", 4096)
        hyper_params.setdefault("h2o_decode_eviction_interval", 128)
        hyper_params.setdefault("h2o_prefill_budget", 8192)
        hyper_params.setdefault("h2o_recent_ratio", 0.5)
        hyper_params.setdefault("h2o_prefill_score_window", 128)

    if args.sparse_method in {"snapkv", "h2o"}:
        configured_mode = hyper_params.get("sparse_prefill_score_mode")
        if args.sparse_prefill_score_mode is None and configured_mode is None:
            raise ValueError(
                f"{args.sparse_method} efficiency probes require an explicit "
                "--sparse-prefill-score-mode "
                "(probability or logits)."
            )
        if (
            args.sparse_prefill_score_mode is not None
            and configured_mode is not None
            and str(configured_mode) != args.sparse_prefill_score_mode
        ):
            raise ValueError(
                "Conflicting sparse prefill score modes in --hyper-params and "
                "--sparse-prefill-score-mode."
            )
        hyper_params["sparse_prefill_score_mode"] = (
            args.sparse_prefill_score_mode
            if args.sparse_prefill_score_mode is not None
            else str(configured_mode)
        )

    sparse_budget = (
        int(hyper_params.get("sink_keep_tokens", 64))
        + int(hyper_params.get("decode_keep_tokens", 2048))
        + int(hyper_params.get("recent_keep_tokens", 64))
        if args.sparse_method == "snapkv"
        else None
    )
    protocol = {
        "score_mode": hyper_params.get("sparse_prefill_score_mode"),
        "score_window": hyper_params.get("snapkv_window_size"),
        "sparse_budget": sparse_budget,
        "max_num_batched_tokens": int(hyper_params["max_num_batched_tokens"]),
    }
    protocol_label = f"sparsevllm-{args.sparse_method}"
    if args.sparse_method == "snapkv":
        protocol_label += (
            f"-{protocol['score_mode']}-budget{protocol['sparse_budget']}"
            f"-window{protocol['score_window']}"
        )
    elif args.sparse_method == "h2o":
        protocol.update(
            {
                "score_window": int(hyper_params["h2o_prefill_score_window"]),
                "decode_budget": int(hyper_params["h2o_decode_budget"]),
                "decode_eviction_interval": int(
                    hyper_params["h2o_decode_eviction_interval"]
                ),
                "prefill_budget": int(hyper_params["h2o_prefill_budget"]),
                "recent_ratio": float(hyper_params["h2o_recent_ratio"]),
            }
        )
        protocol_label += (
            f"-{protocol['score_mode']}-decode{protocol['decode_budget']}"
            f"-prefill{protocol['prefill_budget']}"
            f"-window{protocol['score_window']}"
        )
    return hyper_params, sparse_budget, protocol, protocol_label


def _format_markdown_report(
    summary_rows: list[dict[str, Any]],
    model_name: str,
    tp_size: int,
) -> str:
    lines = [
        "# Standardized LLM Efficiency Benchmark Report",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Tensor parallel size**: `{tp_size}`",
        "- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates",
        f"- **Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    def _number(value: Any, precision: int) -> str:
        return "n/a" if value is None else f"{float(value):.{precision}f}"

    for r in summary_rows:
        fallback_label = f"{r['engine']}-{r.get('sparse_method', 'vanilla')}"
        sys_label = f"`{r.get('protocol_label', fallback_label)}`"
        prompt_range = f"{r['prompt_len_min']}-{r['prompt_len_max']}"
        output_range = f"{r['output_len_min']}-{r['output_len_max']}"
        lines.append(
            f"| {sys_label} | {r['scenario']} | {prompt_range} | {output_range} "
            f"| {r['concurrency']} | {_number(r.get('request_throughput_rps'), 2)} "
            f"| {_number(r.get('output_token_throughput_tps'), 2)} "
            f"| {_number(r.get('output_tps_pct_of_observed_sweep_peak'), 1)}% "
            f"| {_number(r.get('output_tps_scaling_efficiency_pct_vs_min_concurrency'), 1)}% "
            f"| {_number(r.get('ttft_ms_p50'), 2)}/{_number(r.get('ttft_ms_p99'), 2)} "
            f"| {_number(r.get('gpu_compute_activity_pct_mean'), 1)}% "
            f"| {_number(r.get('gpu_memory_io_activity_pct_mean'), 1)}% "
            f"| {_number(r.get('peak_vram_gb_max'), 2)} | {r.get('status', 'success')} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_sparsevllm_probe(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from sparsevllm import LLM, SamplingParams
    from sparsevllm.utils.profiler import profiler

    if args.scenario == "churn":
        return run_sparsevllm_churn(args, model_specs)

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples_file = output_dir / "raw_samples.jsonl"

    hyper_params, sparse_budget, protocol, protocol_label = _resolve_sparse_probe_protocol(args)
    sparse_kwargs: dict[str, Any] = {"sparse_method": args.sparse_method}

    max_len_needed = max(args.prompt_lens) + max(args.output_lens) + 128
    engine_kwargs = {
        **hyper_params,
        "max_model_len": max_len_needed,
        "enable_prefix_caching": False,
        **sparse_kwargs,
    }

    print(f"[Sparse-vLLM Probe] Initializing LLM with method={args.sparse_method}, max_model_len={max_len_needed}...")
    llm = LLM(args.model_path, **engine_kwargs)

    for p_len in args.prompt_lens:
        for o_len in args.output_lens:
            for bs in args.batch_sizes:
                print(f"\n---> Benchmarking Sweep: PromptLen={p_len}, OutputLen={o_len}, BatchSize={bs} <---")

                # 1. Warmup
                for w_idx in range(args.num_warmups):
                    print(f"  [Warmup {w_idx + 1}/{args.num_warmups}]...")
                    warmup_trace = _trace_for_iteration(
                        args,
                        model_specs,
                        scenario="fixed_batch",
                        phase="warmup",
                        prompt_len=p_len,
                        output_len=o_len,
                        concurrency=bs,
                        iteration=w_idx,
                        request_count=bs,
                        vary_output_lengths=False,
                    )
                    for request in warmup_trace:
                        llm.add_request(
                            request.prompt_token_ids,
                            SamplingParams(
                                temperature=0.0,
                                top_p=1.0,
                                top_k=1,
                                ignore_eos=True,
                                max_tokens=request.output_len,
                            ),
                        )
                    while not llm.is_finished():
                        llm.step()

                torch.cuda.synchronize()

                # 2. Benchmark Iterations
                iter_records = []
                case_name = f"fixed-p{p_len}-o{o_len}-c{bs}"
                monitor = _case_monitor(args, case_name)
                try:
                    for it in range(args.num_iters):
                        profiler.reset()
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="fixed_batch",
                            phase="measure",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            request_count=bs,
                            vary_output_lengths=False,
                        )
                        decode_times = []
                        ttft_ms = None
                        t_start = time.perf_counter()

                        seq_to_request: dict[int, Any] = {}
                        for request in trace:
                            seq_id = int(
                                llm.add_request(
                                    request.prompt_token_ids,
                                    SamplingParams(
                                        temperature=0.0,
                                        top_p=1.0,
                                        top_k=1,
                                        ignore_eos=True,
                                        max_tokens=request.output_len,
                                    ),
                                )
                            )
                            if seq_id in seq_to_request:
                                raise RuntimeError(
                                    "Sparse-vLLM returned a duplicate request sequence ID: "
                                    f"seq_id={seq_id}."
                                )
                            seq_to_request[seq_id] = request
                        request_seq_ids = set(seq_to_request)
                        first_token_seq_ids: set[int] = set()
                        finished_by_seq: dict[int, int] = {}

                        while not llm.is_finished():
                            step_s = time.perf_counter()
                            finished_outputs, num_tokens = llm.step()
                            torch.cuda.synchronize()
                            step_dt = time.perf_counter() - step_s
                            now = time.perf_counter()

                            if num_tokens < 0:
                                decode_times.append(step_dt)
                            if ttft_ms is None and _record_batch_first_tokens(
                                request_seq_ids,
                                first_token_seq_ids,
                                getattr(llm, "last_step_token_outputs", []),
                                finished_outputs,
                            ):
                                ttft_ms = (now - t_start) * 1000.0
                            for seq_id, token_ids, _token_logprobs, _top_logprobs in finished_outputs:
                                finished_by_seq[int(seq_id)] = len(token_ids)

                        elapsed_s = time.perf_counter() - t_start
                        if ttft_ms is None:
                            missing = sorted(request_seq_ids - first_token_seq_ids)
                            raise RuntimeError(
                                "Sparse-vLLM finished without publishing a first token for every "
                                f"request; missing_seq_ids={missing}."
                            )
                        if set(finished_by_seq) != request_seq_ids:
                            raise RuntimeError(
                                "Sparse-vLLM fixed-batch completion coverage mismatch: "
                                f"expected={sorted(request_seq_ids)}, actual={sorted(finished_by_seq)}."
                            )
                        for seq_id, generated in finished_by_seq.items():
                            expected = int(seq_to_request[seq_id].output_len)
                            if generated != expected:
                                raise RuntimeError(
                                    f"Sparse-vLLM generated {generated} tokens for seq_id={seq_id}, "
                                    f"expected {expected}."
                                )
                        tpot_ms = (
                            statistics.fmean(decode_times) * 1000.0
                            if decode_times
                            else None
                        )
                        total_input = sum(request.prompt_len for request in trace)
                        total_output = sum(request.output_len for request in trace)
                        profiler_snap = profiler.snapshot()

                        rec = {
                            "engine": "sparsevllm",
                            "sparse_method": args.sparse_method,
                            "scenario": "fixed_batch",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "concurrency": bs,
                            "iteration": it,
                            "status": "success",
                            "elapsed_s": elapsed_s,
                            "ttft_ms": round(ttft_ms, 2),
                            "tpot_ms": None if tpot_ms is None else round(tpot_ms, 2),
                            "request_throughput_rps": bs / elapsed_s,
                            "input_token_throughput_tps": total_input / elapsed_s,
                            "output_token_throughput_tps": total_output / elapsed_s,
                            "total_token_throughput_tps": (total_input + total_output) / elapsed_s,
                            "profiler_breakdown": profiler_snap,
                            "profiler_status": "success" if profiler_snap else "skipped_by_policy",
                            "protocol": protocol,
                            "protocol_label": protocol_label,
                            "decode_metric_status": "success" if tpot_ms is not None else "skipped_by_policy",
                            "trace": trace_metadata(trace),
                        }
                        iter_records.append(rec)
                        with open(raw_samples_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        _append_request_samples(
                            output_dir,
                            engine="sparsevllm",
                            sparse_method=args.sparse_method,
                            scenario="fixed_batch",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            requests=rec["trace"]["requests"],
                        )

                        print(
                            f"  Iter {it + 1}/{args.num_iters}: TTFT={ttft_ms:.1f}ms | "
                            f"TPOT={tpot_ms if tpot_ms is not None else 'n/a'}ms | "
                            f"Output={total_output / elapsed_s:.1f} tok/s"
                        )
                finally:
                    hardware_summary = monitor.stop()
                hardware = _actual_hardware_metrics(hardware_summary)

                # Aggregated summary
                ttft_vals = [r["ttft_ms"] for r in iter_records]
                tpot_vals = [r["tpot_ms"] for r in iter_records if r["tpot_ms"] is not None]
                request_rates = [r["request_throughput_rps"] for r in iter_records]
                input_rates = [r["input_token_throughput_tps"] for r in iter_records]
                output_rates = [r["output_token_throughput_tps"] for r in iter_records]
                total_rates = [r["total_token_throughput_tps"] for r in iter_records]
                prompt_lengths = [
                    length for record in iter_records for length in record["trace"]["prompt_lengths"]
                ]
                output_lengths = [
                    length for record in iter_records for length in record["trace"]["output_lengths"]
                ]

                summary_row = {
                    "engine": "sparsevllm",
                    "sparse_method": args.sparse_method,
                    "scenario": "fixed_batch",
                    "prompt_len": p_len,
                    "output_len": o_len,
                    "prompt_len_min": min(prompt_lengths),
                    "prompt_len_max": max(prompt_lengths),
                    "output_len_min": min(output_lengths),
                    "output_len_max": max(output_lengths),
                    "concurrency": bs,
                    "request_count": bs,
                    "ttft_ms_mean": round(statistics.fmean(ttft_vals), 2),
                    "ttft_ms_p50": round(_percentile(ttft_vals, 0.50), 2),
                    "ttft_ms_p99": round(_percentile(ttft_vals, 0.99), 2),
                    "tpot_ms_mean": round(statistics.fmean(tpot_vals), 2) if tpot_vals else None,
                    "request_throughput_rps": statistics.fmean(request_rates),
                    "input_token_throughput_tps": statistics.fmean(input_rates),
                    "output_token_throughput_tps": statistics.fmean(output_rates),
                    "total_token_throughput_tps": statistics.fmean(total_rates),
                    "sequence_replacements": 0,
                    "status": "success",
                    "protocol": protocol,
                    "protocol_label": protocol_label,
                    "decode_metric_status": "success" if tpot_vals else "skipped_by_policy",
                    "actual_hardware_metrics": hardware,
                    **{key: value for key, value in hardware.items() if key != "per_gpu"},
                }
                results.append(summary_row)

    if hasattr(llm, "exit"):
        llm.exit()
    return results


def run_sparsevllm_churn(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from sparsevllm import LLM, SamplingParams
    from sparsevllm.utils.profiler import profiler

    results: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    raw_samples_file = output_dir / "raw_samples.jsonl"
    base_hyper_params, _sparse_budget, protocol, protocol_label = (
        _resolve_sparse_probe_protocol(args)
    )
    max_len_needed = max(args.prompt_lens) + max(args.output_lens) + 128

    for concurrency in args.batch_sizes:
        engine_kwargs = {
            **base_hyper_params,
            "max_model_len": max_len_needed,
            "sparse_method": args.sparse_method,
            "enable_prefix_caching": False,
            "max_num_seqs_in_batch": concurrency,
            "max_decoding_seqs": concurrency,
            "max_num_seqs_in_gpu": concurrency,
        }
        print(
            "[Sparse-vLLM Churn] Initializing "
            f"method={args.sparse_method}, max_concurrency={concurrency}..."
        )
        llm = LLM(args.model_path, **engine_kwargs)
        try:
            request_count = concurrency * args.churn_request_multiplier
            for p_len in args.prompt_lens:
                for o_len in args.output_lens:
                    for warmup_index in range(args.num_warmups):
                        warmup_count = min(request_count, max(concurrency, 2 * concurrency))
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="oversubscribed_churn",
                            phase="warmup",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=concurrency,
                            iteration=warmup_index,
                            request_count=warmup_count,
                            vary_output_lengths=True,
                        )
                        for request in trace:
                            llm.add_request(
                                request.prompt_token_ids,
                                SamplingParams(
                                    temperature=0.0,
                                    top_p=1.0,
                                    top_k=1,
                                    ignore_eos=True,
                                    max_tokens=request.output_len,
                                ),
                            )
                        while not llm.is_finished():
                            llm.step()
                    torch.cuda.synchronize()

                    case_name = f"churn-p{p_len}-o{o_len}-c{concurrency}"
                    monitor = _case_monitor(args, case_name)
                    iter_records: list[dict[str, Any]] = []
                    try:
                        for iteration in range(args.num_iters):
                            profiler.reset()
                            trace = _trace_for_iteration(
                                args,
                                model_specs,
                                scenario="oversubscribed_churn",
                                phase="measure",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                request_count=request_count,
                                vary_output_lengths=True,
                            )
                            seq_to_request: dict[int, Any] = {}
                            arrival_times: dict[int, float] = {}
                            first_token_times: dict[int, float] = {}
                            finished_times: dict[int, float] = {}
                            generated_counts: dict[int, int] = {}
                            started = time.perf_counter()
                            for request in trace:
                                seq_id = int(
                                    llm.add_request(
                                        request.prompt_token_ids,
                                        SamplingParams(
                                            temperature=0.0,
                                            top_p=1.0,
                                            top_k=1,
                                            ignore_eos=True,
                                            max_tokens=request.output_len,
                                        ),
                                    )
                                )
                                if seq_id in seq_to_request:
                                    raise RuntimeError(
                                        f"Duplicate Sparse-vLLM churn sequence ID: {seq_id}."
                                    )
                                seq_to_request[seq_id] = request
                                arrival_times[seq_id] = time.perf_counter()

                            step_count = 0
                            while not llm.is_finished():
                                finished_outputs, _num_tokens = llm.step()
                                torch.cuda.synchronize()
                                now = time.perf_counter()
                                step_count += 1
                                for seq_id, token_ids in getattr(
                                    llm, "last_step_token_outputs", []
                                ):
                                    seq_id = int(seq_id)
                                    if token_ids and seq_id in seq_to_request:
                                        first_token_times.setdefault(seq_id, now)
                                for (
                                    seq_id,
                                    token_ids,
                                    _token_logprobs,
                                    _top_logprobs,
                                ) in finished_outputs:
                                    seq_id = int(seq_id)
                                    if token_ids and seq_id in seq_to_request:
                                        first_token_times.setdefault(seq_id, now)
                                    finished_times[seq_id] = now
                                    generated_counts[seq_id] = len(token_ids)
                            elapsed_s = time.perf_counter() - started

                            expected_seq_ids = set(seq_to_request)
                            for name, observed in (
                                ("first-token", set(first_token_times)),
                                ("completion", set(finished_times)),
                                ("generated-count", set(generated_counts)),
                            ):
                                if observed != expected_seq_ids:
                                    raise RuntimeError(
                                        f"Sparse-vLLM churn {name} coverage mismatch: "
                                        f"missing={sorted(expected_seq_ids - observed)}, "
                                        f"unexpected={sorted(observed - expected_seq_ids)}."
                                    )

                            request_results = []
                            for seq_id, request in seq_to_request.items():
                                generated = generated_counts[seq_id]
                                if generated != request.output_len:
                                    raise RuntimeError(
                                        f"Sparse-vLLM churn seq_id={seq_id} generated "
                                        f"{generated} tokens, expected {request.output_len}."
                                    )
                                first = first_token_times[seq_id]
                                finished = finished_times[seq_id]
                                request_results.append(
                                    {
                                        **request.metadata(),
                                        "seq_id": seq_id,
                                        "ttft_ms": (first - arrival_times[seq_id]) * 1000.0,
                                        "latency_ms": (finished - arrival_times[seq_id]) * 1000.0,
                                        "tpot_ms": (
                                            (finished - first) * 1000.0 / (generated - 1)
                                            if generated > 1
                                            else None
                                        ),
                                        "generated_tokens": generated,
                                    }
                                )

                            total_input = sum(request.prompt_len for request in trace)
                            total_output = sum(request.output_len for request in trace)
                            profiler_snap = profiler.snapshot()
                            record = {
                                "engine": "sparsevllm",
                                "sparse_method": args.sparse_method,
                                "scenario": "oversubscribed_churn",
                                "prompt_len": p_len,
                                "output_len": o_len,
                                "concurrency": concurrency,
                                "request_count": request_count,
                                "queued_request_count": request_count - concurrency,
                                "iteration": iteration,
                                "status": "success",
                                "elapsed_s": elapsed_s,
                                "step_count": step_count,
                                "request_throughput_rps": request_count / elapsed_s,
                                "input_token_throughput_tps": total_input / elapsed_s,
                                "output_token_throughput_tps": total_output / elapsed_s,
                                "total_token_throughput_tps": (total_input + total_output) / elapsed_s,
                                "profiler_breakdown": profiler_snap,
                                "profiler_status": "success" if profiler_snap else "skipped_by_policy",
                                "protocol": protocol,
                                "protocol_label": protocol_label,
                                "trace": trace_metadata(trace),
                                "request_results": request_results,
                            }
                            iter_records.append(record)
                            with raw_samples_file.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            _append_request_samples(
                                output_dir,
                                engine="sparsevllm",
                                sparse_method=args.sparse_method,
                                scenario="oversubscribed_churn",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                requests=request_results,
                            )
                    finally:
                        hardware_summary = monitor.stop()
                    hardware = _actual_hardware_metrics(hardware_summary)

                    request_results = [
                        request
                        for record in iter_records
                        for request in record["request_results"]
                    ]
                    ttfts = [float(request["ttft_ms"]) for request in request_results]
                    latencies = [float(request["latency_ms"]) for request in request_results]
                    tpots = [
                        float(request["tpot_ms"])
                        for request in request_results
                        if request["tpot_ms"] is not None
                    ]
                    prompt_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["prompt_lengths"]
                    ]
                    output_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["output_lengths"]
                    ]
                    results.append(
                        {
                            "engine": "sparsevllm",
                            "sparse_method": args.sparse_method,
                            "scenario": "oversubscribed_churn",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "prompt_len_min": min(prompt_lengths),
                            "prompt_len_max": max(prompt_lengths),
                            "output_len_min": min(output_lengths),
                            "output_len_max": max(output_lengths),
                            "concurrency": concurrency,
                            "request_count": request_count,
                            "sequence_replacements": request_count - concurrency,
                            "request_throughput_rps": statistics.fmean(
                                record["request_throughput_rps"] for record in iter_records
                            ),
                            "input_token_throughput_tps": statistics.fmean(
                                record["input_token_throughput_tps"] for record in iter_records
                            ),
                            "output_token_throughput_tps": statistics.fmean(
                                record["output_token_throughput_tps"] for record in iter_records
                            ),
                            "total_token_throughput_tps": statistics.fmean(
                                record["total_token_throughput_tps"] for record in iter_records
                            ),
                            "ttft_ms_mean": statistics.fmean(ttfts),
                            "ttft_ms_p50": _percentile(ttfts, 0.50),
                            "ttft_ms_p99": _percentile(ttfts, 0.99),
                            "latency_ms_p50": _percentile(latencies, 0.50),
                            "latency_ms_p99": _percentile(latencies, 0.99),
                            "tpot_ms_mean": statistics.fmean(tpots) if tpots else None,
                            "status": "success",
                            "protocol": protocol,
                            "protocol_label": protocol_label,
                            "actual_hardware_metrics": hardware,
                            **{key: value for key, value in hardware.items() if key != "per_gpu"},
                        }
                    )
        finally:
            if hasattr(llm, "exit"):
                llm.exit()
    del llm
    return results


def run_vllm_probe(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from vllm import LLM, SamplingParams

    if args.scenario == "churn":
        return run_vllm_churn(args, model_specs)

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
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(args.batch_sizes),
        enable_prefix_caching=False,
        disable_log_stats=False,
        trust_remote_code=True,
        seed=args.seed,
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
                    warmup_trace = _trace_for_iteration(
                        args,
                        model_specs,
                        scenario="fixed_batch",
                        phase="warmup",
                        prompt_len=p_len,
                        output_len=o_len,
                        concurrency=bs,
                        iteration=w_idx,
                        request_count=bs,
                        vary_output_lengths=False,
                    )
                    warmup_prompts = [
                        {"prompt_token_ids": request.prompt_token_ids}
                        for request in warmup_trace
                    ]
                    llm.generate(warmup_prompts, sampling_params=sampling_params, use_tqdm=False)

                torch.cuda.synchronize()

                iter_records = []
                case_name = f"fixed-p{p_len}-o{o_len}-c{bs}"
                monitor = _case_monitor(args, case_name)
                try:
                    for it in range(args.num_iters):
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="fixed_batch",
                            phase="measure",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            request_count=bs,
                            vary_output_lengths=False,
                        )
                        prompt_tokens = [
                            {"prompt_token_ids": request.prompt_token_ids}
                            for request in trace
                        ]
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        outputs = llm.generate(
                            prompt_tokens,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                        torch.cuda.synchronize()
                        elapsed_s = time.perf_counter() - started
                        if len(outputs) != bs:
                            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for batch_size={bs}.")

                        ttft_ms, tpot_ms = _vllm_phase_metrics(outputs, o_len)
                        total_input = sum(request.prompt_len for request in trace)
                        total_output = sum(request.output_len for request in trace)
                        rec = {
                            "engine": "vllm",
                            "sparse_method": "vanilla",
                            "scenario": "fixed_batch",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "concurrency": bs,
                            "iteration": it,
                            "status": "success",
                            "elapsed_s": elapsed_s,
                            "ttft_ms": round(ttft_ms, 2),
                            "tpot_ms": None if tpot_ms is None else round(tpot_ms, 2),
                            "request_throughput_rps": bs / elapsed_s,
                            "input_token_throughput_tps": total_input / elapsed_s,
                            "output_token_throughput_tps": total_output / elapsed_s,
                            "total_token_throughput_tps": (total_input + total_output) / elapsed_s,
                            "decode_metric_status": "success" if tpot_ms is not None else "skipped_by_policy",
                            "protocol_label": "vllm-vanilla",
                            "trace": trace_metadata(trace),
                        }
                        iter_records.append(rec)
                        with open(raw_samples_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        _append_request_samples(
                            output_dir,
                            engine="vllm",
                            sparse_method="vanilla",
                            scenario="fixed_batch",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=bs,
                            iteration=it,
                            requests=rec["trace"]["requests"],
                        )

                        print(
                            f"  Iter {it + 1}/{args.num_iters}: TTFT={ttft_ms:.1f}ms | "
                            f"TPOT={tpot_ms if tpot_ms is not None else 'n/a'}ms | "
                            f"Output={total_output / elapsed_s:.1f} tok/s"
                        )
                finally:
                    hardware_summary = monitor.stop()
                hardware = _actual_hardware_metrics(hardware_summary)

                ttft_vals = [r["ttft_ms"] for r in iter_records]
                tpot_vals = [r["tpot_ms"] for r in iter_records if r["tpot_ms"] is not None]
                request_rates = [r["request_throughput_rps"] for r in iter_records]
                input_rates = [r["input_token_throughput_tps"] for r in iter_records]
                output_rates = [r["output_token_throughput_tps"] for r in iter_records]
                total_rates = [r["total_token_throughput_tps"] for r in iter_records]
                prompt_lengths = [
                    length for record in iter_records for length in record["trace"]["prompt_lengths"]
                ]
                output_lengths = [
                    length for record in iter_records for length in record["trace"]["output_lengths"]
                ]

                summary_row = {
                    "engine": "vllm",
                    "sparse_method": "vanilla",
                    "scenario": "fixed_batch",
                    "prompt_len": p_len,
                    "output_len": o_len,
                    "prompt_len_min": min(prompt_lengths),
                    "prompt_len_max": max(prompt_lengths),
                    "output_len_min": min(output_lengths),
                    "output_len_max": max(output_lengths),
                    "concurrency": bs,
                    "request_count": bs,
                    "ttft_ms_mean": round(statistics.fmean(ttft_vals), 2),
                    "ttft_ms_p50": round(_percentile(ttft_vals, 0.50), 2),
                    "ttft_ms_p99": round(_percentile(ttft_vals, 0.99), 2),
                    "tpot_ms_mean": round(statistics.fmean(tpot_vals), 2) if tpot_vals else None,
                    "request_throughput_rps": statistics.fmean(request_rates),
                    "input_token_throughput_tps": statistics.fmean(input_rates),
                    "output_token_throughput_tps": statistics.fmean(output_rates),
                    "total_token_throughput_tps": statistics.fmean(total_rates),
                    "sequence_replacements": 0,
                    "status": "success",
                    "decode_metric_status": "success" if tpot_vals else "skipped_by_policy",
                    "protocol_label": "vllm-vanilla",
                    "actual_hardware_metrics": hardware,
                    **{key: value for key, value in hardware.items() if key != "per_gpu"},
                }
                results.append(summary_row)

    del llm
    return results


def run_vllm_churn(
    args: argparse.Namespace,
    model_specs: ModelArchitectureSpecs,
) -> list[dict[str, Any]]:
    import torch
    from vllm import LLM, SamplingParams

    results: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    raw_samples_file = output_dir / "raw_samples.jsonl"
    max_len_needed = max(args.prompt_lens) + max(args.output_lens) + 128

    for concurrency in args.batch_sizes:
        print(f"[vLLM Churn] Initializing max_concurrency={concurrency}...")
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_len_needed,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=concurrency,
            enable_prefix_caching=False,
            disable_log_stats=False,
            trust_remote_code=True,
            seed=args.seed,
        )
        try:
            request_count = concurrency * args.churn_request_multiplier
            for p_len in args.prompt_lens:
                for o_len in args.output_lens:
                    for warmup_index in range(args.num_warmups):
                        warmup_count = min(request_count, 2 * concurrency)
                        trace = _trace_for_iteration(
                            args,
                            model_specs,
                            scenario="oversubscribed_churn",
                            phase="warmup",
                            prompt_len=p_len,
                            output_len=o_len,
                            concurrency=concurrency,
                            iteration=warmup_index,
                            request_count=warmup_count,
                            vary_output_lengths=True,
                        )
                        prompts = [
                            {"prompt_token_ids": request.prompt_token_ids}
                            for request in trace
                        ]
                        sampling = [
                            SamplingParams(
                                temperature=0.0,
                                top_p=1.0,
                                top_k=1,
                                ignore_eos=True,
                                max_tokens=request.output_len,
                                detokenize=False,
                            )
                            for request in trace
                        ]
                        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                        if len(outputs) != warmup_count:
                            raise RuntimeError(
                                f"vLLM churn warmup returned {len(outputs)} outputs, "
                                f"expected {warmup_count}."
                            )
                    torch.cuda.synchronize()

                    case_name = f"churn-p{p_len}-o{o_len}-c{concurrency}"
                    monitor = _case_monitor(args, case_name)
                    iter_records: list[dict[str, Any]] = []
                    try:
                        for iteration in range(args.num_iters):
                            trace = _trace_for_iteration(
                                args,
                                model_specs,
                                scenario="oversubscribed_churn",
                                phase="measure",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                request_count=request_count,
                                vary_output_lengths=True,
                            )
                            prompts = [
                                {"prompt_token_ids": request.prompt_token_ids}
                                for request in trace
                            ]
                            sampling = [
                                SamplingParams(
                                    temperature=0.0,
                                    top_p=1.0,
                                    top_k=1,
                                    ignore_eos=True,
                                    max_tokens=request.output_len,
                                    detokenize=False,
                                )
                                for request in trace
                            ]
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            outputs = llm.generate(
                                prompts,
                                sampling_params=sampling,
                                use_tqdm=False,
                            )
                            torch.cuda.synchronize()
                            elapsed_s = time.perf_counter() - started
                            if len(outputs) != request_count:
                                raise RuntimeError(
                                    f"vLLM churn returned {len(outputs)} outputs, "
                                    f"expected {request_count}."
                                )

                            request_results = []
                            for request, output in zip(trace, outputs):
                                metrics = getattr(output, "metrics", None)
                                if metrics is None:
                                    raise RuntimeError(
                                        "vLLM churn requires timing metrics for every request."
                                    )
                                if len(output.outputs) != 1:
                                    raise RuntimeError(
                                        f"vLLM churn request {request.request_index} returned "
                                        f"{len(output.outputs)} candidates, expected one."
                                    )
                                generated = len(output.outputs[0].token_ids)
                                if generated != request.output_len:
                                    raise RuntimeError(
                                        f"vLLM churn request {request.request_index} generated "
                                        f"{generated} tokens, expected {request.output_len}."
                                    )
                                ttft_s, decode_s, timing_source = (
                                    _vllm_request_phase_seconds(metrics)
                                )
                                request_results.append(
                                    {
                                        **request.metadata(),
                                        "request_id": str(output.request_id),
                                        "ttft_ms": ttft_s * 1000.0,
                                        "latency_ms": (ttft_s + decode_s) * 1000.0,
                                        "tpot_ms": (
                                            decode_s * 1000.0 / (generated - 1)
                                            if generated > 1
                                            else None
                                        ),
                                        "timing_source": timing_source,
                                        "generated_tokens": generated,
                                    }
                                )

                            total_input = sum(request.prompt_len for request in trace)
                            total_output = sum(request.output_len for request in trace)
                            record = {
                                "engine": "vllm",
                                "sparse_method": "vanilla",
                                "scenario": "oversubscribed_churn",
                                "prompt_len": p_len,
                                "output_len": o_len,
                                "concurrency": concurrency,
                                "request_count": request_count,
                                "queued_request_count": request_count - concurrency,
                                "iteration": iteration,
                                "status": "success",
                                "elapsed_s": elapsed_s,
                                "request_throughput_rps": request_count / elapsed_s,
                                "input_token_throughput_tps": total_input / elapsed_s,
                                "output_token_throughput_tps": total_output / elapsed_s,
                                "total_token_throughput_tps": (total_input + total_output) / elapsed_s,
                                "protocol_label": "vllm-vanilla",
                                "trace": trace_metadata(trace),
                                "request_results": request_results,
                            }
                            iter_records.append(record)
                            with raw_samples_file.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            _append_request_samples(
                                output_dir,
                                engine="vllm",
                                sparse_method="vanilla",
                                scenario="oversubscribed_churn",
                                prompt_len=p_len,
                                output_len=o_len,
                                concurrency=concurrency,
                                iteration=iteration,
                                requests=request_results,
                            )
                    finally:
                        hardware_summary = monitor.stop()
                    hardware = _actual_hardware_metrics(hardware_summary)

                    request_results = [
                        request
                        for record in iter_records
                        for request in record["request_results"]
                    ]
                    ttfts = [float(request["ttft_ms"]) for request in request_results]
                    latencies = [float(request["latency_ms"]) for request in request_results]
                    tpots = [
                        float(request["tpot_ms"])
                        for request in request_results
                        if request["tpot_ms"] is not None
                    ]
                    prompt_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["prompt_lengths"]
                    ]
                    output_lengths = [
                        length
                        for record in iter_records
                        for length in record["trace"]["output_lengths"]
                    ]
                    results.append(
                        {
                            "engine": "vllm",
                            "sparse_method": "vanilla",
                            "scenario": "oversubscribed_churn",
                            "prompt_len": p_len,
                            "output_len": o_len,
                            "prompt_len_min": min(prompt_lengths),
                            "prompt_len_max": max(prompt_lengths),
                            "output_len_min": min(output_lengths),
                            "output_len_max": max(output_lengths),
                            "concurrency": concurrency,
                            "request_count": request_count,
                            "sequence_replacements": request_count - concurrency,
                            "request_throughput_rps": statistics.fmean(
                                record["request_throughput_rps"] for record in iter_records
                            ),
                            "input_token_throughput_tps": statistics.fmean(
                                record["input_token_throughput_tps"] for record in iter_records
                            ),
                            "output_token_throughput_tps": statistics.fmean(
                                record["output_token_throughput_tps"] for record in iter_records
                            ),
                            "total_token_throughput_tps": statistics.fmean(
                                record["total_token_throughput_tps"] for record in iter_records
                            ),
                            "ttft_ms_mean": statistics.fmean(ttfts),
                            "ttft_ms_p50": _percentile(ttfts, 0.50),
                            "ttft_ms_p99": _percentile(ttfts, 0.99),
                            "latency_ms_p50": _percentile(latencies, 0.50),
                            "latency_ms_p99": _percentile(latencies, 0.99),
                            "tpot_ms_mean": statistics.fmean(tpots) if tpots else None,
                            "status": "success",
                            "protocol_label": "vllm-vanilla",
                            "actual_hardware_metrics": hardware,
                            **{key: value for key, value in hardware.items() if key != "per_gpu"},
                        }
                    )
        finally:
            del llm
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Standardized Synthetic Length Sweep & Efficiency Probe.")
    parser.add_argument("--engine", type=str, choices=["sparsevllm", "vllm"], default="sparsevllm")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--sparse-method", type=str, default="vanilla", help="Sparse method name (for sparsevllm)")
    parser.add_argument("--prompt-lens", type=_parse_ints, default=[8192, 16384, 32768])
    parser.add_argument("--output-lens", type=_parse_ints, default=[128])
    parser.add_argument("--batch-sizes", type=_parse_ints, default=[1])
    parser.add_argument(
        "--scenario",
        choices=["fixed", "churn", "all"],
        default="all",
        help="fixed measures one variable-length batch; churn oversubscribes the scheduler.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-length-jitter",
        type=float,
        default=0.10,
        help="Fraction below each requested prompt length used for variable-length batches.",
    )
    parser.add_argument(
        "--output-length-jitter",
        type=float,
        default=0.25,
        help="Fraction below each requested output length used by churn traces.",
    )
    parser.add_argument(
        "--churn-request-multiplier",
        type=int,
        default=4,
        help="Churn request count is max concurrency times this multiplier.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Matched scheduler token budget used by Sparse-vLLM and vLLM.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hyper-params", type=str, default="{}")
    parser.add_argument(
        "--sparse-prefill-score-mode",
        choices=["probability", "logits"],
        default=None,
        help="Required for SnapKV probes so probability and logits results cannot be conflated.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Deprecated compatibility label; theoretical hardware profiles are no longer used.",
    )
    parser.add_argument(
        "--monitor-gpus",
        default=None,
        help="Physical GPU IDs for directly sampled metrics; defaults to CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--hardware-sampling-interval-ms",
        type=int,
        default=100,
        help="nvidia-smi activity sampling interval for each measured case.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("prompt_lens", "output_lens", "batch_sizes"):
        values = getattr(args, name)
        if not values or any(int(value) <= 0 for value in values):
            raise ValueError(f"--{name.replace('_', '-')} requires positive integers, got {values}.")
    if (
        args.tensor_parallel_size <= 0
        or args.max_num_batched_tokens <= 0
        or args.num_warmups < 0
        or args.num_iters <= 0
    ):
        raise ValueError(
            "tensor parallel size, max batched tokens, and iterations must be positive, "
            "and warmups must be non-negative."
        )
    if not 0.0 <= args.prompt_length_jitter < 1.0:
        raise ValueError("--prompt-length-jitter must be in [0, 1).")
    if not 0.0 <= args.output_length_jitter < 1.0:
        raise ValueError("--output-length-jitter must be in [0, 1).")
    if args.churn_request_multiplier < 2:
        raise ValueError("--churn-request-multiplier must be at least 2.")
    if args.hardware_sampling_interval_ms < 50:
        raise ValueError("--hardware-sampling-interval-ms must be at least 50.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "run_manifest.json",
        "raw_samples.jsonl",
        "request_samples.jsonl",
        "summary.json",
        "comparison_report.md",
        "run_status.json",
    )
    collisions = [name for name in artifact_names if (output_dir / name).exists()]
    if collisions:
        raise FileExistsError(
            f"Refusing to mix benchmark runs in {output_dir}: existing artifacts={collisions}. "
            "Use a unique --output-dir."
        )
    (output_dir / "raw_samples.jsonl").write_text("", encoding="utf-8")
    (output_dir / "request_samples.jsonl").write_text("", encoding="utf-8")

    try:
        model_specs = ModelArchitectureSpecs.from_model_path_or_name(args.model_path)
        monitor_gpus = _monitor_gpu_ids(args.monitor_gpus)
    except Exception as exc:
        failure = {
            "status": "metric_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "manifest_version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "command": [sys.executable, *sys.argv],
                    "git": _git_metadata(),
                    "args": vars(args),
                    **failure,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"status": "metric_failed", "summary": [], "error": repr(exc)}, f, indent=2)
        with open(output_dir / "run_status.json", "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2)
        raise

    # 1. Save Run Manifest
    manifest = {
        "manifest_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": _git_metadata(),
        "args": vars(args),
        "environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "VLLM_USE_V1",
                "PROFILER_SVLLM",
                "SPARSEVLLM_SYNC_DEVICE",
            )
            if key in os.environ
        },
        "hardware_metrics": {
            "source": "nvidia-smi sampled activity",
            "physical_gpu_ids": monitor_gpus,
            "sampling_interval_ms": args.hardware_sampling_interval_ms,
            "deprecated_hardware_profile_label": args.hardware,
            "theoretical_mfu_mbu_enabled": False,
        },
        "workload": {
            "trace_generator_version": TRACE_GENERATOR_VERSION,
            "prefix_caching_enabled": False,
            "cross_engine_trace_contract": "same seed, token IDs, and per-request lengths",
            "iteration_prompt_reuse_allowed": False,
        },
        "model_specs": {
            "hidden_size": model_specs.hidden_size,
            "num_hidden_layers": model_specs.num_hidden_layers,
            "is_moe": model_specs.is_moe,
            "num_experts": model_specs.num_experts,
            "num_experts_per_tok": model_specs.num_experts_per_tok,
            "num_attention_heads": model_specs.num_attention_heads,
            "num_key_value_heads": model_specs.num_key_value_heads,
            "head_dim": model_specs.head_dim,
            "vocab_size": model_specs.vocab_size,
        },
        "status": "running",
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 2. Run Benchmarks
    try:
        scenarios = ["fixed", "churn"] if args.scenario == "all" else [args.scenario]
        summary_rows = []
        requested_scenario = args.scenario
        for scenario in scenarios:
            args.scenario = scenario
            if args.engine == "sparsevllm":
                summary_rows.extend(run_sparsevllm_probe(args, model_specs))
            else:
                summary_rows.extend(run_vllm_probe(args, model_specs))
        args.scenario = requested_scenario
        _attach_churn_comparisons(summary_rows)
        _attach_saturation_metrics(summary_rows)
    except Exception as exc:
        failure_status = "metric_failed" if isinstance(exc, HardwareMetricError) else "model_failed"
        failure = {
            "status": failure_status,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        manifest.update(failure)
        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"status": failure_status, "summary": [], "error": repr(exc)}, f, indent=2)
        with open(output_dir / "run_status.json", "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2)
        raise

    # 3. Save Summary & Markdown Report
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"status": "success", "summary": summary_rows}, f, indent=2, ensure_ascii=False)

    md_report = _format_markdown_report(
        summary_rows,
        Path(args.model_path).name,
        args.tensor_parallel_size,
    )
    with open(output_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    manifest["status"] = "success"
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(output_dir / "run_status.json", "w", encoding="utf-8") as f:
        json.dump({"status": "success"}, f, indent=2)

    print("\n" + "=" * 100)
    print(md_report)
    print("=" * 100)
    print(f"\n[Bench Probe] Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
