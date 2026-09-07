"""Shared benchmark statistics; also reaggregate probe request_samples.jsonl.

No engine imports: request latency and measured execution stages are separate
contracts. Engine adapters own observation boundaries and CUDA synchronization.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


REQUEST_TPOT_SCOPE = "mean_per_request_first_token_to_finish_v2"
BATCH_DECODE_WINDOW_SCOPE = "earliest_first_token_to_latest_completion_v2"
TPOT_CONCURRENCY_PROXY_SCOPE = "concurrency_times_1000_over_request_tpot_ms_v1"

REQUEST_METRIC_CONTRACT = "per_request_distribution_v3"


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Percentiles require nonempty values and quantile in [0, 1].")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Metric values must be finite.")
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def request_metrics(ttft_s: float, after_first_s: float, generated_tokens: int) -> dict:
    """Include all waiting after first token; do not subtract other stage work."""
    if (
        not all(math.isfinite(value) and value >= 0 for value in (ttft_s, after_first_s))
        or isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens < 1
        or (generated_tokens > 1 and after_first_s <= 0)
    ):
        raise ValueError("Invalid request duration or generated token count.")
    return {
        "ttft_ms": ttft_s * 1000,
        "latency_ms": (ttft_s + after_first_s) * 1000,
        "tpot_ms": after_first_s * 1000 / (generated_tokens - 1) if generated_tokens > 1 else None,
        "generated_tokens": generated_tokens,
    }


def request_summary(requests: list[dict]) -> dict:
    """Pool requests, never per-iteration maxima or means. Linear quantiles."""
    if not requests:
        raise ValueError("Request summary requires at least one request.")
    sources = {row.get("timing_source") for row in requests}
    if len(sources) > 1:
        raise ValueError("Cannot pool requests with different timing_source boundaries.")
    values = {"ttft_ms": [], "tpot_ms": [], "latency_ms": []}
    for row in requests:
        if row.get("status", "success") != "success":
            raise ValueError("Cannot silently include or omit a failed request.")
        generated = row["generated_tokens"]
        ttft = float(row["ttft_ms"])
        latency = float(row["latency_ms"])
        checked = request_metrics(ttft / 1000, (latency - ttft) / 1000, generated)
        tpot = row["tpot_ms"]
        expected = checked["tpot_ms"]
        if (expected is None and tpot is not None) or (
            expected is not None
            and (tpot is None or not math.isclose(float(tpot), expected, rel_tol=1e-7, abs_tol=1e-7))
        ):
            raise ValueError("TPOT disagrees with request latency, TTFT, and token count.")
        for key in values:
            if row[key] is not None:
                values[key].append(float(row[key]))
    result = {
        "request_metric_contract": REQUEST_METRIC_CONTRACT,
        "measured_request_count": len(requests),
        "tpot_request_count": len(values["tpot_ms"]),
    }
    for key, samples in values.items():
        result[f"{key}_mean"] = statistics.fmean(samples) if samples else None
        for pct in (50, 95, 99):
            result[f"{key}_p{pct}"] = percentile(samples, pct / 100) if samples else None
    return result


def stage_throughput(tokens: int, elapsed_s: float) -> float | None:
    """Actual stage token work / non-overlapping, synchronized stage time.

    Count each step once, including scheduling, sampling, scoring, compaction,
    and cleanup. Decode token work excludes tokens sampled during prefill.
    Prefix hits are excluded from computed prefill tokens; replay work is included.
    """
    if (
        isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
        or not math.isfinite(elapsed_s) or elapsed_s < 0
    ):
        raise ValueError("Invalid stage token count or elapsed time.")
    if not tokens:
        return None
    if elapsed_s <= 0:
        raise ValueError("Stage token work requires positive measured time.")
    return tokens / elapsed_s


def request_timeline_metrics(
    *,
    arrival_times: dict[int, float],
    first_token_times: dict[int, float],
    finished_times: dict[int, float],
    generated_counts: dict[int, int],
) -> dict[str, Any]:
    """Build matched request TPOT and batch phase windows from one timeline."""
    expected = set(arrival_times)
    if not expected:
        raise RuntimeError("Request timing requires at least one request.")
    for name, values in (
        ("first-token", first_token_times),
        ("completion", finished_times),
        ("generated-count", generated_counts),
    ):
        actual = set(values)
        if actual != expected:
            raise RuntimeError(
                f"Request timing {name} coverage mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )

    request_timings = []
    for request_id in sorted(expected):
        arrival = float(arrival_times[request_id])
        first = float(first_token_times[request_id])
        finished = float(finished_times[request_id])
        generated = int(generated_counts[request_id])
        if generated <= 0:
            raise RuntimeError(
                f"Request {request_id} has non-positive generated token count {generated}."
            )
        if first < arrival or finished < first:
            raise RuntimeError(
                f"Request {request_id} has invalid timing order: "
                f"arrival={arrival}, first={first}, finished={finished}."
            )
        decode_s = finished - first
        if generated > 1:
            if decode_s <= 0:
                raise RuntimeError(
                    f"Request {request_id} generated {generated} tokens without a "
                    f"positive decode duration: {decode_s}."
                )
        request_timings.append(
            {
                "request_id": request_id,
                **request_metrics(first - arrival, decode_s, generated),
            }
        )

    summary = request_summary(request_timings)
    return {
        "ttft_ms": summary["ttft_ms_mean"],
        "tpot_ms": summary["tpot_ms_mean"],
        "tpot_timing_scope": REQUEST_TPOT_SCOPE,
        "prefill_elapsed_s": max(
            first_token_times[request_id] - arrival_times[request_id]
            for request_id in expected
        ),
        "decode_elapsed_s": (
            max(finished_times.values()) - min(first_token_times.values())
        ),
        "request_timings": request_timings,
    }


def tpot_concurrency_proxy_tps(
    *,
    concurrency: int,
    tpot_ms: float | None,
) -> float | None:
    """Return the TPOT-equivalent concurrent token-rate proxy.

    This is intentionally distinct from observed batch decode-window throughput.
    For matched concurrency it is algebraically equivalent to TPOT speedup.
    """
    if tpot_ms is None:
        return None
    if concurrency <= 0 or tpot_ms <= 0:
        raise RuntimeError(
            f"Invalid TPOT proxy inputs: concurrency={concurrency}, tpot_ms={tpot_ms}."
        )
    return concurrency * 1000.0 / tpot_ms


def event_window_metrics(
    *,
    total_input_tokens: int,
    total_output_tokens: int,
    request_count: int,
    prefill_elapsed_s: float,
    decode_elapsed_s: float,
) -> dict[str, Any]:
    """Build request-event window diagnostics, never execution-stage throughput.

    The first generated token is produced by the final prefill step, so decode
    throughput counts only the remaining output tokens.
    """
    if total_input_tokens <= 0 or request_count <= 0 or prefill_elapsed_s <= 0:
        raise RuntimeError(
            "Invalid prefill throughput inputs: "
            f"tokens={total_input_tokens}, requests={request_count}, "
            f"elapsed_s={prefill_elapsed_s}."
        )
    decode_tokens = total_output_tokens - request_count
    if decode_tokens < 0:
        raise RuntimeError(
            f"Output token count {total_output_tokens} is smaller than request count "
            f"{request_count}."
        )
    if decode_tokens > 0 and decode_elapsed_s <= 0:
        raise RuntimeError(
            "Decode tokens were generated without a positive decode window: "
            f"tokens={decode_tokens}, elapsed_s={decode_elapsed_s}."
        )
    batch_decode_tps = decode_tokens / decode_elapsed_s if decode_tokens > 0 else None
    return {
        "stage_metrics_status": "not_measured",
        "phase_timing_scope": "matched_request_event_wall_time_windows_v2",
        "batch_decode_window_scope": BATCH_DECODE_WINDOW_SCOPE,
        "prefill_elapsed_s": prefill_elapsed_s,
        "decode_elapsed_s": decode_elapsed_s if decode_tokens > 0 else None,
        "prefill_token_count": total_input_tokens,
        "decode_token_count": decode_tokens,
        "first_token_window_throughput_tps": total_input_tokens / prefill_elapsed_s,
        "prefill_token_throughput_tps": total_input_tokens / prefill_elapsed_s,
        "batch_decode_token_throughput_tps": batch_decode_tps,
        "decode_token_throughput_tps": batch_decode_tps,
    }


def mean_event_window_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    decode_rates = [
        record["decode_token_throughput_tps"]
        for record in records
        if record["decode_token_throughput_tps"] is not None
    ]
    decode_times = [
        record["decode_elapsed_s"]
        for record in records
        if record["decode_elapsed_s"] is not None
    ]
    return {
        "stage_metrics_status": "not_measured",
        "phase_timing_scope": "matched_request_event_wall_time_windows_v2",
        "batch_decode_window_scope": BATCH_DECODE_WINDOW_SCOPE,
        "first_token_window_throughput_tps": statistics.fmean(
            record["first_token_window_throughput_tps"] for record in records
        ),
        "prefill_token_throughput_tps": statistics.fmean(
            record["prefill_token_throughput_tps"] for record in records
        ),
        "decode_token_throughput_tps": (
            statistics.fmean(decode_rates) if decode_rates else None
        ),
        "batch_decode_token_throughput_tps": (
            statistics.fmean(decode_rates) if decode_rates else None
        ),
        "prefill_elapsed_s_mean": statistics.fmean(
            record["prefill_elapsed_s"] for record in records
        ),
        "decode_elapsed_s_mean": statistics.fmean(decode_times) if decode_times else None,
        "output_token_throughput_tps": sum(
            sum(row["generated_tokens"] for row in record["request_results"])
            for record in records
        ) / sum(record["elapsed_s"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_samples", type=Path, help="Probe request_samples.jsonl")
    args = parser.parse_args()
    # Iterations are pooled, but distinct workloads and observation boundaries
    # must never be combined into one distribution.
    keys = ("engine", "sparse_method", "scenario", "nominal_prompt_len",
            "nominal_output_len", "concurrency", "timing_source")
    groups = {}
    with args.request_samples.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["status"] != "success":
                raise ValueError("Input contains non-success requests; no implicit filtering is allowed.")
            group = tuple(row[key] for key in keys)
            groups.setdefault(group, []).append(row)
    if not groups:
        raise ValueError("Request artifact is empty.")
    rows = [{**dict(zip(keys, group)), **request_summary(requests)}
            for group, requests in groups.items()]
    print(json.dumps({"status": "success", "records": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
