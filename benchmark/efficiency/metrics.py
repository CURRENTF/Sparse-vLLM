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


class DecodeOnlyWindow:
    """Contiguous full-residency engine window, synchronized only at its edges.

    Adapters call boundary before scheduling and observe after each engine step.
    All intervening driver, scheduling and GPU work remains in the denominator.
    This opt-in diagnostic perturbs its two boundary iterations, not every step.
    """

    def __init__(self, concurrency, steps, warmup_steps, *, synchronize, clock,
                 graph_stats):
        if concurrency < 1 or steps < 1 or warmup_steps < 1:
            raise ValueError("Decode window requires positive concurrency, steps and warmup")
        self.concurrency = concurrency
        self.target_steps = steps
        self.warmup_steps = warmup_steps
        self.synchronize = synchronize
        self.clock = clock
        self.graph_stats = graph_stats
        self.ids = None
        self.warmed = 0
        self.steps = 0
        self.tokens = 0
        self.started = None
        self.result = None

    def boundary(self):
        if self.result is not None:
            return
        if self.started is not None and self.steps == self.target_steps:
            self.synchronize()
            finished = self.clock()
            after = self.graph_stats()
            delta = decode_graph_delta(self.graph_before, after, self.steps)
            elapsed = finished - self.started
            self.result = {
                "status": "success",
                "scope": "full_residency_contiguous_decode_only_boundary_sync_v1",
                "timing_boundary": "before_scheduler_iteration_to_before_scheduler_iteration",
                "concurrency": self.concurrency,
                "request_ids": list(self.ids),
                "discarded_full_decode_steps": self.warmup_steps,
                "decode_steps": self.steps,
                "decode_stage_tokens": self.tokens,
                "decode_stage_elapsed_s": elapsed,
                "decode_stage_throughput_tps": stage_throughput(self.tokens, elapsed),
                "prefill_steps": 0,
                "cuda_synchronizations": 2,
                "graph_counter_delta": delta,
                "started_monotonic_s": self.started,
                "finished_monotonic_s": finished,
            }
        elif self.started is None and self.warmed == self.warmup_steps:
            self.synchronize()
            self.graph_before = self.graph_stats()
            self.started = self.clock()

    def observe(self, *, is_decode, request_ids, tokens, admission_complete):
        if self.result is not None:
            return
        ids = tuple(sorted(request_ids))
        eligible = (is_decode and admission_complete and len(ids) == self.concurrency
                    and len(set(ids)) == self.concurrency and tokens == self.concurrency)
        if self.started is not None:
            if not eligible or ids != self.ids:
                raise RuntimeError("Prefill, admission, request turnover or partial batch inside decode-only window")
            self.steps += 1
            self.tokens += tokens
        elif eligible:
            if ids != self.ids:
                self.ids, self.warmed = ids, 0
            self.warmed += 1
        else:
            self.ids, self.warmed = None, 0

    def require_result(self):
        if self.result is None:
            raise RuntimeError("Workload ended without the requested full-residency decode-only window")
        return self.result


def decode_graph_delta(before, after, steps):
    """Validate every participating rank, retaining the legacy single-rank form."""
    if isinstance(before, list):
        if not before or not isinstance(after, list) or len(before) != len(after):
            raise RuntimeError("Decode window lost participating ranks")
        return [decode_graph_delta(a, b, steps) for a, b in zip(before, after)]
    keys = ("capture_count", "replay_count", "eager_decode_count")
    delta = {key: after[key] - before[key] for key in keys}
    if delta["capture_count"] or delta["eager_decode_count"] or delta["replay_count"] != steps:
        raise RuntimeError(f"Decode-only window violated captured graph contract: {delta}")
    return delta


class PipelinedDecodeWindow(DecodeOnlyWindow):
    """Stop submissions only at two edges; drain and account for completed work.

    Adapters submit before/after a forward with CPU-owned shape metadata and
    complete after normal result processing. An edge must drain all in-flight
    work before calling boundary(); no timed per-step synchronization is needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = {}
        self.records = []
        self.submitted = 0
        self.last_measured_contexts = None

    @property
    def needs_boundary(self):
        return self.result is None and (
            (self.started is None and self.warmed == self.warmup_steps)
            or (self.started is not None and self.steps == self.target_steps))

    def boundary(self):
        if not self.needs_boundary:
            return
        if self.pending:
            raise RuntimeError("Decode boundary has undrained in-flight work")
        super().boundary()
        if self.result is not None:
            selected = [row for row in self.records if row["measured"]]
            if len(selected) != self.target_steps or not all(row["completed"] for row in selected):
                raise RuntimeError("Incomplete decode window work")
            self.result.update(
                scope="full_residency_contiguous_decode_only_boundary_sync_v2",
                context_lengths_start=selected[0]["context_lengths"],
                context_lengths_end=[n + 1 for n in selected[-1]["context_lengths"]],
                context_length_semantics="logical attention KV length including current query token; end is next-step length",
                completed_steps=len(selected),
            )

    def submit(self, *, is_decode, request_ids, tokens, admission_complete,
               context_lengths, preemptions=0):
        if self.result is not None:
            return None
        if self.needs_boundary:
            raise RuntimeError("Submission crossed an undrained measurement edge")
        if preemptions:
            raise RuntimeError("Full decode batch capacity exceeded: scheduler preemption")
        if len(context_lengths) != len(request_ids) or any(n < 0 for n in context_lengths):
            raise ValueError("Missing request context lengths")
        measured = self.started is not None
        super().observe(is_decode=is_decode, request_ids=request_ids, tokens=tokens,
                        admission_complete=admission_complete)
        pairs = sorted(zip(request_ids, context_lengths))
        ordered_contexts = [p[1] for p in pairs]
        if measured and self.last_measured_contexts is not None:
            if ordered_contexts != [n + 1 for n in self.last_measured_contexts]:
                raise RuntimeError("Decode context lengths did not advance by one token")
        if measured:
            self.last_measured_contexts = ordered_contexts
        ticket = self.submitted
        self.submitted += 1
        row = dict(ticket=ticket, pure_decode=bool(is_decode),
                   request_ids=[p[0] for p in pairs], context_lengths=ordered_contexts,
                   tokens=tokens, measured=measured, completed=False)
        self.pending[ticket] = row
        self.records.append(row)
        return ticket

    def complete(self, ticket, *, decode_tokens=None):
        if ticket is None:
            return
        if ticket not in self.pending:
            raise RuntimeError("Unknown or duplicate completed decode work")
        row = self.pending.pop(ticket)
        if row["pure_decode"] and decode_tokens != row["tokens"]:
            raise RuntimeError("Completed decode tokens disagree with submitted work")
        row["completed_decode_tokens"] = decode_tokens
        row["completed"] = True


def decode_window_fields(result):
    """Canonical row fields; never relabel unsynchronized per-step durations."""
    return dict(
        measurement_scope="full_batch_decode_window", stage_metrics_status="success",
        stage_timing_scope=result["scope"], synchronize_step_timing=False,
        decode_window=result, decode_stage_tokens=result["decode_stage_tokens"],
        decode_stage_elapsed_s=result["decode_stage_elapsed_s"],
        decode_stage_throughput_tps=result["decode_stage_throughput_tps"],
        decode_tp=result["decode_stage_throughput_tps"],
        ttft=None, itl=None, prefill_tp=None,
        ttft_timing_scope="not_measured", itl_timing_scope="not_measured",
        measured_decode_steps_after_full=result["decode_steps"],
        prefill_stage_elapsed_s=None, prefill_stage_throughput_tps=None,
        request_metrics_status="not_measured",
    )


def aggregate_decode_windows(rows):
    if not rows:
        raise ValueError("No decode repetitions")
    identity = ("engine", "method", "length", "output_len", "batch_size",
                "stage_timing_scope", "decode_warmup_steps_after_full",
                "measured_decode_steps_after_full")
    for row in rows:
        if (row.get("status") != "success" or row.get("synchronize_step_timing") is not False
                or row.get("measurement_scope") != "full_batch_decode_window"
                or any(row[k] != rows[0][k] for k in identity)):
            raise ValueError("Inconsistent or failed decode-window repetitions")
        for key in ("resolved_parallel_topology", "actual_async_scheduling", "actual_overlap_scheduling",
                    "admission_wave_size", "wave_decode_gap_steps"):
            if row.get(key) != rows[0].get(key):
                raise ValueError(f"Inconsistent repetition contract: {key}")
        window = row["decode_window"]
        if (row["decode_stage_tokens"] != row["batch_size"] * row["measured_decode_steps_after_full"]
                or window["decode_stage_tokens"] != row["decode_stage_tokens"]
                or window["decode_stage_elapsed_s"] != row["decode_stage_elapsed_s"]
                or not math.isclose(stage_throughput(row["decode_stage_tokens"], row["decode_stage_elapsed_s"]),
                                    row["decode_stage_throughput_tps"], rel_tol=1e-10)):
            raise ValueError("Inconsistent repetition token/time accounting")
    tokens = sum(row["decode_stage_tokens"] for row in rows)
    elapsed = sum(row["decode_stage_elapsed_s"] for row in rows)
    result = dict(rows[0])
    result.pop("decode_window", None)
    result.pop("artifact", None)
    result.pop("repetition", None)
    result.update(repetitions=rows, decode_stage_tokens=tokens, decode_stage_elapsed_s=elapsed,
                  repetition_throughput_stdev_tps=statistics.stdev(r["decode_stage_throughput_tps"] for r in rows) if len(rows) > 1 else None,
                  decode_stage_throughput_tps=stage_throughput(tokens, elapsed),
                  decode_tp=stage_throughput(tokens, elapsed),
                  measured_decode_steps_after_full=sum(r["measured_decode_steps_after_full"] for r in rows))
    return result


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
