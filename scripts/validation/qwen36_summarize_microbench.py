from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


CASES = (
    (1024, 1, 512),
    (4096, 8, 512),
    (32768, 2, 512),
    (65536, 1, 1024),
    (131072, 1, 1024),
)
TOPOLOGIES = ("single", "pure_tp", "tp_ep")
MODES = ("graph", "eager")
METRICS = (
    "ttft_s",
    "prefill_tok_s",
    "decode_tok_s",
    "itl_ms",
    "end_to_end_tok_s",
    "peak_memory_gb",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected an object at {path}:{line_no}.")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stats(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = [float(row[metric]) for row in rows]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "relative_range": (
            (max(values) - min(values)) / statistics.median(values)
            if statistics.median(values)
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a completed Qwen3.6 microbench matrix."
    )
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--modes",
        default="graph,eager",
        help="Comma-separated subset of graph,eager.",
    )
    args = parser.parse_args()
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if not modes or len(set(modes)) != len(modes) or any(
        mode not in MODES for mode in modes
    ):
        raise ValueError(
            "--modes must contain each of 'graph' and 'eager' at most once, "
            f"got {args.modes!r}."
        )
    matrix_dir = args.matrix_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = _read_jsonl(matrix_dir / "per_sample_results.jsonl")

    expected_keys = {
        (topology, mode, length, batch, output)
        for length, batch, output in CASES
        for topology in TOPOLOGIES
        for mode in modes
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    validation_records = []
    for record in records:
        key = (
            record.get("topology"),
            record.get("mode"),
            int(record.get("prompt_tokens")),
            int(record.get("batch_size")),
            int(record.get("max_new_tokens")),
        )
        groups.setdefault(key, []).append(record)
        workers = record.get("worker_runtime_status") or []
        expected_workers = 1 if record.get("topology") == "single" else 2
        graph_active = (
            record.get("mode") != "graph"
            or (
                len(workers) == expected_workers
                and all(worker.get("decode_cuda_graph_active") for worker in workers)
            )
        )
        providers_ok = (
            record.get("moe_expert_provider") == "triton"
            and record.get("moe_router_provider") == "triton"
            and all(
                worker.get("moe_expert_provider") == "triton"
                and worker.get("moe_router_provider") == "triton"
                for worker in workers
            )
        )
        validation_records.append(
            {
                "check": "measured_run",
                "run_name": record.get("run_name"),
                "status": (
                    "success"
                    if record.get("status") == "success"
                    and graph_active
                    and providers_ok
                    else "metric_failed"
                ),
                "graph_active_on_all_workers": graph_active,
                "providers_ok": providers_ok,
                "worker_count": len(workers),
            }
        )

    missing_groups = sorted(expected_keys - set(groups))
    unexpected_groups = sorted(set(groups) - expected_keys)
    summaries = []
    for key in sorted(expected_keys):
        rows = groups.get(key, [])
        successful = [row for row in rows if row.get("status") == "success"]
        summaries.append(
            {
                "topology": key[0],
                "mode": key[1],
                "prompt_tokens": key[2],
                "batch_size": key[3],
                "max_new_tokens": key[4],
                "status": (
                    "success"
                    if len(successful) == args.repeats
                    else "metric_failed"
                ),
                "successful_repeats": len(successful),
                "required_repeats": args.repeats,
                "metrics": (
                    {metric: _stats(successful, metric) for metric in METRICS}
                    if len(successful) == args.repeats
                    else None
                ),
            }
        )

    summary_by_key = {
        (
            row["topology"],
            row["mode"],
            row["prompt_tokens"],
            row["batch_size"],
            row["max_new_tokens"],
        ): row
        for row in summaries
    }
    speedups = []
    if {"graph", "eager"} <= set(modes):
        for length, batch, output in CASES:
            for topology in TOPOLOGIES:
                graph = summary_by_key[(topology, "graph", length, batch, output)]
                eager = summary_by_key[(topology, "eager", length, batch, output)]
                if graph["status"] != "success" or eager["status"] != "success":
                    speedups.append(
                        {
                            "topology": topology,
                            "prompt_tokens": length,
                            "batch_size": batch,
                            "max_new_tokens": output,
                            "status": "metric_failed",
                        }
                    )
                    continue
                graph_decode = graph["metrics"]["decode_tok_s"]["median"]
                eager_decode = eager["metrics"]["decode_tok_s"]["median"]
                graph_itl = graph["metrics"]["itl_ms"]["median"]
                eager_itl = eager["metrics"]["itl_ms"]["median"]
                speedups.append(
                    {
                        "topology": topology,
                        "prompt_tokens": length,
                        "batch_size": batch,
                        "max_new_tokens": output,
                        "status": "success",
                        "decode_throughput_speedup": graph_decode / eager_decode,
                        "itl_speedup": eager_itl / graph_itl,
                        "graph_decode_tok_s": graph_decode,
                        "eager_decode_tok_s": eager_decode,
                    }
                )

    failed = [
        record
        for record in [*validation_records, *summaries, *speedups]
        if record["status"] != "success"
    ]
    aggregate = {
        "status": (
            "success"
            if not failed and not missing_groups and not unexpected_groups
            else "metric_failed"
        ),
        "num_runs": len(records),
        "expected_runs": len(expected_keys) * args.repeats,
        "missing_groups": missing_groups,
        "unexpected_groups": unexpected_groups,
        "failed_checks": len(failed),
        "summaries": summaries,
        "graph_speedups": speedups,
    }
    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "matrix_dir": str(matrix_dir),
        "repeats": args.repeats,
        "modes": list(modes),
    }
    _write_json(output_dir / "run_info.json", run_info)
    _write_json(output_dir / "raw_outputs.json", {"matrix_dir": str(matrix_dir)})
    _write_json(
        output_dir / "parsed_outputs.json",
        {"run_checks": validation_records, "summaries": summaries, "speedups": speedups},
    )
    with (output_dir / "per_sample_results.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in [*validation_records, *summaries, *speedups]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if aggregate["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
