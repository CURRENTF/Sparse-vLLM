from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
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
TOPOLOGIES = (
    ("single", "0", 1, 1),
    ("pure_tp", "0,1", 2, 1),
    ("tp_ep", "0,1", 2, 2),
)
MODES = ("graph", "eager")
METRICS = (
    "ttft_s",
    "prefill_tok_s",
    "decode_tok_s",
    "itl_ms",
    "end_to_end_tok_s",
    "peak_memory_gb",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_snapshot() -> dict[str, Any]:
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    gpu_lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().splitlines()
    gpus = []
    for line in gpu_lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {line!r}.")
        gpus.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    return {"compute_processes": processes.splitlines() if processes else [], "gpus": gpus}


def _wait_for_all_devices_idle(timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        snapshot = _gpu_snapshot()
        idle = not snapshot["compute_processes"] and all(
            gpu["utilization_percent"] <= 1 for gpu in snapshot["gpus"]
        )
        if idle:
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"GPUs remained busy for {timeout_s}s; last snapshot={snapshot!r}."
            )
        print(f"[gpu-idle] busy; waiting 10s: {snapshot}", flush=True)
        time.sleep(10)


def _read_single_record(run_dir: Path) -> dict[str, Any]:
    aggregate_path = run_dir / "aggregate_metrics.json"
    if not aggregate_path.is_file():
        return {
            "status": "model_failed",
            "error": f"missing artifact {aggregate_path}",
        }
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    records = aggregate.get("records")
    if not isinstance(records, list) or len(records) != 1:
        return {
            "status": "model_failed",
            "error": f"expected exactly one microbench record, got {records!r}",
        }
    return records[0]


def _aggregate(records: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["topology"],
            record["mode"],
            record["prompt_tokens"],
            record["batch_size"],
            record["max_new_tokens"],
        )
        groups.setdefault(key, []).append(record)
    summaries = []
    failed_groups = 0
    for key, rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("status") == "success"]
        status = "success" if len(successful) == repeats else "model_failed"
        if status != "success":
            failed_groups += 1
        metrics = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in successful if metric in row]
            metrics[metric] = (
                None
                if len(values) != repeats
                else {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
        summaries.append(
            {
                "topology": key[0],
                "mode": key[1],
                "prompt_tokens": key[2],
                "batch_size": key[3],
                "max_new_tokens": key[4],
                "status": status,
                "successful_repeats": len(successful),
                "required_repeats": repeats,
                "metrics": metrics,
            }
        )
    return {
        "status": "success" if not failed_groups else "model_failed",
        "num_groups": len(summaries),
        "successful_groups": len(summaries) - failed_groups,
        "failed_groups": failed_groups,
        "required_repeats": repeats,
        "summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Qwen3.6 MoE benchmark matrix.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--modes",
        default="graph,eager",
        help="Comma-separated subset of graph,eager.",
    )
    parser.add_argument("--idle-timeout", type=int, default=600)
    parser.add_argument("--run-timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("The acceptance matrix requires at least two measured repeats.")
    if args.run_timeout <= 0:
        raise ValueError("--run-timeout must be positive.")
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if not modes or len(set(modes)) != len(modes) or any(
        mode not in MODES for mode in modes
    ):
        raise ValueError(
            "--modes must contain each of 'graph' and 'eager' at most once, "
            f"got {args.modes!r}."
        )
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "model.safetensors.index.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(f"Required model metadata is missing: {model / name}.")

    initial_gpu = _wait_for_all_devices_idle(args.idle_timeout)
    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "model": str(model),
        "model_config_sha256": _sha256(model / "config.json"),
        "model_index_sha256": _sha256(model / "model.safetensors.index.json"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "git_branch": subprocess.run(
            ["git", "branch", "--show-current"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        ),
        "cases": [
            {"prompt_tokens": length, "batch_size": batch, "max_new_tokens": output}
            for length, batch, output in CASES
        ],
        "topologies": [
            {
                "name": name,
                "cuda_visible_devices": devices,
                "outer_tp_size": tp,
                "expert_parallel_size": ep,
                "moe_tp_size": tp // ep,
            }
            for name, devices, tp, ep in TOPOLOGIES
        ],
        "modes": list(modes),
        "repeats": args.repeats,
        "seed": 20260810,
        "warmup_output_len": 8,
        "run_timeout_s": args.run_timeout,
        "initial_gpu_snapshot": initial_gpu,
    }
    _write_json(output_dir / "run_info.json", run_info)

    records: list[dict[str, Any]] = []
    total = len(CASES) * len(TOPOLOGIES) * len(modes) * args.repeats
    run_index = 0
    for length, batch, max_new_tokens in CASES:
        for topology, devices, tp_size, ep_size in TOPOLOGIES:
            for mode in modes:
                for repeat in range(1, args.repeats + 1):
                    run_index += 1
                    idle_snapshot = _wait_for_all_devices_idle(args.idle_timeout)
                    run_name = (
                        f"{length}_bs{batch}_out{max_new_tokens}_"
                        f"{topology}_{mode}_r{repeat}"
                    )
                    run_dir = output_dir / "runs" / run_name
                    hyper_params = {
                        "tensor_parallel_size": tp_size,
                        "expert_parallel_size": ep_size,
                        "enforce_eager": mode == "eager",
                        "decode_cuda_graph": mode == "graph",
                        "gpu_memory_utilization": 0.9,
                        "engine_prefill_chunk_size": 8192,
                        "max_num_batched_tokens": 65536,
                        "weight_loading_workers": 16,
                    }
                    command = [
                        sys.executable,
                        "benchmark/microbench.py",
                        "--model_path",
                        str(model),
                        "--lengths",
                        str(length),
                        "--batch_sizes",
                        str(batch),
                        "--output_len",
                        str(max_new_tokens),
                        "--methods",
                        "vanilla",
                        "--temperature",
                        "0",
                        "--top_p",
                        "1",
                        "--seed",
                        "20260810",
                        "--synchronize_step_timing",
                        "--warmup_output_len",
                        "8",
                        "--hyper_params",
                        json.dumps(hyper_params, separators=(",", ":")),
                        "--output_dir",
                        str(run_dir),
                    ]
                    env = os.environ.copy()
                    env.update(
                        {
                            "CUDA_VISIBLE_DEVICES": devices,
                            "PYTHONUNBUFFERED": "1",
                            "SPARSEVLLM_MOE_PROVIDER": "triton",
                            "SPARSEVLLM_MOE_ROUTER_PROVIDER": "triton",
                        }
                    )
                    print(
                        f"[matrix {run_index}/{total}] {run_name}: {' '.join(command)}",
                        flush=True,
                    )
                    timed_out = False
                    try:
                        process = subprocess.run(
                            command,
                            cwd=Path(__file__).resolve().parents[2],
                            env=env,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                            timeout=args.run_timeout,
                        )
                        console_output = process.stdout
                        returncode = process.returncode
                    except subprocess.TimeoutExpired as exc:
                        timed_out = True
                        console_output = exc.stdout or ""
                        if isinstance(console_output, bytes):
                            console_output = console_output.decode(
                                "utf-8", errors="replace"
                            )
                        console_output += (
                            "\nMatrix runner timed out this run after "
                            f"{args.run_timeout}s.\n"
                        )
                        returncode = None
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "console.log").write_text(
                        console_output, encoding="utf-8"
                    )
                    record = _read_single_record(run_dir)
                    record.update(
                        {
                            "run_name": run_name,
                            "topology": topology,
                            "mode": mode,
                            "repeat": repeat,
                            "prompt_tokens": length,
                            "batch_size": batch,
                            "max_new_tokens": max_new_tokens,
                            "returncode": returncode,
                            "timed_out": timed_out,
                            "command": command,
                            "idle_snapshot_before": idle_snapshot,
                            "artifact_dir": str(run_dir),
                        }
                    )
                    if timed_out or returncode != 0:
                        record["status"] = "model_failed"
                    records.append(record)
                    with (output_dir / "per_sample_results.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        f"[matrix {run_index}/{total}] status={record['status']}",
                        flush=True,
                    )

    aggregate = _aggregate(records, args.repeats)
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    _write_json(output_dir / "parsed_outputs.json", {"records": records})
    _write_json(
        output_dir / "raw_outputs.json",
        {"run_artifact_dirs": [record["artifact_dir"] for record in records]},
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return 0 if aggregate["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
