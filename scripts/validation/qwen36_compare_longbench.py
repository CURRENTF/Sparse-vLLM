from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_SCORE_MAX_DROP = 20.0
MEAN_SCORE_MAX_DROP = 3.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected an object at {path}:{line_no}.")
        rows.append(value)
    return rows


def _validate_worker_providers(
    workers: list[dict[str, Any]],
    *,
    precision: str,
    path: Path,
) -> None:
    if precision == "bf16":
        valid = all(
            worker.get("moe_expert_provider") == "triton"
            and worker.get("moe_router_provider") == "triton"
            for worker in workers
        )
    else:
        valid_expert_providers = {
            "flashinfer_cutlass_fp8_sm90",
            "triton",
        }
        valid = all(
            worker.get("moe_expert_provider") in valid_expert_providers
            and worker.get("moe_router_provider") == "triton"
            and worker.get("moe_weight_dtype") == "torch.float8_e4m3fn"
            and isinstance(worker.get("fp8_linear_provider"), str)
            and bool(worker["fp8_linear_provider"])
            for worker in workers
        )
    if not valid:
        raise RuntimeError(
            f"LongBench run {path} has invalid {precision.upper()} providers."
        )


def _load_run(path: Path, *, precision: str) -> dict[str, Any]:
    required = (
        "resolved_config.json",
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "per_sample_results.jsonl",
        "aggregate_metrics.json",
        "decode_cuda_graph_status_rank0.json",
        "tokenizer_runtime.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"LongBench run {path} is missing {missing}.")
    metrics = _read_json(path / "aggregate_metrics.json")
    if metrics.get("status") != "success":
        raise RuntimeError(f"LongBench run {path} failed: {metrics!r}.")
    samples = _read_jsonl(path / "per_sample_results.jsonl")
    if not samples or any(sample.get("status") != "success" for sample in samples):
        raise RuntimeError(f"LongBench run {path} contains non-success samples.")
    graph = _read_json(path / "decode_cuda_graph_status_rank0.json")
    if not graph.get("configured_on_all_workers") or not graph.get(
        "active_on_all_workers"
    ):
        raise RuntimeError(f"LongBench run {path} did not activate Graph everywhere.")
    workers = graph.get("workers")
    if not isinstance(workers, list) or not workers:
        raise RuntimeError(f"LongBench run {path} has no worker status records.")
    _validate_worker_providers(workers, precision=precision, path=path)
    config = _read_json(path / "resolved_config.json")
    expected_per_task = int(config["selection"]["samples_per_task"])
    counts = {
        str(dataset): sum(
            sample.get("dataset") == dataset for sample in samples
        )
        for dataset in config["datasets"]
    }
    if any(count != expected_per_task for count in counts.values()):
        raise RuntimeError(
            f"LongBench run {path} has incomplete task samples: "
            f"expected_per_task={expected_per_task}, counts={counts}."
        )
    return {
        "path": path,
        "config": config,
        "metrics": metrics,
        "samples": samples,
        "graph": graph,
        "tokenizer_runtime": _read_json(path / "tokenizer_runtime.json"),
    }


def _selection(run: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (str(sample["dataset"]), int(sample["source_idx"]))
        for sample in run["samples"]
    ]


def _rendered_prompt_hashes(run: dict[str, Any]) -> list[str]:
    hashes = [sample.get("rendered_prompt_sha256") for sample in run["samples"]]
    if any(not isinstance(value, str) or not value for value in hashes):
        raise RuntimeError(
            f"LongBench run {run['path']} is missing rendered prompt hashes."
        )
    return hashes


def _dataset_fingerprints(run: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["dataset"]): str(item["sha256"])
        for item in run["config"]["dataset_files"]
    }


def _task_scores(run: dict[str, Any]) -> dict[str, float]:
    datasets = [str(dataset) for dataset in run["config"]["datasets"]]
    scores = {}
    for dataset in datasets:
        score = run["metrics"].get(dataset)
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"LongBench run {run['path']} has no numeric score for {dataset}: {score!r}."
            )
        scores[dataset] = float(score)
    return scores


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the fixed Qwen3.6 MoE LongBench subset across topologies."
    )
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--pure-tp", type=Path, required=True)
    parser.add_argument("--tp-ep", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp8"),
        default="bf16",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        "single": _load_run(args.single.resolve(), precision=args.precision),
        "pure_tp": _load_run(args.pure_tp.resolve(), precision=args.precision),
        "tp_ep": _load_run(args.tp_ep.resolve(), precision=args.precision),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    reference = runs["single"]
    reference_selection = _selection(reference)
    reference_prompt_hashes = _rendered_prompt_hashes(reference)
    reference_fingerprints = _dataset_fingerprints(reference)
    reference_scores = _task_scores(reference)
    reference_mean = sum(reference_scores.values()) / len(reference_scores)
    records: list[dict[str, Any]] = []
    for topology, run in runs.items():
        config_equal = all(
            run["config"].get(key) == reference["config"].get(key)
            for key in (
                "datasets",
                "seed",
                "decoding",
                "selection",
                "model_files",
                "tokenizer_files",
                "provider_env",
                "prompt_config",
                "maxlen_config",
            )
        )
        same_selection = _selection(run) == reference_selection
        same_rendered_prompts = (
            _rendered_prompt_hashes(run) == reference_prompt_hashes
        )
        same_fingerprints = _dataset_fingerprints(run) == reference_fingerprints
        same_tokenizer_runtime = (
            run["tokenizer_runtime"] == reference["tokenizer_runtime"]
        )
        records.append(
            {
                "check": "fixed_inputs",
                "topology": topology,
                "status": (
                    "success"
                    if config_equal
                    and same_selection
                    and same_rendered_prompts
                    and same_fingerprints
                    and same_tokenizer_runtime
                    else "metric_failed"
                ),
                "config_equal": config_equal,
                "same_sample_ids": same_selection,
                "same_rendered_prompts": same_rendered_prompts,
                "same_dataset_fingerprints": same_fingerprints,
                "same_tokenizer_runtime": same_tokenizer_runtime,
                "num_samples": len(run["samples"]),
            }
        )
        scores = _task_scores(run)
        for task, reference_score in reference_scores.items():
            score = scores[task]
            drop = reference_score - score
            records.append(
                {
                    "check": "task_quality",
                    "topology": topology,
                    "task": task,
                    "status": (
                        "success" if drop <= TASK_SCORE_MAX_DROP else "metric_failed"
                    ),
                    "score": score,
                    "single_reference_score": reference_score,
                    "score_drop": drop,
                    "max_allowed_drop": TASK_SCORE_MAX_DROP,
                }
            )
        mean_score = sum(scores.values()) / len(scores)
        mean_drop = reference_mean - mean_score
        records.append(
            {
                "check": "mean_quality",
                "topology": topology,
                "status": (
                    "success" if mean_drop <= MEAN_SCORE_MAX_DROP else "metric_failed"
                ),
                "mean_score": mean_score,
                "single_reference_mean_score": reference_mean,
                "score_drop": mean_drop,
                "max_allowed_drop": MEAN_SCORE_MAX_DROP,
            }
        )

    failed = [record for record in records if record["status"] != "success"]
    aggregate = {
        "status": "success" if not failed else "metric_failed",
        "num_checks": len(records),
        "success_checks": len(records) - len(failed),
        "failed_checks": len(failed),
        "thresholds": {
            "task_score_max_drop": TASK_SCORE_MAX_DROP,
            "mean_score_max_drop": MEAN_SCORE_MAX_DROP,
        },
        "scores": {
            topology: _task_scores(run) for topology, run in runs.items()
        },
    }
    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "sources": {topology: str(run["path"]) for topology, run in runs.items()},
        "precision": args.precision,
    }
    _write_json(output_dir / "run_info.json", run_info)
    _write_json(output_dir / "raw_outputs.json", run_info["sources"])
    _write_json(output_dir / "parsed_outputs.json", {"checks": records})
    with (output_dir / "per_sample_results.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
