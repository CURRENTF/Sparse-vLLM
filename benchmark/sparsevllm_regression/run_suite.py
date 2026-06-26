#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.sparsevllm_regression.grading import (
    GateGrade,
    grade_logits,
    grade_memory,
    grade_perf,
    grade_quality,
    grade_stress,
    worst_required_grade,
)
from benchmark.sparsevllm_regression.manifest import (
    compressor_path_for,
    load_manifest,
    missing_runtime_inputs,
    resolve_manifest_paths,
    select_entries,
)
from sparsevllm.method_registry import is_decode_cuda_graph_supported, is_tp_decode_cuda_graph_supported


DEFAULT_OUTPUT_ROOT = os.getenv("DELTAKV_OUTPUT_DIR", "/root/autodl-fs/deltakv_outputs")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _append_jsonl_file(dst: Path, src: Path, extra: dict[str, Any]) -> None:
    if not src.exists():
        return
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object rows in {src}, got {type(payload).__name__}.")
        _append_jsonl(dst, {**extra, **payload})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _ensure_artifacts(output_root: Path, outputs: list[str]) -> None:
    for name in outputs:
        path = output_root / name
        if path.exists():
            continue
        if name.endswith(".jsonl"):
            path.write_text("", encoding="utf-8")
        elif name.endswith(".json"):
            _write_json(path, {})


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _git_status_short() -> str:
    return subprocess.check_output(["git", "status", "--short"], text=True).strip()


def _run_command(cmd: list[str], *, cwd: Path, dry_run: bool, log_path: Path) -> dict[str, Any]:
    record = {"cmd": cmd, "cwd": str(cwd), "log_path": str(log_path), "dry_run": dry_run}
    if dry_run:
        return {**record, "status": "skipped_by_policy", "returncode": None}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pythonpath_parts = [str(cwd), str(cwd / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    record["returncode"] = int(proc.returncode)
    record["status"] = "success" if proc.returncode == 0 else "model_failed"
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return record


def _method_config(
    method: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
    model_id: str | None = None,
    include_method: bool = True,
) -> dict[str, Any]:
    cfg = dict(method.get("config") or {})
    if model_id:
        cfg.update((method.get("model_configs") or {}).get(model_id, {}))
    if include_method:
        cfg["sparse_method"] = method["sparse_method"]
    compressor_path = compressor_path_for(model or {}, method)
    if method.get("requires_compressor") and compressor_path:
        cfg["deltakv_checkpoint_path"] = compressor_path
    return cfg


def _tensor_parallel_size_from_config(*configs: dict[str, Any] | None) -> int:
    for cfg in configs:
        if cfg and "tensor_parallel_size" in cfg:
            value = int(cfg["tensor_parallel_size"])
            if value <= 0:
                raise ValueError(f"tensor_parallel_size must be > 0, got {value}.")
            return value
    return 1


def _decode_cuda_graph_for_method(
    method: dict[str, Any],
    requested: bool,
    *,
    tensor_parallel_size: int = 1,
) -> bool:
    if not requested:
        return False
    if int(tensor_parallel_size) > 1:
        if not is_tp_decode_cuda_graph_supported(method["sparse_method"]):
            raise ValueError(
                "decode_cuda_graph with tensor_parallel_size > 1 is a v1 gate for "
                "vanilla, streamingllm, snapkv, pyramidkv, omnikv, rkv, and skipkv only; "
                f"got sparse_method={method['sparse_method']!r}."
            )
        return True
    return is_decode_cuda_graph_supported(method["sparse_method"])


def _quality_command(
    *,
    model_id: str,
    method_id: str,
    model: dict[str, Any],
    method: dict[str, Any],
    quality: dict[str, Any],
    performance: dict[str, Any] | None = None,
    output_root: Path,
) -> list[str]:
    cfg = _method_config(method, model=model, model_id=model_id)
    # Quality runs only the SparseVLLM backend.  HF reference keys are consumed
    # by the logits comparator and should not be forwarded to SparseVLLM config.
    cfg.pop("hf_sparse_method", None)
    tensor_parallel_size = _tensor_parallel_size_from_config(quality, performance)
    cfg["tensor_parallel_size"] = int(tensor_parallel_size)
    cfg["decode_cuda_graph"] = _decode_cuda_graph_for_method(
        method,
        bool((performance or {}).get("decode_cuda_graph", False)),
        tensor_parallel_size=tensor_parallel_size,
    )
    if tensor_parallel_size > 1:
        cfg["decode_cuda_graph_capture_sampling"] = False
    cfg["enforce_eager"] = bool((performance or {}).get("enforce_eager", False))
    if "sparsevllm_max_num_seqs_in_batch" in quality:
        cfg["max_num_seqs_in_batch"] = int(quality["sparsevllm_max_num_seqs_in_batch"])
    if "sparsevllm_max_decoding_seqs" in quality:
        cfg["max_decoding_seqs"] = int(quality["sparsevllm_max_decoding_seqs"])
    return [
        sys.executable,
        "benchmark/long_bench/pred.py",
        "--model",
        f"{model_id}-{method_id}",
        "--model_path",
        model["model_path"],
        "--tokenizer_path",
        model["tokenizer_path"],
        "--ws",
        str(int(quality.get("worker_world_size", quality.get("ws", 1)))),
        "--batch_size",
        str(int(quality.get("batch_size", 1))),
        "--backend",
        "sparsevllm",
        "--sparse_method",
        method["sparse_method"],
        "--task",
        ",".join(quality["tasks"]),
        "--min_prompt_tokens",
        str(int(quality["min_prompt_tokens"])),
        "--samples_per_task",
        str(int(quality["samples_per_task"])),
        "--min_required_samples",
        str(int(quality["min_required_samples"])),
        "--temperature",
        str(float(quality["temperature"])),
        "--top_p",
        str(float(quality["top_p"])),
        "--top_k",
        str(int(quality["top_k"])),
        "--hyper_param",
        json.dumps(cfg, sort_keys=True),
        "--output_root",
        str(output_root),
    ]


def _logits_command(
    *,
    model_id: str | None = None,
    model: dict[str, Any],
    method: dict[str, Any],
    logits: dict[str, Any],
    performance: dict[str, Any] | None = None,
    output_dir: Path,
) -> list[str]:
    cfg = _method_config(method, model=model, model_id=model_id)
    cmd = [
        sys.executable,
        "scripts/debug/compare_logits_hf_sparsevllm.py",
        "--model_path",
        model["model_path"],
        "--output_dir",
        str(output_dir),
        "--cases",
        str(logits["cases"]),
        "--methods",
        method["sparse_method"],
        "--sparse_method",
        method["sparse_method"],
        "--hf_sparse_method",
        cfg.get("hf_sparse_method", cfg.get("sparse_method", method["sparse_method"])),
        "--longbench_task",
        str(logits["longbench_task"]),
        "--longbench_sample_idx",
        str(int(logits["longbench_sample_idx"])),
        "--teacher_forced_decode_steps",
        str(int(logits["teacher_forced_decode_steps"])),
    ]
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible:
        cmd.extend(["--cuda_visible_devices", visible])
    compressor_path = compressor_path_for(model, method)
    if compressor_path:
        cmd.extend(["--compressor_path", compressor_path])
    if _decode_cuda_graph_for_method(method, bool((performance or {}).get("decode_cuda_graph", False))):
        cmd.append("--decode_cuda_graph")

    arg_map = {
        "decode_keep_tokens": "--decode_keep_tokens",
        "sink_keep_tokens": "--sink_keep_tokens",
        "recent_keep_tokens": "--recent_keep_tokens",
        "snapkv_window_size": "--snapkv_window_size",
        "full_attention_layers": "--full_attention_layers",
        "deltakv_center_ratio": "--deltakv_center_ratio",
        "deltakv_neighbor_count": "--deltakv_neighbor_count",
        "deltakv_latent_dim": "--deltakv_latent_dim",
        "deltakv_latent_quant_bits": "--deltakv_latent_quant_bits",
        "deltakv_latent_quant_group_size": "--deltakv_latent_quant_group_size",
        "full_layer_kv_quant_bits": "--full_layer_kv_quant_bits",
        "full_layer_kivi_group_size": "--full_layer_kivi_group_size",
        "full_layer_kivi_residual_length": "--full_layer_kivi_residual_length",
        "engine_prefill_chunk_size": "--engine_prefill_chunk_size",
        "gpu_memory_utilization": "--gpu_memory_utilization",
        "deltakv_full_pool_reserve_ratio": "--deltakv_full_pool_reserve_ratio",
    }
    for key, flag in arg_map.items():
        if key in cfg:
            cmd.extend([flag, str(cfg[key])])
    if cfg.get("use_compression") is False:
        cmd.append("--no-use_compression")
    return cmd


def _perf_command(
    *,
    model_id: str,
    model: dict[str, Any],
    method_id: str,
    method: dict[str, Any],
    performance: dict[str, Any],
    output_jsonl: Path,
) -> list[str]:
    tensor_parallel_size = _tensor_parallel_size_from_config(performance)
    hyper_params = {
        "enforce_eager": bool(performance["enforce_eager"]),
        "decode_cuda_graph": _decode_cuda_graph_for_method(
            method,
            bool(performance["decode_cuda_graph"]),
            tensor_parallel_size=tensor_parallel_size,
        ),
        "tensor_parallel_size": int(tensor_parallel_size),
        "throughput_log_interval_s": 0.0,
    }
    if tensor_parallel_size > 1:
        hyper_params["decode_cuda_graph_capture_sampling"] = False
    method_cfg = _method_config(method, model=model, model_id=model_id, include_method=False)
    # HF reference routing is only meaningful for the logits comparator.  Do
    # not forward it into SparseVLLM perf runs, where unknown keys fail fast.
    method_cfg.pop("hf_sparse_method", None)
    hyper_params.update(method_cfg)
    methods_arg = "vanilla" if method_id == "vanilla" else f"vanilla,{method_id}"
    return [
        sys.executable,
        "scripts/benchmarks/bench_sparse_vllm.py",
        "--model_path",
        model["model_path"],
        "--lengths",
        ",".join(str(int(x)) for x in performance["lengths"]),
        "--batch_sizes",
        ",".join(str(int(x)) for x in performance["batch_sizes"]),
        "--methods",
        methods_arg,
        "--output_len",
        str(int(performance["output_len"])),
        "--temperature",
        "0.0",
        "--hyper_params",
        json.dumps(hyper_params, sort_keys=True),
        "--output_jsonl",
        str(output_jsonl),
    ]


def _stress_command(
    *,
    model_id: str,
    model: dict[str, Any],
    method_id: str,
    method: dict[str, Any],
    performance: dict[str, Any],
    stress: dict[str, Any],
    output_jsonl: Path,
) -> list[str]:
    request_counts = [int(x) for x in stress["request_counts"]]
    tensor_parallel_size = _tensor_parallel_size_from_config(stress, performance)
    hyper_params = {
        "enforce_eager": bool(performance.get("enforce_eager", False)),
        "decode_cuda_graph": _decode_cuda_graph_for_method(
            method,
            bool(performance.get("decode_cuda_graph", True)),
            tensor_parallel_size=tensor_parallel_size,
        ),
        "tensor_parallel_size": int(tensor_parallel_size),
        "throughput_log_interval_s": 0.0,
        "max_num_seqs_in_batch": int(stress.get("max_num_seqs_in_batch", max(request_counts))),
        "max_decoding_seqs": int(stress.get("max_decoding_seqs", max(request_counts))),
    }
    if tensor_parallel_size > 1:
        hyper_params["decode_cuda_graph_capture_sampling"] = False
    method_cfg = _method_config(method, model=model, model_id=model_id, include_method=False)
    # HF reference routing is only meaningful for the logits comparator.  Do
    # not forward it into SparseVLLM stress runs, where unknown keys fail fast.
    method_cfg.pop("hf_sparse_method", None)
    hyper_params.update(method_cfg)
    return [
        sys.executable,
        "scripts/benchmarks/bench_sparse_vllm.py",
        "--model_path",
        model["model_path"],
        "--lengths",
        str(int(stress["length"])),
        "--batch_sizes",
        ",".join(str(value) for value in request_counts),
        "--methods",
        method_id,
        "--output_len",
        str(int(stress["output_len"])),
        "--temperature",
        "0.0",
        "--hyper_params",
        json.dumps(hyper_params, sort_keys=True),
        "--max_decode_steps_after_full",
        str(int(stress["max_decode_steps_after_full"])),
        "--output_jsonl",
        str(output_jsonl),
    ]


def _scbench_command(
    *,
    manifest_path: Path,
    model_id: str,
    method_ids: list[str],
    scbench: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/benchmarks/run_scbench_sparsevllm_methods.py",
        "--manifest",
        str(manifest_path),
        "--model_id",
        model_id,
        "--methods",
        ",".join(method_ids),
        "--tasks",
        ",".join(str(task) for task in scbench["tasks"]),
        "--output_dir",
        str(output_dir),
        "--num_eval_examples",
        str(int(scbench["num_eval_examples"])),
        "--max_turns",
        str(int(scbench["max_turns"])),
        "--max_seq_length",
        str(int(scbench["max_seq_length"])),
        "--batch_size",
        str(int(scbench["batch_size"])),
        "--prefix_cache_block_size",
        str(int(scbench.get("prefix_cache_block_size", 16))),
    ]
    if scbench.get("trust_remote_code", False):
        cmd.append("--trust_remote_code")
    if scbench.get("use_chat_template", False):
        cmd.append("--use_chat_template")
    if scbench.get("disable_golden_context", False):
        cmd.append("--disable_golden_context")
    if "context_min_tokens" in scbench:
        cmd.extend(["--context_min_tokens", str(int(scbench["context_min_tokens"]))])
    if "context_max_tokens" in scbench:
        cmd.extend(["--context_max_tokens", str(int(scbench["context_max_tokens"]))])
    if "gpu_memory_utilization" in scbench:
        cmd.extend(["--gpu_memory_utilization", str(float(scbench["gpu_memory_utilization"]))])
    return cmd


def _load_result_json(path: Path) -> dict[str, Any] | None:
    result_path = path / "result.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _overall_score(result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    value = result.get("overall_category_avg")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _grade_quality_pair(vanilla_root: Path, sparse_root: Path) -> GateGrade:
    vanilla_score = _overall_score(_load_result_json(vanilla_root))
    sparse_score = _overall_score(_load_result_json(sparse_root))
    if vanilla_score is None or sparse_score is None:
        return GateGrade(
            "quality",
            "D",
            "failed",
            {"vanilla_score": vanilla_score, "sparse_score": sparse_score},
            "Missing LongBench-mini aggregate score.",
        )
    return grade_quality(vanilla_score, sparse_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Sparse-VLLM regression gates.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--layer", default="validate", choices=["validate", "quality", "logits", "perf", "stress", "scbench", "nightly", "pre-refactor"])
    parser.add_argument("--models", default=None, help="Comma-separated model ids from the manifest.")
    parser.add_argument("--methods", default=None, help="Comma-separated method ids from the manifest.")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help=(
            "Override Sparse-VLLM engine tensor_parallel_size for regression commands. "
            "This is separate from LongBench --ws data-worker parallelism."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_skipped_policy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    resolved = resolve_manifest_paths(manifest)
    if args.tensor_parallel_size is not None:
        if int(args.tensor_parallel_size) <= 0:
            raise ValueError(f"--tensor_parallel_size must be > 0, got {args.tensor_parallel_size}.")
        resolved.setdefault("performance", {})["tensor_parallel_size"] = int(args.tensor_parallel_size)
    model_ids, method_ids = select_entries(
        resolved,
        [item for item in (args.models or "").split(",") if item] or None,
        [item for item in (args.methods or "").split(",") if item] or None,
    )

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or DEFAULT_OUTPUT_ROOT) / "sparsevllm_regression" / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "resolved_manifest.json", resolved)
    for jsonl_name in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "perf.jsonl"):
        (output_root / jsonl_name).write_text("", encoding="utf-8")

    summary: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "layer": args.layer,
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "models": model_ids,
        "methods": method_ids,
        "tensor_parallel_size": _tensor_parallel_size_from_config(resolved.get("performance")),
        "dry_run": bool(args.dry_run),
        "grades": [],
        "commands": [],
        "skipped": [],
    }
    metrics_records: list[dict[str, Any]] = []
    logits_records: list[dict[str, Any]] = []
    memory_records: list[dict[str, Any]] = []
    stress_records: list[dict[str, Any]] = []
    scbench_records: list[dict[str, Any]] = []

    cwd = Path.cwd()
    try:
        if args.layer == "validate":
            summary["status"] = "completed"
            _write_json(output_root / "metrics.json", {"records": metrics_records})
            _write_json(output_root / "logits_alignment.json", {"records": logits_records})
            _write_json(output_root / "memory.json", {"records": memory_records})
            _write_json(output_root / "stress.json", {"records": stress_records})
            _write_json(output_root / "scbench.json", {"records": scbench_records})
            _write_json(output_root / "grade_summary.json", summary)
            _ensure_artifacts(output_root, list(resolved["outputs"]))
            print(f"[validate] manifest ok: {output_root}")
            return 0

        selected_pairs: list[tuple[str, str]] = []
        for model_id in model_ids:
            for method_id in method_ids:
                missing = missing_runtime_inputs(resolved, model_id, method_id)
                if missing:
                    record = {
                        "model": model_id,
                        "method": method_id,
                        "status": "skipped_by_policy",
                        "missing": missing,
                    }
                    summary["skipped"].append(record)
                    if not args.allow_skipped_policy:
                        raise FileNotFoundError(f"Missing runtime inputs for {model_id}/{method_id}: {missing}")
                    continue
                selected_pairs.append((model_id, method_id))

        run_quality = args.layer in {"quality", "nightly", "pre-refactor"}
        run_logits = args.layer in {"logits", "nightly", "pre-refactor"}
        run_perf = args.layer in {"perf", "nightly", "pre-refactor"}
        run_stress = args.layer in {"stress", "pre-refactor"}
        run_scbench = args.layer == "scbench"

        quality_roots: dict[tuple[str, str], Path] = {}
        if run_quality:
            for model_id, method_id in selected_pairs:
                model = resolved["models"][model_id]
                method = resolved["methods"][method_id]
                out_dir = output_root / "quality" / model_id / method_id
                cmd = _quality_command(
                    model_id=model_id,
                    method_id=method_id,
                    model=model,
                    method=method,
                    quality=resolved["quality"],
                    performance=resolved["performance"],
                    output_root=out_dir,
                )
                summary["commands"].append(
                    _run_command(cmd, cwd=cwd, dry_run=args.dry_run, log_path=out_dir / "run.log")
                )
                quality_roots[(model_id, method_id)] = out_dir
                _append_jsonl_file(
                    output_root / "raw_outputs.jsonl",
                    out_dir / "raw_outputs.jsonl",
                    {"model": model_id, "method": method_id},
                )
                _append_jsonl_file(
                    output_root / "parsed_outputs.jsonl",
                    out_dir / "parsed_outputs.jsonl",
                    {"model": model_id, "method": method_id},
                )
                _append_jsonl_file(
                    output_root / "sample_results.jsonl",
                    out_dir / "sample_results.jsonl",
                    {"model": model_id, "method": method_id},
                )
                result = _load_result_json(out_dir)
                if result is not None:
                    metrics_records.append({"model": model_id, "method": method_id, "result": result})

            for model_id in model_ids:
                vanilla_root = quality_roots.get((model_id, "vanilla"))
                if vanilla_root is None:
                    continue
                for method_id in method_ids:
                    if method_id == "vanilla" or (model_id, method_id) not in quality_roots:
                        continue
                    grade = _grade_quality_pair(vanilla_root, quality_roots[(model_id, method_id)])
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})

        if run_logits:
            for model_id, method_id in selected_pairs:
                method = resolved["methods"][method_id]
                if not method.get("hf_logits_reference"):
                    grade = grade_logits(None)
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                out_dir = output_root / "logits" / model_id / method_id
                cmd = _logits_command(
                    model_id=model_id,
                    model=resolved["models"][model_id],
                    method=method,
                    logits=resolved["logits"],
                    performance=resolved["performance"],
                    output_dir=out_dir,
                )
                summary["commands"].append(
                    _run_command(cmd, cwd=cwd, dry_run=args.dry_run, log_path=out_dir / "run.log")
                )
                summary_path = out_dir / "summary.json"
                metrics = None
                if summary_path.exists():
                    with summary_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    logits_records.append({"model": model_id, "method": method_id, "summary": payload})
                    if payload.get("results"):
                        metrics = payload["results"][0].get("comparisons")
                grade = grade_logits(metrics, p99_threshold=resolved["logits"].get("p99_abs_diff_threshold"))
                summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})

        if run_perf:
            for model_id in model_ids:
                method_ids_for_model = [
                    method_id
                    for pair_model_id, method_id in selected_pairs
                    if pair_model_id == model_id
                ]
                if not method_ids_for_model:
                    continue
                for method_id in method_ids_for_model:
                    out_path = output_root / "perf" / model_id / f"{method_id}.jsonl"
                    cmd = _perf_command(
                        model_id=model_id,
                        model=resolved["models"][model_id],
                        method_id=method_id,
                        method=resolved["methods"][method_id],
                        performance=resolved["performance"],
                        output_jsonl=out_path,
                    )
                    summary["commands"].append(
                        _run_command(
                            cmd,
                            cwd=cwd,
                            dry_run=args.dry_run,
                            log_path=output_root / "perf" / model_id / f"{method_id}.log",
                        )
                    )
                    rows = _read_jsonl(out_path)
                    for row in rows:
                        _append_jsonl(output_root / "perf.jsonl", {"model": model_id, **row})
                    vanilla_by_shape = {
                        (row["length"], row["batch_size"]): row
                        for row in rows
                        if row.get("method") == "vanilla" and row.get("status") == "SUCCESS"
                    }
                    for row in rows:
                        if row.get("method") == "vanilla" or row.get("status") != "SUCCESS":
                            continue
                        vanilla = vanilla_by_shape.get((row["length"], row["batch_size"]))
                        if not vanilla:
                            continue
                        speedup = float(row["decode_tp"]) / max(float(vanilla["decode_tp"]), 1e-9)
                        tensor_parallel_size = _tensor_parallel_size_from_config(resolved.get("performance"))
                        grade = grade_perf(
                            speedup,
                            graph_expected=bool(row.get("decode_cuda_graph_expected")),
                            graph_active=bool(row.get("decode_cuda_graph_active")),
                            require_speedup=tensor_parallel_size <= 1,
                        )
                        summary["grades"].append(
                            {
                                **grade.to_dict(),
                                "model": model_id,
                                "method": row["method"],
                                "length": row["length"],
                                "batch_size": row["batch_size"],
                            }
                        )
                        accounting = row.get("memory_accounting") or {}
                        expected = resolved["methods"].get(row["method"], {}).get("memory", {}).get("expected_savings")
                        observed = accounting.get("observed_savings")
                        mem_grade = grade_memory(expected_savings=expected, observed_savings=observed)
                        memory_record = {
                            "model": model_id,
                            "method": row["method"],
                            "length": row["length"],
                            "batch_size": row["batch_size"],
                            "memory_accounting": accounting,
                            "grade": mem_grade.to_dict(),
                        }
                        memory_records.append(memory_record)
                        summary["grades"].append(
                            {
                                **mem_grade.to_dict(),
                                "model": model_id,
                                "method": row["method"],
                                "length": row["length"],
                                "batch_size": row["batch_size"],
                            }
                        )

        if run_stress:
            for model_id, method_id in selected_pairs:
                out_path = output_root / "stress" / model_id / f"{method_id}.jsonl"
                cmd = _stress_command(
                    model_id=model_id,
                    model=resolved["models"][model_id],
                    method_id=method_id,
                    method=resolved["methods"][method_id],
                    performance=resolved["performance"],
                    stress=resolved["stress"],
                    output_jsonl=out_path,
                )
                summary["commands"].append(
                    _run_command(
                        cmd,
                        cwd=cwd,
                        dry_run=args.dry_run,
                        log_path=output_root / "stress" / model_id / f"{method_id}.log",
                    )
                )
                rows = _read_jsonl(out_path)
                if args.dry_run:
                    grade = GateGrade("stress", "N/A", "skipped_by_policy", {}, "dry run")
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                if not rows:
                    grade = grade_stress(
                        completed=False,
                        crashed=True,
                        preemptions=0,
                        full_admission_window=False,
                        utilization_ok=False,
                    )
                    stress_records.append({"model": model_id, "method": method_id, "rows": [], "grade": grade.to_dict()})
                    summary["grades"].append({**grade.to_dict(), "model": model_id, "method": method_id})
                    continue
                for row in rows:
                    if row.get("status") == "SKIPPED_BY_POLICY":
                        grade = GateGrade(
                            "stress",
                            "N/A",
                            "skipped_by_policy",
                            row,
                            str(row.get("reason") or "stress case skipped by policy"),
                        )
                    else:
                        grade = grade_stress(
                            completed=row.get("status") == "SUCCESS",
                            crashed=row.get("status") != "SUCCESS",
                            preemptions=int(row.get("scheduler_preemptions", 0) or 0),
                            full_admission_window=bool(row.get("full_admission_reached")),
                            utilization_ok=bool(row.get("utilization_ok", False)),
                        )
                    stress_record = {
                        "model": model_id,
                        "method": method_id,
                        "length": row.get("length"),
                        "batch_size": row.get("batch_size"),
                        "row": row,
                        "grade": grade.to_dict(),
                    }
                    stress_records.append(stress_record)
                    summary["grades"].append(
                        {
                            **grade.to_dict(),
                            "model": model_id,
                            "method": method_id,
                            "length": row.get("length"),
                            "batch_size": row.get("batch_size"),
                        }
                    )

        if run_scbench:
            scbench = resolved["scbench"]
            scbench_model_id = str(scbench["model"])
            selected_pair_set = set(selected_pairs)
            method_ids_for_scbench = [
                method_id
                for method_id in scbench["methods"]
                if method_id in method_ids and (scbench_model_id, method_id) in selected_pair_set
            ]
            if scbench_model_id not in model_ids or not method_ids_for_scbench:
                summary["skipped"].append(
                    {
                        "model": scbench_model_id,
                        "methods": method_ids_for_scbench,
                        "status": "skipped_by_policy",
                        "reason": "SCBench configured model/methods are not selected or lack runtime inputs.",
                    }
                )
            else:
                out_dir = output_root / "scbench" / scbench_model_id
                manifest_path = Path(args.manifest) if args.manifest else Path(__file__).with_name("manifest.json")
                cmd = _scbench_command(
                    manifest_path=manifest_path,
                    model_id=scbench_model_id,
                    method_ids=method_ids_for_scbench,
                    scbench=scbench,
                    output_dir=out_dir,
                )
                summary["commands"].append(
                    _run_command(cmd, cwd=cwd, dry_run=args.dry_run, log_path=out_dir / "run.log")
                )
                summary_path = out_dir / "scbench_methods_summary.json"
                if summary_path.exists():
                    with summary_path.open("r", encoding="utf-8") as handle:
                        scbench_summary = json.load(handle)
                    scbench_records.append(scbench_summary)
                    for sample_path in sorted(out_dir.glob("*/*/sample_results_*_multi_turn.jsonl")):
                        _append_jsonl_file(
                            output_root / "sample_results.jsonl",
                            sample_path,
                            {"model": scbench_model_id},
                        )

        grade_objs = [
            GateGrade(item["name"], item["grade"], item["status"], item["metrics"], item.get("reason", ""))
            for item in summary["grades"]
        ]
        summary["worst_required_grade"] = worst_required_grade(grade_objs)
        summary["status"] = "completed"
        _write_json(output_root / "metrics.json", {"records": metrics_records})
        _write_json(output_root / "logits_alignment.json", {"records": logits_records})
        _write_json(output_root / "memory.json", {"records": memory_records})
        _write_json(output_root / "stress.json", {"records": stress_records})
        _write_json(output_root / "scbench.json", {"records": scbench_records})
        _write_json(output_root / "grade_summary.json", summary)
        _ensure_artifacts(output_root, list(resolved["outputs"]))
        print(f"[done] wrote {output_root}")
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        _write_json(output_root / "metrics.json", {"records": metrics_records})
        _write_json(output_root / "logits_alignment.json", {"records": logits_records})
        _write_json(output_root / "memory.json", {"records": memory_records})
        _write_json(output_root / "stress.json", {"records": stress_records})
        _write_json(output_root / "scbench.json", {"records": scbench_records})
        _write_json(output_root / "grade_summary.json", summary)
        _ensure_artifacts(output_root, list(resolved["outputs"]))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
