import os
import torch
import argparse
import multiprocessing as mp
import traceback
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from deltakv.configs.runtime_params import normalize_runtime_params
from sparsevllm.method_registry import is_decode_cuda_graph_supported, normalize_sparse_method


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) # GB


def _load_json_arg(value: str) -> dict[str, Any]:
    """Load a JSON object from a CLI arg.

    Supports:
      - Inline JSON: '{"gpu_memory_utilization": 0.9}'
      - File JSON: '@config.json'
    """
    if value is None:
        return {}
    value = str(value).strip()
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        value = path.read_text(encoding="utf-8")

    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"Invalid JSON for --hyper_params: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("--hyper_params must be a JSON object (dict).")
    return parsed


def _build_engine_hyper_params(args) -> dict[str, Any]:
    # Keep benchmark defaults stable (do not rely on sparsevllm.Config defaults).
    hyper_params: dict[str, Any] = {
        "enforce_eager": False,
        "decode_cuda_graph": True,
        "gpu_memory_utilization": 0.8,
        "engine_prefill_chunk_size": 4096,
        "tensor_parallel_size": 1,
    }

    hyper_params.update(_load_json_arg(args.hyper_params))

    normalized = normalize_runtime_params(hyper_params, backend="sparsevllm")
    for warning in normalized.warnings:
        print(f"[param-normalize] {warning}")

    return hyper_params


def _write_jsonl(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
            handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
            handle.write("\n")


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _git_metadata() -> dict[str, Any]:
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }


def _selected_env_snapshot() -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "SVLLM_BENCHMARK_OUTPUT_DIR",
        "SVLLM_BENCHMARK_DATA_DIR",
        "DELTAKV_OUTPUT_DIR",
        "DELTAKV_DATA_DIR",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _standard_status(status: Any) -> str:
    normalized = str(status or "unknown").strip().upper()
    if normalized == "SUCCESS":
        return "success"
    if normalized == "SKIPPED_BY_POLICY":
        return "skipped_by_policy"
    if normalized in {"FAILED", "OOM"}:
        return "model_failed"
    return "model_failed"


def _artifact_records(args, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        record = dict(row)
        record["case_id"] = idx
        record["benchmark"] = "microbench"
        record["raw_status"] = row.get("status", "UNKNOWN")
        record["status"] = _standard_status(row.get("status"))
        record.setdefault("prompt_tokens", row.get("length"))
        record.setdefault("max_new_tokens", int(args.output_len))
        record.setdefault("sampling_temperature", float(args.temperature))
        record.setdefault("sampling_top_p", float(args.top_p))
        if "prefill_tp" in row:
            record.setdefault("prefill_tok_s", row["prefill_tp"])
        if "decode_tp" in row:
            record.setdefault("decode_tok_s", row["decode_tp"])
        if "ttft" in row:
            record.setdefault("ttft_s", row["ttft"])
        if "itl" in row:
            record.setdefault("itl_ms", row["itl"])
        if "mem" in row:
            record.setdefault("peak_memory_gb", row["mem"])
        records.append(record)
    return records


def _write_output_dir(args, rows: list[dict[str, Any]]) -> None:
    if not args.output_dir:
        return

    output_dir = Path(args.output_dir).expanduser()
    records = _artifact_records(args, rows)
    success_records = [row for row in records if row["status"] == "success"]
    skipped_records = [row for row in records if row["status"] == "skipped_by_policy"]
    failed_records = [row for row in records if row["status"] not in {"success", "skipped_by_policy"}]

    run_info = {
        **_git_metadata(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "methods": [part.strip() for part in args.methods.split(",") if part.strip()],
        "lengths": [int(part) for part in args.lengths.split(",") if part.strip()],
        "batch_sizes": [int(part) for part in args.batch_sizes.split(",") if part.strip()],
        "output_len": int(args.output_len),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "hyper_params": args.hyper_params_dict,
        "env": _selected_env_snapshot(),
    }
    aggregate_status = "success"
    if failed_records:
        aggregate_status = "model_failed"
    elif skipped_records:
        aggregate_status = "skipped_by_policy"

    aggregate = {
        "benchmark": "microbench",
        "status": aggregate_status,
        "num_cases": len(records),
        "success_cases": len(success_records),
        "skipped_cases": len(skipped_records),
        "failed_cases": len(failed_records),
        "records": records,
    }

    report_lines = [
        "# Sparse-vLLM Microbenchmark",
        "",
        f"- Model: `{args.model_path}`",
        f"- Methods: `{args.methods}`",
        f"- Lengths: `{args.lengths}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Output length: `{args.output_len}`",
        "",
        "| Method | Prompt tokens | Batch | Status | TTFT s | Prefill tok/s | Decode tok/s | Peak GB | Decode speedup |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        ok = record["status"] == "success"
        report_lines.append(
            "| {method} | {prompt} | {batch} | {status} | {ttft} | {prefill} | {decode} | {mem} | {speedup} |".format(
                method=record.get("method", ""),
                prompt=record.get("prompt_tokens", ""),
                batch=record.get("batch_size", ""),
                status=record["status"],
                ttft=f"{record.get('ttft_s', 0.0):.3f}" if ok else "",
                prefill=f"{record.get('prefill_tok_s', 0.0):.1f}" if ok else "",
                decode=f"{record.get('decode_tok_s', 0.0):.1f}" if ok else "",
                mem=f"{record.get('peak_memory_gb', 0.0):.2f}" if ok else "",
                speedup=f"{record.get('speedup_vs_vanilla_decode', 0.0):.2f}" if ok else "",
            )
        )

    _write_json(output_dir / "run_info.json", run_info)
    _write_jsonl_rows(output_dir / "performance.jsonl", records)
    _write_jsonl_rows(output_dir / "per_sample_results.jsonl", records)
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _decode_cuda_graph_status(llm) -> dict[str, Any]:
    runner = getattr(getattr(llm, "model_runner", None), "decode_cuda_graph_runner", None)
    states = getattr(runner, "_graphs", {}) if runner is not None else {}
    graph_count = sum(
        1
        for state in states.values()
        if getattr(state, "graph", None) is not None
    )
    configured = bool(getattr(getattr(llm, "config", None), "decode_cuda_graph", False))
    return {
        "decode_cuda_graph_configured": configured,
        "decode_cuda_graph_runner_initialized": runner is not None,
        "decode_cuda_graph_state_count": int(len(states)),
        "decode_cuda_graph_graph_count": int(graph_count),
        "decode_cuda_graph_last_state_key": str(getattr(runner, "last_state_key", None)) if runner is not None else None,
        "decode_cuda_graph_active": bool(configured and graph_count > 0),
    }


def benchmark_task(method, length, bs, args, results_dict):
    # 为每个子进程重置显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    print(f"\n>>> Starting: {method.upper()} | Context: {length} | Batch: {bs}...")
    
    base_hyper_params = args.hyper_params_dict
    sparse_kwargs: dict[str, Any] = {"sparse_method": "vanilla"}
    if method == "vanilla":
        sparse_kwargs["sparse_method"] = "vanilla"
    elif method in (
        "streamingllm",
        "attention-sink",
        "attention_sink",
        "snapkv",
        "pyramidkv",
        "omnikv",
        "quest",
        "rkv",
        "r-kv",
        "r_kv",
        "skipkv",
        "skip-kv",
        "skip_kv",
        "deltakv",
    ):
        sparse_kwargs["sparse_method"] = method
    elif "deltakv" in method:
        sparse_kwargs["sparse_method"] = method

    normalized_method = normalize_sparse_method(sparse_kwargs["sparse_method"])
    if bool(base_hyper_params.get("decode_cuda_graph")) and not is_decode_cuda_graph_supported(normalized_method):
        results_dict[(method, length, bs)] = {
            "method": method,
            "sparse_method": normalized_method,
            "length": int(length),
            "batch_size": int(bs),
            "status": "SKIPPED_BY_POLICY",
            "reason": f"decode_cuda_graph is not supported for sparse_method={normalized_method!r}.",
            "decode_cuda_graph_expected": True,
            "decode_cuda_graph_active": False,
        }
        print(
            f"[{method.upper()}] SKIPPED_BY_POLICY: decode_cuda_graph is not supported "
            f"for sparse_method={normalized_method!r}."
        )
        return
    
    llm = None
    try:
        m_len = length + args.output_len + 100
        # Note: max_model_len is derived from (length, bs, output_len, engine_prefill_chunk_size).
        # They can be passed in --hyper_params, but will be overwritten here to keep the benchmark consistent.
        hyper_params = dict(base_hyper_params)
        hyper_params.pop("max_model_len", None)
        hyper_params.setdefault("max_num_seqs_in_batch", int(bs))
        hyper_params.setdefault("max_decoding_seqs", int(bs))
        
        from sparsevllm import LLM, SamplingParams
        engine_kwargs = {
            **hyper_params,
            "max_model_len": m_len,
            **sparse_kwargs,
        }
        llm = LLM(args.model_path, **engine_kwargs)

        prompt_token_ids = [[100] * length for _ in range(bs)]
        sampling_params = [
            SamplingParams(
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                ignore_eos=True,
                max_tokens=args.output_len,
            )
            for _ in range(bs)
        ]
        admission_wave_size = int(getattr(args, "admission_wave_size", 0) or 0)
        staged_admission = 0 < admission_wave_size < bs
        wave_decode_gap_steps = int(getattr(args, "wave_decode_gap_steps", 0) or 0)

        # --- 关键修改：重置并开始正式测量 ---
        from sparsevllm.utils.profiler import profiler
        profiler.reset()
        
        torch.cuda.synchronize()

        prefill_tokens = 0
        decode_tokens = 0
        prefill_times = []
        decode_times = []
        ttft = None
        decode_tokens_after_full = 0
        decode_times_after_full = []
        decode_bs_after_full = []
        full_admission_reached = not staged_admission
        impossible_full_admission = False
        decode_steps_after_full = 0
        
        t_start = perf_counter()
        decode_started = False

        # Manually run the generation loop to get detailed stats
        next_request_idx = 0
        decode_steps_since_last_wave = 0

        def add_wave(max_new_requests: int):
            nonlocal next_request_idx, decode_steps_since_last_wave
            end_idx = min(bs, next_request_idx + max_new_requests)
            for req_idx in range(next_request_idx, end_idx):
                llm.add_request(prompt_token_ids[req_idx], sampling_params[req_idx])
            added = end_idx - next_request_idx
            next_request_idx = end_idx
            decode_steps_since_last_wave = 0
            return added

        add_wave(admission_wave_size if staged_admission else bs)

        has_queued = False
        zero_steps = 0
        while not llm.is_finished():
            if (
                staged_admission
                and next_request_idx < bs
                and len(llm.scheduler.waiting) == 0
                and len(llm.scheduler.decoding) > 0
                and decode_steps_since_last_wave >= wave_decode_gap_steps
            ):
                add_wave(admission_wave_size)

            step_start = perf_counter()
            finished_outputs, num_tokens = llm.step()
            step_dt = perf_counter() - step_start
            
            if num_tokens > 0:
                prefill_tokens += num_tokens
                prefill_times.append(step_dt)
                if decode_started:
                    has_queued = True
                zero_steps = 0
                # In this engine, the first completion token is sampled during the *last* prefill
                # step of each sequence (Scheduler.postprocess appends it in the prefill branch).
                # So TTFT should be captured on a prefill step, not on the first decode step.
                if ttft is None and (llm.scheduler.decoding or any(tids for _, tids in finished_outputs)):
                    ttft = perf_counter() - t_start
            elif num_tokens < 0:
                # print(f'one decode step ... {perf_counter() - last_time}')
                decode_started = True
                decode_steps_since_last_wave += 1
                decode_times.append(step_dt)
                decode_tokens += (-num_tokens)
                if full_admission_reached:
                    decode_times_after_full.append(step_dt)
                    decode_tokens_after_full += (-num_tokens)
                    decode_bs_after_full.append(len(llm.scheduler.decoding))
                    decode_steps_after_full += 1
                zero_steps = 0
                # Fallback: if output_len==0/1 or internal behavior changes, ensure TTFT is set.
                if ttft is None:
                    ttft = perf_counter() - t_start
            else:
                zero_steps += 1
                if zero_steps >= 50:
                    raise RuntimeError("llm.step() returned 0 tokens repeatedly; scheduler may be stuck.")

            if staged_admission and not full_admission_reached and next_request_idx == bs:
                if len(llm.scheduler.waiting) == 0 and len(llm.scheduler.decoding) == bs:
                    full_admission_reached = True
                elif finished_outputs:
                    impossible_full_admission = True
                    break

            max_decode_steps_after_full = int(getattr(args, "max_decode_steps_after_full", 0) or 0)
            if full_admission_reached and max_decode_steps_after_full > 0 and decode_steps_after_full >= max_decode_steps_after_full:
                break

        print(f'@@@ {decode_tokens=}')
                
        torch.cuda.synchronize()
        t_end = perf_counter()
        
        duration = t_end - t_start
        peak_mem = get_peak_memory()
        graph_status = _decode_cuda_graph_status(llm)
        preemptions = int(getattr(getattr(llm, "scheduler", None), "total_preemptions", 0) or 0)
        cache_manager = getattr(getattr(llm, "model_runner", None), "cache_manager", None)
        if cache_manager is None or not hasattr(cache_manager, "memory_accounting"):
            raise RuntimeError("Sparse-VLLM cache manager does not expose memory_accounting().")
        memory_accounting = cache_manager.memory_accounting()

        ttft = float(ttft or 0.0)
        prefill_s = sum(prefill_times)
        decode_s = sum(decode_times)

        print(f'[debug] {prefill_tokens=} {prefill_s=} {ttft=} {decode_tokens=} {decode_s=} {has_queued=}')
        prefill_tp = prefill_tokens / prefill_s if prefill_s > 0 else 0
        used_full_admission_window = bool(decode_times_after_full)
        decode_s_effective = sum(decode_times_after_full) if used_full_admission_window else decode_s
        decode_tokens_effective = decode_tokens_after_full if used_full_admission_window else decode_tokens
        decode_tp = decode_tokens_effective / decode_s_effective if decode_s_effective > 0 else 0
        # ITL (Inter-token Latency) 是用户感知的生成速度：总解码时间 / 单序列平均生成的 token 数
        avg_itl = (decode_s_effective / (decode_tokens_effective / bs) * 1000) if decode_tokens_effective > 0 else 0
        avg_active_bs = (
            sum(decode_bs_after_full) / len(decode_bs_after_full)
            if decode_bs_after_full
            else (decode_tokens / len(decode_times) if decode_times else 0)
        )
        
        stage_mode = (
            f" | AdmissionWave: {admission_wave_size}"
            f" | WaveGapSteps: {wave_decode_gap_steps}"
            f" | FullAdmit: {'yes' if full_admission_reached else 'no'}"
            f" | DecodeScope: {'full' if used_full_admission_window else 'fallback'}"
            if staged_admission
            else ""
        )
        print(f"[{method.upper()}] TTFT: {ttft:.2f}s | Prefill: {prefill_tp:.2f} tok/s | Decode: {decode_tp:.2f} tok/s | ITL: {avg_itl:.2f}ms | AvgBS: {avg_active_bs:.1f} | Mem: {peak_mem:.2f} GB{stage_mode}")
        
        results_dict[(method, length, bs)] = {
            "method": method,
            "sparse_method": normalized_method,
            "length": int(length),
            "batch_size": int(bs),
            "prefill_tp": prefill_tp,
            "decode_tp": decode_tp,
            "ttft": ttft,
            "itl": avg_itl,
            "avg_bs": avg_active_bs,
            "mem": peak_mem,
            "has_queued": has_queued,
            "full_admission_reached": full_admission_reached,
            "impossible_full_admission": impossible_full_admission,
            "decode_scope": "full" if used_full_admission_window else "fallback",
            "staged_admission": staged_admission,
            "admission_wave_size": admission_wave_size if staged_admission else None,
            "wave_decode_gap_steps": wave_decode_gap_steps if staged_admission else None,
            "max_decode_steps_after_full": int(getattr(args, "max_decode_steps_after_full", 0) or 0),
            "decode_steps_after_full": int(decode_steps_after_full),
            "scheduler_preemptions": preemptions,
            "decode_cuda_graph_expected": bool(base_hyper_params.get("decode_cuda_graph")),
            **graph_status,
            "memory_accounting": memory_accounting,
            "engine_hyper_params": engine_kwargs,
            "status": "SUCCESS"
        }

    except Exception as e:
        print(f"Error at {method}/{length}/{bs}: {e}")
        traceback.print_exc()
        results_dict[(method, length, bs)] = {
            "method": method,
            "sparse_method": normalized_method,
            "length": int(length),
            "batch_size": int(bs),
            "status": "FAILED",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        if llm is not None and hasattr(llm, "exit"):
            llm.exit()


def main():
    parser = argparse.ArgumentParser(description="Professional benchmark for sparsevllm.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lengths", type=str, default="16000,32000,64000", help="Context lengths to test")
    parser.add_argument("--batch_sizes", type=str, default="4", help="Batch sizes to test")
    parser.add_argument(
        "--methods",
        type=str,
        default="vanilla,snapkv,omnikv",
        help=(
            "Methods to test (vanilla, streamingllm, attention-sink, snapkv, pyramidkv, "
            "omnikv, quest, rkv, skipkv, deltakv; deltakv-less-memory* are legacy aliases)"
        ),
    )
    parser.add_argument("--output_len", type=int, default=512, help="Output tokens per request")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. Default 0.0 (greedy) for throughput benchmarking.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p. Only used when temperature > 0.",
    )
    parser.add_argument(
        "--admission_wave_size",
        type=int,
        default=0,
        help="If >0 and < batch size, only admit this many sequences at a time. Decode throughput is then measured after the final wave has fully entered decode.",
    )
    parser.add_argument(
        "--max_decode_steps_after_full",
        type=int,
        default=0,
        help="If >0 in staged mode, stop after this many decode steps after full admission is reached.",
    )
    parser.add_argument(
        "--wave_decode_gap_steps",
        type=int,
        default=0,
        help="In staged admission mode, require this many decode steps before admitting the next wave.",
    )
    parser.add_argument(
        "--hyper_params",
        type=str,
        default="{}",
        help=(
            "LLMEngine/Config hyper-params as JSON (string or @file.json). "
            'Example: \'{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"tensor_parallel_size":1,"decode_keep_tokens":2048}\''
        ),
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional machine-readable JSONL output path with one row per method/length/batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory for run_info.json, performance.jsonl, aggregate_metrics.json, and report.md.",
    )
    
    args = parser.parse_args()
    try:
        args.hyper_params_dict = _build_engine_hyper_params(args)
    except ValueError as e:
        parser.error(str(e))
    
    test_lengths = [int(x) for x in args.lengths.split(",")]
    test_methods = args.methods.split(",")
    test_batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    manager = mp.Manager()
    results_dict = manager.dict()

    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                p = mp.Process(target=benchmark_task, args=(method, length, bs, args, results_dict))
                p.start()
                p.join()

    # 打印最终报表
    print(f"\n\n{'='*140}")
    print(f"{ 'Method':<12} {'Len':<8} {'BS':<4} {'TTFT(s)':<10} {'PreTP':<12} {'DecTP':<12} {'ITL(ms)':<10} {'AvgBS':<8} {'Mem(GB)':<10} {'Speedup'}")
    print("-" * 140)
    
    # 获取 Vanilla 作为基准计算加速比 (按 length 和 BS 匹配)
    vanilla_stats = {}
    for length in test_lengths:
        for bs in test_batch_sizes:
            v_res = results_dict.get(("vanilla", length, bs))
            if v_res and v_res["status"] == "SUCCESS":
                vanilla_stats[(length, bs)] = v_res["decode_tp"]

    jsonl_rows: list[dict[str, Any]] = []
    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                res = results_dict.get((method, length, bs))
                if not res or res["status"] in ["FAILED", "OOM", "SKIPPED_BY_POLICY"]:
                    status_str = res["status"] if res else "UNKNOWN"
                    print(f"{method:<12} {length:<8} {bs:<4} {status_str:<10} {'-':<12} {'-':<12} {'-':<10} {'-':<8} {'-':<10} {'-'}")
                    row = dict(res or {})
                    row.setdefault("method", method)
                    row.setdefault("length", int(length))
                    row.setdefault("batch_size", int(bs))
                    row.setdefault("status", status_str)
                    jsonl_rows.append(row)
                    continue
                
                ttft = res["ttft"]
                pre_tp = res["prefill_tp"]
                dec_tp = res["decode_tp"]
                itl = res["itl"]
                avg_bs = res["avg_bs"]
                mem = res["mem"]
                has_queued = res.get("has_queued", False)
                
                bs_str = f"{bs}*" if has_queued else f"{bs}"
                
                speedup = 1.0
                if (length, bs) in vanilla_stats:
                    speedup = dec_tp / vanilla_stats[(length, bs)]
                
                speedup_str = f"{speedup:.2f}x"
                print(f"{method:<12} {length:<8} {bs_str:<4} {ttft:<10.2f} {pre_tp:<12.1f} {dec_tp:<12.1f} {itl:<10.2f} {avg_bs:<8.1f} {mem:<10.2f} {speedup_str}")
                row = dict(res)
                row["speedup_vs_vanilla_decode"] = float(speedup)
                jsonl_rows.append(row)
    print(f"{ '='*140}\n")
    _write_jsonl(args.output_jsonl, jsonl_rows)
    _write_output_dir(args, jsonl_rows)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
