from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from sparsevllm import LLM, SamplingParams


SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            status = row.get("status")
            if status not in SAMPLE_STATUSES:
                raise ValueError(f"Invalid sample status {status!r}.")
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _graph_status(llm: LLM) -> dict[str, Any]:
    runner = getattr(llm.model_runner, "decode_cuda_graph_runner", None)
    states = getattr(runner, "_graphs", {}) if runner is not None else {}
    graph_count = sum(
        getattr(state, "graph", None) is not None for state in states.values()
    )
    return {
        "configured": bool(llm.config.decode_cuda_graph),
        "state_count": len(states),
        "graph_count": graph_count,
        "active": bool(graph_count),
        "last_state_key": str(getattr(runner, "last_state_key", None)),
    }


def _cache_stats(llm: LLM) -> dict[str, int]:
    manager = getattr(llm.model_runner, "cache_manager", None)
    if manager is None or not hasattr(manager, "free_slot_stats"):
        raise RuntimeError("Cache manager does not expose free_slot_stats().")
    return {
        str(key): int(value)
        for key, value in manager.free_slot_stats().items()
        if isinstance(value, (int, float, bool))
    }


def _provider_names(llm: LLM) -> list[str]:
    names = set()
    for layer in llm.model_runner.model.model.layers:
        moe = getattr(layer, "mlp", None)
        if moe is None:
            moe = getattr(layer, "block_sparse_moe", None)
        experts = getattr(moe, "experts", None)
        if experts is None:
            raise RuntimeError(
                f"Layer {type(layer).__name__} does not expose packed MoE experts."
            )
        names.add(str(experts.provider.name))
    return sorted(names)


def _validate_replica_consistency(llm: LLM) -> list[dict[str, Any]]:
    summaries = llm.debug_sparse_state_summaries()
    failures = []
    for summary in summaries:
        consistency = summary.get("replica_consistency") or {}
        logits_ratio = consistency.get("last_logits_tolerance_ratio")
        if logits_ratio is not None and float(logits_ratio) > 1.0:
            failures.append(
                {
                    "world_rank": summary.get("world_rank"),
                    "field": "last_logits",
                    "value": consistency.get("last_logits_tolerance_ratio"),
                }
            )
        for layer_idx, layer in consistency.get("moe_layers", {}).items():
            if bool(layer.get("topk_ids_mismatch")):
                failures.append(
                    {
                        "world_rank": summary.get("world_rank"),
                        "field": f"layer_{layer_idx}.topk_ids",
                        "value": True,
                    }
                )
            for field in ("topk_weights_tolerance_ratio", "output_tolerance_ratio"):
                if float(layer.get(field, 0.0)) > 1.0:
                    failures.append(
                        {
                            "world_rank": summary.get("world_rank"),
                            "field": f"layer_{layer_idx}.{field}",
                            "value": layer.get(field),
                        }
                    )
    if failures:
        raise RuntimeError(f"Cross-rank MoE consistency validation failed: {failures}")
    return summaries


def _worker_runtime_summaries(llm: LLM) -> list[dict[str, Any]]:
    summaries = llm.debug_sparse_state_summaries()
    if len(summaries) != llm.config.world_size:
        raise RuntimeError(
            "Runtime summary did not cover every worker: "
            f"expected={llm.config.world_size}, got={len(summaries)}."
        )
    return summaries


def _load_reference(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Reference output does not exist: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): list(row["token_ids"]) for row in rows}


def _prompts(tokenizer) -> list[tuple[str, str]]:
    shared_prefix = "Sparse-vLLM 的纯张量并行验证前缀。" * 64
    cases = [
        ("fact", "请只回答：法国的首都是哪里？"),
        ("reasoning", "请计算 17×19，并只给出整数结果。"),
    ]
    prompts = []
    for sample_id, question in cases:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": shared_prefix + "\n" + question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append((sample_id, prompt))
    return prompts


def _generate_sample(
    llm: LLM,
    *,
    sample_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        output = llm.generate(
            [prompt],
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            use_tqdm=False,
        )[0]
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "status": "model_failed",
            "prompt": prompt,
            "text": "",
            "token_ids": [],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    return {
        "sample_id": sample_id,
        "status": "success",
        "prompt": prompt,
        "text": output["text"],
        "token_ids": list(output["token_ids"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--decode-cuda-graph", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--require-prefix-hit", action="store_true")
    parser.add_argument("--check-replica-consistency", action="store_true")
    parser.add_argument("--reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tensor_parallel_size > 1 and (
        args.tensor_parallel_size % args.expert_parallel_size
    ):
        raise ValueError("Outer TP must be divisible by MoE EP.")
    if args.require_prefix_hit and not args.enable_prefix_caching:
        raise ValueError("--require-prefix-hit requires --enable-prefix-caching.")
    if args.check_replica_consistency and args.decode_cuda_graph:
        raise ValueError(
            "Replica debug snapshots are an eager correctness check; run Graph "
            "validation separately."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.check_replica_consistency:
        os.environ["SPARSEVLLM_DEBUG_RUNTIME"] = "1"
        os.environ["SPARSEVLLM_DEBUG_MOE"] = "1"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    reference = _load_reference(args.reference)
    run_info = {
        "command": [sys.executable, *sys.argv],
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "moe_tensor_parallel_size": (
            args.tensor_parallel_size // args.expert_parallel_size
            if args.tensor_parallel_size > 1
            else args.tensor_parallel_size
        ),
        "decode_cuda_graph": args.decode_cuda_graph,
        "enable_prefix_caching": args.enable_prefix_caching,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "temperature": 0.0,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "torch_version": torch.__version__,
    }
    _write_json(output_dir / "run_info.json", run_info)

    llm = None
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any]
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts = _prompts(tokenizer)
        llm = LLM(
            args.model,
            sparse_method="vanilla",
            tensor_parallel_size=args.tensor_parallel_size,
            expert_parallel_size=args.expert_parallel_size,
            data_parallel_size=1,
            enforce_eager=not args.decode_cuda_graph,
            decode_cuda_graph=args.decode_cuda_graph,
            enable_prefix_caching=args.enable_prefix_caching,
            prefix_cache_block_size=16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs_in_batch=1,
            max_decoding_seqs=1,
        )
        cache_before = _cache_stats(llm)
        for sample_id, prompt in prompts:
            row = _generate_sample(
                llm,
                sample_id=sample_id,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            rows.append(row)
            if row["status"] != "success":
                raise RuntimeError(
                    f"Generation failed for sample {sample_id!r}: {row['error']}"
                )
            expected = reference.get(sample_id)
            if args.reference is not None and expected is None:
                row["status"] = "metric_failed"
                row["error"] = "Reference output is missing this sample ID."
            elif expected is not None and row["token_ids"] != expected:
                row["status"] = "metric_failed"
                row["error"] = "Generated token IDs differ from the reference."

        cache_after = _cache_stats(llm)
        cache_delta = {
            key: int(cache_after.get(key, 0)) - int(cache_before.get(key, 0))
            for key in sorted(set(cache_before) | set(cache_after))
        }
        graph_status = _graph_status(llm)
        worker_summaries = _worker_runtime_summaries(llm)
        if args.decode_cuda_graph:
            inactive_workers = [
                summary["world_rank"]
                for summary in worker_summaries
                if not summary["decode_cuda_graph"]["active"]
            ]
            if inactive_workers:
                raise RuntimeError(
                    "Decode CUDA Graph was not active on every worker: "
                    f"inactive_world_ranks={inactive_workers}."
                )
        hit_tokens = int(cache_delta.get("prefix_cache_hit_tokens", 0))
        if args.require_prefix_hit and hit_tokens <= 0:
            raise RuntimeError(f"No real prefix-cache hit was observed: {cache_delta}")
        replica_summaries = (
            _validate_replica_consistency(llm)
            if args.check_replica_consistency
            else None
        )
        failed = sum(row["status"] != "success" for row in rows)
        aggregate = {
            "status": "success" if failed == 0 else "metric_failed",
            "sample_count": len(rows),
            "success_count": len(rows) - failed,
            "failed_count": failed,
            "moe_providers": _provider_names(llm),
            "decode_cuda_graph": graph_status,
            "worker_runtime_summaries": worker_summaries,
            "prefix_cache_stats_before": cache_before,
            "prefix_cache_stats_after": cache_after,
            "prefix_cache_stats_delta": cache_delta,
            "replica_consistency": replica_summaries,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if failed:
            raise RuntimeError("One or more generated samples failed reference comparison.")
    except Exception as exc:
        sample_metric_failed = any(
            row.get("status") == "metric_failed" for row in rows
        )
        if not rows:
            rows.append(
                {
                    "sample_id": "run",
                    "status": "model_failed",
                    "prompt": "",
                    "text": "",
                    "token_ids": [],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        aggregate = {
            "status": "metric_failed" if sample_metric_failed else "model_failed",
            "sample_count": len(rows),
            "success_count": sum(row["status"] == "success" for row in rows),
            "failed_count": sum(row["status"] != "success" for row in rows),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        active_error = sys.exception()
        exit_error = None
        if llm is not None:
            try:
                llm.exit()
            except Exception as exc:
                exit_error = exc
        if exit_error is not None:
            aggregate["shutdown_error"] = repr(exit_error)
            if active_error is None:
                aggregate["status"] = "model_failed"
                aggregate["error"] = f"Engine shutdown failed: {exit_error!r}"
            else:
                active_error.add_note(f"Engine shutdown also failed: {exit_error!r}")
        raw_rows = [
            {
                "sample_id": row["sample_id"],
                "status": row["status"],
                "prompt": row["prompt"],
                "raw_text": row["text"],
                "token_ids": row["token_ids"],
                **({"error": row["error"]} if "error" in row else {}),
            }
            for row in rows
        ]
        parsed_rows = [
            {
                "sample_id": row["sample_id"],
                "status": row["status"],
                "text": row["text"],
                **({"error": row["error"]} if "error" in row else {}),
            }
            for row in rows
        ]
        _write_json(output_dir / "raw_outputs.json", raw_rows)
        _write_json(output_dir / "parsed_outputs.json", parsed_rows)
        _write_jsonl(output_dir / "per_sample_results.jsonl", rows)
        _write_json(output_dir / "aggregate_metrics.json", aggregate)
        if exit_error is not None and active_error is None:
            raise RuntimeError(f"Engine shutdown failed: {exit_error!r}") from exit_error


if __name__ == "__main__":
    main()
