#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sparsevllm import LLM, SamplingParams
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.utils.profiler import profiler


SUPPORTED_METHODS = (
    "vanilla",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
)

EXPECTED_PROFILE_KEYS = {
    "streamingllm": ("streamingllm_prefill_eviction",),
    "snapkv": ("sparse_prepare_attn_score",),
    "pyramidkv": ("sparse_prepare_attn_score", "pyramidkv_staging_materialize_layer"),
    "omnikv": ("sparse_update_dynamic_indices",),
    "quest": ("quest_build_decode_view_static",),
    "rkv": ("rkv_decode_eviction",),
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _make_prompt(length: int, seed: int, *, offset: int = 0) -> list[int]:
    if length <= 0:
        raise ValueError(f"Prompt length must be positive, got {length}.")
    return [100 + ((seed * 997 + offset * 101 + idx * 37) % 140000) for idx in range(length)]


def _find_live_seq(llm: LLM, seq_id: int) -> Any | None:
    for queue in (llm.scheduler.waiting, llm.scheduler.decoding):
        for seq in queue:
            if int(seq.seq_id) == int(seq_id):
                return seq
    return None


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item())
    denominator = expected.float().abs().clamp_min(1.0e-5)
    return max_abs, float((difference / denominator).max().item())


def _rank_sync_error(summaries: list[dict[str, Any]]) -> str | None:
    if not summaries:
        return "No world-rank summaries were returned."
    for summary in summaries[1:]:
        if summary["state"] != summaries[0]["state"]:
            return (
                "Replicated EP runtime state diverged: "
                f"rank0={summaries[0]['world_rank']} rank={summary['world_rank']}."
            )
        for key in ("last_logits", "moe_synced"):
            if summary[key].keys() != summaries[0][key].keys():
                return f"Replicated EP {key} debug structure diverged across ranks."
        if summary["replica_consistency"] != summaries[0]["replica_consistency"]:
            return "Collective replica-consistency metrics differ across ranks."

    consistency = summaries[0]["replica_consistency"]
    if consistency is None:
        return "Replica-consistency metrics were not captured."
    if float(consistency["last_logits_tolerance_ratio"]) > 1.0:
        return (
            "World-rank logits differ beyond tolerance: "
            f"max_abs={consistency['last_logits_max_abs']} "
            f"tolerance_ratio={consistency['last_logits_tolerance_ratio']}."
        )
    for layer_idx, metrics in consistency["moe_layers"].items():
        if metrics["topk_ids_mismatch"]:
            return f"MoE router TopK IDs diverged across ranks at layer {layer_idx}."
        if float(metrics["topk_weights_tolerance_ratio"]) > 1.0:
            return (
                f"MoE routing weights diverged at layer {layer_idx}: "
                f"max_abs={metrics['topk_weights_max_abs']} "
                f"tolerance_ratio={metrics['topk_weights_tolerance_ratio']}."
            )
        if float(metrics["output_tolerance_ratio"]) > 1.0:
            return (
                f"MoE all-reduce output diverged at layer {layer_idx}: "
                f"max_abs={metrics['output_max_abs']} "
                f"tolerance_ratio={metrics['output_tolerance_ratio']}."
            )
    layer_zero_ranges = []
    for summary in summaries:
        layer_zero = summary["moe_local"].get("0")
        if layer_zero is None:
            return f"Rank {summary['world_rank']} did not expose layer-0 MoE debug state."
        layer_zero_ranges.append(
            (int(layer_zero["local_expert_start"]), int(layer_zero["local_expert_end"]))
        )
    for previous, current in zip(sorted(layer_zero_ranges), sorted(layer_zero_ranges)[1:]):
        if previous[1] != current[0]:
            return f"Local expert ranges are not contiguous: {sorted(layer_zero_ranges)}."
    return None


def _cache_max_row_len(summary: dict[str, Any]) -> int | None:
    live_rows = (
        summary.get("state", {})
        .get("cache", {})
        .get("live_rows", {})
    )
    row_lens = [
        int(record["row_len"])
        for records in live_rows.values()
        for record in records
    ]
    return max(row_lens) if row_lens else None


def _engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sparse_method": args.method,
        "enforce_eager": True,
        "decode_cuda_graph": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "engine_prefill_chunk_size": args.chunk_prefill_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs_in_batch": 2 if args.enable_prefix_caching else 1,
        "max_decoding_seqs": 2 if args.enable_prefix_caching else 1,
        "tensor_parallel_size": 1,
        "expert_parallel_size": args.expert_parallel_size,
        "data_parallel_size": 1,
        "moe_backend": "triton",
        "sink_keep_tokens": 4,
        "recent_keep_tokens": 16,
        "decode_keep_tokens": 32,
        "snapkv_window_size": 8,
        "snapkv_num_full_layers": 0,
        "pyramidkv_start_layer": 0,
        "pyramidkv_start_ratio": 0.5,
        "pyramidkv_least_layer": 47,
        "pyramidkv_least_ratio": 0.25,
        "full_attention_layers": "0",
        "quest_chunk_size": 8,
        "quest_token_budget": 32,
        "quest_skip_layers": 0,
        "rkv_compression_interval": 8,
        "rkv_observation_tokens": 4,
        "rkv_max_redundancy_tokens": 128,
        "enable_profiler": True,
        "enable_prefix_caching": args.enable_prefix_caching,
        "prefix_cache_block_size": args.prefix_cache_block_size,
        "prefix_cache_max_blocks": args.prefix_cache_max_blocks,
        "prefix_cache_salt": "qwen3-moe-ep-validation-v1",
    }
    return kwargs


def _capture_step(
    llm: LLM,
    *,
    case_name: str,
    step_idx: int,
    num_tokens: int,
    raw_steps: list[dict[str, Any]],
    per_step: list[dict[str, Any]],
) -> None:
    summaries = llm.debug_sparse_state_summaries()
    sync_error = _rank_sync_error(summaries)
    logits = llm.debug_last_logits().contiguous()
    hidden_states = {
        int(layer_idx): tensor.contiguous()
        for layer_idx, tensor in llm.debug_hidden_states().items()
    }
    moe_states = llm.debug_moe_states()
    raw_steps.append(
        {
            "case_name": case_name,
            "step_idx": int(step_idx),
            "stage": "prefill" if num_tokens > 0 else "decode",
            "logits": logits,
            "hidden_states": hidden_states,
            "moe_states": moe_states,
            "sampled_token_outputs": [
                (int(seq_id), [int(token_id) for token_id in token_ids])
                for seq_id, token_ids in llm.last_step_token_outputs
            ],
        }
    )
    per_step.append(
        {
            "case_name": case_name,
            "step_idx": int(step_idx),
            "stage": "prefill" if num_tokens > 0 else "decode",
            "num_tokens": int(num_tokens),
            "status": "success" if sync_error is None else "metric_failed",
            "error": sync_error,
            "rank_summaries": summaries,
            "cache_max_row_len": _cache_max_row_len(summaries[0]),
        }
    )


def _run_request_batch(
    llm: LLM,
    *,
    case_name: str,
    prompts: list[list[int]],
    output_tokens: int,
    raw_steps: list[dict[str, Any]],
    per_step: list[dict[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_tokens,
    )
    seq_ids = [int(llm.add_request(prompt, sampling)) for prompt in prompts]
    hits = {seq_id: 0 for seq_id in seq_ids}
    generated = {seq_id: [] for seq_id in seq_ids}
    step_idx = 0
    while any(len(generated[seq_id]) < output_tokens for seq_id in seq_ids):
        if step_idx >= max_steps:
            raise RuntimeError(
                f"Exceeded max_steps={max_steps} in case {case_name!r}."
            )
        _finished, num_tokens = llm.step()
        if num_tokens == 0:
            raise RuntimeError(f"Engine made no progress in case {case_name!r}.")
        for seq_id in seq_ids:
            seq = _find_live_seq(llm, seq_id)
            if seq is not None:
                hits[seq_id] = max(hits[seq_id], int(seq.prefix_cache_hit_len or 0))
        for seq_id, token_ids in llm.last_step_token_outputs:
            if int(seq_id) in generated:
                generated[int(seq_id)].extend(int(token_id) for token_id in token_ids)
        _capture_step(
            llm,
            case_name=case_name,
            step_idx=step_idx,
            num_tokens=num_tokens,
            raw_steps=raw_steps,
            per_step=per_step,
        )
        step_idx += 1
    return {
        "case_name": case_name,
        "seq_ids": seq_ids,
        "prompt_token_ids": prompts,
        "generated_token_ids": [generated[seq_id][:output_tokens] for seq_id in seq_ids],
        "prefix_cache_hit_tokens": [hits[seq_id] for seq_id in seq_ids],
        "status": "success",
    }


def _capture_control_state(
    llm: LLM,
    *,
    operation: str,
    result: dict[str, Any],
    controls: list[dict[str, Any]],
) -> None:
    summaries = llm.debug_sparse_state_summaries()
    sync_error = _rank_sync_error(summaries)
    controls.append(
        {
            "operation": operation,
            "result": result,
            "status": "success" if sync_error is None else "metric_failed",
            "error": sync_error,
            "rank_summaries": summaries,
        }
    )


def _run_cases(
    llm: LLM,
    args: argparse.Namespace,
    raw_steps: list[dict[str, Any]],
    per_step: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = _make_prompt(args.prompt_len, args.seed)
    requests = []
    controls: list[dict[str, Any]] = []
    requests.append(
        _run_request_batch(
            llm,
            case_name="primary",
            prompts=[primary],
            output_tokens=args.output_tokens,
            raw_steps=raw_steps,
            per_step=per_step,
            max_steps=args.max_steps,
        )
    )
    if not args.enable_prefix_caching:
        return requests, controls

    _capture_control_state(
        llm,
        operation="inspect_primary",
        result=llm.prefix_cache_inspect(primary, include_subtree=True),
        controls=controls,
    )
    _capture_control_state(
        llm,
        operation="match_primary",
        result=llm.prefix_cache_match(primary),
        controls=controls,
    )
    _capture_control_state(
        llm,
        operation="protect_primary_root",
        result=llm.prefix_cache_set_eviction_priority(
            primary[: args.prefix_cache_block_size],
            -1,
        ),
        controls=controls,
    )
    requests.append(
        _run_request_batch(
            llm,
            case_name="exact_replay",
            prompts=[primary],
            output_tokens=args.output_tokens,
            raw_steps=raw_steps,
            per_step=per_step,
            max_steps=args.max_steps,
        )
    )
    partial = primary[: args.prompt_len // 2] + _make_prompt(
        args.prompt_len - args.prompt_len // 2,
        args.seed,
        offset=1,
    )
    requests.append(
        _run_request_batch(
            llm,
            case_name="partial_replay",
            prompts=[partial],
            output_tokens=args.output_tokens,
            raw_steps=raw_steps,
            per_step=per_step,
            max_steps=args.max_steps,
        )
    )
    requests.append(
        _run_request_batch(
            llm,
            case_name="concurrent_references",
            prompts=[primary, primary],
            output_tokens=args.output_tokens,
            raw_steps=raw_steps,
            per_step=per_step,
            max_steps=args.max_steps,
        )
    )
    for offset in (2, 3):
        requests.append(
            _run_request_batch(
                llm,
                case_name=f"capacity_pressure_{offset}",
                prompts=[_make_prompt(args.prompt_len, args.seed, offset=offset)],
                output_tokens=args.output_tokens,
                raw_steps=raw_steps,
                per_step=per_step,
                max_steps=args.max_steps,
            )
        )
    _capture_control_state(
        llm,
        operation="inspect_protected_after_pressure",
        result=llm.prefix_cache_inspect(primary, include_subtree=True),
        controls=controls,
    )
    _capture_control_state(
        llm,
        operation="unprotect_primary_root",
        result=llm.prefix_cache_set_eviction_priority(
            primary[: args.prefix_cache_block_size],
            0,
        ),
        controls=controls,
    )
    _capture_control_state(
        llm,
        operation="delete_primary_subtree",
        result=llm.prefix_cache_delete_subtree(primary[: args.prefix_cache_block_size]),
        controls=controls,
    )
    return requests, controls


def _compare_reference(
    reference_path: Path,
    *,
    raw_steps: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    reference_steps = reference["steps"]
    errors = []
    if len(reference_steps) != len(raw_steps):
        errors.append(
            f"Step count mismatch: reference={len(reference_steps)} actual={len(raw_steps)}."
        )
    comparisons = []
    for actual, expected in zip(raw_steps, reference_steps):
        identity_matches = (
            actual["case_name"] == expected["case_name"]
            and actual["step_idx"] == expected["step_idx"]
            and actual["stage"] == expected["stage"]
            and tuple(actual["logits"].shape) == tuple(expected["logits"].shape)
        )
        max_abs = max_rel = float("inf")
        hidden_metrics = []
        first_hidden_mismatch = None
        moe_metrics = []
        first_moe_input_mismatch = None
        first_topk_ids_mismatch = None
        first_topk_weights_mismatch = None
        first_moe_output_mismatch = None
        close = False
        hidden_close = False
        moe_close = False
        if identity_matches:
            max_abs, max_rel = _max_errors(actual["logits"], expected["logits"])
            close = bool(
                torch.allclose(
                    actual["logits"].float(),
                    expected["logits"].float(),
                    atol=atol,
                    rtol=rtol,
                )
            )
            actual_hidden = actual.get("hidden_states", {})
            expected_hidden = expected.get("hidden_states", {})
            if actual_hidden.keys() != expected_hidden.keys():
                first_hidden_mismatch = "layer_keys"
            else:
                hidden_close = True
                for layer_idx in sorted(actual_hidden):
                    layer_max_abs, layer_max_rel = _max_errors(
                        actual_hidden[layer_idx],
                        expected_hidden[layer_idx],
                    )
                    layer_close = bool(
                        torch.allclose(
                            actual_hidden[layer_idx].float(),
                            expected_hidden[layer_idx].float(),
                            atol=atol,
                            rtol=rtol,
                        )
                    )
                    hidden_metrics.append(
                        {
                            "layer_idx": int(layer_idx),
                            "max_abs_error": layer_max_abs,
                            "max_rel_error": layer_max_rel,
                            "within_tolerance": layer_close,
                        }
                    )
                    if first_hidden_mismatch is None and not layer_close:
                        first_hidden_mismatch = int(layer_idx)
                        hidden_close = False
            actual_moe = actual.get("moe_states", {})
            expected_moe = expected.get("moe_states", {})
            if actual_moe.keys() != expected_moe.keys():
                first_moe_input_mismatch = "layer_keys"
                first_topk_ids_mismatch = "layer_keys"
                first_topk_weights_mismatch = "layer_keys"
                first_moe_output_mismatch = "layer_keys"
            else:
                moe_close = True
                for layer_idx in sorted(actual_moe):
                    layer_metrics = {"layer_idx": int(layer_idx)}
                    for name in ("input", "topk_weights", "output"):
                        tensor_max_abs, tensor_max_rel = _max_errors(
                            actual_moe[layer_idx][name],
                            expected_moe[layer_idx][name],
                        )
                        tensor_close = bool(
                            torch.allclose(
                                actual_moe[layer_idx][name].float(),
                                expected_moe[layer_idx][name].float(),
                                atol=atol,
                                rtol=rtol,
                            )
                        )
                        layer_metrics[name] = {
                            "max_abs_error": tensor_max_abs,
                            "max_rel_error": tensor_max_rel,
                            "within_tolerance": tensor_close,
                        }
                    ids_equal = bool(
                        torch.equal(
                            actual_moe[layer_idx]["topk_ids"],
                            expected_moe[layer_idx]["topk_ids"],
                        )
                    )
                    layer_metrics["topk_ids_equal"] = ids_equal
                    moe_metrics.append(layer_metrics)
                    if (
                        first_moe_input_mismatch is None
                        and not layer_metrics["input"]["within_tolerance"]
                    ):
                        first_moe_input_mismatch = int(layer_idx)
                        moe_close = False
                    if first_topk_ids_mismatch is None and not ids_equal:
                        first_topk_ids_mismatch = int(layer_idx)
                        moe_close = False
                    if (
                        first_topk_weights_mismatch is None
                        and not layer_metrics["topk_weights"]["within_tolerance"]
                    ):
                        first_topk_weights_mismatch = int(layer_idx)
                        moe_close = False
                    if (
                        first_moe_output_mismatch is None
                        and not layer_metrics["output"]["within_tolerance"]
                    ):
                        first_moe_output_mismatch = int(layer_idx)
                        moe_close = False
        if not identity_matches or not close:
            errors.append(
                f"Logits mismatch at case={actual['case_name']} step={actual['step_idx']}: "
                f"identity_matches={identity_matches} max_abs={max_abs} max_rel={max_rel}."
            )
        if identity_matches and not hidden_close:
            errors.append(
                f"Hidden-state mismatch at case={actual['case_name']} "
                f"step={actual['step_idx']}: first_layer={first_hidden_mismatch}."
            )
        if identity_matches and not moe_close:
            errors.append(
                f"MoE-state mismatch at case={actual['case_name']} "
                f"step={actual['step_idx']}: input={first_moe_input_mismatch} "
                f"topk_ids={first_topk_ids_mismatch} "
                f"topk_weights={first_topk_weights_mismatch} "
                f"output={first_moe_output_mismatch}."
            )
        step_success = identity_matches and close and hidden_close and moe_close
        comparisons.append(
            {
                "case_name": actual["case_name"],
                "step_idx": actual["step_idx"],
                "identity_matches": identity_matches,
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "hidden_states": hidden_metrics,
                "first_hidden_mismatch": first_hidden_mismatch,
                "moe_states": moe_metrics,
                "first_moe_input_mismatch": first_moe_input_mismatch,
                "first_topk_ids_mismatch": first_topk_ids_mismatch,
                "first_topk_weights_mismatch": first_topk_weights_mismatch,
                "first_moe_output_mismatch": first_moe_output_mismatch,
                "status": "success" if step_success else "metric_failed",
            }
        )
    actual_tokens = [item["generated_token_ids"] for item in requests]
    reference_tokens = [item["generated_token_ids"] for item in reference["requests"]]
    if actual_tokens != reference_tokens:
        errors.append("Generated token IDs differ from the EP=1 reference.")
    actual_hits = [item["prefix_cache_hit_tokens"] for item in requests]
    reference_hits = [item["prefix_cache_hit_tokens"] for item in reference["requests"]]
    if actual_hits != reference_hits:
        errors.append(
            f"Prefix hit lengths differ: reference={reference_hits} actual={actual_hits}."
        )
    return {"status": "success" if not errors else "metric_failed", "errors": errors, "steps": comparisons}


def _validate_method_trigger(
    args: argparse.Namespace,
    profiler_stats: dict[str, Any],
    per_step: list[dict[str, Any]],
) -> list[str]:
    errors = []
    method = normalize_sparse_method(args.method)
    expected = EXPECTED_PROFILE_KEYS.get(method, ())
    if expected and not any(
        key in profiler_stats and int(profiler_stats[key]["calls"]) > 0
        for key in expected
    ):
        errors.append(
            f"Method-specific path did not execute; expected one of {expected}, "
            f"observed={sorted(profiler_stats)}."
        )
    row_lengths = [
        int(step["cache_max_row_len"])
        for step in per_step
        if step["cache_max_row_len"] is not None
    ]
    if method in {"streamingllm", "snapkv", "pyramidkv", "rkv"}:
        budget = 20 if method == "streamingllm" else 52
        if not row_lengths or min(row_lengths) > budget:
            errors.append(
                f"Physical KV compression boundary was not observed: method={method} "
                f"budget={budget} row_lengths={row_lengths}."
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Qwen3MoE sparse attention and prefix-cache state under EP."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--expert-parallel-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--output-tokens", type=int, default=12)
    parser.add_argument("--chunk-prefill-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--prefix-cache-block-size", type=int, default=8)
    parser.add_argument("--prefix-cache-max-blocks", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_prefix_caching and args.method not in {"vanilla", "omnikv", "quest"}:
        raise ValueError(
            "Prefix validation supports only vanilla, omnikv, and quest; "
            f"got method={args.method!r}."
        )
    if args.prompt_len + args.output_tokens > args.max_model_len:
        raise ValueError(
            f"prompt_len + output_tokens exceeds max_model_len: "
            f"{args.prompt_len} + {args.output_tokens} > {args.max_model_len}."
        )
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}.")
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_config_path}.")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    num_hidden_layers = int(model_config["num_hidden_layers"])
    if num_hidden_layers <= 0:
        raise ValueError(
            f"num_hidden_layers must be positive, got {num_hidden_layers}."
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference).resolve() if args.reference else None
    if reference_path is not None and not reference_path.is_file():
        raise FileNotFoundError(f"Reference raw output does not exist: {reference_path}.")

    os.environ["SPARSEVLLM_DEBUG_RUNTIME"] = "1"
    os.environ["SPARSEVLLM_DEBUG_MOE"] = "1"
    os.environ["SPARSEVLLM_DEBUG_HIDDEN_LAYERS"] = ",".join(
        str(layer_idx) for layer_idx in range(num_hidden_layers)
    )
    raw_steps: list[dict[str, Any]] = []
    per_step: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    llm = None
    started = time.perf_counter()
    failure: BaseException | None = None
    reference_metrics = None
    trigger_errors: list[str] = []
    try:
        torch.cuda.reset_peak_memory_stats()
        llm = LLM(str(model_path), **_engine_kwargs(args))
        profiler.reset()
        requests, controls = _run_cases(llm, args, raw_steps, per_step)
        profiler_stats = profiler.snapshot()
        trigger_errors = _validate_method_trigger(args, profiler_stats, per_step)
        if reference_path is not None:
            reference_metrics = _compare_reference(
                reference_path,
                raw_steps=raw_steps,
                requests=requests,
                atol=args.atol,
                rtol=args.rtol,
            )
    except BaseException as exc:
        failure = exc
        profiler_stats = profiler.snapshot()
    finally:
        if llm is not None:
            llm.exit()

    rank_errors = [
        step["error"] for step in per_step if step["status"] != "success"
    ] + [control["error"] for control in controls if control["status"] != "success"]
    prefix_errors = []
    if args.enable_prefix_caching and requests:
        if len(requests) < 4 or requests[1]["prefix_cache_hit_tokens"][0] <= 0:
            prefix_errors.append("Exact replay did not report a prefix-cache hit.")
        if len(requests) < 4 or requests[2]["prefix_cache_hit_tokens"][0] <= 0:
            prefix_errors.append("Partial replay did not report a prefix-cache hit.")
        if len(requests) < 4 or any(
            hit <= 0 for hit in requests[3]["prefix_cache_hit_tokens"]
        ):
            prefix_errors.append("Concurrent references did not both acquire cached prefixes.")
        concurrent_steps = [
            step for step in per_step if step["case_name"] == "concurrent_references"
        ]
        if not any(
            block["ref_count"] >= 2
            for step in concurrent_steps
            for block in (
                step["rank_summaries"][0]["state"]["cache"]["prefix_cache"] or {}
            ).get("blocks", [])
        ):
            prefix_errors.append("Concurrent prefix references never reached ref_count >= 2.")
        final_prefix_stats = (
            per_step[-1]["rank_summaries"][0]["state"]["cache"]["prefix_cache"]["stats"]
            if per_step
            else {}
        )
        if int(final_prefix_stats.get("prefix_cache_evicted_blocks", 0)) <= 0:
            prefix_errors.append("Capacity pressure did not evict any prefix block.")
        protected_controls = [
            control
            for control in controls
            if control["operation"] == "inspect_protected_after_pressure"
        ]
        if not protected_controls or not protected_controls[0]["result"].get("matched"):
            prefix_errors.append("Negative-priority protected prefix did not survive pressure.")
        delete_controls = [
            control for control in controls if control["operation"] == "delete_primary_subtree"
        ]
        if not delete_controls or not delete_controls[0]["result"].get("deleted_block_ids"):
            prefix_errors.append("Prefix delete-subtree control deleted no blocks.")

    metric_errors = rank_errors + trigger_errors + prefix_errors
    if reference_metrics is not None and reference_metrics["status"] != "success":
        metric_errors.extend(reference_metrics["errors"])
    status = "model_failed" if failure is not None else ("metric_failed" if metric_errors else "success")

    raw_outputs = {"steps": raw_steps, "requests": requests}
    torch.save(raw_outputs, output_dir / "raw_outputs.pt")
    _write_json(
        output_dir / "parsed_outputs.json",
        {"status": status, "requests": requests, "controls": controls},
    )
    _write_json(output_dir / "per_step_results.json", per_step)
    _write_json(output_dir / "profiler_stats.json", profiler_stats)
    if reference_metrics is not None:
        _write_json(output_dir / "reference_metrics.json", reference_metrics)
    _write_json(
        output_dir / "run_config.json",
        {
            "command": [sys.executable, *sys.argv],
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "model": str(model_path),
            "method": args.method,
            "engine_kwargs": _engine_kwargs(args),
            "expert_parallel_size": args.expert_parallel_size,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "seed": args.seed,
            "prompt_len": args.prompt_len,
            "output_tokens": args.output_tokens,
            "reference": str(reference_path) if reference_path else None,
            "atol": args.atol,
            "rtol": args.rtol,
            "environment": {
                key: os.environ[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "SPARSEVLLM_MASTER_PORT",
                    "SPARSEVLLM_DEBUG_RUNTIME",
                    "SPARSEVLLM_DEBUG_MOE",
                    "SPARSEVLLM_DEBUG_HIDDEN_LAYERS",
                )
                if key in os.environ
            },
        },
    )
    _write_json(
        output_dir / "aggregate_metrics.json",
        {
            "status": status,
            "num_steps": len(per_step),
            "num_requests": sum(len(item["seq_ids"]) for item in requests),
            "num_rank_consistency_failures": len(rank_errors),
            "trigger_errors": trigger_errors,
            "prefix_errors": prefix_errors,
            "reference_status": reference_metrics["status"] if reference_metrics else None,
            "metric_errors": metric_errors,
            "failure": repr(failure) if failure is not None else None,
            "traceback": (
                "".join(traceback.format_exception(failure)) if failure is not None else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    if failure is not None:
        raise failure
    if metric_errors:
        raise RuntimeError(
            f"Qwen3MoE sparse EP validation failed; inspect {output_dir}. Errors: {metric_errors}"
        )


if __name__ == "__main__":
    main()
