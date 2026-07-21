#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISION = "8e095dcb5d87d55e261ea10fef7fc5f4a596f9a8"
SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
DEBUG_ENV_KEYS = (
    "SPARSEVLLM_DEBUG_RUNTIME",
    "SPARSEVLLM_DEBUG_HIDDEN_LAYERS",
    "SPARSEVLLM_DEBUG_MOE",
    "SPARSEVLLM_DEBUG_MINIMAX_M2",
)


class MetricFailure(RuntimeError):
    def __init__(self, message: str, *, metrics: dict[str, Any] | None = None):
        super().__init__(message)
        self.metrics = metrics


class ParseFailure(RuntimeError):
    pass


def _enable_debug_runtime() -> None:
    # Spawned EP workers inherit their environment during LLM construction.
    os.environ["SPARSEVLLM_DEBUG_RUNTIME"] = "1"
    os.environ["SPARSEVLLM_DEBUG_HIDDEN_LAYERS"] = "0,30,61"
    os.environ["SPARSEVLLM_DEBUG_MOE"] = "1"
    os.environ["SPARSEVLLM_DEBUG_MINIMAX_M2"] = "1"


def _parse_int_csv(value: str, *, allow_zero: bool = False) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    lower_bound = 0 if allow_zero else 1
    if not values or any(item < lower_bound for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(
            f"Expected {qualifier} comma-separated integers, got {value!r}."
        )
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate integer values are not allowed: {values}.")
    return values


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"]),
        "branch": _command_output(["git", "branch", "--show-current"]),
        "dirty": bool(_command_output(["git", "status", "--porcelain"])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    missing = [path for path in (config_path, index_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official checkpoint metadata files are missing: {missing}."
        )
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensor shards found in {model_path}.")
    return {
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(index_path),
        "num_safetensor_shards": len(shards),
        "safetensor_bytes": sum(path.stat().st_size for path in shards),
    }


def _query_gpus() -> list[dict[str, Any]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for line in output.splitlines():
        index, name, memory_used, memory_total, utilization = [
            part.strip() for part in line.split(",")
        ]
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mib": int(memory_used),
                "memory_total_mib": int(memory_total),
                "utilization_percent": int(utilization),
            }
        )
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPU rows.")
    return devices


def _validate_gpu_selection(
    indices: list[int],
    *,
    ep_size: int,
    max_memory_used_mib: int,
    max_utilization_percent: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(indices) != ep_size:
        raise ValueError(
            f"gpu-indices must contain exactly EP={ep_size} devices, got {indices}."
        )
    devices = _query_gpus()
    by_index = {device["index"]: device for device in devices}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"GPU indices do not exist: {missing}; devices={devices}.")
    selected = [by_index[index] for index in indices]
    busy = [
        device
        for device in selected
        if device["memory_used_mib"] > max_memory_used_mib
        or device["utilization_percent"] > max_utilization_percent
    ]
    if busy:
        raise RuntimeError(
            f"Selected GPUs are busy; refusing to start: busy={busy}, all={devices}."
        )
    return selected, devices


def _tensor_summary(tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "abs_mean": float(values.abs().mean()),
        "finite": bool(values.isfinite().all()),
    }


def _relative_metrics(torch, actual, expected) -> dict[str, float]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise MetricFailure(
            f"Tensor shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}."
        )
    actual_float = actual.float()
    expected_float = expected.float()
    if not bool(actual_float.isfinite().all()):
        raise MetricFailure("Actual tensor contains non-finite values.")
    if not bool(expected_float.isfinite().all()):
        raise MetricFailure("Oracle tensor contains non-finite values.")
    difference = actual_float - expected_float
    denominator = torch.linalg.vector_norm(expected_float)
    if float(denominator) == 0.0:
        relative_l2 = 0.0 if bool((difference == 0).all()) else math.inf
    else:
        relative_l2 = float(torch.linalg.vector_norm(difference) / denominator)
    return {
        "max_abs_error": float(difference.abs().max()),
        "mean_abs_error": float(difference.abs().mean()),
        "relative_l2_error": relative_l2,
    }


def _compare_case(
    torch,
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    max_relative_l2: float,
    max_router_relative_l2: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "greedy_tokens": {
            "exact": actual["generated_token_ids"]
            == expected["generated_token_ids"],
            "actual": actual["generated_token_ids"],
            "expected": expected["generated_token_ids"],
        }
    }
    failures = []
    if not metrics["greedy_tokens"]["exact"]:
        failures.append(
            "Greedy token mismatch: "
            f"actual={actual['generated_token_ids']}, "
            f"expected={expected['generated_token_ids']}."
        )
    logits_metrics = _relative_metrics(torch, actual["logits"], expected["logits"])
    metrics["logits"] = logits_metrics
    if logits_metrics["relative_l2_error"] > max_relative_l2:
        failures.append(f"Logits correctness gate failed: {logits_metrics}.")

    for group_name in ("hidden_states", "attention_states"):
        if group_name not in actual:
            continue
        if group_name not in expected:
            raise MetricFailure(f"Oracle is missing tensor group {group_name!r}.")
        group_metrics = {}
        for layer_idx, tensors in actual[group_name].items():
            if layer_idx not in expected[group_name]:
                raise MetricFailure(
                    f"Oracle is missing {group_name} layer {layer_idx}."
                )
            if isinstance(tensors, dict):
                tensor_metrics = {}
                for name, tensor in tensors.items():
                    result = _relative_metrics(
                        torch,
                        tensor,
                        expected[group_name][layer_idx][name],
                    )
                    tensor_metrics[name] = result
                    if result["relative_l2_error"] > max_relative_l2:
                        failures.append(
                            f"{group_name}/{layer_idx}/{name} gate failed: {result}."
                        )
                group_metrics[str(layer_idx)] = tensor_metrics
            else:
                result = _relative_metrics(
                    torch,
                    tensors,
                    expected[group_name][layer_idx],
                )
                group_metrics[str(layer_idx)] = result
                if result["relative_l2_error"] > max_relative_l2:
                    failures.append(
                        f"{group_name}/{layer_idx} gate failed: {result}."
                    )
        metrics[group_name] = group_metrics

    if "moe_states" in actual:
        if "moe_states" not in expected:
            raise MetricFailure("Oracle is missing tensor group 'moe_states'.")
        moe_metrics = {}
        for layer_idx, tensors in actual["moe_states"].items():
            oracle_tensors = expected["moe_states"].get(layer_idx)
            if oracle_tensors is None:
                raise MetricFailure(f"Oracle is missing MoE layer {layer_idx}.")
            topk_ids_exact = torch.equal(
                tensors["topk_ids"], oracle_tensors["topk_ids"]
            )
            layer_metrics = {"topk_ids": {"exact": topk_ids_exact}}
            if not topk_ids_exact:
                failures.append(f"Router expert IDs differ at layer {layer_idx}.")
            for name in ("input", "router_logits", "topk_weights", "output"):
                result = _relative_metrics(
                    torch,
                    tensors[name],
                    oracle_tensors[name],
                )
                layer_metrics[name] = result
                threshold = (
                    max_router_relative_l2
                    if name in {"router_logits", "topk_weights"}
                    else max_relative_l2
                )
                if result["relative_l2_error"] > threshold:
                    failures.append(
                        f"moe_states/{layer_idx}/{name} gate failed: {result}."
                    )
            moe_metrics[str(layer_idx)] = layer_metrics
        metrics["moe_states"] = moe_metrics
    if failures:
        raise MetricFailure(" ".join(failures), metrics=metrics)
    return metrics


def _graph_state_snapshot(llm) -> dict[str, Any] | None:
    runner = getattr(llm.model_runner, "decode_cuda_graph_runner", None)
    if runner is None or runner.last_state_key is None:
        return None
    state = runner._graphs.get(runner.last_state_key)
    if state is None:
        raise RuntimeError(f"Last graph key is absent: {runner.last_state_key!r}.")
    key = runner.last_state_key
    return {
        "key": {
            "method": key.method,
            "batch_size": int(key.batch_size),
            "context_capacity": int(key.context_capacity),
            "is_long_text": bool(key.is_long_text),
            "capture_sampling": bool(key.capture_sampling),
        },
        "captured": state.graph is not None,
        "real_batch_size": int(runner.last_real_batch_size or 0),
        "num_graph_states": len(runner._graphs),
    }


def _graph_logits(llm):
    runner = llm.model_runner.decode_cuda_graph_runner
    if runner is None or runner.last_state_key is None:
        raise RuntimeError("No decode CUDA Graph state was used by this case.")
    state = runner._graphs[runner.last_state_key]
    if state.graph is None or state.logits is None:
        raise RuntimeError("Decode CUDA Graph state is not captured or has no logits.")
    real_batch_size = int(runner.last_real_batch_size or 0)
    if real_batch_size <= 0:
        raise RuntimeError("Decode CUDA Graph did not record a real batch size.")
    return state.logits[:real_batch_size].detach().cpu()


def _cache_stats(llm) -> dict[str, int]:
    state = getattr(llm.model_runner, "runtime_state", None)
    if state is None or not hasattr(state, "free_slot_stats"):
        raise RuntimeError("Runtime state does not expose free_slot_stats().")
    return {key: int(value) for key, value in state.free_slot_stats().items()}


def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _run_requests(
    llm,
    SamplingParams,
    prompts: list[list[int]],
    *,
    max_tokens: int,
    max_steps: int,
    capture_debug_steps: bool = False,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=max_tokens,
    )
    stats_before = _cache_stats(llm)
    seq_ids = [int(llm.add_request(prompt, sampling_params)) for prompt in prompts]
    outputs: dict[int, list[int]] = {}
    prefix_hits = {seq_id: 0 for seq_id in seq_ids}
    graph_keys = []
    debug_steps = []
    steps = 0
    while not llm.is_finished():
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"Generation exceeded max_steps={max_steps} for seq_ids={seq_ids}."
            )
        finished, num_tokens = llm.step()
        if capture_debug_steps:
            rank_summaries = llm.debug_sparse_state_summaries()
            debug_steps.append(
                {
                    "step": steps,
                    "num_tokens": int(num_tokens),
                    "logits": llm.debug_last_logits(),
                    "hidden_states": llm.debug_hidden_states(),
                    "attention_states": llm.debug_attention_states(),
                    "moe_states": llm.debug_moe_states(),
                    "rank_moe_summaries": [
                        {
                            "world_rank": int(summary["world_rank"]),
                            "ep_rank": int(summary["ep_rank"]),
                            "moe_local": summary["moe_local"],
                            "moe_synced": summary["moe_synced"],
                        }
                        for summary in rank_summaries
                    ],
                }
            )
        for queue_name in ("waiting", "decoding"):
            for sequence in getattr(llm.scheduler, queue_name, ()):
                seq_id = int(sequence.seq_id)
                if seq_id in prefix_hits:
                    prefix_hits[seq_id] = max(
                        prefix_hits[seq_id],
                        int(getattr(sequence, "prefix_cache_hit_len", 0) or 0),
                    )
        for seq_id, token_ids, _logprobs, _top_logprobs in finished:
            outputs[int(seq_id)] = [int(token_id) for token_id in token_ids]
        if num_tokens < 0:
            graph_state = _graph_state_snapshot(llm)
            if graph_state is not None:
                graph_keys.append(graph_state)
    missing = [seq_id for seq_id in seq_ids if seq_id not in outputs]
    if missing:
        raise RuntimeError(f"Generation completed without outputs for seq_ids={missing}.")
    stats_after = _cache_stats(llm)
    return {
        "generated_token_ids": [outputs[seq_id] for seq_id in seq_ids],
        "prefix_cache_hit_lengths": [prefix_hits[seq_id] for seq_id in seq_ids],
        "steps": steps,
        "cache_stats_before": stats_before,
        "cache_stats_after": stats_after,
        "cache_stats_delta": _stats_delta(stats_before, stats_after),
        "graph_states": graph_keys,
        "debug_steps": debug_steps,
    }


def _make_prompts(
    *,
    prompt_length: int,
    batch_size: int,
    seed: int,
    vocab_size: int,
) -> list[list[int]]:
    if vocab_size <= 1024:
        raise ValueError(f"MiniMax vocabulary is unexpectedly small: {vocab_size}.")
    rng = random.Random(seed + prompt_length * 1009 + batch_size * 9176)
    high = min(vocab_size - 1, 50_000)
    return [
        [rng.randint(1000, high) for _ in range(prompt_length)]
        for _ in range(batch_size)
    ]


def _collect_raw_case(llm, request_result: dict[str, Any], *, use_graph: bool):
    raw_case = {
        "generated_token_ids": request_result["generated_token_ids"],
    }
    if use_graph:
        raw_case["logits"] = _graph_logits(llm)
    else:
        raw_case.update(
            {
                "logits": llm.debug_last_logits(),
                "hidden_states": llm.debug_hidden_states(),
                "attention_states": llm.debug_attention_states(),
                "moe_states": llm.debug_moe_states(),
            }
        )
    if request_result.get("debug_steps"):
        raw_case["debug_steps"] = request_result["debug_steps"]
    return raw_case


def _parsed_case(raw_case: dict[str, Any]) -> dict[str, Any]:
    parsed = {
        "generated_token_ids": raw_case["generated_token_ids"],
        "logits": _tensor_summary(raw_case["logits"]),
    }
    for group_name in ("hidden_states", "attention_states", "moe_states"):
        if group_name not in raw_case:
            continue
        parsed[group_name] = {}
        for layer_idx, tensors in raw_case[group_name].items():
            if isinstance(tensors, dict):
                parsed[group_name][str(layer_idx)] = {
                    name: _tensor_summary(tensor)
                    for name, tensor in tensors.items()
                }
            else:
                parsed[group_name][str(layer_idx)] = _tensor_summary(tensors)
    return parsed


def _record_case_artifacts(
    case_id: str,
    raw_case: dict[str, Any],
    raw_cases: dict[str, dict[str, Any]],
    parsed_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_cases[case_id] = raw_case
    try:
        parsed = _parsed_case(raw_case)
    except Exception as exc:
        raise ParseFailure(f"Failed to parse artifacts for {case_id}.") from exc
    parsed_outputs[case_id] = parsed
    return parsed


def _graph_key_tuple(request_result: dict[str, Any]):
    states = request_result["graph_states"]
    if not states:
        return None
    key = states[-1]["key"]
    return tuple((name, key[name]) for name in sorted(key))


def _validate_prefix_patterns(results: dict[str, dict[str, Any]], *, use_graph: bool):
    warm_hits = results["prefix_warm"]["prefix_cache_hit_lengths"]
    exact_hits = results["prefix_exact"]["prefix_cache_hit_lengths"]
    partial_hits = results["prefix_partial"]["prefix_cache_hit_lengths"]
    no_hit = results["prefix_no_hit"]["prefix_cache_hit_lengths"]
    shared_hits = results["prefix_shared"]["prefix_cache_hit_lengths"]
    if warm_hits != [0] or no_hit != [0]:
        raise MetricFailure(
            f"Prefix no-hit phases unexpectedly reused cache: warm={warm_hits}, "
            f"no_hit={no_hit}."
        )
    if exact_hits[0] <= 0:
        raise MetricFailure(f"Exact prefix hit was not observed: {exact_hits}.")
    if not 0 < partial_hits[0] < exact_hits[0]:
        raise MetricFailure(
            f"Partial prefix hit is invalid: partial={partial_hits}, exact={exact_hits}."
        )
    if len(shared_hits) != 2 or any(hit <= 0 for hit in shared_hits):
        raise MetricFailure(f"Concurrent shared-prefix reuse failed: {shared_hits}.")
    if use_graph:
        keys = {
            phase: _graph_key_tuple(results[phase])
            for phase in (
                "prefix_warm",
                "prefix_exact",
                "prefix_partial",
                "prefix_no_hit",
            )
        }
        if None in keys.values() or len(set(keys.values())) != 1:
            raise MetricFailure(
                "Prefix hit mode changed the decode graph key for one bucket: "
                f"{keys}."
            )
    return {
        "warm_hit_tokens": warm_hits,
        "exact_hit_tokens": exact_hits,
        "partial_hit_tokens": partial_hits,
        "no_hit_tokens": no_hit,
        "shared_hit_tokens": shared_hits,
    }


def _validate_replica_consistency(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        raise MetricFailure("No EP replica-consistency summaries were returned.")
    for summary in summaries:
        consistency = summary.get("replica_consistency")
        if not isinstance(consistency, dict):
            raise MetricFailure(
                "EP replica consistency is unavailable for "
                f"world_rank={summary.get('world_rank')}."
            )
        logits_ratio = float(
            consistency.get("last_logits_tolerance_ratio", math.inf)
        )
        if not math.isfinite(logits_ratio) or logits_ratio > 1.0:
            raise MetricFailure(
                f"EP rank logits differ beyond tolerance: {consistency}."
            )
        for layer_idx, layer in consistency.get("moe_layers", {}).items():
            if layer.get("topk_ids_mismatch"):
                raise MetricFailure(
                    f"EP router IDs differ at layer {layer_idx}: {layer}."
                )
            for metric_name in (
                "topk_weights_tolerance_ratio",
                "output_tolerance_ratio",
            ):
                ratio = float(layer.get(metric_name, math.inf))
                if not math.isfinite(ratio) or ratio > 1.0:
                    raise MetricFailure(
                        f"EP {metric_name} failed at layer {layer_idx}: {layer}."
                    )


def _software_versions(torch) -> dict[str, Any]:
    import kernels
    import transformers
    import triton

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "kernels": getattr(kernels, "__version__", "unknown"),
        "nvidia_smi": _command_output(["nvidia-smi"]),
    }


def _write_report(
    path: Path,
    *,
    aggregate_status: str,
    run_config: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    lines = [
        "# MiniMax M2.7 end-to-end validation",
        "",
        f"- status: `{aggregate_status}`",
        f"- backend: `{run_config['backend']}`",
        f"- EP: `{run_config['ep_size']}`",
        f"- decode CUDA Graph: `{run_config['decode_cuda_graph']}`",
        f"- prefix cache: `{run_config['enable_prefix_caching']}`",
        f"- oracle: `{run_config.get('oracle_artifact')}`",
        "",
        "| case | status | prompt | batch | prefix hits | graph states |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for record in records:
        lines.append(
            "| {case} | {status} | {prompt} | {batch} | {hits} | {graphs} |".format(
                case=record.get("case_id", "runtime"),
                status=record["status"],
                prompt=record.get("prompt_length", ""),
                batch=record.get("batch_size", ""),
                hits=record.get("prefix_cache_hit_lengths", ""),
                graphs=record.get("graph_state_count", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MiniMax M2.7 official-checkpoint inference on AutoDL."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-indices", required=True)
    parser.add_argument("--ep-size", type=int, choices=(4, 8), required=True)
    parser.add_argument(
        "--backend",
        choices=("pytorch", "native", "triton"),
        required=True,
    )
    parser.add_argument(
        "--decode-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--validate-prefix-patterns", action="store_true")
    parser.add_argument("--prompt-lengths", default="16")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-steps-per-case", type=int, default=10_000)
    parser.add_argument("--capture-debug-steps", action="store_true")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--oracle-artifact", type=Path, default=None)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("--max-router-relative-l2", type=float, default=1.0e-4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--chunk-prefill-size", type=int, default=512)
    parser.add_argument("--warmup-prompt-len", type=int, default=16)
    parser.add_argument("--prefix-length", type=int, default=128)
    parser.add_argument("--prefix-cache-block-size", type=int, default=16)
    parser.add_argument("--prefix-cache-max-blocks", type=int, default=None)
    parser.add_argument("--max-memory-used-mib", type=int, default=512)
    parser.add_argument("--max-utilization-percent", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    parsed_outputs: dict[str, Any] = {}
    raw_cases: dict[str, Any] = {}
    run_config: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "git": _git_metadata(),
        "model": "MiniMaxAI/MiniMax-M2.7",
        "model_path": str(args.model_path),
        "model_revision": MODEL_REVISION,
        "backend": args.backend,
        "ep_size": args.ep_size,
        "decode_cuda_graph": args.decode_cuda_graph,
        "enable_prefix_caching": args.enable_prefix_caching,
        "validate_prefix_patterns": args.validate_prefix_patterns,
        "oracle_artifact": (
            str(args.oracle_artifact.expanduser().resolve())
            if args.oracle_artifact is not None
            else None
        ),
        "seed": args.seed,
    }
    torch = None
    llm = None
    oracle_cases = None
    exit_code = 0
    try:
        gpu_indices = _parse_int_csv(args.gpu_indices, allow_zero=True)
        prompt_lengths = _parse_int_csv(args.prompt_lengths)
        batch_sizes = _parse_int_csv(args.batch_sizes)
        if args.decode_cuda_graph and args.backend != "triton":
            raise ValueError(
                "decode CUDA Graph requires --backend triton for MiniMax M2.7."
            )
        if args.validate_prefix_patterns and not args.enable_prefix_caching:
            raise ValueError(
                "--validate-prefix-patterns requires --enable-prefix-caching."
            )
        if args.max_tokens < 2 or args.max_steps_per_case <= 0:
            raise ValueError("max-tokens must be >= 2 and max-steps-per-case > 0.")
        if args.prefix_length < 2 * args.prefix_cache_block_size:
            raise ValueError(
                "prefix-length must cover at least two prefix-cache blocks."
            )
        if args.prefix_length % (2 * args.prefix_cache_block_size):
            raise ValueError(
                "prefix-length/2 must be aligned to prefix-cache-block-size."
            )
        if not 0.0 < args.max_relative_l2 < 1.0:
            raise ValueError("max-relative-l2 must be between 0 and 1.")
        if not 0.0 < args.max_router_relative_l2 < 1.0:
            raise ValueError("max-router-relative-l2 must be between 0 and 1.")
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {args.model_path}.")
        max_planned_prompt = max(
            max(prompt_lengths),
            args.prefix_length if args.validate_prefix_patterns else 0,
        )
        if max_planned_prompt + args.max_tokens > args.max_model_len:
            raise ValueError(
                "Planned prompt plus output exceeds max-model-len: "
                f"{max_planned_prompt} + {args.max_tokens} > {args.max_model_len}."
            )
        max_planned_batch_tokens = max(prompt_lengths) * max(batch_sizes)
        if args.validate_prefix_patterns:
            max_planned_batch_tokens = max(
                max_planned_batch_tokens,
                args.prefix_length * 2,
            )
        if max_planned_batch_tokens > args.max_num_batched_tokens:
            raise ValueError(
                "The largest prompt batch exceeds max-num-batched-tokens: "
                f"{max_planned_batch_tokens} > {args.max_num_batched_tokens}."
            )
        if args.warmup_prompt_len <= 0:
            raise ValueError("warmup-prompt-len must be positive.")
        if not 0.0 < args.gpu_memory_utilization < 1.0:
            raise ValueError("gpu-memory-utilization must be between 0 and 1.")
        if (
            args.prefix_cache_max_blocks is not None
            and args.prefix_cache_max_blocks <= 0
        ):
            raise ValueError("prefix-cache-max-blocks must be positive when provided.")
        selected, all_devices = _validate_gpu_selection(
            gpu_indices,
            ep_size=args.ep_size,
            max_memory_used_mib=args.max_memory_used_mib,
            max_utilization_percent=args.max_utilization_percent,
        )
        run_config.update(
            {
                "gpu_indices": gpu_indices,
                "gpu_preflight": {
                    "selected": selected,
                    "all_devices": all_devices,
                    "max_memory_used_mib": args.max_memory_used_mib,
                    "max_utilization_percent": args.max_utilization_percent,
                },
                "prompt_lengths": prompt_lengths,
                "batch_sizes": batch_sizes,
                "max_tokens": args.max_tokens,
                "max_steps_per_case": args.max_steps_per_case,
                "capture_debug_steps": bool(args.capture_debug_steps),
                "checkpoint_manifest": _checkpoint_manifest(args.model_path),
                "thresholds": {
                    "max_relative_l2": args.max_relative_l2,
                    "max_router_relative_l2": args.max_router_relative_l2,
                    "greedy_tokens": "exact",
                    "router_expert_ids": "exact",
                },
                "input_distribution": {
                    "token_ids": (
                        "Python random.Random uniform integers "
                        "[1000, min(vocab_size-1, 50000)]"
                    ),
                    "sampling": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "ignore_eos": True,
                    },
                },
                "prefix_config": {
                    "prefix_length": args.prefix_length,
                    "block_size": args.prefix_cache_block_size,
                    "max_blocks": args.prefix_cache_max_blocks,
                },
                "warmup_prompt_len": args.warmup_prompt_len,
            }
        )

        existing_debug_env = {
            key: os.environ[key]
            for key in DEBUG_ENV_KEYS
            if os.environ.get(key, "").strip() not in {"", "0"}
        }
        if existing_debug_env:
            raise ValueError(
                "Unset MiniMax debug environment variables before startup; the "
                f"script owns their values for worker startup: {existing_debug_env}."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        os.environ["SPARSEVLLM_DELTAKV_GRAPH_WARMUP_PROMPT_LEN"] = str(
            args.warmup_prompt_len
        )
        _enable_debug_runtime()
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "src"))
        import torch as torch_module

        torch = torch_module
        if not torch.cuda.is_available() or torch.cuda.device_count() != args.ep_size:
            raise RuntimeError(
                f"Expected exactly {args.ep_size} visible CUDA devices after preflight, "
                f"got {torch.cuda.device_count()}."
            )
        run_config["software"] = _software_versions(torch)
        if args.oracle_artifact is not None:
            oracle_path = args.oracle_artifact.expanduser().resolve()
            if not oracle_path.is_file():
                raise FileNotFoundError(f"Oracle artifact does not exist: {oracle_path}.")
            oracle_payload = torch.load(
                oracle_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(oracle_payload, dict) or "cases" not in oracle_payload:
                raise ValueError(
                    f"Oracle artifact has an invalid schema: {oracle_path}."
                )
            oracle_cases = oracle_payload["cases"]

        from sparsevllm import LLM, SamplingParams
        from sparsevllm.quantization.fp8 import (
            FINEGRAINED_FP8_KERNEL_REPO,
            FINEGRAINED_FP8_KERNEL_REVISION,
            FINEGRAINED_FP8_KERNEL_SOURCE_SHA256,
            FINEGRAINED_FP8_KERNEL_VERSION,
        )

        run_config["fp8_kernel"] = {
            "repo": FINEGRAINED_FP8_KERNEL_REPO,
            "version": FINEGRAINED_FP8_KERNEL_VERSION,
            "revision": FINEGRAINED_FP8_KERNEL_REVISION,
            "source_sha256": FINEGRAINED_FP8_KERNEL_SOURCE_SHA256,
            "local_override": os.getenv("SPARSEVLLM_FINEGRAINED_FP8_KERNEL_PATH"),
        }

        engine_kwargs = {
            "tensor_parallel_size": 1,
            "expert_parallel_size": args.ep_size,
            "data_parallel_size": 1,
            "moe_backend": args.backend,
            "sparse_method": "vanilla",
            "enforce_eager": not args.decode_cuda_graph,
            "decode_cuda_graph": args.decode_cuda_graph,
            "decode_cuda_graph_capture_sampling": False,
            "enable_prefix_caching": args.enable_prefix_caching,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs_in_batch": max(batch_sizes + [2]),
            "max_decoding_seqs": max(batch_sizes + [2]),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "engine_prefill_chunk_size": args.chunk_prefill_size,
            "prefix_cache_block_size": args.prefix_cache_block_size,
            "prefix_cache_max_blocks": args.prefix_cache_max_blocks,
            "throughput_log_interval_s": 0.0,
        }
        run_config["engine_kwargs"] = engine_kwargs
        llm = LLM(str(args.model_path), **engine_kwargs)

        def run_case(case_id: str, prompts: list[list[int]]) -> dict[str, Any]:
            request_result = None
            try:
                request_result = _run_requests(
                    llm,
                    SamplingParams,
                    prompts,
                    max_tokens=args.max_tokens,
                    max_steps=args.max_steps_per_case,
                    capture_debug_steps=args.capture_debug_steps,
                )
                raw_case = _collect_raw_case(
                    llm,
                    request_result,
                    use_graph=args.decode_cuda_graph,
                )
                parsed = _record_case_artifacts(
                    case_id,
                    raw_case,
                    raw_cases,
                    parsed_outputs,
                )
                correctness = None
                if oracle_cases is not None:
                    expected = oracle_cases.get(case_id)
                    if expected is None:
                        raise MetricFailure(f"Oracle has no case {case_id!r}.")
                    correctness = _compare_case(
                        torch,
                        raw_case,
                        expected,
                        max_relative_l2=args.max_relative_l2,
                        max_router_relative_l2=args.max_router_relative_l2,
                    )
                elif not parsed["logits"]["finite"]:
                    raise MetricFailure(f"Non-finite logits for {case_id}.")
                record = {
                    "case_id": case_id,
                    "status": "success",
                    "prompt_length": len(prompts[0]),
                    "batch_size": len(prompts),
                    "generated_token_ids": request_result["generated_token_ids"],
                    "prefix_cache_hit_lengths": request_result[
                        "prefix_cache_hit_lengths"
                    ],
                    "cache_stats_delta": request_result["cache_stats_delta"],
                    "graph_state_count": len(request_result["graph_states"]),
                    "last_graph_state": (
                        request_result["graph_states"][-1]
                        if request_result["graph_states"]
                        else None
                    ),
                    "correctness_vs_oracle": correctness,
                }
                records.append(record)
                return request_result
            except ParseFailure as exc:
                records.append(
                    {
                        "case_id": case_id,
                        "status": "parse_failed",
                        "prompt_length": len(prompts[0]),
                        "batch_size": len(prompts),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                raise
            except MetricFailure as exc:
                record = {
                    "case_id": case_id,
                    "status": "metric_failed",
                    "prompt_length": len(prompts[0]),
                    "batch_size": len(prompts),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "correctness_vs_oracle": exc.metrics,
                }
                if request_result is not None:
                    record.update(
                        {
                            "generated_token_ids": request_result[
                                "generated_token_ids"
                            ],
                            "prefix_cache_hit_lengths": request_result[
                                "prefix_cache_hit_lengths"
                            ],
                            "cache_stats_delta": request_result["cache_stats_delta"],
                            "graph_state_count": len(request_result["graph_states"]),
                            "last_graph_state": (
                                request_result["graph_states"][-1]
                                if request_result["graph_states"]
                                else None
                            ),
                        }
                    )
                records.append(record)
                if request_result is None:
                    raise
                return request_result
            except Exception as exc:
                records.append(
                    {
                        "case_id": case_id,
                        "status": "model_failed",
                        "prompt_length": len(prompts[0]),
                        "batch_size": len(prompts),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                raise

        vocab_size = int(llm.config.hf_config.vocab_size)
        for prompt_length in prompt_lengths:
            for batch_size in batch_sizes:
                case_id = f"prompt_{prompt_length}_batch_{batch_size}"
                prompts = _make_prompts(
                    prompt_length=prompt_length,
                    batch_size=batch_size,
                    seed=args.seed,
                    vocab_size=vocab_size,
                )
                run_case(case_id, prompts)

        if args.validate_prefix_patterns:
            rng = random.Random(args.seed + 700_001)
            high = min(vocab_size - 1, 50_000)
            base = [rng.randint(1000, high) for _ in range(args.prefix_length)]
            half = args.prefix_length // 2
            partial = base[:half] + [
                rng.randint(1000, high) for _ in range(args.prefix_length - half)
            ]
            no_hit = [rng.randint(1000, high) for _ in range(args.prefix_length)]
            shared = [
                base[:half]
                + [rng.randint(1000, high) for _ in range(args.prefix_length - half)]
                for _ in range(2)
            ]
            prefix_results = {
                "prefix_warm": run_case("prefix_warm", [base]),
                "prefix_exact": run_case("prefix_exact", [base]),
                "prefix_partial": run_case("prefix_partial", [partial]),
                "prefix_no_hit": run_case("prefix_no_hit", [no_hit]),
                "prefix_shared": run_case("prefix_shared", shared),
            }
            run_config["prefix_validation"] = _validate_prefix_patterns(
                prefix_results,
                use_graph=args.decode_cuda_graph,
            )

        replica_consistency = llm.debug_sparse_state_summaries()
        _validate_replica_consistency(replica_consistency)
        run_config["replica_consistency"] = replica_consistency
    except MetricFailure as exc:
        exit_code = 1
        if not records or records[-1]["status"] == "success":
            records.append(
                {
                    "case_id": "aggregate_validation",
                    "status": "metric_failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    except (ValueError, FileNotFoundError) as exc:
        exit_code = 2
        if not records or records[-1]["status"] == "success":
            records.append(
                {
                    "case_id": "preflight_or_input",
                    "status": "invalid_input",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    except Exception as exc:
        exit_code = 1
        if not records or records[-1]["status"] == "success":
            records.append(
                {
                    "case_id": "runtime",
                    "status": "model_failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    finally:
        if llm is not None:
            try:
                llm.exit()
            except Exception as exc:
                exit_code = 1
                records.append(
                    {
                        "case_id": "engine_exit",
                        "status": "model_failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        statuses = [record["status"] for record in records]
        invalid_statuses = sorted(set(statuses) - SAMPLE_STATUSES)
        if invalid_statuses:
            raise RuntimeError(f"Invalid result statuses: {invalid_statuses}.")
        if any(status == "metric_failed" for status in statuses):
            exit_code = 1
        aggregate_status = "success"
        for status in (
            "model_failed",
            "invalid_input",
            "parse_failed",
            "metric_failed",
            "skipped_by_policy",
        ):
            if status in statuses:
                aggregate_status = status
                break
        if not statuses:
            aggregate_status = "model_failed"
        run_config["completed_at"] = datetime.now().astimezone().isoformat()
        aggregate = {
            "status": aggregate_status,
            "num_cases": len(records),
            "success_cases": sum(record["status"] == "success" for record in records),
            "failed_cases": sum(record["status"] != "success" for record in records),
            "records": records,
        }
        _write_json(args.output_dir / "run_config.json", run_config)
        _write_json(args.output_dir / "parsed_outputs.json", parsed_outputs)
        _write_json(args.output_dir / "per_case_results.json", records)
        _write_json(args.output_dir / "aggregate_metrics.json", aggregate)
        if torch is None:
            import torch as torch_for_artifacts

            torch_for_artifacts.save(
                {"schema_version": 1, "cases": raw_cases},
                args.output_dir / "raw_outputs.pt",
            )
        else:
            torch.save(
                {"schema_version": 1, "cases": raw_cases},
                args.output_dir / "raw_outputs.pt",
            )
        _write_report(
            args.output_dir / "report.md",
            aggregate_status=aggregate_status,
            run_config=run_config,
            records=records,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
