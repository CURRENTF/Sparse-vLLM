#!/usr/bin/env python3
"""Validate one real GLM-4.7-Flash TP/EP runtime variant.

This script deliberately uses the public ``LLM`` entry point so worker launch,
weight loading, scheduling, collectives, and cache allocation follow the same
path as serving.  Cross-variant comparisons are performed separately from the
saved float32 logits artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one real GLM checkpoint TP/EP variant."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--expert_parallel_size", type=int, default=1)
    parser.add_argument(
        "--sparse_method",
        choices=(
            "vanilla",
            "streamingllm",
            "snapkv",
            "h2o",
            "omnikv",
            "rkv",
        ),
        default="vanilla",
    )
    parser.add_argument(
        "--decode_cuda_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--mlp_chunk_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=256)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_output(*args: str) -> str:
    command = ["git", *args]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            f"Git provenance command failed in {REPO_ROOT}: "
            f"{' '.join(command)}: {str(detail).strip()}"
        ) from exc
    return result.stdout.strip()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _tensor_evidence(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu().contiguous()
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "sha256": _tensor_sha256(value),
        "argmax": value.argmax(dim=-1).tolist(),
        "mean": float(value.mean().item()),
        "max": float(value.max().item()),
        "min": float(value.min().item()),
    }


def _prompt_token_ids(*, seed: int, prompt_len: int, vocab_size: int) -> list[int]:
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}.")
    if vocab_size <= 1024:
        raise ValueError(
            "GLM checkpoint validation expects a vocabulary larger than 1024, "
            f"got {vocab_size}."
        )
    usable = vocab_size - 512
    base = 256 + int(seed) % max(1, usable - 256)
    return [256 + ((base + 97 * index) % usable) for index in range(prompt_len)]


def _validate_parallel_evidence(
    summaries: list[dict[str, Any]],
    *,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    last_moe_layer: int,
    require_all_ep_ranks_hit: bool = True,
    require_decode_cuda_graph: bool = False,
    require_decode_cuda_graph_execution: bool = False,
) -> dict[str, Any]:
    from sparsevllm.distributed.parallel_context import (
        hybrid_moe_group_ranks,
        parallel_group_ranks,
    )

    tp_size = int(tensor_parallel_size)
    ep_size = int(expert_parallel_size)
    hybrid_moe = tp_size > 1 and ep_size > 1
    expected_world_size = tp_size if hybrid_moe else tp_size * ep_size
    if len(summaries) != expected_world_size:
        raise RuntimeError(
            "Expected one debug summary per world rank: "
            f"expected={expected_world_size} got={len(summaries)}."
        )

    world_ranks = sorted(int(item["world_rank"]) for item in summaries)
    if world_ranks != list(range(expected_world_size)):
        raise RuntimeError(f"Unexpected world ranks: {world_ranks}.")

    if hybrid_moe:
        raw_groups = hybrid_moe_group_ranks(
            outer_tp_size=tp_size,
            moe_ep_size=ep_size,
        )
        group_ranks = {
            "attention": raw_groups["attention"],
            "expert": raw_groups["moe_expert"],
            "moe_tensor": raw_groups["moe_tensor"],
            "data": raw_groups["data"],
        }
    else:
        raw_groups = parallel_group_ranks(
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=1,
        )
        group_ranks = {
            "attention": raw_groups["tensor"],
            "expert": raw_groups["expert"],
            "moe_tensor": raw_groups["tensor"],
            "data": raw_groups["data"],
        }

    def expected_group(dimension: str, world_rank: int) -> dict[str, Any]:
        ranks = next(
            ranks
            for ranks in group_ranks[dimension]
            if int(world_rank) in ranks
        )
        return {
            "rank": ranks.index(int(world_rank)),
            "size": len(ranks),
            "ranks": list(ranks),
        }

    local_ranges: list[tuple[int, int]] = []
    local_hit_counts: list[int] = []
    replica_checks: list[dict[str, Any]] = []
    layer_key = str(int(last_moe_layer))
    for summary in summaries:
        configured = summary["parallel"]["configured"]
        expected_configured = {
            "tensor_parallel_size": int(tensor_parallel_size),
            "expert_parallel_size": int(expert_parallel_size),
            "data_parallel_size": 1,
            "world_size": expected_world_size,
        }
        if configured != expected_configured:
            raise RuntimeError(
                "Configured parallel evidence mismatch: "
                f"expected={expected_configured} got={configured}."
            )
        effective = summary["parallel"]["effective"]
        world = effective["world"]
        if int(world["size"]) != expected_world_size:
            raise RuntimeError(f"Effective world-size mismatch: {world}.")
        if world != {
            "rank": int(summary["world_rank"]),
            "size": expected_world_size,
            "ranks": list(range(expected_world_size)),
        }:
            raise RuntimeError(
                "Effective world group mismatch: "
                f"world_rank={summary['world_rank']} group={world}."
            )
        for dimension in ("attention", "expert", "moe_tensor", "data"):
            expected = expected_group(dimension, int(summary["world_rank"]))
            if effective[dimension] != expected:
                raise RuntimeError(
                    f"Effective {dimension} group mismatch on world rank "
                    f"{summary['world_rank']}: expected={expected} "
                    f"got={effective[dimension]}."
                )

        layer = summary.get("moe_local", {}).get(layer_key)
        if layer is None:
            raise RuntimeError(
                f"Missing MoE debug evidence for layer={last_moe_layer} "
                f"on world_rank={summary['world_rank']}."
            )
        local_ranges.append(
            (int(layer["local_expert_start"]), int(layer["local_expert_end"]))
        )
        local_hit_counts.append(int(layer["local_hit_count"]))

        consistency = summary.get("replica_consistency")
        if not isinstance(consistency, dict):
            raise RuntimeError(
                "Missing replica-consistency evidence for "
                f"world_rank={summary['world_rank']}."
            )
        replica_checks.append(consistency)
        logits_tolerance_ratio = consistency["last_logits_tolerance_ratio"]
        if tensor_parallel_size > 1:
            if (
                logits_tolerance_ratio is not None
                or consistency.get("last_logits_comparison")
                != "not_applicable_tp_vocab_sharded"
            ):
                raise RuntimeError(
                    "TP-sharded logits must be marked not applicable for cross-rank "
                    f"comparison: world_rank={summary['world_rank']} "
                    f"evidence={consistency}."
                )
        elif logits_tolerance_ratio is None or float(logits_tolerance_ratio) > 1.0:
            raise RuntimeError(
                "Cross-rank logits exceeded the declared tolerance or were not "
                f"compared: world_rank={summary['world_rank']} evidence={consistency}."
            )
        moe_consistency = consistency.get("moe_layers", {}).get(layer_key)
        if moe_consistency is None:
            raise RuntimeError(
                f"Missing cross-rank MoE consistency for layer={layer_key}."
            )
        if bool(moe_consistency["topk_ids_mismatch"]):
            raise RuntimeError(
                f"Router top-k differs across ranks: {moe_consistency}."
            )
        for field in ("topk_weights_tolerance_ratio", "output_tolerance_ratio"):
            if float(moe_consistency[field]) > 1.0:
                raise RuntimeError(
                    f"Cross-rank MoE {field} exceeded tolerance: "
                    f"{moe_consistency}."
                )
        graph = summary.get("decode_cuda_graph")
        if not isinstance(graph, dict):
            raise RuntimeError(
                "Missing decode CUDA Graph evidence on world rank "
                f"{summary['world_rank']}."
            )
        if bool(graph.get("enabled")) != bool(require_decode_cuda_graph):
            raise RuntimeError(
                "Decode CUDA Graph config evidence mismatch on world rank "
                f"{summary['world_rank']}: {graph}."
            )
        if require_decode_cuda_graph_execution:
            if not require_decode_cuda_graph:
                raise ValueError(
                    "Requiring decode CUDA Graph execution also requires the "
                    "graph config to be enabled."
                )
            if int(graph.get("capture_count", 0)) <= 0:
                raise RuntimeError(
                    f"World rank {summary['world_rank']} did not capture a decode graph."
                )
            if int(graph.get("replay_count", 0)) <= 0:
                raise RuntimeError(
                    f"World rank {summary['world_rank']} did not replay a decode graph."
                )
            if int(graph.get("eager_static_count", 0)) != 0 or int(
                graph.get("force_eager_count", 0)
            ) != 0:
                raise RuntimeError(
                    "Decode CUDA Graph used an eager path on world rank "
                    f"{summary['world_rank']}: {graph}."
                )

    if ep_size > 1:
        expected_width = 64 // ep_size
        expected_ranges = [
            (rank * expected_width, (rank + 1) * expected_width)
            for rank in range(ep_size)
            for _ in range(tp_size // ep_size if hybrid_moe else tp_size)
        ]
        if sorted(local_ranges) != expected_ranges:
            raise RuntimeError(
                "EP local expert shards do not partition 64 experts: "
                f"expected={expected_ranges} got={sorted(local_ranges)}."
            )
        hits_by_range: dict[tuple[int, int], int] = {}
        for local_range, hit_count in zip(local_ranges, local_hit_counts):
            hits_by_range[local_range] = (
                hits_by_range.get(local_range, 0) + int(hit_count)
            )
        if require_all_ep_ranks_hit and any(
            hit_count <= 0 for hit_count in hits_by_range.values()
        ):
            raise RuntimeError(
                "At least one EP rank did not execute a local routed expert: "
                f"hits_by_range={hits_by_range}."
            )
        expected_replicated_attention = tp_size == 1
        if any(
            bool(summary["parallel"]["attention_replicated_for_ep"])
            != expected_replicated_attention
            for summary in summaries
        ):
            raise RuntimeError(
                "EP replicated-attention evidence does not match the selected "
                f"layout: TP={tp_size}, EP={ep_size}."
            )

    return {
        "world_ranks": world_ranks,
        "local_expert_ranges": [list(value) for value in local_ranges],
        "local_expert_hit_counts": local_hit_counts,
        "all_ep_ranks_hit": all(hit_count > 0 for hit_count in local_hit_counts),
        "hybrid_moe": hybrid_moe,
        "expected_groups": {
            key: [list(ranks) for ranks in value]
            for key, value in group_ranks.items()
        },
        "replica_checks": replica_checks,
    }


def _run(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    os.environ["SPARSEVLLM_DEBUG_RUNTIME"] = "1"
    os.environ["SPARSEVLLM_DEBUG_MOE"] = "1"

    from sparsevllm import LLM, SamplingParams

    tp_size = int(args.tensor_parallel_size)
    ep_size = int(args.expert_parallel_size)
    if tp_size <= 0 or ep_size <= 0:
        raise ValueError(f"TP and EP must be positive, got TP={tp_size}, EP={ep_size}.")
    if tp_size > 1 and ep_size > 1 and tp_size % ep_size:
        raise ValueError(
            "Joint GLM outer TP and MoE EP requires TP divisible by EP, got "
            f"TP={tp_size}, EP={ep_size}."
        )
    expected_devices = (
        tp_size if tp_size > 1 and ep_size > 1 else tp_size * ep_size
    )
    actual_devices = torch.cuda.device_count()
    if actual_devices != expected_devices:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the requested world size: "
            f"expected={expected_devices} got={actual_devices}."
        )

    torch.manual_seed(int(args.seed))
    max_model_len = int(args.prompt_len) + int(args.max_tokens) + 16
    llm = None
    try:
        llm = LLM(
            args.model_path,
            tensor_parallel_size=tp_size,
            expert_parallel_size=ep_size,
            data_parallel_size=1,
            sparse_method=args.sparse_method,
            enforce_eager=True,
            decode_cuda_graph=bool(args.decode_cuda_graph),
            max_model_len=max_model_len,
            max_num_seqs_in_batch=1,
            max_decoding_seqs=1,
            max_num_batched_tokens=max(int(args.prompt_len), 1),
            engine_prefill_chunk_size=max(int(args.prompt_len), 1),
            mlp_chunk_size=int(args.mlp_chunk_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            throughput_log_interval_s=0.0,
            weight_loading_workers=1,
        )
        prompt = _prompt_token_ids(
            seed=int(args.seed),
            prompt_len=int(args.prompt_len),
            vocab_size=int(llm.config.hf_config.vocab_size),
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=int(args.max_tokens),
        )
        llm.add_request(prompt, sampling_params)

        prefill_logits = None
        prefill_rank_summaries = None
        finished_outputs: list[tuple[int, list[int], Any, Any]] = []
        steps = 0
        while not llm.is_finished():
            step_outputs, num_tokens = llm.step()
            steps += 1
            if steps > int(args.max_steps):
                raise RuntimeError(
                    f"Validation exceeded max_steps={int(args.max_steps)}."
                )
            if (
                num_tokens > 0
                and llm.last_step_token_outputs
                and prefill_rank_summaries is None
            ):
                prefill_logits = llm.debug_last_logits().detach().float().cpu()
                prefill_rank_summaries = llm.debug_sparse_state_summaries()
            finished_outputs.extend(step_outputs)

        if prefill_logits is None or prefill_rank_summaries is None:
            raise RuntimeError("The run completed without final-prefill evidence.")
        if len(finished_outputs) != 1:
            raise RuntimeError(
                f"Expected one finished request, got {len(finished_outputs)}."
            )
        generated_ids = [int(value) for value in finished_outputs[0][1]]
        if len(generated_ids) != int(args.max_tokens):
            raise RuntimeError(
                "Greedy generation returned an unexpected token count: "
                f"expected={int(args.max_tokens)} got={len(generated_ids)}."
            )
        final_logits = llm.debug_last_logits().detach().float().cpu()
        final_rank_summaries = llm.debug_sparse_state_summaries()
        last_moe_layer = int(llm.config.hf_config.num_hidden_layers) - 1

        prefill_parallel = _validate_parallel_evidence(
            prefill_rank_summaries,
            tensor_parallel_size=tp_size,
            expert_parallel_size=ep_size,
            last_moe_layer=last_moe_layer,
            require_decode_cuda_graph=bool(args.decode_cuda_graph),
        )
        final_parallel = _validate_parallel_evidence(
            final_rank_summaries,
            tensor_parallel_size=tp_size,
            expert_parallel_size=ep_size,
            last_moe_layer=last_moe_layer,
            # A single decode token selects only top-k experts and does not
            # have to cover every EP shard. Prefill coverage above proves all
            # local expert partitions execute; keep the final per-rank counts
            # as evidence without imposing an invalid per-token requirement.
            require_all_ep_ranks_hit=False,
            require_decode_cuda_graph=bool(args.decode_cuda_graph),
            require_decode_cuda_graph_execution=bool(args.decode_cuda_graph),
        )

        torch.save(
            {
                "prefill": prefill_logits,
                "final_decode": final_logits,
            },
            output_dir / "logits.pt",
        )
        _write_json(output_dir / "prefill_rank_summaries.json", prefill_rank_summaries)
        _write_json(output_dir / "final_rank_summaries.json", final_rank_summaries)

        return {
            "status": "success",
            "real_checkpoint": True,
            "model_path": str(Path(args.model_path).resolve()),
            "tensor_parallel_size": tp_size,
            "expert_parallel_size": ep_size,
            "data_parallel_size": 1,
            "world_size": expected_devices,
            "sparse_method": str(args.sparse_method),
            "decode_cuda_graph": bool(args.decode_cuda_graph),
            "seed": int(args.seed),
            "prompt_len": len(prompt),
            "prompt_sha256": hashlib.sha256(
                torch.tensor(prompt, dtype=torch.int64).numpy().tobytes()
            ).hexdigest(),
            "max_tokens": int(args.max_tokens),
            "generated_token_ids": generated_ids,
            "steps": steps,
            "prefill_logits": _tensor_evidence(prefill_logits),
            "final_decode_logits": _tensor_evidence(final_logits),
            "prefill_parallel": prefill_parallel,
            "final_parallel": final_parallel,
            "logits_artifact": str(output_dir / "logits.pt"),
        }
    finally:
        if llm is not None:
            llm.exit()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "args": vars(args),
    }
    _write_json(output_dir / "run_info.json", run_info)
    try:
        summary = _run(args, output_dir)
    except BaseException as exc:
        failure = {
            "status": "model_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "summary.json", failure)
        run_info.update(
            status="model_failed",
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            error=failure,
        )
        _write_json(output_dir / "run_info.json", run_info)
        raise
    _write_json(output_dir / "summary.json", summary)
    run_info.update(
        status="success",
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        summary_path=str(output_dir / "summary.json"),
    )
    _write_json(output_dir / "run_info.json", run_info)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
