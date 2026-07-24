#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from deltakv.configs.runtime_params import normalize_runtime_params
from deltakv.configs.default_paths import compressor_path, model_path, output_path
from deltakv.get_chat_api import get_generate_api
from benchmark.long_bench.pred import build_chat
from sparsevllm.config import Config
from sparsevllm.engine.model_runner import ModelRunner, make_tp_shm_name
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import (
    PREFILL_POLICY_ALL_CHUNKED,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    is_deltakv_method,
)
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.debug.tiny_random import (
    TINY_RANDOM_CONFIG_ENV,
    TINY_RANDOM_ENV,
    TINY_RANDOM_SEED_ENV,
    apply_tiny_random_overrides,
    build_tiny_random_hf_model,
)
from sparsevllm.utils.context import get_context, reset_context


DEFAULT_MODEL = model_path("Qwen2.5-7B-Instruct-1M")
DEFAULT_COMPRESSOR = compressor_path("Qwen2.5-7B-Instruct-1M-Compressor")
DEFAULT_OUTPUT_ROOT = output_path("sparsevllm_logits_align")

DIRECT_HF_METHODS = {
    "vanilla",
    "omnikv",
    "snapkv",
    "pyramidkv",
    "quest",
    "streamingllm",
    "attention-sink",
    "attention_sink",
}
STANDARD_SPARSE_METHODS = {
    "vanilla",
    "streamingllm",
    "attention-sink",
    "attention_sink",
    "snapkv",
    "pyramidkv",
    "quest",
    "omnikv",
}
DELTAKV_COMPARE_METHODS = {
    "deltakv",
    "deltakv-less-memory",
    "deltakv-less-memory-cudagraph",
}
HF_FULL_LAYER_KIVI_METHOD = "delta_compressed_quant_kivi_full_fp8_ref"


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _git_status_short() -> str:
    import subprocess

    return subprocess.check_output(["git", "status", "--short"], text=True).strip()


def _require_path(path: str, kind: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _parse_cases(value: str) -> list[str]:
    cases = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"short", "long", "longbench", "longbench_batch", "synthetic_batch"}
    bad = sorted(set(cases) - allowed)
    if bad:
        raise ValueError(f"Unsupported cases: {bad}. Allowed: {sorted(allowed)}")
    return cases


def _parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _load_longbench_task(args: argparse.Namespace) -> tuple[str, str, str, list[dict[str, Any]]]:
    task = args.longbench_task
    if not task:
        raise ValueError("--longbench_task is required when --cases includes longbench or longbench_batch.")

    if not args.longbench_data_dir:
        raise FileNotFoundError(
            "LongBench data root is not configured.\n"
            "Set DELTAKV_LONGBENCH_DATA_DIR or DELTAKV_DATA_DIR, or pass --longbench_data_dir "
            "to the LongBench root directory that contains data/*.jsonl."
        )
    data_root = Path(args.longbench_data_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"LongBench data root does not exist: {data_root}")

    prompt_path = Path("benchmark/long_bench/config/dataset2prompt.json")
    with prompt_path.open("r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    if task not in dataset2prompt:
        raise ValueError(f"Unknown LongBench task {task!r}; available tasks include {sorted(dataset2prompt)[:8]}...")

    data_path = str(data_root / "data" / f"{task}.jsonl")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"LongBench task file does not exist: {data_path}")

    samples: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"LongBench task file is empty: {data_path}")
    return task, data_path, dataset2prompt[task], samples


def _build_longbench_prompt_from_sample(
    args: argparse.Namespace,
    tokenizer,
    *,
    task: str,
    data_path: str,
    prompt_template: str,
    sample_idx: int,
    sample: dict[str, Any],
) -> tuple[str, list[int], dict[str, Any]]:
    if sample_idx < 0:
        raise ValueError(f"LongBench sample index must be >= 0, got {sample_idx}.")
    max_length = int(args.longbench_max_length)
    if max_length <= 0:
        raise ValueError(f"--longbench_max_length must be > 0, got {max_length}.")

    prompt = prompt_template.format(**sample)
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized[-half:], skip_special_tokens=True
        )
    prompt = build_chat(
        tokenizer,
        prompt,
        task,
        no_chat_template=bool(args.no_chat_template),
        thinking_mode=args.thinking_mode,
    )
    add_special_tokens = True
    if tokenizer.bos_token is None or prompt.startswith(tokenizer.bos_token):
        add_special_tokens = False
    token_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not token_ids:
        raise ValueError(f"LongBench task {task} sample {sample_idx} tokenized to an empty sequence.")
    meta = {
        "task": task,
        "sample_idx": int(sample_idx),
        "answers": sample.get("answers"),
        "all_classes": sample.get("all_classes"),
        "length": sample.get("length"),
        "data_path": data_path,
    }
    return prompt, token_ids, meta


def _build_longbench_prompt(args: argparse.Namespace, tokenizer) -> tuple[str, list[int], dict[str, Any]]:
    task, data_path, prompt_template, samples = _load_longbench_task(args)
    sample_idx = int(args.longbench_sample_idx)
    if sample_idx >= len(samples):
        raise IndexError(f"LongBench task {task} has no sample index {sample_idx}.")
    return _build_longbench_prompt_from_sample(
        args,
        tokenizer,
        task=task,
        data_path=data_path,
        prompt_template=prompt_template,
        sample_idx=sample_idx,
        sample=samples[sample_idx],
    )


def _build_prompt(tokenizer, case_name: str, target_tokens: int, args: argparse.Namespace) -> tuple[str, list[int], dict[str, Any] | None]:
    if case_name == "short":
        prompt = "The capital of France is"
        meta = None
    elif case_name == "long":
        unit = (
            "Sparse long-context inference compares cache layouts, attention masks, "
            "position ids, and compressed DeltaKV reconstruction. "
        )
        prompt = unit
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        while len(token_ids) < target_tokens:
            prompt += unit
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt = tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=False)
        meta = None
    elif case_name == "longbench":
        return _build_longbench_prompt(args, tokenizer)
    else:
        raise ValueError(f"Unsupported case_name={case_name!r}")

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"{case_name} prompt tokenized to an empty sequence.")
    return prompt, token_ids, meta


def _parse_longbench_batch_sample_indices(value: str | None) -> list[list[int]]:
    raw = str(value or "").strip()
    if not raw:
        return []
    batches: list[list[int]] = []
    for batch_part in raw.split(";"):
        batch_part = batch_part.strip()
        if not batch_part:
            continue
        batch = [int(part.strip()) for part in batch_part.split(",") if part.strip()]
        if not batch:
            raise ValueError(f"Empty batch in --longbench_batch_sample_indices={value!r}.")
        if any(idx < 0 for idx in batch):
            raise ValueError(f"LongBench batch sample indices must be >= 0, got {batch}.")
        if len(set(batch)) != len(batch):
            raise ValueError(f"Duplicate sample index inside one batch is not allowed: {batch}.")
        batches.append(batch)
    if not batches:
        raise ValueError(f"--longbench_batch_sample_indices did not contain any sample indices: {value!r}.")
    return batches


def _parse_longbench_batch_sizes(value: str | None) -> list[int]:
    sizes = _parse_int_list(value)
    if not sizes:
        raise ValueError("--longbench_batch_sizes must contain at least one positive batch size.")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"--longbench_batch_sizes must contain positive integers, got {sizes}.")
    return sizes


def _validate_longbench_batches(args: argparse.Namespace, batches: list[list[dict[str, Any]]]) -> None:
    if not batches:
        raise ValueError("LongBench batch alignment requires at least one batch.")
    batch_sizes = [len(batch) for batch in batches]
    if any(size <= 0 for size in batch_sizes):
        raise ValueError(f"LongBench batch alignment got an empty batch: batch_sizes={batch_sizes}.")
    if bool(args.longbench_batch_require_varied_batch_sizes) and len(batch_sizes) > 1 and len(set(batch_sizes)) == 1:
        raise ValueError(
            "LongBench batch alignment requires varied batch sizes by default; "
            f"got batch_sizes={batch_sizes}. Pass --no-longbench_batch_require_varied_batch_sizes to disable."
        )
    if bool(args.longbench_batch_require_varied_lengths):
        for batch_idx, batch in enumerate(batches):
            lengths = [int(row["prompt_tokens"]) for row in batch]
            if len(lengths) > 1 and len(set(lengths)) != len(lengths):
                raise ValueError(
                    "LongBench batch alignment requires unique prompt token lengths inside each multi-row batch; "
                    f"batch_idx={batch_idx}, prompt_tokens={lengths}. "
                    "Pass --no-longbench_batch_require_varied_lengths to disable."
                )


def _longbench_row_from_sample(
    args: argparse.Namespace,
    tokenizer,
    *,
    task: str,
    data_path: str,
    prompt_template: str,
    sample_idx: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    prompt, token_ids, meta = _build_longbench_prompt_from_sample(
        args,
        tokenizer,
        task=task,
        data_path=data_path,
        prompt_template=prompt_template,
        sample_idx=sample_idx,
        sample=sample,
    )
    return {
        "sample_idx": int(sample_idx),
        "prompt": prompt,
        "input_ids": token_ids,
        "prompt_tokens": int(len(token_ids)),
        "prompt_preview": prompt[:240],
        "prompt_meta": meta,
    }


def _build_longbench_batch_rows(args: argparse.Namespace, tokenizer) -> list[list[dict[str, Any]]]:
    task, data_path, prompt_template, samples = _load_longbench_task(args)
    explicit_batches = _parse_longbench_batch_sample_indices(args.longbench_batch_sample_indices)
    batches: list[list[dict[str, Any]]] = []
    if explicit_batches:
        for batch_indices in explicit_batches:
            batch: list[dict[str, Any]] = []
            for sample_idx in batch_indices:
                if sample_idx >= len(samples):
                    raise IndexError(f"LongBench task {task} has no sample index {sample_idx}.")
                batch.append(
                    _longbench_row_from_sample(
                        args,
                        tokenizer,
                        task=task,
                        data_path=data_path,
                        prompt_template=prompt_template,
                        sample_idx=sample_idx,
                        sample=samples[sample_idx],
                    )
                )
            batches.append(batch)
        _validate_longbench_batches(args, batches)
        return batches

    sizes = _parse_longbench_batch_sizes(args.longbench_batch_sizes)
    start_idx = int(args.longbench_batch_start_idx)
    candidate_count = int(args.longbench_batch_candidate_count)
    if start_idx < 0:
        raise ValueError(f"--longbench_batch_start_idx must be >= 0, got {start_idx}.")
    if candidate_count <= 0:
        raise ValueError(f"--longbench_batch_candidate_count must be > 0, got {candidate_count}.")
    end_idx = min(len(samples), start_idx + candidate_count)
    if start_idx >= end_idx:
        raise IndexError(
            f"LongBench task {task} has no samples in requested candidate window: "
            f"start_idx={start_idx}, candidate_count={candidate_count}, total={len(samples)}."
        )

    candidates = [
        _longbench_row_from_sample(
            args,
            tokenizer,
            task=task,
            data_path=data_path,
            prompt_template=prompt_template,
            sample_idx=sample_idx,
            sample=samples[sample_idx],
        )
        for sample_idx in range(start_idx, end_idx)
    ]
    cursor = 0
    used_sample_indices: set[int] = set()
    for batch_size in sizes:
        batch: list[dict[str, Any]] = []
        used_lengths: set[int] = set()
        while cursor < len(candidates) and len(batch) < batch_size:
            row = candidates[cursor]
            cursor += 1
            if int(row["sample_idx"]) in used_sample_indices:
                continue
            if (
                bool(args.longbench_batch_require_varied_lengths)
                and batch_size > 1
                and int(row["prompt_tokens"]) in used_lengths
            ):
                continue
            batch.append(row)
            used_sample_indices.add(int(row["sample_idx"]))
            used_lengths.add(int(row["prompt_tokens"]))
        if len(batch) != batch_size:
            raise RuntimeError(
                "Could not auto-select enough LongBench samples for a varied-length batch. "
                f"requested_batch_size={batch_size}, selected={len(batch)}, "
                f"candidate_window=[{start_idx}, {end_idx}). "
                "Increase --longbench_batch_candidate_count, disable varied lengths, or pass explicit "
                "--longbench_batch_sample_indices."
            )
        batches.append(batch)
    _validate_longbench_batches(args, batches)
    return batches


def _validate_longbench_batch_debug_args(args: argparse.Namespace) -> None:
    unsupported: list[str] = []
    for name in (
        "compressed_state_layers",
        "full_kivi_debug_layers",
        "hidden_debug_layers",
        "qk_debug_layers",
    ):
        if getattr(args, name):
            unsupported.append(f"--{name}")
    if bool(args.save_hidden_debug_vectors):
        unsupported.append("--save_hidden_debug_vectors")
    if unsupported:
        raise ValueError(
            "longbench_batch currently supports logits and sparse metadata alignment only. "
            f"Unsupported debug args in batch mode: {', '.join(unsupported)}."
        )


def _parse_synthetic_batch_lengths(value: str | None) -> list[list[int]]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("--synthetic_batch_lengths must not be empty.")
    batches: list[list[int]] = []
    for batch_part in raw.split(";"):
        batch_part = batch_part.strip()
        if not batch_part:
            continue
        batch = [int(part.strip()) for part in batch_part.split(",") if part.strip()]
        if not batch:
            raise ValueError(f"Empty batch in --synthetic_batch_lengths={value!r}.")
        if any(length <= 0 for length in batch):
            raise ValueError(f"Synthetic batch lengths must be positive, got {batch}.")
        batches.append(batch)
    if not batches:
        raise ValueError(f"--synthetic_batch_lengths did not contain any lengths: {value!r}.")
    return batches


def _synthetic_token_bounds(args: argparse.Namespace, tokenizer) -> tuple[int, int, int]:
    vocab_size = int(len(tokenizer))
    token_low = int(args.synthetic_token_low)
    token_high = vocab_size if args.synthetic_token_high is None else int(args.synthetic_token_high)
    if token_low < 0:
        raise ValueError(f"--synthetic_token_low must be >= 0, got {token_low}.")
    if token_high > vocab_size:
        raise ValueError(f"--synthetic_token_high must be <= tokenizer size {vocab_size}, got {token_high}.")
    if token_high <= token_low:
        raise ValueError(
            f"Synthetic token range must be non-empty, got low={token_low}, high={token_high}."
        )

    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None}
    replacement = None
    for token_id in range(token_low, token_high):
        if token_id not in special_ids:
            replacement = int(token_id)
            break
    if replacement is None:
        raise ValueError(
            "Synthetic token range contains only tokenizer special ids: "
            f"low={token_low}, high={token_high}."
        )
    return token_low, token_high, replacement


def _synthetic_token_ids(
    *,
    length: int,
    seed: int,
    token_low: int,
    token_high: int,
    replacement_token_id: int,
    tokenizer,
) -> list[int]:
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    ids = torch.randint(
        int(token_low),
        int(token_high),
        (int(length),),
        generator=generator,
        dtype=torch.long,
    ).tolist()
    if special_ids:
        ids = [int(token_id) if int(token_id) not in special_ids else int(replacement_token_id) for token_id in ids]
    return ids


def _validate_synthetic_decode_batches(
    args: argparse.Namespace,
    batches: list[list[dict[str, Any]]],
    method: str,
) -> None:
    sparse_method = method if method in STANDARD_SPARSE_METHODS or is_deltakv_method(method) else args.sparse_method
    if sparse_method in {"streamingllm", "attention-sink", "attention_sink"}:
        decode_threshold = int(args.sink_keep_tokens) + int(args.recent_keep_tokens)
    else:
        decode_threshold = int(args.sink_keep_tokens) + int(args.recent_keep_tokens) + int(args.decode_keep_tokens)
    first_decode_extra = 1
    for batch_idx, batch in enumerate(batches):
        flags = [int(row["prompt_tokens"]) + first_decode_extra > decode_threshold for row in batch]
        if len(set(flags)) > 1:
            raise ValueError(
                "Synthetic batch mixes short and long decode rows, which Sparse-VLLM scheduler rejects. "
                f"batch_idx={batch_idx}, prompt_tokens={[int(row['prompt_tokens']) for row in batch]}, "
                f"decode_threshold={decode_threshold}. Split these lengths into separate batches."
            )


def _build_synthetic_batch_rows(
    args: argparse.Namespace,
    tokenizer,
    method: str,
) -> list[list[dict[str, Any]]]:
    length_batches = _parse_synthetic_batch_lengths(args.synthetic_batch_lengths)
    token_low, token_high, replacement_token_id = _synthetic_token_bounds(args, tokenizer)
    max_model_len = int(args.max_model_len)
    if max_model_len <= 0:
        raise ValueError(f"--max_model_len must be > 0 for synthetic_batch, got {max_model_len}.")

    batches: list[list[dict[str, Any]]] = []
    sample_idx = 0
    for batch_idx, lengths in enumerate(length_batches):
        batch: list[dict[str, Any]] = []
        for row_idx, length in enumerate(lengths):
            length = int(length)
            if length > max_model_len:
                raise ValueError(
                    "Synthetic prompt length exceeds --max_model_len: "
                    f"length={length}, max_model_len={max_model_len}."
                )
            seed = int(args.synthetic_batch_seed) + batch_idx * 1_000_003 + row_idx * 10_009 + length
            token_ids = _synthetic_token_ids(
                length=length,
                seed=seed,
                token_low=token_low,
                token_high=token_high,
                replacement_token_id=replacement_token_id,
                tokenizer=tokenizer,
            )
            meta = {
                "source": "synthetic_random_token_ids",
                "sample_idx": sample_idx,
                "length": length,
                "seed": seed,
                "token_low": token_low,
                "token_high": token_high,
                "replacement_token_id": replacement_token_id,
                "first_token_ids": token_ids[:16],
            }
            batch.append(
                {
                    "sample_idx": int(sample_idx),
                    "input_ids": token_ids,
                    "prompt_tokens": length,
                    "prompt_preview": (
                        f"synthetic_random_token_ids length={length} seed={seed} "
                        f"first_token_ids={token_ids[:16]}"
                    ),
                    "prompt_meta": meta,
                }
            )
            sample_idx += 1
        batches.append(batch)
    _validate_longbench_batches(args, batches)
    _validate_synthetic_decode_batches(args, batches, method)
    return batches


def _compare_logits(
    hf_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
    *,
    tokenizer,
    topk_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, Any]:
    hf = hf_logits.detach().float().cpu().view(-1)
    sv = sparse_logits.detach().float().cpu().view(-1)
    if hf.shape != sv.shape:
        raise ValueError(f"Logit shape mismatch: hf={tuple(hf.shape)} sparse={tuple(sv.shape)}")

    diff = (hf - sv).abs()
    hf_argmax = int(torch.argmax(hf).item())
    sv_argmax = int(torch.argmax(sv).item())
    metrics: dict[str, Any] = {
        "shape": list(hf.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "median_abs_diff": float(torch.quantile(diff, 0.5).item()),
        "p99_abs_diff": float(torch.quantile(diff, 0.99).item()),
        "argmax_match": hf_argmax == sv_argmax,
        "hf_argmax": hf_argmax,
        "sparse_argmax": sv_argmax,
        "hf_argmax_text": tokenizer.decode([hf_argmax], skip_special_tokens=False),
        "sparse_argmax_text": tokenizer.decode([sv_argmax], skip_special_tokens=False),
        "hf_argmax_logit": float(hf[hf_argmax].item()),
        "sparse_argmax_logit": float(sv[sv_argmax].item()),
        "topk_overlap": {},
    }

    vocab = int(hf.numel())
    for k in topk_values:
        kk = min(int(k), vocab)
        hf_top = set(int(x) for x in torch.topk(hf, kk).indices.tolist())
        sv_top = set(int(x) for x in torch.topk(sv, kk).indices.tolist())
        metrics["topk_overlap"][str(k)] = {
            "intersection": len(hf_top & sv_top),
            "ratio": len(hf_top & sv_top) / max(kk, 1),
        }

    union_top = []
    for token_id in torch.topk(hf, min(10, vocab)).indices.tolist():
        if int(token_id) not in union_top:
            union_top.append(int(token_id))
    for token_id in torch.topk(sv, min(10, vocab)).indices.tolist():
        if int(token_id) not in union_top:
            union_top.append(int(token_id))

    metrics["top_token_diffs"] = [
        {
            "token_id": token_id,
            "text": tokenizer.decode([token_id], skip_special_tokens=False),
            "hf_logit": float(hf[token_id].item()),
            "sparse_logit": float(sv[token_id].item()),
            "abs_diff": float(diff[token_id].item()),
        }
        for token_id in union_top[:20]
    ]
    return metrics


def _tensor_preview(tensor: torch.Tensor | None, limit: int = 16) -> dict[str, Any] | None:
    if tensor is None:
        return None
    t = tensor.detach().cpu()
    flat = t.reshape(-1)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "preview": [int(x) if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long) else float(x) for x in flat[:limit].tolist()],
    }


def _collect_hf_deltakv_metadata(past_key_values) -> dict[str, Any]:
    meta: dict[str, Any] = {"cache_class": type(past_key_values).__name__}
    if hasattr(past_key_values, "get_compressed_length"):
        compressed = {}
        for layer_idx in range(28):
            try:
                compressed[str(layer_idx)] = int(past_key_values.get_compressed_length(layer_idx))
            except Exception:
                break
        meta["compressed_lens"] = compressed
    if hasattr(past_key_values, "top_token_idx"):
        top = {}
        for layer_idx, tensor in sorted(past_key_values.top_token_idx.items()):
            top[str(layer_idx)] = _tensor_preview(tensor, limit=4096)
        meta["top_token_idx"] = top
    if hasattr(past_key_values, "token_scores"):
        scores = {}
        for layer_idx, tensor in sorted(past_key_values.token_scores.items()):
            scores[str(layer_idx)] = _tensor_preview(tensor, limit=50000)
        meta["token_scores"] = scores
    if hasattr(past_key_values, "get_buffer_valid_lengths"):
        buffers = {}
        for layer_idx in range(28):
            try:
                lens = past_key_values.get_buffer_valid_lengths(layer_idx)
            except Exception:
                continue
            if lens is not None:
                buffers[str(layer_idx)] = _tensor_preview(lens)
        meta["buffer_lens"] = buffers
    return meta


def _collect_sparse_deltakv_metadata(runner: ModelRunner, seq: Sequence) -> dict[str, Any]:
    cm = runner.cache_manager
    sc = runner.sparse_controller
    meta: dict[str, Any] = {
        "cache_manager_class": type(cm).__name__,
        "row_idx": None,
        "row_seq_len": None,
        "row_deltakv_compressed_len": None,
        "pre_rope_store_debug": {
            "calls": int(getattr(cm, "_debug_pre_rope_store_calls", 0) or 0),
            "writes": int(getattr(cm, "_debug_pre_rope_store_writes", 0) or 0),
            "source_max_abs_diff": float(getattr(cm, "_debug_pre_rope_source_max_abs_diff", 0.0) or 0.0),
            "source_mean_abs_diff_last": float(getattr(cm, "_debug_pre_rope_source_mean_abs_diff_last", 0.0) or 0.0),
            "layers": getattr(cm, "_debug_pre_rope_layers", {}),
        },
        "active_compressed_indices": {},
        "attn_score": {},
        "dynamic_selection_debug": getattr(sc, "debug_dynamic_selection", {}),
    }
    seq_id_to_row = getattr(cm, "seq_id_to_row", {})
    if isinstance(seq_id_to_row, list):
        seq_id_to_row = seq_id_to_row[0] if seq_id_to_row else {}
    row_idx = seq_id_to_row.get(seq.seq_id)
    if row_idx is not None:
        row_idx = int(row_idx)
        meta["row_idx"] = row_idx
        if hasattr(cm, "row_seq_lens"):
            row_seq_lens = cm.row_seq_lens
            if isinstance(row_seq_lens, list):
                row_seq_lens = row_seq_lens[0] if row_seq_lens else None
            if row_seq_lens is not None:
                meta["row_seq_len"] = int(row_seq_lens[row_idx])
        if hasattr(cm, "row_deltakv_compressed_lens"):
            meta["row_deltakv_compressed_len"] = int(cm.row_deltakv_compressed_lens[row_idx])
        if hasattr(cm, "sparse_layer_raw_slots_map"):
            cur_len = int(meta["row_seq_len"] or 0)
            raw = cm.sparse_layer_raw_slots_map[row_idx, :cur_len]
            latent = cm.sparse_layer_latent_slots_map[row_idx, :cur_len]
            meta["raw_slot_count"] = int((raw >= 0).sum().item())
            meta["latent_slot_count"] = int((latent >= 0).sum().item())
    for layer_idx, state in sorted(sc.layer_batch_sparse_states.items()):
        if state.active_compressed_indices is not None:
            meta["active_compressed_indices"][str(layer_idx)] = _tensor_preview(state.active_compressed_indices, limit=4096)
        if state.attn_score is not None:
            meta["attn_score"][str(layer_idx)] = _tensor_preview(state.attn_score, limit=50000)
    return meta


def _center_positions_cpu(
    *,
    start: int,
    seq_len: int,
    base_step: int,
) -> torch.Tensor:
    if seq_len <= 0:
        return torch.empty((0,), dtype=torch.long)
    pos = int(start)
    end = int(start) + int(seq_len)
    centers: list[int] = []
    while pos < end:
        centers.append(pos)
        pos += max(1, int(base_step))
    return torch.tensor(centers, dtype=torch.long)


def _dequantize_hf_payload(past_key_values, layer_idx: int, payload: torch.Tensor, k_dim: int) -> torch.Tensor:
    quant_bits = int(past_key_values._layer_quant_bits(layer_idx))
    dim = 2 * k_dim if past_key_values._layer_origin_codec(layer_idx) else int(past_key_values.config.kv_compressed_size)
    if quant_bits in (2, 4):
        scale = past_key_values.comp_kv_scales[layer_idx]
        mn = past_key_values.comp_kv_mins[layer_idx]
        group_size = past_key_values._layer_quant_group_size(layer_idx, k_dim=k_dim, payload_dim=dim)
        payload = past_key_values._dequantize(payload, scale, mn, dim, quant_bits, group_size=group_size)
    return payload


def _collect_hf_compressed_state_debug(past_key_values, model, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    if not layers:
        return {}
    out: dict[int, dict[str, torch.Tensor]] = {}
    config = past_key_values.config
    sink = int(config.num_sink_tokens)
    base_step = max(1, int(1 / float(config.cluster_ratio)))
    for layer_idx in layers:
        if layer_idx not in getattr(past_key_values, "comp_kv_cache", {}):
            continue
        payload = past_key_values.comp_kv_cache[layer_idx]
        pos = past_key_values.comp_pos_cache[layer_idx]
        father_idx = past_key_values.token_father_idx[layer_idx]
        if payload.shape[0] != 1 or pos.shape[0] != 1 or father_idx.shape[0] != 1:
            raise NotImplementedError("Compressed-state debug currently expects HF batch_size=1.")
        k_dim = int(model.model.layers[layer_idx].self_attn.head_dim * config.num_key_value_heads)
        payload = _dequantize_hf_payload(past_key_values, layer_idx, payload, k_dim).squeeze(0)
        pos_1d = pos.squeeze(0).to(torch.long)
        father_idx_2d = father_idx.squeeze(0).to(torch.long)
        sink_pos = past_key_values.sink_pos_cache[layer_idx].squeeze(0).to(torch.long)
        centers = _center_positions_cpu(
            start=int(pos_1d[0].item()) if pos_1d.numel() else sink,
            seq_len=int(pos_1d.numel()),
            base_step=base_step,
        ).to(father_idx_2d.device)
        base_pos = torch.cat([sink_pos.to(father_idx_2d.device), centers], dim=0)
        if father_idx_2d.numel() and int(father_idx_2d.max().item()) >= int(base_pos.numel()):
            raise RuntimeError(
                "HF compressed-state debug inferred too few base positions: "
                f"layer={layer_idx}, max_father_idx={int(father_idx_2d.max().item())}, bases={int(base_pos.numel())}."
            )
        father_pos = base_pos[father_idx_2d]
        out[layer_idx] = {
            "positions": pos_1d.detach().cpu(),
            "payload": payload.detach().float().cpu(),
            "father_pos": father_pos.detach().cpu(),
            "base_positions": base_pos.detach().cpu(),
            "base_vectors": past_key_values.bases_cache[layer_idx].squeeze(0).detach().float().cpu(),
        }
    return out


def _dequantize_sparse_payload(cm, l_idx: int, latent_slots: torch.Tensor, kv_dim: int) -> torch.Tensor:
    residual = cm.deltakv_latent_cache[l_idx, latent_slots]
    if int(cm.config.kv_quant_bits or 0) in (2, 4):
        scales = cm.deltakv_latent_scales[l_idx, latent_slots]
        mins = cm.deltakv_latent_mins[l_idx, latent_slots]
        payload_dim = cm._sparse_payload_dim(kv_dim)
        residual = cm._dequantize_residual(residual, scales, mins, payload_dim)
    return residual


def _collect_sparse_compressed_state_debug(runner: ModelRunner, seq: Sequence, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    if not layers:
        return {}
    cm = runner.cache_manager
    seq_id_to_row = getattr(cm, "seq_id_to_row", {})
    if isinstance(seq_id_to_row, list):
        seq_id_to_row = seq_id_to_row[0] if seq_id_to_row else {}
    row_idx = seq_id_to_row.get(seq.seq_id)
    if row_idx is None:
        return {}
    row_idx = int(row_idx)
    cur_len = int(cm.row_seq_lens[row_idx])
    kv_dim = 2 * int(cm.num_kv_heads) * int(cm.head_dim)
    out: dict[int, dict[str, torch.Tensor]] = {}
    latent_map = cm.sparse_layer_latent_slots_map[row_idx, :cur_len]
    latent_positions = torch.nonzero(latent_map >= 0, as_tuple=False).flatten().to(torch.long)
    if latent_positions.numel() == 0:
        return out
    latent_slots = latent_map[latent_positions].to(torch.long)
    for layer_idx in layers:
        if layer_idx not in cm.deltakv_layer_to_idx:
            continue
        l_idx = cm.deltakv_layer_to_idx[layer_idx]
        payload = _dequantize_sparse_payload(cm, l_idx, latent_slots, kv_dim)
        father_slots = cm.deltakv_latent_to_full_slots[l_idx, latent_slots].to(torch.long)
        father_pos = cm.deltakv_slot_to_pos[father_slots].to(torch.long)
        base_slots = cm.row_deltakv_center_slots[row_idx][layer_idx]
        base_pos = base_vectors = base_roped_vectors = base_pre_rope_vectors = None
        if base_slots is not None and base_slots.numel() > 0:
            base_slots = base_slots.to(torch.int32)
            base_pos = cm.deltakv_slot_to_pos[base_slots.to(torch.long)].to(torch.long)
            k_rope = cm.deltakv_full_kv_cache[0, l_idx, base_slots.to(torch.long)]
            v_raw = cm.deltakv_full_kv_cache[1, l_idx, base_slots.to(torch.long)]
            half_dim = int(cm.num_kv_heads) * int(cm.head_dim)
            base_roped_vectors = torch.cat(
                [
                    k_rope.reshape(-1, half_dim),
                    v_raw.reshape(-1, half_dim),
                ],
                dim=-1,
            )
            if hasattr(cm, "_gather_sparse_ref_raw_kv_by_slots"):
                base_vectors = cm._gather_sparse_ref_raw_kv_by_slots(l_idx, base_slots)
            elif hasattr(cm, "_gather_raw_kv_by_slots"):
                base_vectors = cm._gather_raw_kv_by_slots(layer_idx, base_slots)
            if (
                base_vectors is not None
                and hasattr(cm, "_sparse_ref_fp8_enabled")
                and cm._sparse_ref_fp8_enabled()
            ):
                base_vectors = cm._fp8_roundtrip(base_vectors)
                base_roped_vectors = cm._fp8_roundtrip(base_roped_vectors)
            layers = getattr(getattr(runner.model, "model", None), "layers", None)
            if layers is not None and layer_idx < len(layers):
                attn = getattr(layers[layer_idx], "self_attn", None)
                pre_k = getattr(attn, "debug_last_pre_rope_k", None)
                pre_v = getattr(attn, "debug_last_pre_rope_v", None)
                pre_pos = getattr(attn, "debug_last_pre_rope_positions", None)
                if pre_k is not None and pre_v is not None and pre_pos is not None:
                    pre_pos = pre_pos.to(device=base_pos.device, dtype=torch.long)
                    pre_index = {int(pos): i for i, pos in enumerate(pre_pos.tolist())}
                    if all(int(pos) in pre_index for pos in base_pos.tolist()):
                        take = torch.tensor([pre_index[int(pos)] for pos in base_pos.tolist()], device=pre_k.device)
                        pre_k_take = pre_k.index_select(0, take)
                        pre_v_take = pre_v.index_select(0, take)
                        base_pre_rope_vectors = torch.cat(
                            [
                                pre_k_take.reshape(-1, half_dim),
                                pre_v_take.reshape(-1, half_dim),
                            ],
                            dim=-1,
                        )
                        if (
                            hasattr(cm, "_sparse_ref_fp8_enabled")
                            and cm._sparse_ref_fp8_enabled()
                        ):
                            base_pre_rope_vectors = cm._fp8_roundtrip(base_pre_rope_vectors)
        out[layer_idx] = {
            "positions": latent_positions.detach().cpu(),
            "payload": payload.detach().float().cpu(),
            "father_pos": father_pos.detach().cpu(),
        }
        if base_pos is not None and base_vectors is not None:
            out[layer_idx]["base_positions"] = base_pos.detach().cpu()
            out[layer_idx]["base_vectors"] = base_vectors.detach().float().cpu()
        if base_roped_vectors is not None:
            out[layer_idx]["base_roped_vectors"] = base_roped_vectors.detach().float().cpu()
        if base_pre_rope_vectors is not None:
            out[layer_idx]["base_pre_rope_vectors"] = base_pre_rope_vectors.detach().float().cpu()
    return out


def _compare_compressed_state_debug(
    hf_state: dict[int, dict[str, torch.Tensor]],
    sparse_state: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    def _quantile_1d(flat: torch.Tensor, q: float) -> torch.Tensor:
        flat = flat.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Cannot compute quantile of an empty tensor.")
        if flat.numel() <= 16_000_000:
            return torch.quantile(flat, q)
        k = min(max(int(q * (flat.numel() - 1)) + 1, 1), flat.numel())
        return flat.kthvalue(k).values

    def _add_split_diff_stats(prefix: str, diff: torch.Tensor, summary: dict[str, Any]) -> None:
        if diff.ndim != 2 or diff.shape[1] % 2 != 0:
            return
        half = diff.shape[1] // 2
        for name, part in (("key", diff[:, :half]), ("value", diff[:, half:])):
            flat = part.reshape(-1)
            summary[f"{prefix}_{name}_mean_abs_diff"] = float(flat.mean().item())
            summary[f"{prefix}_{name}_max_abs_diff"] = float(flat.max().item())
            summary[f"{prefix}_{name}_p99_abs_diff"] = float(_quantile_1d(flat, 0.99).item())

    out: dict[str, Any] = {}
    for layer_idx in sorted(set(hf_state) & set(sparse_state)):
        hf = hf_state[layer_idx]
        sv = sparse_state[layer_idx]
        hf_pos = hf["positions"].to(torch.long)
        sv_pos = sv["positions"].to(torch.long)
        hf_index = {int(pos): i for i, pos in enumerate(hf_pos.tolist())}
        sv_index = {int(pos): i for i, pos in enumerate(sv_pos.tolist())}
        common_pos = sorted(set(hf_index) & set(sv_index))
        hf_only = sorted(set(hf_index) - set(sv_index))
        sv_only = sorted(set(sv_index) - set(hf_index))
        summary: dict[str, Any] = {
            "hf_tokens": int(hf_pos.numel()),
            "sparse_latent_tokens": int(sv_pos.numel()),
            "common_tokens": len(common_pos),
            "hf_only_tokens": len(hf_only),
            "sparse_only_tokens": len(sv_only),
            "hf_only_preview": hf_only[:32],
            "sparse_only_preview": sv_only[:32],
        }
        if common_pos:
            hf_take = torch.tensor([hf_index[p] for p in common_pos], dtype=torch.long)
            sv_take = torch.tensor([sv_index[p] for p in common_pos], dtype=torch.long)
            payload_diff = (hf["payload"].index_select(0, hf_take) - sv["payload"].index_select(0, sv_take)).abs()
            hf_father = hf["father_pos"].index_select(0, hf_take)
            sv_father = sv["father_pos"].index_select(0, sv_take)
            father_exact = (hf_father == sv_father).all(dim=1)
            hf_father_sorted = torch.sort(hf_father, dim=1).values
            sv_father_sorted = torch.sort(sv_father, dim=1).values
            father_set = (hf_father_sorted == sv_father_sorted).all(dim=1)
            summary.update(
                {
                    "payload_mean_abs_diff": float(payload_diff.mean().item()),
                    "payload_max_abs_diff": float(payload_diff.max().item()),
                    "payload_p99_abs_diff": float(_quantile_1d(payload_diff, 0.99).item()),
                    "father_exact_match_ratio": float(father_exact.float().mean().item()),
                    "father_exact_match_count": int(father_exact.sum().item()),
                    "father_set_match_ratio": float(father_set.float().mean().item()),
                    "father_set_match_count": int(father_set.sum().item()),
                    "father_first_mismatches": [
                        {
                            "position": int(common_pos[i]),
                            "hf": [int(x) for x in hf_father[i].tolist()],
                            "sparse": [int(x) for x in sv_father[i].tolist()],
                        }
                        for i in torch.nonzero(~father_exact, as_tuple=False).flatten()[:16].tolist()
                    ],
                    "father_first_set_mismatches": [
                        {
                            "position": int(common_pos[i]),
                            "hf": [int(x) for x in hf_father[i].tolist()],
                            "sparse": [int(x) for x in sv_father[i].tolist()],
                        }
                        for i in torch.nonzero(~father_set, as_tuple=False).flatten()[:16].tolist()
                    ],
                }
            )
            _add_split_diff_stats("payload", payload_diff, summary)
            hf_base_pos = hf.get("base_positions")
            sv_base_pos = sv.get("base_positions")
            hf_base_vec = hf.get("base_vectors")
            sv_base_vec = sv.get("base_vectors")
            if (
                hf_base_pos is not None
                and sv_base_pos is not None
                and hf_base_vec is not None
                and sv_base_vec is not None
            ):
                hf_base_index = {int(pos): i for i, pos in enumerate(hf_base_pos.to(torch.long).tolist())}
                sv_base_index = {int(pos): i for i, pos in enumerate(sv_base_pos.to(torch.long).tolist())}
                common_base_pos = sorted(set(hf_base_index) & set(sv_base_index))
                summary["base_common_tokens"] = len(common_base_pos)
                summary["base_hf_only_tokens"] = len(set(hf_base_index) - set(sv_base_index))
                summary["base_sparse_only_tokens"] = len(set(sv_base_index) - set(hf_base_index))
                if common_base_pos:
                    hf_base_take = torch.tensor([hf_base_index[p] for p in common_base_pos], dtype=torch.long)
                    sv_base_take = torch.tensor([sv_base_index[p] for p in common_base_pos], dtype=torch.long)
                    base_diff = (
                        hf_base_vec.index_select(0, hf_base_take)
                        - sv_base_vec.index_select(0, sv_base_take)
                    ).abs()
                    summary["base_mean_abs_diff"] = float(base_diff.mean().item())
                    summary["base_max_abs_diff"] = float(base_diff.max().item())
                    summary["base_p99_abs_diff"] = float(_quantile_1d(base_diff, 0.99).item())
                    _add_split_diff_stats("base", base_diff, summary)
                    sv_base_roped_vec = sv.get("base_roped_vectors")
                    if sv_base_roped_vec is not None:
                        base_roped_diff = (
                            hf_base_vec.index_select(0, hf_base_take)
                            - sv_base_roped_vec.index_select(0, sv_base_take)
                        ).abs()
                        summary["base_roped_mean_abs_diff"] = float(base_roped_diff.mean().item())
                        summary["base_roped_max_abs_diff"] = float(base_roped_diff.max().item())
                        summary["base_roped_p99_abs_diff"] = float(_quantile_1d(base_roped_diff, 0.99).item())
                        _add_split_diff_stats("base_roped", base_roped_diff, summary)
                    sv_base_pre_rope_vec = sv.get("base_pre_rope_vectors")
                    if sv_base_pre_rope_vec is not None:
                        base_pre_rope_diff = (
                            hf_base_vec.index_select(0, hf_base_take)
                            - sv_base_pre_rope_vec.index_select(0, sv_base_take)
                        ).abs()
                        summary["base_pre_rope_mean_abs_diff"] = float(base_pre_rope_diff.mean().item())
                        summary["base_pre_rope_max_abs_diff"] = float(base_pre_rope_diff.max().item())
                        summary["base_pre_rope_p99_abs_diff"] = float(_quantile_1d(base_pre_rope_diff, 0.99).item())
                        _add_split_diff_stats("base_pre_rope", base_pre_rope_diff, summary)
                if hf["payload"].shape[-1] == sv["payload"].shape[-1] == hf_base_vec.shape[-1] == sv_base_vec.shape[-1]:
                    hf_ref_take = torch.tensor(
                        [[hf_base_index[int(pos)] for pos in row.tolist()] for row in hf_father],
                        dtype=torch.long,
                    )
                    sv_ref_take = torch.tensor(
                        [[sv_base_index[int(pos)] for pos in row.tolist()] for row in sv_father],
                        dtype=torch.long,
                    )
                    hf_refs = hf_base_vec.index_select(0, hf_ref_take.reshape(-1)).view(
                        len(common_pos),
                        hf_ref_take.shape[1],
                        -1,
                    ).mean(dim=1)
                    sv_refs = sv_base_vec.index_select(0, sv_ref_take.reshape(-1)).view(
                        len(common_pos),
                        sv_ref_take.shape[1],
                        -1,
                    ).mean(dim=1)
                    hf_raw = hf["payload"].index_select(0, hf_take) + hf_refs
                    sv_raw = sv["payload"].index_select(0, sv_take) + sv_refs
                    raw_diff = (hf_raw - sv_raw).abs()
                    summary["raw_reconstructed_mean_abs_diff"] = float(raw_diff.mean().item())
                    summary["raw_reconstructed_max_abs_diff"] = float(raw_diff.max().item())
                    summary["raw_reconstructed_p99_abs_diff"] = float(_quantile_1d(raw_diff, 0.99).item())
                    _add_split_diff_stats("raw_reconstructed", raw_diff, summary)
        out[str(layer_idx)] = summary
    missing_hf = sorted(set(sparse_state) - set(hf_state))
    missing_sparse = sorted(set(hf_state) - set(sparse_state))
    if missing_hf:
        out["_missing_hf_layers"] = missing_hf
    if missing_sparse:
        out["_missing_sparse_layers"] = missing_sparse
    return out


def _collect_hf_full_kivi_debug(past_key_values, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    if not layers or not hasattr(past_key_values, "_full_layer_kivi_quantized_lens"):
        return {}
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        quant_len = int(past_key_values._full_layer_kivi_quantized_lens.get(layer_idx, 0))
        if quant_len <= 0 or layer_idx not in getattr(past_key_values, "buffer_key_cache", {}):
            continue
        pos = past_key_values.buffer_pos_cache[layer_idx][:, :quant_len]
        key = past_key_values.buffer_key_cache[layer_idx][:, :quant_len]
        value = past_key_values.buffer_value_cache[layer_idx][:, :quant_len]
        if pos.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise NotImplementedError("Full-layer KIVI debug expects HF batch_size=1.")
        out[layer_idx] = {
            "positions": pos.squeeze(0).detach().cpu().to(torch.long),
            "key_postrope": key.squeeze(0).detach().float().cpu(),
            "value": value.squeeze(0).detach().float().cpu(),
        }
        debug = getattr(past_key_values, "debug_full_layer_kivi_roundtrip", {}).get(layer_idx)
        if debug:
            if "key_before" in debug:
                out[layer_idx]["key_before"] = debug["key_before"].detach().float().cpu()
            if "value_before" in debug:
                out[layer_idx]["value_before"] = debug["value_before"].detach().float().cpu()
    return out


def _collect_sparse_full_kivi_debug(runner: ModelRunner, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    if not layers:
        return {}
    cm = runner.cache_manager
    snapshots = getattr(cm, "debug_full_layer_kivi_roundtrip", {})
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        snap = snapshots.get(int(layer_idx))
        if not snap:
            continue
        sparse_key = snap.get("key_pre_rope")
        sparse_key_name = "key_pre_rope"
        if sparse_key is None:
            sparse_key = snap.get("key_postrope")
            sparse_key_name = "key_postrope"
        if sparse_key is None:
            raise KeyError(f"Full-layer KIVI debug snapshot missing key tensor for layer {layer_idx}.")
        out[int(layer_idx)] = {
            "positions": snap["positions"].detach().cpu().to(torch.long),
            sparse_key_name: sparse_key.detach().float().cpu(),
            "key_rope": snap["key_rope"].detach().float().cpu(),
            "value": snap["value"].detach().float().cpu(),
        }
        if "key_before" in snap:
            out[int(layer_idx)]["key_before"] = snap["key_before"].detach().float().cpu()
        if "value_before" in snap:
            out[int(layer_idx)]["value_before"] = snap["value_before"].detach().float().cpu()
    return out


def _compare_full_kivi_debug(
    hf_state: dict[int, dict[str, torch.Tensor]],
    sparse_state: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer_idx in sorted(set(hf_state) & set(sparse_state)):
        hf = hf_state[layer_idx]
        sv = sparse_state[layer_idx]
        hf_pos = hf["positions"].to(torch.long)
        sv_pos = sv["positions"].to(torch.long)
        hf_index = {int(pos): i for i, pos in enumerate(hf_pos.tolist())}
        sv_index = {int(pos): i for i, pos in enumerate(sv_pos.tolist())}
        common_pos = sorted(set(hf_index) & set(sv_index))
        summary: dict[str, Any] = {
            "hf_tokens": int(hf_pos.numel()),
            "sparse_tokens": int(sv_pos.numel()),
            "common_tokens": len(common_pos),
            "hf_only_tokens": len(set(hf_index) - set(sv_index)),
            "sparse_only_tokens": len(set(sv_index) - set(hf_index)),
        }
        if common_pos:
            hf_take = torch.tensor([hf_index[p] for p in common_pos], dtype=torch.long)
            sv_take = torch.tensor([sv_index[p] for p in common_pos], dtype=torch.long)
            for name in ("key_before", "value_before", "key_pre_rope", "key_postrope", "value"):
                if name not in hf or name not in sv:
                    continue
                hf_tensor = hf[name].index_select(0, hf_take).reshape(len(common_pos), -1)
                sv_tensor = sv[name].index_select(0, sv_take).reshape(len(common_pos), -1)
                diff = (hf_tensor - sv_tensor).abs()
                summary[f"{name}_mean_abs_diff"] = float(diff.mean().item())
                summary[f"{name}_max_abs_diff"] = float(diff.max().item())
                flat = diff.reshape(-1)
                kth = min(flat.numel(), max(1, int(math.ceil(0.99 * flat.numel()))))
                summary[f"{name}_p99_abs_diff"] = float(flat.kthvalue(kth).values.item())
            if "key_rope" in sv and "key_postrope" in hf:
                hf_tensor = hf["key_postrope"].index_select(0, hf_take).reshape(len(common_pos), -1)
                sv_tensor = sv["key_rope"].index_select(0, sv_take).reshape(len(common_pos), -1)
                diff = (hf_tensor - sv_tensor).abs()
                summary["key_rope_vs_hf_postrope_mean_abs_diff"] = float(diff.mean().item())
                summary["key_rope_vs_hf_postrope_max_abs_diff"] = float(diff.max().item())
        out[str(layer_idx)] = summary
    missing_hf = sorted(set(sparse_state) - set(hf_state))
    missing_sparse = sorted(set(hf_state) - set(sparse_state))
    if missing_hf:
        out["_missing_hf_layers"] = missing_hf
    if missing_sparse:
        out["_missing_sparse_layers"] = missing_sparse
    return out


def _collect_hf_hidden_debug(outputs, layers: list[int], num_hidden_layers: int) -> dict[int, torch.Tensor]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if not layers or hidden_states is None:
        return {}
    out: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        if layer_idx == -1:
            source_idx = 0
        elif layer_idx == num_hidden_layers:
            source_idx = len(hidden_states) - 1
        elif 0 <= layer_idx < num_hidden_layers:
            source_idx = layer_idx + 1
        else:
            raise ValueError(
                f"hidden debug layer must be -1, 0..{num_hidden_layers - 1}, or {num_hidden_layers}; "
                f"got {layer_idx}."
            )
        out[int(layer_idx)] = hidden_states[source_idx][:, -1, :].detach().float().cpu()
    return out


def _last_token_hidden_from_output(output) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"Cannot capture hidden state from module output type {type(output).__name__}.")
    if hidden.dim() != 3:
        raise ValueError(f"Expected hidden state with shape (batch, seq, hidden), got {tuple(hidden.shape)}.")
    return hidden[:, -1, :].detach().float().cpu()


def _install_hf_hidden_debug_hooks(model, layers: list[int]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    if not layers:
        return {}, []

    inner_model = getattr(model, "model", None)
    decoder_layers = getattr(inner_model, "layers", None)
    if inner_model is None or decoder_layers is None:
        raise AttributeError("HF hidden debug expected model.model.layers to exist.")

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers"))
    captures: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    for layer_idx in layers:
        if layer_idx == -1:
            module = getattr(inner_model, "embed_tokens", None)
        elif layer_idx == num_hidden_layers:
            module = getattr(inner_model, "norm", None)
        elif 0 <= layer_idx < num_hidden_layers:
            module = decoder_layers[layer_idx]
        else:
            raise ValueError(
                f"hidden debug layer must be -1, 0..{num_hidden_layers - 1}, or {num_hidden_layers}; "
                f"got {layer_idx}."
            )
        if module is None:
            raise AttributeError(f"Cannot find HF module for hidden debug layer {layer_idx}.")

        def _hook(_module, _inputs, output, *, captured_layer=int(layer_idx)):
            captures[captured_layer] = _last_token_hidden_from_output(output)

        handles.append(module.register_forward_hook(_hook))

    return captures, handles


def _collect_sparse_hidden_debug(runner: ModelRunner, layers: list[int]) -> dict[int, torch.Tensor]:
    if not layers:
        return {}
    model = getattr(runner.model, "model", None)
    snapshots = getattr(model, "debug_last_hidden_states", {}) if model is not None else {}
    out: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        snap = snapshots.get(int(layer_idx))
        if snap is not None:
            out[int(layer_idx)] = snap.detach().float().cpu()
    return out


def _compare_hidden_debug(
    hf_state: dict[int, torch.Tensor],
    sparse_state: dict[int, torch.Tensor],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer_idx in sorted(set(hf_state) & set(sparse_state)):
        hf = hf_state[layer_idx].reshape(-1)
        sv = sparse_state[layer_idx].reshape(-1)
        if hf.shape != sv.shape:
            out[str(layer_idx)] = {
                "shape_mismatch": {
                    "hf": list(hf.shape),
                    "sparse": list(sv.shape),
                }
            }
            continue
        diff = (hf - sv).abs()
        denom = hf.abs().mean().clamp_min(1e-6)
        out[str(layer_idx)] = {
            "shape": list(hf.shape),
            "mean_abs_diff": float(diff.mean().item()),
            "max_abs_diff": float(diff.max().item()),
            "p99_abs_diff": float(torch.quantile(diff, 0.99).item()),
            "relative_mean_abs_diff": float((diff.mean() / denom).item()),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(hf, sv, dim=0).item()),
        }
    missing_hf = sorted(set(sparse_state) - set(hf_state))
    missing_sparse = sorted(set(hf_state) - set(sparse_state))
    if missing_hf:
        out["_missing_hf_layers"] = missing_hf
    if missing_sparse:
        out["_missing_sparse_layers"] = missing_sparse
    return out


def _collect_hf_qk_debug(model, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    if not layers:
        return {}
    decoder_layers = getattr(getattr(model, "model", None), "layers", None)
    if decoder_layers is None:
        return {}
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(decoder_layers):
            continue
        attn = getattr(decoder_layers[layer_idx], "self_attn", None)
        q = getattr(attn, "debug_last_q_postrope", None)
        k = getattr(attn, "debug_last_k_postrope", None)
        k_raw = getattr(attn, "debug_last_k_raw", None)
        k_norm = getattr(attn, "debug_last_k_norm", None)
        k_full_kivi_postrope_input = getattr(attn, "debug_last_k_full_kivi_postrope_input", None)
        v = getattr(attn, "debug_last_v", None)
        pos = getattr(attn, "debug_last_qk_positions", None)
        attn_output = getattr(attn, "debug_last_attn_output", None)
        o_proj_output = getattr(attn, "debug_last_o_proj_output", None)
        if q is not None:
            out.setdefault(int(layer_idx), {})["q"] = q.detach().float().cpu()
        if k is not None:
            out.setdefault(int(layer_idx), {})["k"] = k.detach().float().cpu()
        if k_raw is not None:
            out.setdefault(int(layer_idx), {})["k_raw"] = k_raw.detach().float().cpu()
        if k_norm is not None:
            out.setdefault(int(layer_idx), {})["k_norm"] = k_norm.detach().float().cpu()
        if k_full_kivi_postrope_input is not None:
            out.setdefault(int(layer_idx), {})["k_full_kivi_postrope_input"] = (
                k_full_kivi_postrope_input.detach().float().cpu()
            )
        if v is not None:
            out.setdefault(int(layer_idx), {})["v"] = v.detach().float().cpu()
        if pos is not None:
            out.setdefault(int(layer_idx), {})["positions"] = pos.detach().cpu()
        if attn_output is not None:
            attn_output = attn_output.detach().float().cpu()
            if attn_output.dim() == 4:
                attn_output = attn_output.transpose(1, 2)
            out.setdefault(int(layer_idx), {})["attn_output"] = attn_output
        if o_proj_output is not None:
            o_proj_output = o_proj_output.detach().float().cpu()
            if o_proj_output.dim() == 2:
                o_proj_output = o_proj_output.unsqueeze(0)
            out.setdefault(int(layer_idx), {})["o_proj_output"] = o_proj_output
    return out


def _collect_sparse_qk_debug(runner: ModelRunner, seq: Sequence, layers: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    if not layers:
        return {}
    model_layers = getattr(getattr(runner.model, "model", None), "layers", None)
    cm = runner.cache_manager
    row_idx = getattr(cm, "seq_id_to_row", {}).get(seq.seq_id)
    reconstruct_debug = getattr(cm, "debug_last_reconstruct", {})
    reconstruct_alternatives = getattr(cm, "debug_last_reconstruct_alternatives", {})
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        if model_layers is not None and 0 <= layer_idx < len(model_layers):
            attn = getattr(model_layers[layer_idx], "self_attn", None)
            q = getattr(attn, "debug_last_q_postrope", None)
            if q is not None:
                q_out = q.detach().float().cpu()
                if q_out.dim() == 3:
                    q_out = q_out.unsqueeze(0).transpose(1, 2)
                out.setdefault(int(layer_idx), {})["q"] = q_out
            for debug_name, output_name in (
                ("debug_last_k_raw", "k_raw"),
                ("debug_last_k_norm", "k_norm"),
                ("debug_last_k_full_kivi_postrope_input", "k_full_kivi_postrope_input"),
            ):
                tensor = getattr(attn, debug_name, None)
                if tensor is not None:
                    tensor_out = tensor.detach().float().cpu()
                    if tensor_out.dim() == 3:
                        tensor_out = tensor_out.unsqueeze(0).transpose(1, 2)
                    out.setdefault(int(layer_idx), {})[output_name] = tensor_out
            v = getattr(attn, "debug_last_v", None)
            if v is not None:
                v_out = v.detach().float().cpu()
                if v_out.dim() == 3:
                    v_out = v_out.unsqueeze(0).transpose(1, 2)
                out.setdefault(int(layer_idx), {})["v_current"] = v_out
            attn_output = getattr(attn, "debug_last_attn_output", None)
            if attn_output is not None:
                attn_out = attn_output.detach().float().cpu()
                if attn_out.dim() == 3:
                    attn_out = attn_out.unsqueeze(0).transpose(1, 2)
                out.setdefault(int(layer_idx), {})["attn_output"] = attn_out
            o_proj_output = getattr(attn, "debug_last_o_proj_output", None)
            if o_proj_output is not None:
                proj_out = o_proj_output.detach().float().cpu()
                if proj_out.dim() == 2:
                    proj_out = proj_out.unsqueeze(0)
                out.setdefault(int(layer_idx), {})["o_proj_output"] = proj_out
        if (
            row_idx is not None
            and hasattr(cm, "full_layer_to_idx")
            and layer_idx in cm.full_layer_to_idx
            and hasattr(cm, "full_layer_slots_map")
            and hasattr(cm, "full_kv_cache")
        ):
            comp_len = int(cm.get_compressed_lens(torch.tensor([int(row_idx)], device="cuda", dtype=torch.int32))[0].item())
            sink = int(getattr(cm.config, "num_sink_tokens", 0) or 0)
            slots = cm.full_layer_slots_map[int(row_idx), sink : sink + comp_len].to(torch.long)
            l_idx = cm.full_layer_to_idx[layer_idx]
            k = cm.full_kv_cache[0, l_idx, slots].permute(1, 0, 2).unsqueeze(0)
            v = cm.full_kv_cache[1, l_idx, slots].permute(1, 0, 2).unsqueeze(0)
            out.setdefault(int(layer_idx), {})["k"] = k.detach().float().cpu()
            out.setdefault(int(layer_idx), {})["v"] = v.detach().float().cpu()
        if layer_idx in reconstruct_debug:
            snap = reconstruct_debug[layer_idx]
            out.setdefault(int(layer_idx), {})["positions"] = snap["positions"].detach().cpu()
            out.setdefault(int(layer_idx), {})["k"] = snap["k"].detach().float().cpu()
            out.setdefault(int(layer_idx), {})["v"] = snap["v"].detach().float().cpu()
        if layer_idx in reconstruct_alternatives:
            snap = reconstruct_alternatives[layer_idx]
            out.setdefault(int(layer_idx), {})["alt_positions"] = snap["positions"].detach().cpu()
            if "k_raw" in snap:
                out.setdefault(int(layer_idx), {})["k_raw"] = snap["k_raw"].detach().float().cpu()
            if "k_norm" in snap:
                out.setdefault(int(layer_idx), {})["k_norm"] = snap["k_norm"].detach().float().cpu()
            out.setdefault(int(layer_idx), {})["k_raw_rope"] = snap["k_raw_rope"].detach().float().cpu()
            out.setdefault(int(layer_idx), {})["k_norm_rope"] = snap["k_norm_rope"].detach().float().cpu()
    return out


def _compare_qk_debug(
    hf_state: dict[int, dict[str, torch.Tensor]],
    sparse_state: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer_idx in sorted(set(hf_state) & set(sparse_state)):
        layer_out: dict[str, Any] = {}
        aligned_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in ("q", "k_raw", "k_norm", "k_full_kivi_postrope_input", "k", "v", "attn_output", "o_proj_output"):
            if name not in hf_state[layer_idx] or name not in sparse_state[layer_idx]:
                continue
            hf = hf_state[layer_idx][name]
            sv = sparse_state[layer_idx][name]
            if name == "q":
                hf = hf[:, :, -1:, :]
                sv = sv[:, :, -1:, :]
            elif name == "attn_output":
                hf = hf[:, :, -1:, :]
                sv = sv[:, :, -1:, :]
            elif name == "o_proj_output":
                hf = hf[:, -1:, :]
                sv = sv[:, -1:, :]
            elif (
                name in ("k_raw", "k_norm", "k_full_kivi_postrope_input", "k", "v")
                and "positions" in hf_state[layer_idx]
                and "positions" in sparse_state[layer_idx]
                and hf.dim() == 4
                and sv.dim() == 4
                and hf.shape[2] == int(hf_state[layer_idx]["positions"].reshape(-1).numel())
                and sv.shape[2] == int(sparse_state[layer_idx]["positions"].reshape(-1).numel())
            ):
                hf_pos = hf_state[layer_idx]["positions"].reshape(-1).to(torch.long)
                sv_pos = sparse_state[layer_idx]["positions"].reshape(-1).to(torch.long)
                hf_index = {int(pos): idx for idx, pos in enumerate(hf_pos.tolist())}
                take_hf = []
                take_sv = []
                for sv_idx, pos in enumerate(sv_pos.tolist()):
                    hf_idx = hf_index.get(int(pos))
                    if hf_idx is not None:
                        take_hf.append(hf_idx)
                        take_sv.append(sv_idx)
                if take_hf:
                    hf = hf.index_select(2, torch.tensor(take_hf, dtype=torch.long))
                    sv = sv.index_select(2, torch.tensor(take_sv, dtype=torch.long))
                    hf_set = {int(pos) for pos in hf_pos.tolist()}
                    sv_set = {int(pos) for pos in sv_pos.tolist()}
                    hf_only = sorted(hf_set - sv_set)
                    sparse_only = sorted(sv_set - hf_set)
                    layer_out.setdefault("position_alignment", {})[name] = {
                        "common_positions": len(take_hf),
                        "hf_positions": int(hf_pos.numel()),
                        "sparse_positions": int(sv_pos.numel()),
                        "hf_only_count": len(hf_only),
                        "sparse_only_count": len(sparse_only),
                        "hf_only_preview": hf_only[:32],
                        "sparse_only_preview": sparse_only[:32],
                    }
            elif (
                name in ("k_raw", "k_norm", "k_full_kivi_postrope_input", "k", "v")
                and hf.dim() == 4
                and sv.dim() == 4
                and hf.shape[2] != sv.shape[2]
            ):
                sink = 8
                if hf.shape[2] >= sink + sv.shape[2]:
                    hf = hf[:, :, sink : sink + sv.shape[2], :]
            if hf.shape != sv.shape:
                layer_out[name] = {"shape_mismatch": {"hf": list(hf.shape), "sparse": list(sv.shape)}}
                continue
            aligned_tensors[name] = (hf, sv)
            hf_f = hf.reshape(-1)
            sv_f = sv.reshape(-1)
            diff = (hf_f - sv_f).abs()
            denom = hf_f.abs().mean().clamp_min(1e-6)
            if diff.numel() > 10_000_000:
                p99 = diff.kthvalue(max(1, int(diff.numel() * 0.99))).values
            else:
                p99 = torch.quantile(diff, 0.99)
            layer_out[name] = {
                "shape": list(hf.shape),
                "mean_abs_diff": float(diff.mean().item()),
                "max_abs_diff": float(diff.max().item()),
                "p99_abs_diff": float(p99.item()),
                "relative_mean_abs_diff": float((diff.mean() / denom).item()),
                "cosine_similarity": float(torch.nn.functional.cosine_similarity(hf_f, sv_f, dim=0).item()),
            }
            if name in ("k_raw", "k_norm", "k_full_kivi_postrope_input", "k", "v") and diff.numel() > 0:
                token_diff = (hf - sv).abs().mean(dim=(0, 1, 3))
                token_max = (hf - sv).abs().amax(dim=(0, 1, 3))
                topn = min(16, int(token_diff.numel()))
                if topn > 0:
                    vals, idx = torch.topk(token_diff, topn)
                    layer_out[name]["top_token_mean_abs_diff"] = [
                        {
                            "idx": int(i.item()),
                            "mean_abs_diff": float(v.item()),
                            "max_abs_diff": float(token_max[int(i.item())].item()),
                        }
                        for v, i in zip(vals, idx)
                    ]
        if "q" in aligned_tensors and "k" in aligned_tensors:
            hf_q, sv_q = aligned_tensors["q"]
            hf_k, sv_k = aligned_tensors["k"]
            hf_q = hf_q[:, :, -1:, :]
            sv_q = sv_q[:, :, -1:, :]
            num_heads = int(hf_q.shape[1])
            num_kv_heads = int(hf_k.shape[1])
            if num_heads % num_kv_heads == 0:
                group = num_heads // num_kv_heads

                def _repeat_kv(x: torch.Tensor) -> torch.Tensor:
                    return x[:, :, None, :, :].expand(x.shape[0], num_kv_heads, group, x.shape[2], x.shape[3]).reshape(
                        x.shape[0], num_heads, x.shape[2], x.shape[3]
                    )

                scale = float(hf_q.shape[-1]) ** -0.5
                hf_scores = torch.softmax(
                    torch.matmul(hf_q, _repeat_kv(hf_k).transpose(2, 3)) * scale,
                    dim=-1,
                    dtype=torch.float32,
                ).to(hf_q.dtype).mean(dim=2).max(dim=1).values
                sv_scores = torch.softmax(
                    torch.matmul(sv_q, _repeat_kv(sv_k).transpose(2, 3)) * scale,
                    dim=-1,
                    dtype=torch.float32,
                ).to(sv_q.dtype).mean(dim=2).max(dim=1).values
                score_diff = (hf_scores - sv_scores).abs().reshape(-1)
                k = min(1342, int(hf_scores.shape[-1]))
                overlap = None
                boundary = None
                diff_token_scores = []
                if k > 0:
                    hf_flat = hf_scores.reshape(-1)
                    sv_flat = sv_scores.reshape(-1)
                    hf_top_vals, hf_top_idx = torch.topk(hf_flat, k)
                    sv_top_vals, sv_top_idx = torch.topk(sv_flat, k)
                    hf_top = set(int(x) for x in hf_top_idx.tolist())
                    sv_top = set(int(x) for x in sv_top_idx.tolist())
                    overlap = len(hf_top & sv_top)
                    boundary = {
                        "hf_kth_score": float(hf_top_vals[-1].item()),
                        "sparse_kth_score": float(sv_top_vals[-1].item()),
                        "hf_top_min_idx": int(hf_top_idx[-1].item()),
                        "sparse_top_min_idx": int(sv_top_idx[-1].item()),
                    }
                    for label, indices in (
                        ("hf_only", sorted(hf_top - sv_top)[:16]),
                        ("sparse_only", sorted(sv_top - hf_top)[:16]),
                    ):
                        for idx_i in indices:
                            diff_token_scores.append(
                                {
                                    "set": label,
                                    "idx": int(idx_i),
                                    "hf": float(hf_flat[idx_i].item()),
                                    "sparse": float(sv_flat[idx_i].item()),
                                    "abs_diff": float((hf_flat[idx_i] - sv_flat[idx_i]).abs().item()),
                                }
                            )
                topn = min(16, int(score_diff.numel()))
                vals, idx = torch.topk(score_diff, topn) if topn > 0 else (torch.empty(0), torch.empty(0, dtype=torch.long))
                layer_out["qk_replay_score"] = {
                    "shape": list(hf_scores.shape),
                    "mean_abs_diff": float(score_diff.mean().item()) if score_diff.numel() else 0.0,
                    "max_abs_diff": float(score_diff.max().item()) if score_diff.numel() else 0.0,
                    "top1342_overlap": overlap,
                    "top1342_boundary": boundary,
                    "top1342_diff_token_scores": diff_token_scores,
                    "top_score_diff": [
                        {
                            "idx": int(i.item()),
                            "hf": float(hf_scores.reshape(-1)[int(i.item())].item()),
                            "sparse": float(sv_scores.reshape(-1)[int(i.item())].item()),
                            "abs_diff": float(v.item()),
                        }
                        for v, i in zip(vals, idx)
                    ],
                }
                if "v" in aligned_tensors:
                    hf_v, sv_v = aligned_tensors["v"]
                    if hf_v.shape == hf_k.shape and sv_v.shape == sv_k.shape:
                        hf_attn = torch.softmax(
                            torch.matmul(hf_q, _repeat_kv(hf_k).transpose(2, 3)) * scale,
                            dim=-1,
                            dtype=torch.float32,
                        ).to(hf_q.dtype)
                        sv_attn = torch.softmax(
                            torch.matmul(sv_q, _repeat_kv(sv_k).transpose(2, 3)) * scale,
                            dim=-1,
                            dtype=torch.float32,
                        ).to(sv_q.dtype)
                        hf_out = torch.matmul(hf_attn, _repeat_kv(hf_v))
                        sv_out = torch.matmul(sv_attn, _repeat_kv(sv_v))
                        out_diff = (hf_out - sv_out).abs().reshape(-1)
                        denom = hf_out.reshape(-1).abs().mean().clamp_min(1e-6)
                        layer_out["qkv_replay_attn_output"] = {
                            "shape": list(hf_out.shape),
                            "mean_abs_diff": float(out_diff.mean().item()) if out_diff.numel() else 0.0,
                            "max_abs_diff": float(out_diff.max().item()) if out_diff.numel() else 0.0,
                            "relative_mean_abs_diff": float((out_diff.mean() / denom).item()) if out_diff.numel() else 0.0,
                            "cosine_similarity": float(
                                torch.nn.functional.cosine_similarity(
                                    hf_out.reshape(-1), sv_out.reshape(-1), dim=0
                                ).item()
                            ) if out_diff.numel() else 0.0,
                        }
        if (
            "k" in hf_state[layer_idx]
            and "positions" in hf_state[layer_idx]
            and "alt_positions" in sparse_state[layer_idx]
        ):
            hf_k = hf_state[layer_idx]["k"]
            hf_pos = hf_state[layer_idx]["positions"].reshape(-1).to(torch.long)
            alt_pos = sparse_state[layer_idx]["alt_positions"].reshape(-1).to(torch.long)
            hf_index = {int(pos): idx for idx, pos in enumerate(hf_pos.tolist())}
            for alt_name in ("k_raw_rope", "k_norm_rope"):
                if alt_name not in sparse_state[layer_idx]:
                    continue
                sv = sparse_state[layer_idx][alt_name]
                take_hf = []
                take_sv = []
                for sv_idx, pos in enumerate(alt_pos.tolist()):
                    hf_idx = hf_index.get(int(pos))
                    if hf_idx is not None:
                        take_hf.append(hf_idx)
                        take_sv.append(sv_idx)
                if not take_hf:
                    continue
                hf = hf_k.index_select(2, torch.tensor(take_hf, dtype=torch.long))
                sv = sv.index_select(2, torch.tensor(take_sv, dtype=torch.long))
                if hf.shape != sv.shape:
                    layer_out[alt_name] = {"shape_mismatch": {"hf": list(hf.shape), "sparse": list(sv.shape)}}
                    continue
                diff = (hf.reshape(-1) - sv.reshape(-1)).abs()
                layer_out[alt_name] = {
                    "shape": list(hf.shape),
                    "mean_abs_diff": float(diff.mean().item()),
                    "max_abs_diff": float(diff.max().item()),
                    "cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(hf.reshape(-1), sv.reshape(-1), dim=0).item()
                    ),
                    "common_positions": len(take_hf),
                }
        if (
            "positions" in hf_state[layer_idx]
            and "alt_positions" in sparse_state[layer_idx]
            and (
                ("k_raw" in hf_state[layer_idx] and "k_raw" in sparse_state[layer_idx])
                or ("k_norm" in hf_state[layer_idx] and "k_norm" in sparse_state[layer_idx])
            )
        ):
            hf_pos = hf_state[layer_idx]["positions"].reshape(-1).to(torch.long)
            alt_pos = sparse_state[layer_idx]["alt_positions"].reshape(-1).to(torch.long)
            hf_index = {int(pos): idx for idx, pos in enumerate(hf_pos.tolist())}
            take_hf = []
            take_sv = []
            for sv_idx, pos in enumerate(alt_pos.tolist()):
                hf_idx = hf_index.get(int(pos))
                if hf_idx is not None:
                    take_hf.append(hf_idx)
                    take_sv.append(sv_idx)
            for raw_name in ("k_raw", "k_norm"):
                if raw_name not in hf_state[layer_idx] or raw_name not in sparse_state[layer_idx] or not take_hf:
                    continue
                hf_raw = hf_state[layer_idx][raw_name]
                sv_raw = sparse_state[layer_idx][raw_name]
                hf = hf_raw.index_select(2, torch.tensor(take_hf, dtype=torch.long))
                sv = sv_raw.index_select(2, torch.tensor(take_sv, dtype=torch.long))
                if hf.shape != sv.shape:
                    layer_out[raw_name] = {"shape_mismatch": {"hf": list(hf.shape), "sparse": list(sv.shape)}}
                else:
                    diff = (hf.reshape(-1) - sv.reshape(-1)).abs()
                    layer_out[raw_name] = {
                        "shape": list(hf.shape),
                        "mean_abs_diff": float(diff.mean().item()),
                        "max_abs_diff": float(diff.max().item()),
                        "relative_mean_abs_diff": float((diff.mean() / hf.reshape(-1).abs().mean().clamp_min(1e-6)).item()),
                        "cosine_similarity": float(
                            torch.nn.functional.cosine_similarity(hf.reshape(-1), sv.reshape(-1), dim=0).item()
                        ),
                        "common_positions": len(take_hf),
                    }
        if layer_out:
            out[str(layer_idx)] = layer_out
    return out


def _serialize_hidden_debug(state: dict[int, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer_idx, tensor in sorted(state.items()):
        t = tensor.detach().float().cpu().reshape(-1)
        out[str(layer_idx)] = {
            "shape": list(t.shape),
            "values": [float(x) for x in t.tolist()],
        }
    return out


def _hf_infer_config(args: argparse.Namespace, method: str, prompt_len: int) -> dict[str, Any]:
    if method == "vanilla":
        return {
            "sparse_method": "vanilla",
        }

    if not isinstance(args.decode_keep_tokens, int) or not isinstance(args.prefill_keep_tokens, int):
        raise TypeError("HF prefill and shared decode token budgets must be explicit integers.")

    if method == "omnikv":
        hf_prefill_chunk_size = int(args.hf_prefill_chunk_size or args.engine_prefill_chunk_size)
        return {
            "sparse_method": "omnikv",
            "decode_keep_tokens": int(args.decode_keep_tokens),
            "prefill_keep_tokens": int(args.prefill_keep_tokens),
            "sink_keep_tokens": int(args.sink_keep_tokens),
            "recent_keep_tokens": int(args.recent_keep_tokens),
            "full_attention_layers": args.full_attention_layers,
            "hf_prefill_chunk_size": hf_prefill_chunk_size,
        }

    if method in {"snapkv", "pyramidkv"}:
        hf_prefill_chunk_size = int(args.hf_prefill_chunk_size or args.engine_prefill_chunk_size)
        return {
            "sparse_method": method,
            "decode_keep_tokens": int(args.decode_keep_tokens),
            "prefill_keep_tokens": int(args.prefill_keep_tokens),
            "sink_keep_tokens": int(args.sink_keep_tokens),
            "recent_keep_tokens": int(args.recent_keep_tokens),
            "snapkv_window_size": int(args.snapkv_window_size),
            "pool_kernel_size": int(args.pool_kernel_size),
            "hf_prefill_chunk_size": hf_prefill_chunk_size,
            "pyramidkv_start_layer": int(args.pyramidkv_start_layer),
            "pyramidkv_start_ratio": float(args.pyramidkv_start_ratio),
            "pyramidkv_least_layer": args.pyramidkv_least_layer,
            "pyramidkv_least_ratio": float(args.pyramidkv_least_ratio),
        }

    if method in {"streamingllm", "attention-sink", "attention_sink"}:
        return {
            "sparse_method": method,
            "decode_keep_tokens": int(args.decode_keep_tokens),
            "prefill_keep_tokens": int(args.prefill_keep_tokens),
            "sink_keep_tokens": int(args.sink_keep_tokens),
            "recent_keep_tokens": int(args.recent_keep_tokens),
        }

    if method == "quest":
        return {
            "sparse_method": "quest",
            "decode_keep_tokens": int(args.quest_token_budget),
            "chunk_size": int(args.quest_chunk_size),
        }

    # HF DeltaKV should not chunk this logits-alignment prefill unless explicitly requested.
    hf_prefill_chunk_size = int(args.hf_prefill_chunk_size or 100_000_000)
    if hf_prefill_chunk_size <= prompt_len:
        raise ValueError(
            "hf_prefill_chunk_size must exceed the prompt length for this DeltaKV alignment run. "
            f"got hf_prefill_chunk_size={hf_prefill_chunk_size}, prompt_len={prompt_len}"
        )

    config = {
        "sparse_method": args.hf_sparse_method,
        "use_cluster": True,
        "use_compression": bool(args.use_compression),
        "decode_keep_tokens": int(args.decode_keep_tokens),
        "prefill_keep_tokens": int(args.prefill_keep_tokens),
        "sink_keep_tokens": int(args.sink_keep_tokens),
        "recent_keep_tokens": int(args.recent_keep_tokens),
        "snapkv_window_size": int(args.snapkv_window_size),
        "full_attention_layers": args.full_attention_layers,
        "deltakv_center_ratio": float(args.deltakv_center_ratio),
        "deltakv_latent_quant_bits": int(args.deltakv_latent_quant_bits),
        "deltakv_latent_quant_group_size": int(args.deltakv_latent_quant_group_size),
        "deltakv_neighbor_count": int(args.deltakv_neighbor_count),
        "full_layer_kv_quant_bits": int(args.full_layer_kv_quant_bits),
        "full_layer_kivi_group_size": int(args.full_layer_kivi_group_size),
        "full_layer_kivi_residual_length": int(args.full_layer_kivi_residual_length),
        "enable_full_layer_kivi_quant": bool(args.enable_full_layer_kivi_quant),
        "enable_sparse_ref_fp8": bool(args.enable_sparse_ref_fp8),
        "hf_prefill_chunk_size": hf_prefill_chunk_size,
    }
    if args.deltakv_latent_dim is not None:
        config["deltakv_latent_dim"] = int(args.deltakv_latent_dim)
    if bool(args.use_compression):
        config["deltakv_checkpoint_path"] = args.compressor_path
    return config


def _sparse_infer_config(args: argparse.Namespace, method: str) -> dict[str, Any]:
    config = {
        "max_model_len": args.max_model_len,
        "max_num_seqs_in_batch": int(args.max_num_seqs_in_batch),
        "max_decoding_seqs": int(args.max_decoding_seqs),
        "max_num_batched_tokens": int(
            args.max_num_batched_tokens
            if args.max_num_batched_tokens is not None
            else max(args.long_tokens + 8, args.engine_prefill_chunk_size * 2 + 8)
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "enforce_eager": bool(args.enforce_eager),
        "decode_cuda_graph": bool(args.decode_cuda_graph),
        "decode_cuda_graph_capture_sizes": args.decode_cuda_graph_capture_sizes,
        "throughput_log_interval_s": 0.0,
        "sparse_method": method,
        "engine_prefill_chunk_size": int(args.engine_prefill_chunk_size),
        "mlp_chunk_size": int(args.mlp_chunk_size),
    }
    if method in {"streamingllm", "attention-sink", "attention_sink"}:
        config.update(
            {
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
            }
        )
    elif method in {"snapkv", "pyramidkv"}:
        config.update(
            {
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
                "snapkv_window_size": int(args.snapkv_window_size),
                "pool_kernel_size": int(args.pool_kernel_size),
                "pyramidkv_start_layer": int(args.pyramidkv_start_layer),
                "pyramidkv_start_ratio": float(args.pyramidkv_start_ratio),
                "pyramidkv_least_layer": args.pyramidkv_least_layer,
                "pyramidkv_least_ratio": float(args.pyramidkv_least_ratio),
            }
        )
    elif method == "quest":
        config.update(
            {
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
                "quest_chunk_size": int(args.quest_chunk_size),
                "quest_token_budget": int(args.quest_token_budget),
                "quest_skip_layers": int(args.quest_skip_layers),
            }
        )
    elif method == "omnikv":
        config.update(
            {
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
                "full_attention_layers": args.full_attention_layers,
            }
        )
    elif method != "vanilla":
        config.update(
            {
                "deltakv_checkpoint_path": args.compressor_path,
                "use_compression": bool(args.use_compression),
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
                "snapkv_window_size": int(args.snapkv_window_size),
                "full_attention_layers": args.full_attention_layers,
                "deltakv_center_ratio": float(args.deltakv_center_ratio),
                "deltakv_latent_quant_bits": int(args.deltakv_latent_quant_bits),
                "deltakv_latent_quant_group_size": int(args.deltakv_latent_quant_group_size),
                "deltakv_neighbor_count": int(args.deltakv_neighbor_count),
                "full_layer_kv_quant_bits": int(args.full_layer_kv_quant_bits),
                "full_layer_kivi_group_size": int(args.full_layer_kivi_group_size),
                "full_layer_kivi_residual_length": int(args.full_layer_kivi_residual_length),
                "enable_full_layer_kivi_quant": bool(args.enable_full_layer_kivi_quant),
                "enable_full_layer_kivi_fused_decode": bool(args.enable_full_layer_kivi_fused_decode),
                "enable_full_layer_kivi_grouped_decode": bool(args.enable_full_layer_kivi_grouped_decode),
                "enable_full_layer_kivi_dense_decode": bool(args.enable_full_layer_kivi_dense_decode),
                "deltakv_full_pool_reserve_ratio": float(args.deltakv_full_pool_reserve_ratio),
                "deltakv_cluster_gather_chunk_size": int(args.deltakv_cluster_gather_chunk_size),
            }
        )
        if args.deltakv_latent_dim is not None:
            config["deltakv_latent_dim"] = int(args.deltakv_latent_dim)
        if not bool(args.use_compression):
            config.pop("deltakv_checkpoint_path", None)
    return config


def _load_hf_model(args: argparse.Namespace, method: str, prompt_len: int):
    infer_config = _hf_infer_config(args, method, prompt_len)
    if args.tiny_random_config:
        if method != "vanilla":
            raise NotImplementedError(
                "Tiny random HF comparison currently supports --methods vanilla only."
            )
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        overrides = apply_tiny_random_overrides(config, args.tiny_random_config)
        config._attn_implementation = "eager"
        model = build_tiny_random_hf_model(
            config,
            seed=int(args.tiny_random_seed),
        )
        model.to(torch.device("cuda", 0))
        model.eval()
        return model, {
            "weight_mode": "tiny_random",
            "seed": int(args.tiny_random_seed),
            "config_path": os.path.abspath(args.tiny_random_config),
            "overrides": overrides,
        }
    _, model = get_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=None,
        sparse_method=None,
        cuda_device=0,
        backend="hf",
        return_model=True,
    )
    model.eval()
    return model, infer_config


def _hf_method_for_compare(args: argparse.Namespace, method: str) -> str:
    if method in DIRECT_HF_METHODS:
        return method
    if is_deltakv_method(method):
        return str(args.hf_sparse_method or HF_FULL_LAYER_KIVI_METHOD)
    return "deltakv"


def _hf_method_is_deltakv(method: str) -> bool:
    return method in {"deltakv", HF_FULL_LAYER_KIVI_METHOD}


def _hf_logits_for_prompt(
    model,
    input_ids: list[int],
    decode_steps: int,
    compressed_state_layers: list[int],
    full_kivi_debug_layers: list[int],
    hidden_debug_layers: list[int],
    hidden_debug_stage: str,
    qk_debug_layers: list[int],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[int],
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
    dict[int, torch.Tensor],
    dict[int, dict[str, torch.Tensor]],
]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be > 0, got {decode_steps}.")
    if hidden_debug_stage not in {"prefill", "decode"}:
        raise ValueError(f"hidden_debug_stage must be 'prefill' or 'decode', got {hidden_debug_stage!r}.")
    with torch.inference_mode():
        prefill_hidden_layers = hidden_debug_layers if hidden_debug_stage == "prefill" else []
        hidden_hook_state, hidden_hook_handles = _install_hf_hidden_debug_hooks(model, prefill_hidden_layers)
        try:
            prefill = model(
                input_ids=ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=False,
            )
        finally:
            for handle in hidden_hook_handles:
                handle.remove()
        prefill_logits = prefill.logits[:, -1, :].detach().cpu()
        past_key_values = prefill.past_key_values
        metadata = {"after_prefill": _collect_hf_deltakv_metadata(prefill.past_key_values)}
        compressed_state_debug = _collect_hf_compressed_state_debug(
            prefill.past_key_values,
            model,
            compressed_state_layers,
        )
        full_kivi_debug = _collect_hf_full_kivi_debug(prefill.past_key_values, full_kivi_debug_layers)
        hidden_debug = dict(hidden_hook_state)
        forced_token_ids: list[int] = []
        decode_logits_steps: list[torch.Tensor] = []
        qk_debug: dict[int, dict[str, torch.Tensor]] = (
            _collect_hf_qk_debug(model, qk_debug_layers) if hidden_debug_stage == "prefill" else {}
        )
        current_logits = prefill_logits
        for step_idx in range(decode_steps):
            forced_token_id = int(torch.argmax(current_logits[0]).item())
            forced_token_ids.append(forced_token_id)
            forced = torch.tensor([[forced_token_id]], dtype=torch.long, device=model.device)
            decode_hidden_layers = hidden_debug_layers if hidden_debug_stage == "decode" and step_idx == 0 else []
            hidden_hook_state, hidden_hook_handles = _install_hf_hidden_debug_hooks(model, decode_hidden_layers)
            try:
                decode = model(
                    input_ids=forced,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            finally:
                for handle in hidden_hook_handles:
                    handle.remove()
            if decode_hidden_layers:
                hidden_debug = dict(hidden_hook_state)
            if hidden_debug_stage == "decode" and step_idx == 0:
                qk_debug = _collect_hf_qk_debug(model, qk_debug_layers)
            current_logits = decode.logits[:, -1, :].detach().cpu()
            decode_logits_steps.append(current_logits)
            past_key_values = decode.past_key_values
        metadata["after_decode"] = _collect_hf_deltakv_metadata(past_key_values)
    return (
        {"prefill": prefill_logits, "decode_steps": decode_logits_steps},
        metadata,
        forced_token_ids,
        compressed_state_debug,
        full_kivi_debug,
        hidden_debug,
        qk_debug,
    )


def _make_sparse_runner(args: argparse.Namespace, method: str) -> tuple[ModelRunner, dict[str, Any]]:
    public_config = _sparse_infer_config(args, method)
    normalized = normalize_runtime_params(public_config, backend="sparsevllm")
    config_fields = {field.name for field in fields(Config)}
    unknown = sorted(set(normalized.infer_config) - config_fields)
    if unknown:
        raise ValueError(f"Unknown Sparse-VLLM config keys after normalization: {unknown}")
    config = Config(args.model_path, **normalized.infer_config)
    processes = []
    events = []
    ctx = mp.get_context("spawn")
    tp_shm_name = make_tp_shm_name() if config.world_size > 1 else None
    try:
        for rank in range(1, int(config.tensor_parallel_size)):
            event = (ctx.Event(), ctx.Event())
            process = ctx.Process(
                target=ModelRunner,
                args=(config, rank, event, tp_shm_name),
            )
            process.start()
            processes.append(process)
            events.append(event)
        runner = ModelRunner(config, 0, events, tp_shm_name)
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        raise
    runner._compare_tp_processes = processes
    return runner, public_config


def _sparse_long_text_threshold(config: Config, *, is_prefill: bool) -> int:
    if (
        is_prefill
        and config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
    ):
        return int(config.chunk_prefill_size)
    if config.vllm_sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
        base = config.num_sink_tokens + config.num_recent_tokens
    else:
        base = config.num_sink_tokens + config.num_recent_tokens + config.decode_keep_tokens
    return int(base) + (int(config.chunk_prefill_size) if is_prefill else 0)


def _sparse_prefill_chunk_size(config: Config, seq: Sequence) -> int:
    remaining = int(seq.num_prompt_tokens) - int(seq.num_prefilled_tokens)
    if remaining <= 0:
        raise ValueError(f"Invalid sparse prefill state: remaining={remaining}")

    if config.prefill_schedule_policy == PREFILL_POLICY_ALL_CHUNKED:
        return min(int(config.chunk_prefill_size), remaining)

    if config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
        threshold = _sparse_long_text_threshold(config, is_prefill=True)
        if int(seq.num_prompt_tokens) > int(threshold):
            return min(int(config.chunk_prefill_size), remaining)
        return min(int(config.chunk_prefill_size), remaining)

    raise ValueError(f"Unknown Sparse-VLLM prefill_schedule_policy={config.prefill_schedule_policy!r}")


def _sparse_prefill(
    runner: ModelRunner,
    input_ids: list[int],
    max_tokens: int,
) -> tuple[torch.Tensor, Sequence]:
    seq = Sequence(input_ids, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
    last_logits = None
    while int(seq.num_prefilled_tokens) < int(seq.num_prompt_tokens):
        seq.current_chunk_size = _sparse_prefill_chunk_size(runner.config, seq)
        try:
            logits = runner.call("run_logits_for_compare", [seq], True)
            if logits is None:
                raise RuntimeError("Sparse-VLLM logits compare step returned no rank-0 logits.")
            logits = logits.detach().cpu()
        finally:
            reset_context()
        seq.num_prefilled_tokens += int(seq.current_chunk_size)
        last_logits = logits[-1:].contiguous()

    if last_logits is None:
        raise RuntimeError("Sparse-VLLM prefill produced no logits.")
    return last_logits, seq


def _sparse_decode(
    runner: ModelRunner,
    seq: Sequence,
    forced_token_id: int,
) -> torch.Tensor:
    seq.append_token(forced_token_id)
    if runner.config.decode_cuda_graph:
        if runner.decode_cuda_graph_runner is None:
            raise RuntimeError("decode_cuda_graph is enabled but the runner was not initialized.")
        try:
            logits, _ = runner.decode_cuda_graph_runner.run(
                [seq],
                capture_sampling=runner.config.decode_cuda_graph_capture_sampling,
            )
            runner.sparse_controller.post_forward([seq], is_prefill=False)
            return logits[-1:].detach().cpu().contiguous()
        finally:
            reset_context()
    try:
        logits = runner.call("run_logits_for_compare", [seq], False)
        if logits is None:
            raise RuntimeError("Sparse-VLLM logits compare decode returned no rank-0 logits.")
        logits = logits.detach().cpu()
        return logits[-1:].contiguous()
    finally:
        reset_context()


def _sparse_prefill_batch(
    runner: ModelRunner,
    input_ids_batch: list[list[int]],
    max_tokens: int,
) -> tuple[torch.Tensor, list[Sequence]]:
    if not input_ids_batch:
        raise ValueError("Sparse-VLLM batch prefill requires a non-empty batch.")
    seqs = [
        Sequence(input_ids, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
        for input_ids in input_ids_batch
    ]
    last_logits_by_seq: list[torch.Tensor | None] = [None for _ in seqs]
    while True:
        active = [(idx, seq) for idx, seq in enumerate(seqs) if int(seq.num_prefilled_tokens) < int(seq.num_prompt_tokens)]
        if not active:
            break
        for _, seq in active:
            seq.current_chunk_size = _sparse_prefill_chunk_size(runner.config, seq)
        full_prefill_singletons = [
            (idx, seq)
            for idx, seq in active
            if bool(getattr(runner.cache_manager, "is_full_prefill_step", lambda _seqs: False)([seq]))
        ]
        if full_prefill_singletons:
            # DeltaKV long full-prefill staging is intentionally singleton-only.
            # Keep the logical batch for decode, but respect the engine scheduler
            # constraint during prefill.
            active = [full_prefill_singletons[0]]
        active_seqs = [seq for _, seq in active]
        expected_rows = len(active_seqs)
        try:
            logits = runner.call("run_logits_for_compare", active_seqs, True)
            if logits is None:
                raise RuntimeError("Sparse-VLLM logits compare batch prefill returned no rank-0 logits.")
            logits = logits.detach().cpu()
        finally:
            reset_context()
        if int(logits.shape[0]) != int(expected_rows):
            raise RuntimeError(
                "Sparse-VLLM batch prefill logits row count mismatch: "
                f"expected={expected_rows}, got={int(logits.shape[0])}."
            )
        for row_idx, (seq_idx, seq) in enumerate(active):
            chunk_size = int(seq.current_chunk_size)
            # ParallelLMHead returns one prefill logit row per active sequence:
            # the final token of each chunk selected via cu_seqlens_q.
            last_logits_by_seq[seq_idx] = logits[row_idx : row_idx + 1].contiguous()
            seq.num_prefilled_tokens += chunk_size

    if any(logits is None for logits in last_logits_by_seq):
        raise RuntimeError("Sparse-VLLM batch prefill produced no logits for at least one sequence.")
    return torch.cat([logits for logits in last_logits_by_seq if logits is not None], dim=0), seqs


def _sparse_decode_batch(
    runner: ModelRunner,
    seqs: list[Sequence],
    forced_token_ids: list[int],
) -> torch.Tensor:
    if not seqs:
        raise ValueError("Sparse-VLLM batch decode requires a non-empty batch.")
    if len(seqs) != len(forced_token_ids):
        raise ValueError(f"Decode batch size mismatch: seqs={len(seqs)} forced_token_ids={len(forced_token_ids)}.")
    for seq, forced_token_id in zip(seqs, forced_token_ids):
        seq.append_token(int(forced_token_id))
    if runner.config.decode_cuda_graph:
        if runner.decode_cuda_graph_runner is None:
            raise RuntimeError("decode_cuda_graph is enabled but the runner was not initialized.")
        try:
            logits, _ = runner.decode_cuda_graph_runner.run(
                seqs,
                capture_sampling=runner.config.decode_cuda_graph_capture_sampling,
            )
            runner.sparse_controller.post_forward(seqs, is_prefill=False)
            logits = logits.detach().cpu().contiguous()
        finally:
            reset_context()
    else:
        try:
            logits = runner.call("run_logits_for_compare", seqs, False)
            if logits is None:
                raise RuntimeError("Sparse-VLLM logits compare batch decode returned no rank-0 logits.")
            logits = logits.detach().cpu().contiguous()
        finally:
            reset_context()
    if int(logits.shape[0]) != len(seqs):
        raise RuntimeError(
            "Sparse-VLLM batch decode logits row count mismatch: "
            f"expected={len(seqs)}, got={int(logits.shape[0])}."
        )
    return logits


def _token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _sparse_batch_capacity_args(
    args: argparse.Namespace,
    batches: list[list[dict[str, Any]]],
) -> tuple[argparse.Namespace, dict[str, Any]]:
    sparse_args = argparse.Namespace(**vars(args))
    max_batch_size = max(len(batch) for batch in batches)
    max_prefill_tokens = max(sum(int(row["prompt_tokens"]) for row in batch) for batch in batches)
    overrides: dict[str, Any] = {}

    if int(sparse_args.max_num_seqs_in_batch) < max_batch_size:
        overrides["max_num_seqs_in_batch"] = {
            "from": int(sparse_args.max_num_seqs_in_batch),
            "to": int(max_batch_size),
        }
        sparse_args.max_num_seqs_in_batch = int(max_batch_size)
    if int(sparse_args.max_decoding_seqs) < max_batch_size:
        overrides["max_decoding_seqs"] = {
            "from": int(sparse_args.max_decoding_seqs),
            "to": int(max_batch_size),
        }
        sparse_args.max_decoding_seqs = int(max_batch_size)

    greedy_rollout_steps = int(getattr(args, "greedy_rollout_steps", 0) or 0)
    decode_steps_for_capacity = max(int(args.teacher_forced_decode_steps), greedy_rollout_steps)
    required_batched_tokens = int(max_prefill_tokens) + int(decode_steps_for_capacity) + 8
    if sparse_args.max_num_batched_tokens is None:
        default_batched_tokens = max(args.long_tokens + 8, args.engine_prefill_chunk_size * 2 + 8)
        sparse_args.max_num_batched_tokens = max(int(default_batched_tokens), int(required_batched_tokens))
        overrides["max_num_batched_tokens"] = {
            "from": None,
            "to": int(sparse_args.max_num_batched_tokens),
            "required_for_largest_prefill_batch": int(required_batched_tokens),
        }
    elif int(sparse_args.max_num_batched_tokens) < required_batched_tokens:
        raise ValueError(
            "--max_num_batched_tokens is too small for the selected LongBench batch alignment plan: "
            f"configured={int(sparse_args.max_num_batched_tokens)}, required>={required_batched_tokens}."
        )

    capture_sizes = str(getattr(sparse_args, "decode_cuda_graph_capture_sizes", "auto") or "auto").strip().lower()
    if bool(sparse_args.decode_cuda_graph) and capture_sizes not in {"", "auto"}:
        parsed_capture_sizes = _parse_int_list(str(sparse_args.decode_cuda_graph_capture_sizes))
        if not parsed_capture_sizes or max(parsed_capture_sizes) < max_batch_size:
            raise ValueError(
                "--decode_cuda_graph_capture_sizes does not cover the selected LongBench batch plan: "
                f"capture_sizes={parsed_capture_sizes}, max_batch_size={max_batch_size}."
            )
    return sparse_args, overrides


def _sparse_resolved_config_summary(
    config: Config,
    cache_manager_class: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "vllm_sparse_method": config.vllm_sparse_method,
        "cache_manager_class": cache_manager_class,
        "prefill_schedule_policy": config.prefill_schedule_policy,
        "hidden_debug_stage": args.hidden_debug_stage,
        "decode_keep_tokens": config.decode_keep_tokens,
        "num_sink_tokens": config.num_sink_tokens,
        "num_recent_tokens": config.num_recent_tokens,
        "chunk_prefill_size": config.chunk_prefill_size,
        "cluster_ratio": config.cluster_ratio,
        "kv_compressed_size": config.kv_compressed_size,
        "kv_quant_bits": config.kv_quant_bits,
        "kv_quant_group_size": config.kv_quant_group_size,
        "use_compression": config.use_compression,
        "max_num_seqs_in_batch": config.max_num_seqs_in_batch,
        "max_decoding_seqs": config.max_decoding_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "decode_cuda_graph": config.decode_cuda_graph,
        "decode_cuda_graph_capture_sizes": config.decode_cuda_graph_capture_sizes,
        "full_layer_kv_quant_bits": config.full_layer_kv_quant_bits,
        "full_layer_kivi_group_size": config.full_layer_kivi_group_size,
        "full_layer_kivi_residual_length": config.full_layer_kivi_residual_length,
        "enable_full_layer_kivi_quant": config.enable_full_layer_kivi_quant,
        "enable_full_layer_kivi_grouped_decode": config.enable_full_layer_kivi_grouped_decode,
        "enable_full_layer_kivi_dense_decode": config.enable_full_layer_kivi_dense_decode,
        "enable_sparse_ref_fp8": config.enable_sparse_ref_fp8,
        "deltakv_k_neighbors": config.deltakv_k_neighbors,
        "full_attn_layers": config.full_attn_layers,
    }


def _sparse_logits_for_longbench_batches(
    args: argparse.Namespace,
    method: str,
    batches: list[list[dict[str, Any]]],
    hf_rows_by_batch: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    sparse_args, capacity_overrides = _sparse_batch_capacity_args(args, batches)
    runner = None
    batch_outputs: list[dict[str, Any]] = []
    try:
        runner, public_config = _make_sparse_runner(sparse_args, method)
        cache_manager_class = type(runner.cache_manager).__name__
        for batch_idx, batch in enumerate(batches):
            seqs: list[Sequence] = []
            try:
                input_ids_batch = [row["input_ids"] for row in batch]
                prefill_logits, seqs = _sparse_prefill_batch(
                    runner,
                    input_ids_batch,
                    max_tokens=int(args.teacher_forced_decode_steps),
                )
                metadata = {
                    "after_prefill": [_collect_sparse_deltakv_metadata(runner, seq) for seq in seqs],
                }
                decode_logits_steps: list[torch.Tensor] = []
                for step_idx in range(int(args.teacher_forced_decode_steps)):
                    forced_token_ids = [
                        int(hf_rows_by_batch[batch_idx][row_idx]["forced_token_ids"][step_idx])
                        for row_idx in range(len(batch))
                    ]
                    decode_logits_steps.append(_sparse_decode_batch(runner, seqs, forced_token_ids))
                metadata["after_decode"] = [_collect_sparse_deltakv_metadata(runner, seq) for seq in seqs]
                batch_outputs.append(
                    {
                        "prefill": prefill_logits,
                        "decode_steps": decode_logits_steps,
                        "metadata": metadata,
                    }
                )
            finally:
                if seqs:
                    runner.call("free_slots_batch", [seq.seq_id for seq in seqs])
                    reset_context()
        return {
            "batches": batch_outputs,
            "public_config": public_config,
            "resolved_config": runner.config,
            "cache_manager_class": cache_manager_class,
            "capacity_overrides": capacity_overrides,
        }
    finally:
        if runner is not None:
            runner.call("exit")
            for process in getattr(runner, "_compare_tp_processes", []):
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        _cleanup_cuda()


def _sparse_logits_for_prompt(
    args: argparse.Namespace,
    method: str,
    input_ids: list[int],
    forced_token_ids: list[int],
    compressed_state_layers: list[int],
    full_kivi_debug_layers: list[int],
    hidden_debug_layers: list[int],
    hidden_debug_stage: str,
    qk_debug_layers: list[int],
):
    if not forced_token_ids:
        raise ValueError("forced_token_ids must be non-empty.")
    if hidden_debug_stage not in {"prefill", "decode"}:
        raise ValueError(f"hidden_debug_stage must be 'prefill' or 'decode', got {hidden_debug_stage!r}.")
    runner = None
    try:
        runner, public_config = _make_sparse_runner(args, method)
        cache_manager_class = type(runner.cache_manager).__name__
        prefill_logits, seq = _sparse_prefill(runner, input_ids, max_tokens=len(forced_token_ids))
        metadata = {"after_prefill": _collect_sparse_deltakv_metadata(runner, seq)}
        compressed_state_debug = _collect_sparse_compressed_state_debug(runner, seq, compressed_state_layers)
        full_kivi_debug = _collect_sparse_full_kivi_debug(runner, full_kivi_debug_layers)
        if hidden_debug_stage == "prefill":
            hidden_debug = _collect_sparse_hidden_debug(runner, hidden_debug_layers)
        else:
            hidden_debug = {}
        decode_logits_steps = []
        qk_debug: dict[int, dict[str, torch.Tensor]] = (
            _collect_sparse_qk_debug(runner, seq, qk_debug_layers) if hidden_debug_stage == "prefill" else {}
        )
        for step_idx, forced_token_id in enumerate(forced_token_ids):
            decode_logits_steps.append(_sparse_decode(runner, seq, forced_token_id))
            if hidden_debug_stage == "decode" and step_idx == 0:
                hidden_debug = _collect_sparse_hidden_debug(runner, hidden_debug_layers)
            if hidden_debug_stage == "decode" and step_idx == 0:
                qk_debug = _collect_sparse_qk_debug(runner, seq, qk_debug_layers)
        metadata["after_decode"] = _collect_sparse_deltakv_metadata(runner, seq)
        return (
            {"prefill": prefill_logits, "decode_steps": decode_logits_steps},
            metadata,
            public_config,
            runner.config,
            cache_manager_class,
            compressed_state_debug,
            full_kivi_debug,
            hidden_debug,
            qk_debug,
        )
    finally:
        if runner is not None:
            runner.call("exit")
            for process in getattr(runner, "_compare_tp_processes", []):
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        _cleanup_cuda()


def _hf_logits_for_longbench_batches(
    args: argparse.Namespace,
    method: str,
    batches: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    hf_method = _hf_method_for_compare(args, method)
    max_prompt_len = max(int(row["prompt_tokens"]) for batch in batches for row in batch)
    hf_model, hf_config = _load_hf_model(args, hf_method, max_prompt_len)
    rows_by_batch: list[list[dict[str, Any]]] = []
    try:
        for batch in batches:
            batch_rows: list[dict[str, Any]] = []
            for row in batch:
                (
                    hf_logits,
                    hf_metadata,
                    forced_token_ids,
                    _hf_compressed_state,
                    _hf_full_kivi_state,
                    _hf_hidden_state,
                    _hf_qk_state,
                ) = _hf_logits_for_prompt(
                    hf_model,
                    row["input_ids"],
                    int(args.teacher_forced_decode_steps),
                    [],
                    [],
                    [],
                    "prefill",
                    [],
                )
                batch_rows.append(
                    {
                        "logits": hf_logits,
                        "metadata": hf_metadata,
                        "forced_token_ids": forced_token_ids,
                    }
                )
                _cleanup_cuda()
            rows_by_batch.append(batch_rows)
    finally:
        del hf_model
        _cleanup_cuda()
    return {
        "method": hf_method,
        "config": hf_config,
        "batches": rows_by_batch,
    }


def _hf_greedy_rollout_for_longbench_batches(
    args: argparse.Namespace,
    method: str,
    batches: list[list[dict[str, Any]]],
    rollout_steps: int,
) -> dict[str, Any]:
    if rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be > 0, got {rollout_steps}.")
    hf_method = _hf_method_for_compare(args, method)
    max_prompt_len = max(int(row["prompt_tokens"]) for batch in batches for row in batch)
    hf_model, hf_config = _load_hf_model(args, hf_method, max_prompt_len)
    rows_by_batch: list[list[dict[str, Any]]] = []
    try:
        for batch in batches:
            batch_rows: list[dict[str, Any]] = []
            for row in batch:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=hf_model.device)
                with torch.inference_mode():
                    prefill = hf_model(
                        input_ids=ids,
                        use_cache=True,
                        return_dict=True,
                        output_hidden_states=False,
                    )
                    past_key_values = prefill.past_key_values
                    current_logits = prefill.logits[:, -1, :].detach().cpu()
                    next_token_logits_steps: list[torch.Tensor] = []
                    generated_token_ids: list[int] = []
                    for _step_idx in range(rollout_steps):
                        next_token_logits_steps.append(current_logits)
                        token_id = int(torch.argmax(current_logits[0]).item())
                        generated_token_ids.append(token_id)
                        token = torch.tensor([[token_id]], dtype=torch.long, device=hf_model.device)
                        decode = hf_model(
                            input_ids=token,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = decode.past_key_values
                        current_logits = decode.logits[:, -1, :].detach().cpu()
                    metadata = {"after_rollout": _collect_hf_deltakv_metadata(past_key_values)}
                batch_rows.append(
                    {
                        "next_token_logits_steps": next_token_logits_steps,
                        "generated_token_ids": generated_token_ids,
                        "metadata": metadata,
                    }
                )
                _cleanup_cuda()
            rows_by_batch.append(batch_rows)
    finally:
        del hf_model
        _cleanup_cuda()
    return {
        "method": hf_method,
        "config": hf_config,
        "batches": rows_by_batch,
    }


def _sparse_greedy_rollout_for_longbench_batches(
    args: argparse.Namespace,
    method: str,
    batches: list[list[dict[str, Any]]],
    rollout_steps: int,
) -> dict[str, Any]:
    if rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be > 0, got {rollout_steps}.")
    sparse_args, capacity_overrides = _sparse_batch_capacity_args(args, batches)
    runner = None
    batch_outputs: list[dict[str, Any]] = []
    try:
        runner, public_config = _make_sparse_runner(sparse_args, method)
        cache_manager_class = type(runner.cache_manager).__name__
        for batch in batches:
            seqs: list[Sequence] = []
            try:
                input_ids_batch = [row["input_ids"] for row in batch]
                current_logits, seqs = _sparse_prefill_batch(
                    runner,
                    input_ids_batch,
                    max_tokens=int(rollout_steps),
                )
                metadata = {
                    "after_prefill": [_collect_sparse_deltakv_metadata(runner, seq) for seq in seqs],
                }
                next_token_logits_steps: list[torch.Tensor] = []
                generated_token_steps: list[list[int]] = []
                for _step_idx in range(rollout_steps):
                    current_logits = current_logits.detach().cpu().contiguous()
                    next_token_logits_steps.append(current_logits)
                    token_ids = [
                        int(torch.argmax(current_logits[row_idx]).item())
                        for row_idx in range(len(batch))
                    ]
                    generated_token_steps.append(token_ids)
                    current_logits = _sparse_decode_batch(runner, seqs, token_ids)
                metadata["after_rollout"] = [_collect_sparse_deltakv_metadata(runner, seq) for seq in seqs]
                batch_outputs.append(
                    {
                        "next_token_logits_steps": next_token_logits_steps,
                        "generated_token_steps": generated_token_steps,
                        "metadata": metadata,
                    }
                )
            finally:
                if seqs:
                    runner.call("free_slots_batch", [seq.seq_id for seq in seqs])
                    reset_context()
        return {
            "batches": batch_outputs,
            "public_config": public_config,
            "resolved_config": runner.config,
            "cache_manager_class": cache_manager_class,
            "capacity_overrides": capacity_overrides,
        }
    finally:
        if runner is not None:
            runner.call("exit")
            for process in getattr(runner, "_compare_tp_processes", []):
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        _cleanup_cuda()


def _summarize_logit_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"count": 0}
    argmax_matches = [bool(metric["argmax_match"]) for metric in metrics]
    return {
        "count": len(metrics),
        "max_abs_diff_max": max(float(metric["max_abs_diff"]) for metric in metrics),
        "mean_abs_diff_mean": sum(float(metric["mean_abs_diff"]) for metric in metrics) / len(metrics),
        "p99_abs_diff_max": max(float(metric["p99_abs_diff"]) for metric in metrics),
        "argmax_match_count": sum(1 for matched in argmax_matches if matched),
        "argmax_match_ratio": sum(1 for matched in argmax_matches if matched) / len(argmax_matches),
    }


def _summarize_rollout_rows(rows: list[dict[str, Any]], rollout_steps: int) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "rollout_steps": rollout_steps}
    all_steps = [step for row in rows for step in row["steps"]]
    per_step = []
    for step_idx in range(rollout_steps):
        step_metrics = [row["steps"][step_idx] for row in rows]
        token_matches = [bool(step["generated_token_match"]) for step in step_metrics]
        top10 = [int(step["topk_overlap"]["10"]["intersection"]) for step in step_metrics]
        top50 = [int(step["topk_overlap"]["50"]["intersection"]) for step in step_metrics]
        per_step.append(
            {
                "step": int(step_idx),
                "count": len(step_metrics),
                "generated_token_match_count": sum(1 for matched in token_matches if matched),
                "generated_token_match_ratio": sum(1 for matched in token_matches if matched) / len(token_matches),
                "top10_min": min(top10),
                "top10_mean": sum(top10) / len(top10),
                "top10_exact_count": sum(1 for value in top10 if value == 10),
                "top50_min": min(top50),
                "top50_mean": sum(top50) / len(top50),
                "max_abs_diff_max": max(float(step["max_abs_diff"]) for step in step_metrics),
                "p99_abs_diff_max": max(float(step["p99_abs_diff"]) for step in step_metrics),
            }
        )

    first_divergence_steps = [
        row["first_divergence_step"]
        for row in rows
        if row["first_divergence_step"] is not None
    ]
    top10_all = [int(step["topk_overlap"]["10"]["intersection"]) for step in all_steps]
    top50_all = [int(step["topk_overlap"]["50"]["intersection"]) for step in all_steps]
    token_match_count = sum(1 for step in all_steps if bool(step["generated_token_match"]))
    return {
        "row_count": len(rows),
        "rollout_steps": int(rollout_steps),
        "step_count": len(all_steps),
        "generated_token_match_count": token_match_count,
        "generated_token_match_ratio": token_match_count / len(all_steps),
        "rows_without_divergence": sum(1 for row in rows if row["first_divergence_step"] is None),
        "first_divergence_steps": first_divergence_steps,
        "first_divergence_min": min(first_divergence_steps) if first_divergence_steps else None,
        "first_divergence_max": max(first_divergence_steps) if first_divergence_steps else None,
        "first_divergence_mean": (
            sum(first_divergence_steps) / len(first_divergence_steps)
            if first_divergence_steps
            else None
        ),
        "top10_min": min(top10_all),
        "top10_mean": sum(top10_all) / len(top10_all),
        "top10_exact_count": sum(1 for value in top10_all if value == 10),
        "top10_ge9_count": sum(1 for value in top10_all if value >= 9),
        "top50_min": min(top50_all),
        "top50_mean": sum(top50_all) / len(top50_all),
        "top50_exact_count": sum(1 for value in top50_all if value == 50),
        "max_abs_diff_max": max(float(step["max_abs_diff"]) for step in all_steps),
        "p99_abs_diff_max": max(float(step["p99_abs_diff"]) for step in all_steps),
        "per_step": per_step,
    }


def _build_greedy_rollout_result(
    args: argparse.Namespace,
    tokenizer,
    method: str,
    sparse_method: str,
    batches: list[list[dict[str, Any]]],
) -> dict[str, Any] | None:
    rollout_steps = int(getattr(args, "greedy_rollout_steps", 0) or 0)
    if rollout_steps <= 0:
        return None
    hf = _hf_greedy_rollout_for_longbench_batches(args, method, batches, rollout_steps)
    sparse = _sparse_greedy_rollout_for_longbench_batches(args, sparse_method, batches, rollout_steps)

    all_rows: list[dict[str, Any]] = []
    result_batches: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches):
        sparse_batch = sparse["batches"][batch_idx]
        hf_batch = hf["batches"][batch_idx]
        rows_out: list[dict[str, Any]] = []
        for row_idx, row in enumerate(batch):
            hf_row = hf_batch[row_idx]
            hf_tokens = [int(token_id) for token_id in hf_row["generated_token_ids"]]
            sparse_tokens = [
                int(sparse_batch["generated_token_steps"][step_idx][row_idx])
                for step_idx in range(rollout_steps)
            ]
            steps: list[dict[str, Any]] = []
            first_divergence_step = None
            first_divergence: dict[str, Any] | None = None
            for step_idx in range(rollout_steps):
                sparse_step = sparse_batch["next_token_logits_steps"][step_idx]
                if int(sparse_step.shape[0]) != len(batch):
                    raise RuntimeError(
                        "Sparse greedy rollout batch row count mismatch while serializing result: "
                        f"batch_idx={batch_idx}, step={step_idx}, expected={len(batch)}, "
                        f"got={int(sparse_step.shape[0])}."
                    )
                step_metrics = _compare_logits(
                    hf_row["next_token_logits_steps"][step_idx],
                    sparse_step[row_idx : row_idx + 1],
                    tokenizer=tokenizer,
                )
                hf_token = int(hf_tokens[step_idx])
                sparse_token = int(sparse_tokens[step_idx])
                token_match = hf_token == sparse_token
                step_metrics.update(
                    {
                        "step": int(step_idx),
                        "hf_generated_token_id": hf_token,
                        "sparse_generated_token_id": sparse_token,
                        "hf_generated_token_text": _token_text(tokenizer, hf_token),
                        "sparse_generated_token_text": _token_text(tokenizer, sparse_token),
                        "generated_token_match": token_match,
                    }
                )
                if not token_match and first_divergence_step is None:
                    first_divergence_step = int(step_idx)
                    first_divergence = {
                        "step": int(step_idx),
                        "hf_generated_token_id": hf_token,
                        "sparse_generated_token_id": sparse_token,
                        "hf_generated_token_text": _token_text(tokenizer, hf_token),
                        "sparse_generated_token_text": _token_text(tokenizer, sparse_token),
                        "top10_overlap": step_metrics["topk_overlap"]["10"],
                        "top50_overlap": step_metrics["topk_overlap"]["50"],
                        "max_abs_diff": step_metrics["max_abs_diff"],
                        "p99_abs_diff": step_metrics["p99_abs_diff"],
                    }
                steps.append(step_metrics)

            row_out = {
                "row_index": int(row_idx),
                "status": "success",
                "sample_idx": int(row["sample_idx"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "prompt_preview": row["prompt_preview"],
                "prompt_meta": row["prompt_meta"],
                "rollout_steps": int(rollout_steps),
                "first_divergence_step": first_divergence_step,
                "first_divergence": first_divergence,
                "hf_generated_token_ids": hf_tokens,
                "sparse_generated_token_ids": sparse_tokens,
                "hf_generated_text": tokenizer.decode(hf_tokens, skip_special_tokens=False),
                "sparse_generated_text": tokenizer.decode(sparse_tokens, skip_special_tokens=False),
                "steps": steps,
                "hf_cache_metadata": hf_row["metadata"],
                "sparse_cache_metadata": {
                    "after_prefill": sparse_batch["metadata"]["after_prefill"][row_idx],
                    "after_rollout": sparse_batch["metadata"]["after_rollout"][row_idx],
                },
            }
            rows_out.append(row_out)
            all_rows.append(row_out)
        result_batches.append(
            {
                "batch_index": int(batch_idx),
                "batch_size": len(batch),
                "sample_indices": [int(row["sample_idx"]) for row in batch],
                "prompt_tokens": [int(row["prompt_tokens"]) for row in batch],
                "rows": rows_out,
                "summary": _summarize_rollout_rows(rows_out, rollout_steps),
            }
        )

    return {
        "rollout_steps": int(rollout_steps),
        "hf_method": hf["method"],
        "sparse_method": sparse_method,
        "hf_config": hf["config"],
        "sparse_public_config": sparse["public_config"],
        "sparse_capacity_overrides": sparse["capacity_overrides"],
        "sparse_resolved_config": _sparse_resolved_config_summary(
            sparse["resolved_config"],
            sparse["cache_manager_class"],
            args,
        ),
        "summary": _summarize_rollout_rows(all_rows, rollout_steps),
        "batches": result_batches,
    }


def _run_longbench_batch(
    args: argparse.Namespace,
    tokenizer,
    method: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_longbench_batch_debug_args(args)
    decode_steps = int(args.teacher_forced_decode_steps)
    if decode_steps <= 0:
        raise ValueError(f"--teacher_forced_decode_steps must be > 0, got {decode_steps}.")

    batches = _build_longbench_batch_rows(args, tokenizer)
    batch_sizes = [len(batch) for batch in batches]
    batch_lengths = [[int(row["prompt_tokens"]) for row in batch] for batch in batches]
    print(
        f"[Case] longbench_batch/{method}: "
        f"batch_sizes={batch_sizes} prompt_tokens={batch_lengths}"
    )

    hf = _hf_logits_for_longbench_batches(args, method, batches)
    sparse_method = method if method in STANDARD_SPARSE_METHODS or is_deltakv_method(method) else args.sparse_method
    sparse = _sparse_logits_for_longbench_batches(args, sparse_method, batches, hf["batches"])

    result_batches: list[dict[str, Any]] = []
    all_prefill_metrics: list[dict[str, Any]] = []
    all_decode_metrics_by_step: list[list[dict[str, Any]]] = [[] for _ in range(decode_steps)]
    for batch_idx, batch in enumerate(batches):
        sparse_batch = sparse["batches"][batch_idx]
        sparse_prefill = sparse_batch["prefill"]
        sparse_decode_steps = sparse_batch["decode_steps"]
        if int(sparse_prefill.shape[0]) != len(batch):
            raise RuntimeError(
                "Sparse prefill batch row count mismatch while serializing result: "
                f"batch_idx={batch_idx}, expected={len(batch)}, got={int(sparse_prefill.shape[0])}."
            )
        if len(sparse_decode_steps) != decode_steps:
            raise RuntimeError(
                "Sparse decode step count mismatch while serializing result: "
                f"batch_idx={batch_idx}, expected={decode_steps}, got={len(sparse_decode_steps)}."
            )
        rows_out: list[dict[str, Any]] = []
        batch_prefill_metrics: list[dict[str, Any]] = []
        batch_decode_metrics_by_step: list[list[dict[str, Any]]] = [[] for _ in range(decode_steps)]
        for row_idx, row in enumerate(batch):
            hf_row = hf["batches"][batch_idx][row_idx]
            prefill_metrics = _compare_logits(
                hf_row["logits"]["prefill"],
                sparse_prefill[row_idx : row_idx + 1],
                tokenizer=tokenizer,
            )
            all_prefill_metrics.append(prefill_metrics)
            batch_prefill_metrics.append(prefill_metrics)

            decode_metrics: list[dict[str, Any]] = []
            forced_token_ids = [int(token_id) for token_id in hf_row["forced_token_ids"]]
            for step_idx, forced_token_id in enumerate(forced_token_ids):
                sparse_step = sparse_decode_steps[step_idx]
                if int(sparse_step.shape[0]) != len(batch):
                    raise RuntimeError(
                        "Sparse decode batch row count mismatch while serializing result: "
                        f"batch_idx={batch_idx}, step={step_idx}, expected={len(batch)}, "
                        f"got={int(sparse_step.shape[0])}."
                    )
                step_metrics = _compare_logits(
                    hf_row["logits"]["decode_steps"][step_idx],
                    sparse_step[row_idx : row_idx + 1],
                    tokenizer=tokenizer,
                )
                step_metrics["step"] = int(step_idx)
                step_metrics["forced_token_id"] = int(forced_token_id)
                step_metrics["forced_token_text"] = tokenizer.decode([int(forced_token_id)], skip_special_tokens=False)
                decode_metrics.append(step_metrics)
                all_decode_metrics_by_step[step_idx].append(step_metrics)
                batch_decode_metrics_by_step[step_idx].append(step_metrics)

            sparse_metadata = sparse_batch["metadata"]
            rows_out.append(
                {
                    "row_index": int(row_idx),
                    "status": "success",
                    "sample_idx": int(row["sample_idx"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prompt_preview": row["prompt_preview"],
                    "prompt_meta": row["prompt_meta"],
                    "forced_decode_token_ids": forced_token_ids,
                    "forced_decode_token_texts": tokenizer.batch_decode(
                        [[token_id] for token_id in forced_token_ids],
                        skip_special_tokens=False,
                    ),
                    "hf_cache_metadata": hf_row["metadata"],
                    "sparse_cache_metadata": {
                        "after_prefill": sparse_metadata["after_prefill"][row_idx],
                        "after_decode": sparse_metadata["after_decode"][row_idx],
                    },
                    "comparisons": {
                        "prefill": prefill_metrics,
                        "decode": decode_metrics[0] if decode_metrics else None,
                        "decode_steps": decode_metrics,
                    },
                }
            )

        result_batches.append(
            {
                "batch_index": int(batch_idx),
                "batch_size": len(batch),
                "sample_indices": [int(row["sample_idx"]) for row in batch],
                "prompt_tokens": [int(row["prompt_tokens"]) for row in batch],
                "rows": rows_out,
                "summary": {
                    "prefill": _summarize_logit_metrics(batch_prefill_metrics),
                    "decode_steps": [
                        {
                            "step": step_idx,
                            **_summarize_logit_metrics(batch_decode_metrics_by_step[step_idx]),
                        }
                        for step_idx in range(decode_steps)
                    ],
                    "decode_all": _summarize_logit_metrics(
                        [metric for step_metrics in batch_decode_metrics_by_step for metric in step_metrics]
                    ),
                },
            }
        )

    result = {
        "case": "longbench_batch",
        "method": method,
        "sparse_method": sparse_method,
        "status": "success",
        "batch_count": len(batches),
        "batch_sizes": batch_sizes,
        "batch_prompt_tokens": batch_lengths,
        "batch_sample_indices": [[int(row["sample_idx"]) for row in batch] for batch in batches],
        "teacher_forced_decode_steps": decode_steps,
        "hf_config": hf["config"],
        "sparse_public_config": sparse["public_config"],
        "sparse_capacity_overrides": sparse["capacity_overrides"],
        "sparse_resolved_config": _sparse_resolved_config_summary(
            sparse["resolved_config"],
            sparse["cache_manager_class"],
            args,
        ),
        "summary": {
            "prefill": _summarize_logit_metrics(all_prefill_metrics),
            "decode_steps": [
                {
                    "step": step_idx,
                    **_summarize_logit_metrics(all_decode_metrics_by_step[step_idx]),
                }
                for step_idx in range(decode_steps)
            ],
            "decode_all": _summarize_logit_metrics(
                [metric for step_metrics in all_decode_metrics_by_step for metric in step_metrics]
            ),
        },
        "batches": result_batches,
    }
    greedy_rollout = _build_greedy_rollout_result(args, tokenizer, method, sparse_method, batches)
    if greedy_rollout is not None:
        result["greedy_rollout"] = greedy_rollout
    _json_dump(output_dir / f"longbench_batch_{method}.json", result)
    return result


def _run_synthetic_batch(
    args: argparse.Namespace,
    tokenizer,
    method: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_longbench_batch_debug_args(args)
    decode_steps = int(args.teacher_forced_decode_steps)
    if decode_steps <= 0:
        raise ValueError(f"--teacher_forced_decode_steps must be > 0, got {decode_steps}.")

    batches = _build_synthetic_batch_rows(args, tokenizer, method)
    batch_sizes = [len(batch) for batch in batches]
    batch_lengths = [[int(row["prompt_tokens"]) for row in batch] for batch in batches]
    print(
        f"[Case] synthetic_batch/{method}: "
        f"batch_sizes={batch_sizes} prompt_tokens={batch_lengths}"
    )

    hf = _hf_logits_for_longbench_batches(args, method, batches)
    sparse_method = method if method in STANDARD_SPARSE_METHODS or is_deltakv_method(method) else args.sparse_method
    sparse = _sparse_logits_for_longbench_batches(args, sparse_method, batches, hf["batches"])

    result_batches: list[dict[str, Any]] = []
    all_prefill_metrics: list[dict[str, Any]] = []
    all_decode_metrics_by_step: list[list[dict[str, Any]]] = [[] for _ in range(decode_steps)]
    for batch_idx, batch in enumerate(batches):
        sparse_batch = sparse["batches"][batch_idx]
        sparse_prefill = sparse_batch["prefill"]
        sparse_decode_steps = sparse_batch["decode_steps"]
        if int(sparse_prefill.shape[0]) != len(batch):
            raise RuntimeError(
                "Sparse prefill batch row count mismatch while serializing result: "
                f"batch_idx={batch_idx}, expected={len(batch)}, got={int(sparse_prefill.shape[0])}."
            )
        if len(sparse_decode_steps) != decode_steps:
            raise RuntimeError(
                "Sparse decode step count mismatch while serializing result: "
                f"batch_idx={batch_idx}, expected={decode_steps}, got={len(sparse_decode_steps)}."
            )
        rows_out: list[dict[str, Any]] = []
        batch_prefill_metrics: list[dict[str, Any]] = []
        batch_decode_metrics_by_step: list[list[dict[str, Any]]] = [[] for _ in range(decode_steps)]
        for row_idx, row in enumerate(batch):
            hf_row = hf["batches"][batch_idx][row_idx]
            prefill_metrics = _compare_logits(
                hf_row["logits"]["prefill"],
                sparse_prefill[row_idx : row_idx + 1],
                tokenizer=tokenizer,
            )
            all_prefill_metrics.append(prefill_metrics)
            batch_prefill_metrics.append(prefill_metrics)

            decode_metrics: list[dict[str, Any]] = []
            forced_token_ids = [int(token_id) for token_id in hf_row["forced_token_ids"]]
            for step_idx, forced_token_id in enumerate(forced_token_ids):
                sparse_step = sparse_decode_steps[step_idx]
                if int(sparse_step.shape[0]) != len(batch):
                    raise RuntimeError(
                        "Sparse decode batch row count mismatch while serializing result: "
                        f"batch_idx={batch_idx}, step={step_idx}, expected={len(batch)}, "
                        f"got={int(sparse_step.shape[0])}."
                    )
                step_metrics = _compare_logits(
                    hf_row["logits"]["decode_steps"][step_idx],
                    sparse_step[row_idx : row_idx + 1],
                    tokenizer=tokenizer,
                )
                step_metrics["step"] = int(step_idx)
                step_metrics["forced_token_id"] = int(forced_token_id)
                step_metrics["forced_token_text"] = tokenizer.decode([int(forced_token_id)], skip_special_tokens=False)
                decode_metrics.append(step_metrics)
                all_decode_metrics_by_step[step_idx].append(step_metrics)
                batch_decode_metrics_by_step[step_idx].append(step_metrics)

            sparse_metadata = sparse_batch["metadata"]
            rows_out.append(
                {
                    "row_index": int(row_idx),
                    "status": "success",
                    "sample_idx": int(row["sample_idx"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prompt_preview": row["prompt_preview"],
                    "prompt_meta": row["prompt_meta"],
                    "forced_decode_token_ids": forced_token_ids,
                    "forced_decode_token_texts": tokenizer.batch_decode(
                        [[token_id] for token_id in forced_token_ids],
                        skip_special_tokens=False,
                    ),
                    "hf_cache_metadata": hf_row["metadata"],
                    "sparse_cache_metadata": {
                        "after_prefill": sparse_metadata["after_prefill"][row_idx],
                        "after_decode": sparse_metadata["after_decode"][row_idx],
                    },
                    "comparisons": {
                        "prefill": prefill_metrics,
                        "decode": decode_metrics[0] if decode_metrics else None,
                        "decode_steps": decode_metrics,
                    },
                }
            )

        result_batches.append(
            {
                "batch_index": int(batch_idx),
                "batch_size": len(batch),
                "sample_indices": [int(row["sample_idx"]) for row in batch],
                "prompt_tokens": [int(row["prompt_tokens"]) for row in batch],
                "rows": rows_out,
                "summary": {
                    "prefill": _summarize_logit_metrics(batch_prefill_metrics),
                    "decode_steps": [
                        {
                            "step": step_idx,
                            **_summarize_logit_metrics(batch_decode_metrics_by_step[step_idx]),
                        }
                        for step_idx in range(decode_steps)
                    ],
                    "decode_all": _summarize_logit_metrics(
                        [metric for step_metrics in batch_decode_metrics_by_step for metric in step_metrics]
                    ),
                },
            }
        )

    result = {
        "case": "synthetic_batch",
        "method": method,
        "sparse_method": sparse_method,
        "status": "success",
        "batch_count": len(batches),
        "batch_sizes": batch_sizes,
        "batch_prompt_tokens": batch_lengths,
        "batch_sample_indices": [[int(row["sample_idx"]) for row in batch] for batch in batches],
        "teacher_forced_decode_steps": decode_steps,
        "hf_config": hf["config"],
        "sparse_public_config": sparse["public_config"],
        "sparse_capacity_overrides": sparse["capacity_overrides"],
        "sparse_resolved_config": _sparse_resolved_config_summary(
            sparse["resolved_config"],
            sparse["cache_manager_class"],
            args,
        ),
        "summary": {
            "prefill": _summarize_logit_metrics(all_prefill_metrics),
            "decode_steps": [
                {
                    "step": step_idx,
                    **_summarize_logit_metrics(all_decode_metrics_by_step[step_idx]),
                }
                for step_idx in range(decode_steps)
            ],
            "decode_all": _summarize_logit_metrics(
                [metric for step_metrics in all_decode_metrics_by_step for metric in step_metrics]
            ),
        },
        "batches": result_batches,
    }
    _json_dump(output_dir / f"synthetic_batch_{method}.json", result)
    return result


def _run_one(args: argparse.Namespace, tokenizer, case_name: str, method: str, output_dir: Path) -> dict[str, Any]:
    prompt, input_ids, prompt_meta = _build_prompt(tokenizer, case_name, args.long_tokens, args)
    print(f"[Case] {case_name}/{method}: prompt_tokens={len(input_ids)}")
    compressed_state_layers = _parse_int_list(args.compressed_state_layers)
    full_kivi_debug_layers = _parse_int_list(args.full_kivi_debug_layers)
    hidden_debug_layers = _parse_int_list(args.hidden_debug_layers)
    qk_debug_layers = _parse_int_list(args.qk_debug_layers)

    hf_method = _hf_method_for_compare(args, method)
    hf_model, hf_config = _load_hf_model(args, hf_method, len(input_ids))
    try:
        (
            hf_logits,
            hf_metadata,
            forced_token_ids,
            hf_compressed_state,
            hf_full_kivi_state,
            hf_hidden_state,
            hf_qk_state,
        ) = _hf_logits_for_prompt(
            hf_model,
            input_ids,
            int(args.teacher_forced_decode_steps),
            compressed_state_layers if _hf_method_is_deltakv(hf_method) else [],
            full_kivi_debug_layers if _hf_method_is_deltakv(hf_method) else [],
            hidden_debug_layers,
            args.hidden_debug_stage,
            qk_debug_layers,
        )
    finally:
        del hf_model
        _cleanup_cuda()

    sparse_method = method if method in STANDARD_SPARSE_METHODS or is_deltakv_method(method) else args.sparse_method
    (
        sparse_logits,
        sparse_metadata,
        sparse_public_config,
        sparse_resolved_config,
        sparse_cache_manager_class,
        sparse_compressed_state,
        sparse_full_kivi_state,
        sparse_hidden_state,
        sparse_qk_state,
    ) = _sparse_logits_for_prompt(
        args,
        sparse_method,
        input_ids,
        forced_token_ids,
        compressed_state_layers if sparse_method.startswith("deltakv") else [],
        full_kivi_debug_layers if sparse_method.startswith("deltakv") else [],
        hidden_debug_layers,
        args.hidden_debug_stage,
        qk_debug_layers,
    )

    comparisons = {
        "prefill": _compare_logits(hf_logits["prefill"], sparse_logits["prefill"], tokenizer=tokenizer),
        "decode": _compare_logits(hf_logits["decode_steps"][0], sparse_logits["decode_steps"][0], tokenizer=tokenizer),
    }
    comparisons["decode_steps"] = []
    for step_idx, (forced_token_id, hf_step, sparse_step) in enumerate(
        zip(forced_token_ids, hf_logits["decode_steps"], sparse_logits["decode_steps"])
    ):
        step_metrics = _compare_logits(hf_step, sparse_step, tokenizer=tokenizer)
        step_metrics["step"] = step_idx
        step_metrics["forced_token_id"] = int(forced_token_id)
        step_metrics["forced_token_text"] = tokenizer.decode([int(forced_token_id)], skip_special_tokens=False)
        comparisons["decode_steps"].append(step_metrics)
    compressed_state_comparison = _compare_compressed_state_debug(hf_compressed_state, sparse_compressed_state)
    full_kivi_state_comparison = _compare_full_kivi_debug(hf_full_kivi_state, sparse_full_kivi_state)
    hidden_state_comparison = _compare_hidden_debug(hf_hidden_state, sparse_hidden_state)
    qk_debug_comparison = _compare_qk_debug(hf_qk_state, sparse_qk_state)

    first_forced_token_id = int(forced_token_ids[0])
    result = {
        "case": case_name,
        "method": method,
        "status": "success",
        "prompt_tokens": len(input_ids),
        "prompt_preview": prompt[:240],
        "prompt_meta": prompt_meta,
        "forced_decode_token_id": first_forced_token_id,
        "forced_decode_token_text": tokenizer.decode([first_forced_token_id], skip_special_tokens=False),
        "forced_decode_token_ids": forced_token_ids,
        "forced_decode_token_texts": tokenizer.batch_decode([[token_id] for token_id in forced_token_ids], skip_special_tokens=False),
        "hf_config": hf_config,
        "hf_cache_metadata": hf_metadata,
        "sparse_public_config": sparse_public_config,
        "sparse_cache_metadata": sparse_metadata,
        "compressed_state_comparison": compressed_state_comparison,
        "full_kivi_state_comparison": full_kivi_state_comparison,
        "hidden_state_comparison": hidden_state_comparison,
        "qk_debug_comparison": qk_debug_comparison,
        "sparse_resolved_config": {
            "vllm_sparse_method": sparse_resolved_config.vllm_sparse_method,
            "cache_manager_class": sparse_cache_manager_class,
            "prefill_schedule_policy": sparse_resolved_config.prefill_schedule_policy,
            "hidden_debug_stage": args.hidden_debug_stage,
            "decode_keep_tokens": sparse_resolved_config.decode_keep_tokens,
            "num_sink_tokens": sparse_resolved_config.num_sink_tokens,
            "num_recent_tokens": sparse_resolved_config.num_recent_tokens,
            "chunk_prefill_size": sparse_resolved_config.chunk_prefill_size,
            "cluster_ratio": sparse_resolved_config.cluster_ratio,
            "kv_compressed_size": sparse_resolved_config.kv_compressed_size,
            "kv_quant_bits": sparse_resolved_config.kv_quant_bits,
            "kv_quant_group_size": sparse_resolved_config.kv_quant_group_size,
            "use_compression": sparse_resolved_config.use_compression,
            "full_layer_kv_quant_bits": sparse_resolved_config.full_layer_kv_quant_bits,
            "full_layer_kivi_group_size": sparse_resolved_config.full_layer_kivi_group_size,
            "full_layer_kivi_residual_length": sparse_resolved_config.full_layer_kivi_residual_length,
            "enable_full_layer_kivi_quant": sparse_resolved_config.enable_full_layer_kivi_quant,
            "enable_full_layer_kivi_grouped_decode": sparse_resolved_config.enable_full_layer_kivi_grouped_decode,
            "enable_full_layer_kivi_dense_decode": sparse_resolved_config.enable_full_layer_kivi_dense_decode,
            "enable_sparse_ref_fp8": sparse_resolved_config.enable_sparse_ref_fp8,
            "deltakv_k_neighbors": sparse_resolved_config.deltakv_k_neighbors,
            "full_attn_layers": sparse_resolved_config.full_attn_layers,
        },
        "comparisons": comparisons,
    }
    if bool(args.save_hidden_debug_vectors):
        result["hf_hidden_debug_vectors"] = _serialize_hidden_debug(hf_hidden_state)
        result["sparse_hidden_debug_vectors"] = _serialize_hidden_debug(sparse_hidden_state)
    _json_dump(output_dir / f"{case_name}_{method}.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HF and Sparse-VLLM logits on GPU 6.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--compressor_path", default=DEFAULT_COMPRESSOR)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cases", default="short,long")
    parser.add_argument("--methods", default="vanilla,deltakv")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--master_port", type=int, default=29561)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--long_tokens", type=int, default=9000)
    parser.add_argument("--longbench_task", default=None)
    parser.add_argument("--longbench_sample_idx", type=int, default=0)
    parser.add_argument(
        "--longbench_batch_sample_indices",
        default="",
        help="Semicolon-separated LongBench batches, e.g. '0;1,2;3,4,5'. Overrides auto selection.",
    )
    parser.add_argument("--longbench_batch_sizes", default="1,2,4")
    parser.add_argument("--longbench_batch_start_idx", type=int, default=0)
    parser.add_argument("--longbench_batch_candidate_count", type=int, default=200)
    parser.add_argument("--longbench_batch_require_varied_lengths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--longbench_batch_require_varied_batch_sizes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--synthetic_batch_lengths",
        default="32,96;512,1024;2048,4096;16384,65536;105000",
        help="Semicolon-separated synthetic token-id batches, e.g. '32,96;2048,4096;105000'.",
    )
    parser.add_argument("--synthetic_batch_seed", type=int, default=20260617)
    parser.add_argument("--synthetic_token_low", type=int, default=0)
    parser.add_argument("--synthetic_token_high", type=int, default=None)
    parser.add_argument(
        "--longbench_data_dir",
        default=os.getenv("DELTAKV_LONGBENCH_DATA_DIR") or os.getenv("DELTAKV_DATA_DIR"),
    )
    parser.add_argument("--longbench_max_length", type=int, default=121000)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--thinking_mode", default="off", choices=["off", "on_strip"])
    parser.add_argument("--teacher_forced_decode_steps", type=int, default=1)
    parser.add_argument(
        "--greedy_rollout_steps",
        type=int,
        default=0,
        help="If >0 for longbench_batch, run independent HF/sparse greedy rollout and record first token divergence.",
    )
    parser.add_argument("--compressed_state_layers", default="")
    parser.add_argument("--full_kivi_debug_layers", default="")
    parser.add_argument("--hidden_debug_layers", default="")
    parser.add_argument("--hidden_debug_stage", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--save_hidden_debug_vectors", action="store_true")
    parser.add_argument("--qk_debug_layers", default="")
    parser.add_argument("--sparse_method", default="deltakv-less-memory")
    parser.add_argument("--hf_sparse_method", default="delta_compressed_quant_kivi_full_fp8_ref")
    parser.add_argument("--decode_keep_tokens", type=int, default=4096)
    parser.add_argument("--prefill_keep_tokens", type=int, default=4096)
    parser.add_argument("--sink_keep_tokens", type=int, default=8)
    parser.add_argument("--recent_keep_tokens", type=int, default=128)
    parser.add_argument("--snapkv_window_size", type=int, default=32)
    parser.add_argument("--pool_kernel_size", type=int, default=1)
    parser.add_argument("--pyramidkv_start_layer", type=int, default=0)
    parser.add_argument("--pyramidkv_start_ratio", type=float, default=0.6)
    parser.add_argument("--pyramidkv_least_layer", type=int, default=None)
    parser.add_argument("--pyramidkv_least_ratio", type=float, default=0.01)
    parser.add_argument("--quest_chunk_size", type=int, default=16)
    parser.add_argument("--quest_token_budget", type=int, default=1024)
    parser.add_argument("--quest_skip_layers", type=int, default=2)
    parser.add_argument("--full_attention_layers", default="0,1,2,4,7,14")
    parser.add_argument("--deltakv_center_ratio", type=float, default=0.1)
    parser.add_argument("--use_compression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deltakv_latent_dim", type=int, default=None)
    parser.add_argument("--deltakv_latent_quant_bits", type=int, default=0)
    parser.add_argument("--deltakv_latent_quant_group_size", type=int, default=32)
    parser.add_argument("--deltakv_neighbor_count", type=int, default=4)
    parser.add_argument("--full_layer_kv_quant_bits", type=int, default=4)
    parser.add_argument("--full_layer_kivi_group_size", type=int, default=32)
    parser.add_argument("--full_layer_kivi_residual_length", type=int, default=32)
    parser.add_argument("--enable_full_layer_kivi_quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_full_layer_kivi_fused_decode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_full_layer_kivi_grouped_decode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_full_layer_kivi_dense_decode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_sparse_ref_fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--engine_prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--hf_prefill_chunk_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--deltakv_full_pool_reserve_ratio", type=float, default=0.2)
    parser.add_argument("--deltakv_cluster_gather_chunk_size", type=int, default=16384)
    parser.add_argument("--mlp_chunk_size", type=int, default=16384)
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode_cuda_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode_cuda_graph_capture_sizes", default="auto")
    parser.add_argument("--max_num_seqs_in_batch", type=int, default=1)
    parser.add_argument("--max_decoding_seqs", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--tiny_random_config",
        default=None,
        help="JSON architecture overrides for deterministic random-weight comparison.",
    )
    parser.add_argument("--tiny_random_seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.cuda_visible_devices is not None and visible != args.cuda_visible_devices:
        raise RuntimeError(
            "This script must be launched with the intended visible GPU. "
            f"Expected CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r}, got {visible!r}."
        )

    os.environ.setdefault("SPARSEVLLM_MASTER_PORT", str(args.master_port))
    _require_path(args.model_path, "model_path")
    if args.tiny_random_config:
        _require_path(args.tiny_random_config, "tiny_random_config")
        if int(args.tiny_random_seed) < 0:
            raise ValueError("--tiny_random_seed must be >= 0.")
        os.environ[TINY_RANDOM_ENV] = "1"
        os.environ[TINY_RANDOM_CONFIG_ENV] = os.path.abspath(args.tiny_random_config)
        os.environ[TINY_RANDOM_SEED_ENV] = str(int(args.tiny_random_seed))
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    if args.tiny_random_config and set(methods) != {"vanilla"}:
        raise NotImplementedError(
            "Tiny random HF comparison currently requires --methods vanilla."
        )
    uses_deltakv_compare = any(method in DELTAKV_COMPARE_METHODS for method in methods)
    if bool(args.use_compression) and uses_deltakv_compare:
        _require_path(args.compressor_path, "compressor_path")
    if not isinstance(args.decode_keep_tokens, int) or not isinstance(args.prefill_keep_tokens, int):
        raise TypeError("HF prefill and shared decode token budgets must be integers.")
    if args.full_kivi_debug_layers:
        os.environ["SPARSEVLLM_DEBUG_FULL_KIVI_LAYERS"] = args.full_kivi_debug_layers
    if args.hidden_debug_layers:
        os.environ["SPARSEVLLM_DEBUG_HIDDEN_LAYERS"] = args.hidden_debug_layers
    if args.qk_debug_layers:
        os.environ["DELTAKV_DEBUG_QK_LAYERS"] = args.qk_debug_layers
        os.environ["SPARSEVLLM_DEBUG_QK_LAYERS"] = args.qk_debug_layers

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_info = {
        "status": "running",
        "created_at": timestamp,
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "cuda_visible_devices": visible,
        "torch_version": torch.__version__,
        "model_path": args.model_path,
        "compressor_path": args.compressor_path,
        "args": vars(args),
    }
    _json_dump(output_dir / "run_info.json", run_info)

    cases = _parse_cases(args.cases)
    allowed_methods = STANDARD_SPARSE_METHODS | DELTAKV_COMPARE_METHODS
    bad_methods = sorted(set(methods) - allowed_methods)
    if bad_methods:
        raise ValueError(f"Unsupported methods: {bad_methods}. Allowed: {sorted(allowed_methods)}")

    results = []
    try:
        for method in methods:
            for case_name in cases:
                if case_name == "longbench_batch":
                    results.append(_run_longbench_batch(args, tokenizer, method, output_dir))
                elif case_name == "synthetic_batch":
                    results.append(_run_synthetic_batch(args, tokenizer, method, output_dir))
                else:
                    results.append(_run_one(args, tokenizer, case_name, method, output_dir))
        run_info["status"] = "completed"
        run_info["completed_at"] = time.strftime("%Y%m%d_%H%M%S")
        run_info["results"] = results
        _json_dump(output_dir / "summary.json", run_info)
        _json_dump(output_dir / "run_info.json", run_info)
    except Exception as exc:
        run_info["status"] = "failed"
        run_info["error"] = repr(exc)
        _json_dump(output_dir / "run_info.json", run_info)
        raise

    print(f"[Done] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
