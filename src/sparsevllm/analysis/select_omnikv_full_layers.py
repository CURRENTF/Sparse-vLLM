from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from benchmark.long_bench.pred import build_chat
from deltakv.baseline_adapters import load_omnikv_model
from deltakv.configs.default_paths import longbench_root, model_path, output_path


DEFAULT_MODEL_PATH = model_path("Qwen2.5-7B-Instruct-1M")
DEFAULT_LONGBENCH_ROOT = longbench_root()
DEFAULT_OUTPUT_ROOT = output_path("deltakv", "omnikv_full_layer_calibration")
DEFAULT_CONFIG_DIR = "benchmark/long_bench/config"


@dataclass(frozen=True)
class CalibrationPoint:
    sample_idx: int
    point_idx: int
    kind: str
    prefix_len: int
    query_token_id: int


def parse_int_list(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def require_path(path: str | Path, kind: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{kind} does not exist: {resolved}")
    return resolved


def git_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError(f"num_samples must be > 0, got {count}.")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx >= count:
                break
            rows.append(json.loads(line))
    if len(rows) < count:
        raise ValueError(f"Requested {count} samples from {path}, but found only {len(rows)}.")
    return rows


def read_jsonl_indices(path: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    if not indices:
        raise ValueError("No sample indices requested.")
    max_idx = max(indices)
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx > max_idx:
                break
            if line_idx in indices:
                rows[line_idx] = json.loads(line)
    missing = sorted(indices - set(rows))
    if missing:
        raise ValueError(f"Dataset {path} is missing requested sample indices: {missing[:10]}")
    return rows


def token_ids_for_prompt(tokenizer, prompt: str) -> list[int]:
    add_special_tokens = True
    if tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
        add_special_tokens = False
    ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not ids:
        raise ValueError("Prompt tokenized to an empty sequence.")
    return ids


def first_answer_token_id(tokenizer, sample: dict[str, Any]) -> tuple[int, str]:
    answers = sample.get("answers")
    if isinstance(answers, list):
        if not answers:
            raise ValueError("Sample has an empty `answers` list; cannot build answer-boundary point.")
        answer = str(answers[0])
    elif answers is not None:
        answer = str(answers)
    else:
        raise ValueError("Sample has no `answers` field; cannot build answer-boundary point.")
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"First answer tokenized to an empty sequence: {answer!r}")
    return int(ids[0]), answer


def build_longbench_prompt_and_ids(
    *,
    tokenizer,
    sample: dict[str, Any],
    dataset: str,
    prompt_format: str,
    max_length: int,
    no_chat_template: bool,
    thinking_mode: str,
) -> tuple[str, list[int], bool]:
    prompt = prompt_format.format(**sample)
    raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    truncated = False
    if len(raw_ids) > max_length:
        half = int(max_length / 2)
        if half <= 0:
            raise ValueError(f"max_length must be > 1 for middle truncation, got {max_length}.")
        prompt = tokenizer.decode(raw_ids[:half], skip_special_tokens=True) + tokenizer.decode(
            raw_ids[-half:], skip_special_tokens=True
        )
        truncated = True
    prompt = build_chat(
        tokenizer,
        prompt,
        dataset,
        no_chat_template=no_chat_template,
        thinking_mode=thinking_mode,
    )
    return prompt, token_ids_for_prompt(tokenizer, prompt), truncated


def sample_decode_points(
    *,
    sample_idx: int,
    prompt_token_ids: list[int],
    answer_query_token_id: int,
    random_points_per_sample: int,
    rng: random.Random,
    num_sink_tokens: int,
    num_recent_tokens: int,
    min_prefix_tokens: int,
) -> list[CalibrationPoint]:
    if random_points_per_sample < 0:
        raise ValueError(f"random_points_per_sample must be >= 0, got {random_points_per_sample}.")
    prompt_len = len(prompt_token_ids)
    min_prefix = max(int(min_prefix_tokens), int(num_sink_tokens) + int(num_recent_tokens))
    max_prefix = prompt_len - 1
    if max_prefix < min_prefix:
        raise ValueError(
            "Prompt is too short for decode-point sampling after sink/recent exclusion: "
            f"prompt_len={prompt_len}, min_prefix={min_prefix}."
        )

    candidates = list(range(min_prefix, max_prefix + 1))
    if random_points_per_sample > len(candidates):
        raise ValueError(
            "Requested more unique random decode points than available positions: "
            f"requested={random_points_per_sample}, available={len(candidates)}."
        )

    random_prefix_lens = sorted(rng.sample(candidates, random_points_per_sample))
    points: list[CalibrationPoint] = []
    for point_idx, prefix_len in enumerate(random_prefix_lens):
        points.append(
            CalibrationPoint(
                sample_idx=sample_idx,
                point_idx=point_idx,
                kind="random",
                prefix_len=prefix_len,
                query_token_id=int(prompt_token_ids[prefix_len]),
            )
        )

    points.append(
        CalibrationPoint(
            sample_idx=sample_idx,
            point_idx=len(points),
            kind="answer_boundary",
            prefix_len=prompt_len,
            query_token_id=int(answer_query_token_id),
        )
    )
    return sorted(points, key=lambda point: (point.prefix_len, point.kind != "random"))


def topk_indices_from_decode_attentions(
    attentions: tuple[torch.Tensor, ...],
    *,
    topk: int,
    num_sink_tokens: int,
    num_recent_tokens: int,
) -> tuple[list[list[int]], int]:
    if not attentions:
        raise RuntimeError("Model did not return attentions for the decode point.")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}.")

    layer_topk: list[list[int]] = []
    k_eff: int | None = None
    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            raise RuntimeError(f"Attention tensor for layer {layer_idx} is None.")
        if attn.dim() != 4 or attn.shape[0] != 1 or attn.shape[2] != 1:
            raise RuntimeError(
                "Expected decode attention shape (1, num_heads, 1, kv_len), "
                f"got layer {layer_idx} shape {tuple(attn.shape)}."
            )
        scores = attn[0, :, 0, :].detach().float().max(dim=0).values
        kv_len = int(scores.numel())
        search_start = int(num_sink_tokens)
        search_end = kv_len - int(num_recent_tokens)
        if search_end <= search_start:
            raise RuntimeError(
                "Decode point has no searchable history after sink/recent exclusion: "
                f"kv_len={kv_len}, num_sink_tokens={num_sink_tokens}, num_recent_tokens={num_recent_tokens}."
            )
        search_scores = scores[search_start:search_end]
        cur_k = min(int(topk), int(search_scores.numel()))
        if k_eff is None:
            k_eff = cur_k
        elif k_eff != cur_k:
            raise RuntimeError(f"Inconsistent top-k length across layers: first={k_eff}, layer{layer_idx}={cur_k}.")
        indices = torch.topk(search_scores, k=cur_k, dim=-1, sorted=False).indices + search_start
        layer_topk.append([int(x) for x in indices.cpu().tolist()])

    assert k_eff is not None
    return layer_topk, k_eff


def add_topk_to_pair_scores(pair_scores: np.ndarray, layer_topk: list[list[int]]) -> None:
    num_layers = len(layer_topk)
    if pair_scores.shape != (num_layers, num_layers):
        raise ValueError(f"pair_scores shape {pair_scores.shape} does not match {num_layers} layers.")
    sets = [set(indices) for indices in layer_topk]
    for anchor in range(num_layers):
        anchor_set = sets[anchor]
        for target in range(anchor + 1, num_layers):
            pair_scores[anchor, target] += len(anchor_set & sets[target])


def compute_segment_scores(pair_scores: np.ndarray) -> np.ndarray:
    if pair_scores.ndim != 2 or pair_scores.shape[0] != pair_scores.shape[1]:
        raise ValueError(f"pair_scores must be a square matrix, got shape {pair_scores.shape}.")
    num_layers = int(pair_scores.shape[0])
    segment_scores = np.zeros((num_layers, num_layers + 1), dtype=np.int64)
    for anchor in range(num_layers):
        running = 0
        for next_full in range(anchor + 1, num_layers + 1):
            target = next_full - 1
            if target > anchor:
                running += int(pair_scores[anchor, target])
            segment_scores[anchor, next_full] = running
    return segment_scores


def select_full_layers_dp(segment_scores: np.ndarray, num_full_layers: int) -> tuple[list[int], int]:
    if segment_scores.ndim != 2 or segment_scores.shape[1] != segment_scores.shape[0] + 1:
        raise ValueError(f"segment_scores must have shape (num_layers, num_layers + 1), got {segment_scores.shape}.")
    num_layers = int(segment_scores.shape[0])
    if num_full_layers <= 0 or num_full_layers > num_layers:
        raise ValueError(f"num_full_layers must be in [1, {num_layers}], got {num_full_layers}.")

    memo: dict[tuple[int, int, int], tuple[int, tuple[int, ...]]] = {}

    def best(prev_full: int, min_next: int, remaining: int) -> tuple[int, tuple[int, ...]]:
        key = (prev_full, min_next, remaining)
        if key in memo:
            return memo[key]
        if remaining == 0:
            result = (int(segment_scores[prev_full, num_layers]), ())
            memo[key] = result
            return result

        best_score: int | None = None
        best_suffix: tuple[int, ...] | None = None
        max_candidate = num_layers - remaining
        for candidate in range(min_next, max_candidate + 1):
            suffix_score, suffix = best(candidate, candidate + 1, remaining - 1)
            score = int(segment_scores[prev_full, candidate]) + suffix_score
            candidate_suffix = (candidate,) + suffix
            if best_score is None or score > best_score or (score == best_score and candidate_suffix < best_suffix):
                best_score = score
                best_suffix = candidate_suffix
        assert best_score is not None and best_suffix is not None
        result = (best_score, best_suffix)
        memo[key] = result
        return result

    score, suffix = best(0, 1, num_full_layers - 1)
    return [0, *suffix], int(score)


def selected_segment_breakdown(segment_scores: np.ndarray, selected_layers: list[int]) -> list[dict[str, Any]]:
    num_layers = int(segment_scores.shape[0])
    out = []
    for idx, anchor in enumerate(selected_layers):
        next_full = selected_layers[idx + 1] if idx + 1 < len(selected_layers) else num_layers
        out.append(
            {
                "anchor": int(anchor),
                "next_full_or_end": int(next_full),
                "sparse_layers": list(range(anchor + 1, next_full)),
                "score": int(segment_scores[anchor, next_full]),
            }
        )
    return out


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype {name!r}. Use bfloat16, float16, or float32.")


def move_inputs(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], dtype=torch.long, device=device)


@torch.no_grad()
def advance_cache(model, past_key_values, token_ids: list[int], *, device: torch.device, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError(f"prefill_chunk_size must be > 0, got {chunk_size}.")
    past = past_key_values
    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        if not chunk:
            continue
        outputs = model(
            input_ids=move_inputs(chunk, device),
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = outputs.past_key_values
    return past


@torch.no_grad()
def collect_sample_topk(
    *,
    model,
    device: torch.device,
    prompt_token_ids: list[int],
    points: list[CalibrationPoint],
    topk: int,
    num_sink_tokens: int,
    num_recent_tokens: int,
    prefill_chunk_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    num_layers = int(model.config.num_hidden_layers)
    pair_scores = np.zeros((num_layers, num_layers), dtype=np.int64)
    point_records: list[dict[str, Any]] = []
    saveable_topk: list[dict[str, Any]] = []

    past = None
    processed = 0
    for point in points:
        if point.prefix_len < processed:
            raise RuntimeError(
                f"Calibration points must be nondecreasing by prefix_len; got {point.prefix_len} after {processed}."
            )
        past = advance_cache(
            model,
            past,
            prompt_token_ids[processed : point.prefix_len],
            device=device,
            chunk_size=prefill_chunk_size,
        )
        processed = point.prefix_len

        outputs = model(
            input_ids=move_inputs([point.query_token_id], device),
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        processed += 1

        layer_topk, k_eff = topk_indices_from_decode_attentions(
            outputs.attentions,
            topk=topk,
            num_sink_tokens=num_sink_tokens,
            num_recent_tokens=num_recent_tokens,
        )
        add_topk_to_pair_scores(pair_scores, layer_topk)
        record = asdict(point)
        record.update({"status": "success", "effective_topk": int(k_eff)})
        point_records.append(record)
        saveable_topk.append({"point": record, "topk_indices_by_layer": layer_topk})

    return point_records, pair_scores, saveable_topk


def parse_policy_arg(value: str) -> tuple[str, list[int]]:
    if "=" in value:
        name, layers = value.split("=", 1)
    elif ":" in value:
        name, layers = value.split(":", 1)
    else:
        raise ValueError(f"Policy must be NAME=0,1,2 form, got {value!r}.")
    name = name.strip()
    if not name:
        raise ValueError(f"Policy name is empty in {value!r}.")
    parsed = parse_int_list(layers)
    if not parsed:
        raise ValueError(f"Policy {name!r} has no layers.")
    if parsed[0] != 0:
        raise ValueError(f"Policy {name!r} must include layer 0 first, got {parsed}.")
    if parsed != sorted(set(parsed)):
        raise ValueError(f"Policy {name!r} must be sorted and unique, got {parsed}.")
    return name, parsed


def load_selector_point_records(selector_output_dir: Path, *, include_answer_boundary: bool) -> list[dict[str, Any]]:
    path = require_path(selector_output_dir / "per_sample_points.jsonl", "selector point record")
    points: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sample_record = json.loads(line)
            for point in sample_record.get("points", []):
                if point.get("kind") == "random" or include_answer_boundary:
                    points.append(point)
    if not points:
        raise ValueError(f"No calibration points found in {path}.")
    return points


def group_points_by_sample(points: list[dict[str, Any]]) -> dict[int, list[CalibrationPoint]]:
    grouped: dict[int, list[CalibrationPoint]] = {}
    for point in points:
        sample_idx = int(point["sample_idx"])
        grouped.setdefault(sample_idx, []).append(
            CalibrationPoint(
                sample_idx=sample_idx,
                point_idx=int(point["point_idx"]),
                kind=str(point["kind"]),
                prefix_len=int(point["prefix_len"]),
                query_token_id=int(point["query_token_id"]),
            )
        )
    return {
        sample_idx: sorted(sample_points, key=lambda item: (item.prefix_len, item.point_idx))
        for sample_idx, sample_points in grouped.items()
    }


def top128_kl_from_logits(full_logits: torch.Tensor, sparse_logits: torch.Tensor, top_k: int) -> dict[str, Any]:
    if full_logits.shape != sparse_logits.shape:
        raise ValueError(f"Logit shape mismatch: full={tuple(full_logits.shape)} sparse={tuple(sparse_logits.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")
    vocab = int(full_logits.numel())
    k_eff = min(int(top_k), vocab)
    full = full_logits.detach().float().view(-1)
    sparse = sparse_logits.detach().float().view(-1)
    top_idx = torch.topk(full, k=k_eff, dim=-1).indices
    full_top = full.index_select(0, top_idx)
    sparse_top = sparse.index_select(0, top_idx)
    full_log_probs = torch.log_softmax(full_top, dim=-1)
    sparse_log_probs = torch.log_softmax(sparse_top, dim=-1)
    full_probs = full_log_probs.exp()
    kl = torch.sum(full_probs * (full_log_probs - sparse_log_probs))
    return {
        "top_k": k_eff,
        "kl": float(kl.item()),
        "full_argmax": int(torch.argmax(full).item()),
        "sparse_argmax": int(torch.argmax(sparse).item()),
        "argmax_match": int(torch.argmax(full).item()) == int(torch.argmax(sparse).item()),
        "top128_overlap": len(set(int(x) for x in top_idx.cpu().tolist()) & set(int(x) for x in torch.topk(sparse, k=k_eff, dim=-1).indices.cpu().tolist())) / k_eff,
    }


@torch.no_grad()
def collect_point_logits(
    *,
    model,
    device: torch.device,
    samples_by_idx: dict[int, dict[str, Any]],
    points_by_sample: dict[int, list[CalibrationPoint]],
    tokenizer,
    dataset: str,
    prompt_format: str,
    max_length: int,
    no_chat_template: bool,
    thinking_mode: str,
    prefill_chunk_size: int,
) -> dict[tuple[int, int], torch.Tensor]:
    logits_by_point: dict[tuple[int, int], torch.Tensor] = {}
    for sample_idx in tqdm(sorted(points_by_sample), desc="Collecting logits"):
        sample = samples_by_idx[sample_idx]
        _, prompt_token_ids, _ = build_longbench_prompt_and_ids(
            tokenizer=tokenizer,
            sample=sample,
            dataset=dataset,
            prompt_format=prompt_format,
            max_length=max_length,
            no_chat_template=no_chat_template,
            thinking_mode=thinking_mode,
        )
        past = None
        processed = 0
        for point in points_by_sample[sample_idx]:
            if point.prefix_len < processed:
                raise RuntimeError(
                    f"KL points must be nondecreasing by prefix_len; got {point.prefix_len} after {processed}."
                )
            past = advance_cache(
                model,
                past,
                prompt_token_ids[processed : point.prefix_len],
                device=device,
                chunk_size=prefill_chunk_size,
            )
            processed = point.prefix_len
            outputs = model(
                input_ids=move_inputs([point.query_token_id], device),
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past = outputs.past_key_values
            processed += 1
            logits_by_point[(sample_idx, point.point_idx)] = outputs.logits[:, -1, :].detach().float().cpu()
        del past
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return logits_by_point


def summarize_kl_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize empty KL records.")
    values = np.array([float(record["kl"]) for record in records], dtype=np.float64)
    return {
        "num_points": int(len(records)),
        "mean_kl": float(values.mean()),
        "median_kl": float(np.median(values)),
        "p90_kl": float(np.quantile(values, 0.90)),
        "max_kl": float(values.max()),
        "argmax_match_rate": float(np.mean([1.0 if record["argmax_match"] else 0.0 for record in records])),
        "mean_top128_overlap": float(np.mean([float(record["top128_overlap"]) for record in records])),
    }


def cuda_device_map_arg(device: torch.device) -> int | str:
    if device.type == "cuda":
        return 0 if device.index is None else int(device.index)
    return "cpu"


def run_top128_kl_validation(args: argparse.Namespace) -> dict[str, Any]:
    selector_output_dir = require_path(args.selector_output_dir, "selector output dir")
    selected_path = require_path(selector_output_dir / "selected_full_layers.json", "selected full-layer result")
    run_info_path = selector_output_dir / "run_info.json"
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    prior_run_info = json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.exists() else {}

    model_path = require_path(args.model_path or prior_run_info.get("model_path") or DEFAULT_MODEL_PATH, "model path")
    longbench_root = require_path(args.longbench_root or prior_run_info.get("longbench_root") or DEFAULT_LONGBENCH_ROOT, "LongBench root")
    config_dir = require_path(args.config_dir or DEFAULT_CONFIG_DIR, "LongBench config dir")
    dataset = args.dataset or selected_payload.get("dataset") or "narrativeqa"
    data_path = require_path(longbench_root / "data" / f"{dataset}.jsonl", "LongBench dataset file")
    prompt_path = require_path(config_dir / "dataset2prompt.json", "LongBench prompt config")
    with prompt_path.open("r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    if dataset not in dataset2prompt:
        raise ValueError(f"Dataset {dataset!r} is missing from {prompt_path}.")

    points = load_selector_point_records(
        selector_output_dir,
        include_answer_boundary=bool(args.top128_kl_include_answer_boundary),
    )
    if args.top128_kl_max_points is not None:
        max_points = int(args.top128_kl_max_points)
        if max_points <= 0:
            raise ValueError(f"top128_kl_max_points must be > 0, got {max_points}.")
        points = points[:max_points]
    points_by_sample = group_points_by_sample(points)
    samples_by_idx = read_jsonl_indices(data_path, set(points_by_sample))

    output_dir = selector_output_dir / "top128_kl"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    base_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    max_length = int(args.max_length or prior_run_info.get("max_length") or getattr(base_config, "max_position_embeddings", 32000))
    no_chat_template = bool(args.no_chat_template or prior_run_info.get("no_chat_template", False))
    thinking_mode = args.thinking_mode or prior_run_info.get("thinking_mode", "off")
    dtype = torch_dtype_from_name(args.torch_dtype)
    device = torch.device(args.device)

    full_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.top128_kl_attn_implementation,
    )
    full_model.to(device)
    full_model.eval()
    full_logits = collect_point_logits(
        model=full_model,
        device=device,
        samples_by_idx=samples_by_idx,
        points_by_sample=points_by_sample,
        tokenizer=tokenizer,
        dataset=dataset,
        prompt_format=dataset2prompt[dataset],
        max_length=max_length,
        no_chat_template=no_chat_template,
        thinking_mode=thinking_mode,
        prefill_chunk_size=int(args.prefill_chunk_size),
    )
    del full_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    policies: list[tuple[str, list[int]]] = [
        ("selected", [int(layer) for layer in selected_payload["selected_full_layers"]])
    ]
    for item in args.top128_kl_policy or []:
        policies.append(parse_policy_arg(item))

    policy_results = {}
    for policy_name, layers in policies:
        infer_config = {
            "full_attn_layers": ",".join(str(layer) for layer in layers),
            "decode_keep_tokens": int(args.topk or selected_payload.get("topk", 2048)),
            "num_sink_tokens": int(args.num_sink_tokens if args.num_sink_tokens is not None else selected_payload.get("num_sink_tokens", 0)),
            "num_recent_tokens": int(args.num_recent_tokens if args.num_recent_tokens is not None else selected_payload.get("num_recent_tokens", 32)),
            "chunk_prefill_size": int(args.prefill_chunk_size),
            "pool_kernel_size": 1,
        }
        sparse_model = load_omnikv_model(str(model_path), infer_config, cuda_device_map_arg(device))
        sparse_model.eval()
        sparse_logits = collect_point_logits(
            model=sparse_model,
            device=device,
            samples_by_idx=samples_by_idx,
            points_by_sample=points_by_sample,
            tokenizer=tokenizer,
            dataset=dataset,
            prompt_format=dataset2prompt[dataset],
            max_length=max_length,
            no_chat_template=no_chat_template,
            thinking_mode=thinking_mode,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )
        records = []
        for key in sorted(full_logits):
            metric = top128_kl_from_logits(full_logits[key], sparse_logits[key], int(args.top128_kl_topk))
            records.append(
                {
                    "sample_idx": int(key[0]),
                    "point_idx": int(key[1]),
                    "policy": policy_name,
                    **metric,
                }
            )
        policy_results[policy_name] = {
            "full_attention_layers": ",".join(str(layer) for layer in layers),
            "summary": summarize_kl_records(records),
            "records": records,
        }
        del sparse_model, sparse_logits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "metric": "top128_kl",
        "selector_output_dir": str(selector_output_dir),
        "model_path": str(model_path),
        "dataset": dataset,
        "data_path": str(data_path),
        "num_points": int(len(points)),
        "include_answer_boundary": bool(args.top128_kl_include_answer_boundary),
        "topk": int(args.top128_kl_topk),
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "attention_implementation": args.top128_kl_attn_implementation,
        "policies": policy_results,
        "created_at": datetime.now().isoformat(),
        "command": sys.argv,
        "git_commit": git_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_text(["git", "status", "--short"]),
    }
    json_dump(output_dir / "top128_kl_metrics.json", result)
    return result


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    model_path = require_path(args.model_path, "model path")
    longbench_root = require_path(args.longbench_root, "LongBench root")
    config_dir = require_path(args.config_dir, "LongBench config dir")
    data_path = require_path(longbench_root / "data" / f"{args.dataset}.jsonl", "LongBench dataset file")
    prompt_path = require_path(config_dir / "dataset2prompt.json", "LongBench prompt config")

    with prompt_path.open("r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    if args.dataset not in dataset2prompt:
        raise ValueError(f"Dataset {args.dataset!r} is missing from {prompt_path}.")
    prompt_format = dataset2prompt[args.dataset]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    base_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    max_length = int(args.max_length or getattr(base_config, "max_position_embeddings", 32000))
    if max_length <= 0:
        raise ValueError(f"Resolved max_length must be > 0, got {max_length}.")

    dtype = torch_dtype_from_name(args.torch_dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    if args.num_full_layers > num_layers:
        raise ValueError(f"num_full_layers={args.num_full_layers} exceeds num_hidden_layers={num_layers}.")

    samples = read_jsonl_prefix(data_path, args.num_samples)
    rng = random.Random(args.seed)
    total_pair_scores = np.zeros((num_layers, num_layers), dtype=np.int64)
    all_point_records: list[dict[str, Any]] = []
    all_topk: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"Calibrating {args.dataset}")):
        prompt, prompt_token_ids, truncated = build_longbench_prompt_and_ids(
            tokenizer=tokenizer,
            sample=sample,
            dataset=args.dataset,
            prompt_format=prompt_format,
            max_length=max_length,
            no_chat_template=bool(args.no_chat_template),
            thinking_mode=args.thinking_mode,
        )
        answer_token_id, answer_text = first_answer_token_id(tokenizer, sample)
        points = sample_decode_points(
            sample_idx=sample_idx,
            prompt_token_ids=prompt_token_ids,
            answer_query_token_id=answer_token_id,
            random_points_per_sample=args.random_decode_points_per_sample,
            rng=rng,
            num_sink_tokens=args.num_sink_tokens,
            num_recent_tokens=args.num_recent_tokens,
            min_prefix_tokens=args.min_prefix_tokens,
        )
        point_records, sample_pair_scores, sample_topk = collect_sample_topk(
            model=model,
            device=device,
            prompt_token_ids=prompt_token_ids,
            points=points,
            topk=args.topk,
            num_sink_tokens=args.num_sink_tokens,
            num_recent_tokens=args.num_recent_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        total_pair_scores += sample_pair_scores
        all_point_records.extend(point_records)
        if args.save_topk:
            all_topk.extend(sample_topk)
        prompt_records.append(
            {
                "sample_idx": sample_idx,
                "status": "success",
                "prompt_token_length": len(prompt_token_ids),
                "truncated": truncated,
                "answer_boundary_token_id": answer_token_id,
                "answer_boundary_answer": answer_text,
                "point_count": len(points),
                "points": point_records,
            }
        )
        del prompt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    segment_scores = compute_segment_scores(total_pair_scores)
    selected_layers, best_score = select_full_layers_dp(segment_scores, args.num_full_layers)
    full_layers_str = ",".join(str(layer) for layer in selected_layers)

    point_topk_sum = sum(int(record["effective_topk"]) for record in all_point_records)
    denominator = point_topk_sum * (num_layers - len(selected_layers))
    normalized = float(best_score / denominator) if denominator else 0.0

    np.save(output_dir / "pair_scores.npy", total_pair_scores)
    np.save(output_dir / "segment_scores.npy", segment_scores)

    with (output_dir / "per_sample_points.jsonl").open("w", encoding="utf-8") as f:
        for record in prompt_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.save_topk:
        torch.save(all_topk, output_dir / "topk_indices.pt")

    selected_payload = {
        "selected_full_layers": selected_layers,
        "full_attention_layers": full_layers_str,
        "num_hidden_layers": num_layers,
        "num_full_layers": int(args.num_full_layers),
        "forced_full_layers": [0],
        "topk": int(args.topk),
        "num_sink_tokens": int(args.num_sink_tokens),
        "num_recent_tokens": int(args.num_recent_tokens),
        "dataset": args.dataset,
        "num_samples": int(args.num_samples),
        "random_decode_points_per_sample": int(args.random_decode_points_per_sample),
        "answer_boundary_points_per_sample": 1,
        "seed": int(args.seed),
        "token_coverage_score": int(best_score),
        "coverage_denominator": int(denominator),
        "normalized_token_coverage": normalized,
        "segment_breakdown": selected_segment_breakdown(segment_scores, selected_layers),
    }
    json_dump(output_dir / "selected_full_layers.json", selected_payload)

    run_info = {
        "command": sys.argv,
        "created_at": datetime.now().isoformat(),
        "cwd": os.getcwd(),
        "model_path": str(model_path),
        "longbench_root": str(longbench_root),
        "data_path": str(data_path),
        "prompt_path": str(prompt_path),
        "output_dir": str(output_dir),
        "model_config_model_type": getattr(base_config, "model_type", None),
        "model_config_num_hidden_layers": getattr(base_config, "num_hidden_layers", None),
        "attention_implementation": "eager",
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "max_length": int(max_length),
        "no_chat_template": bool(args.no_chat_template),
        "thinking_mode": args.thinking_mode,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "git_commit": git_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_text(["git", "status", "--short"]),
    }
    json_dump(output_dir / "run_info.json", run_info)
    return {"output_dir": str(output_dir), **selected_payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OmniKV full-layer selector using decode-style token coverage.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench-root", default=DEFAULT_LONGBENCH_ROOT)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--dataset", default="narrativeqa")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-full-layers", type=int, default=6)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--random-decode-points-per-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sink-tokens", type=int, default=0)
    parser.add_argument("--num-recent-tokens", type=int, default=32)
    parser.add_argument("--min-prefix-tokens", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--thinking-mode", default="off", choices=("off", "on"))
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--save-topk", action="store_true")
    parser.add_argument("--top128-kl-only", action="store_true")
    parser.add_argument("--selector-output-dir", default=None)
    parser.add_argument("--top128-kl-policy", action="append", default=[])
    parser.add_argument("--top128-kl-topk", type=int, default=128)
    parser.add_argument("--top128-kl-max-points", type=int, default=None)
    parser.add_argument("--top128-kl-include-answer-boundary", action="store_true")
    parser.add_argument("--top128-kl-attn-implementation", default="flash_attention_2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.top128_kl_only:
        if not args.selector_output_dir:
            raise ValueError("--selector-output-dir is required with --top128-kl-only.")
        result = run_top128_kl_validation(args)
    else:
        result = run_calibration(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
