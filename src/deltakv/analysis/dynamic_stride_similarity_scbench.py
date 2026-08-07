#!/usr/bin/env python3
"""Measure dynamic-stride reference-center similarity on SCBench prompts.

The metric compares each non-sink token with its best historical reference
center under a fixed-stride schedule and dynamic-stride schedules. It is meant
to support the paper claim that dynamic stride sharply reduces reference-token
count without a large drop in KV-cache similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltakv.analysis.colors import (
    COLOR_BLACK,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_PRIMARY_LIGHT,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
)
from deltakv.configs.default_paths import dataset_path as default_dataset_path


SUCCESS = "success"
SKIPPED = "skipped_by_policy"
MODEL_FAILED = "model_failed"
METRIC_FAILED = "metric_failed"
INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class PromptSample:
    task: str
    row_idx: int
    text: str
    token_count: int


class KVProjectionCollector:
    def __init__(self) -> None:
        self.target_layers: set[int] = set()
        self.k_states: dict[int, list[torch.Tensor]] = {}
        self.v_states: dict[int, list[torch.Tensor]] = {}

    def configure(self, target_layers: Iterable[int]) -> None:
        self.target_layers = {int(layer) for layer in target_layers}
        self.clear()

    def clear(self) -> None:
        self.k_states = {}
        self.v_states = {}

    def hook(self, layer_idx: int, state_name: str):
        def _hook(_module, _input, output):
            if layer_idx not in self.target_layers:
                return
            if isinstance(output, tuple):
                output = output[0]
            if state_name == "k":
                self.k_states.setdefault(layer_idx, []).append(output.detach().cpu())
            elif state_name == "v":
                self.v_states.setdefault(layer_idx, []).append(output.detach().cpu())
            else:
                raise ValueError(f"Unsupported state name: {state_name}")

        return _hook


collector = KVProjectionCollector()


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one string.")
    return values


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}"


def alpha_slug(alpha: float) -> str:
    return alpha_label(alpha).replace("-", "m").replace(".", "p")


def prompt_to_text(value, mode: str) -> str:
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    items = [str(item) for item in items if item is not None]
    if not items:
        return ""
    if mode == "first":
        return items[0]
    if mode == "join":
        return "\n\n".join(items)
    raise ValueError(f"Unsupported prompt mode: {mode}")


def load_scbench_samples(
    data_dir: Path,
    tasks: list[str],
    tokenizer,
    *,
    max_rows_per_task: int,
    prompt_mode: str,
    min_tokens: int,
    seed: int,
) -> list[PromptSample]:
    rng = random.Random(seed)
    samples: list[PromptSample] = []
    for task in tasks:
        path = data_dir / f"{task}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing SCBench parquet file: {path}")
        df = pd.read_parquet(path)
        if "prompts" not in df.columns:
            raise ValueError(f"{path} does not contain a 'prompts' column.")

        row_indices = list(range(len(df)))
        rng.shuffle(row_indices)
        row_indices = row_indices[: min(max_rows_per_task, len(row_indices))]
        for row_idx in tqdm(row_indices, desc=f"Tokenizing {task}", leave=False):
            text = prompt_to_text(df.iloc[row_idx]["prompts"], prompt_mode)
            if not text.strip():
                continue
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids
            token_count = len(token_ids)
            if token_count < min_tokens:
                continue
            samples.append(
                PromptSample(
                    task=task,
                    row_idx=int(row_idx),
                    text=text,
                    token_count=int(token_count),
                )
            )
    if not samples:
        raise RuntimeError(
            f"No SCBench samples with at least {min_tokens} tokens were found in {data_dir}."
        )
    samples.sort(key=lambda item: (item.task, item.row_idx))
    return samples


def choose_samples_by_context(
    samples: list[PromptSample],
    context_lengths: list[int],
    samples_per_length: int,
    seed: int,
) -> dict[int, list[PromptSample]]:
    rng = random.Random(seed)
    by_len: dict[int, list[PromptSample]] = {}
    for ctx_len in context_lengths:
        candidates = [sample for sample in samples if sample.token_count >= ctx_len]
        rng.shuffle(candidates)
        chosen = candidates[:samples_per_length]
        if not chosen:
            raise RuntimeError(f"No samples have at least {ctx_len} tokens.")
        by_len[ctx_len] = sorted(chosen, key=lambda item: (item.task, item.row_idx))
    return by_len


def resolve_layers(target_layers: str, num_layers: int) -> list[int]:
    if target_layers == "auto":
        layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    else:
        layers = parse_csv_ints(target_layers)
    layers = sorted({layer for layer in layers if 0 <= layer < num_layers})
    if not layers:
        raise ValueError(f"No valid target layers for a model with {num_layers} layers.")
    return layers


def patch_model(model, target_layers: list[int]):
    collector.configure(target_layers)
    handles = []
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        handles.append(layer.self_attn.k_proj.register_forward_hook(collector.hook(layer_idx, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(collector.hook(layer_idx, "v")))
    return handles


def build_token_states(layer_idx: int, state_kind: str, device: str) -> torch.Tensor:
    if state_kind not in {"k", "v", "kv"}:
        raise ValueError(f"Unsupported state_kind: {state_kind}")
    if state_kind in {"k", "kv"} and layer_idx not in collector.k_states:
        raise KeyError("hook did not collect K states")
    if state_kind in {"v", "kv"} and layer_idx not in collector.v_states:
        raise KeyError("hook did not collect V states")

    states = []
    if state_kind in {"k", "kv"}:
        states.append(torch.cat(collector.k_states[layer_idx], dim=1).squeeze(0))
    if state_kind in {"v", "kv"}:
        states.append(torch.cat(collector.v_states[layer_idx], dim=1).squeeze(0))
    if len(states) == 1:
        return states[0].float().to(device)
    return torch.cat(states, dim=-1).float().to(device)


def build_center_indices(
    seq_len: int,
    *,
    sink_keep_tokens: int,
    base_stride: int,
    schedule_kind: str,
    stride_alpha: float = 0.0,
    stride_increment: int = 0,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    if sink_keep_tokens < 0:
        raise ValueError(f"sink_keep_tokens must be non-negative, got {sink_keep_tokens}.")
    if base_stride <= 0:
        raise ValueError(f"base_stride must be positive, got {base_stride}.")
    if schedule_kind not in {"position_linear", "selection_linear"}:
        raise ValueError(f"Unsupported schedule_kind: {schedule_kind}")
    if stride_increment < 0:
        raise ValueError(f"stride_increment must be non-negative, got {stride_increment}.")

    sink = min(int(sink_keep_tokens), int(seq_len))
    centers = list(range(sink))
    pos = sink
    selection_idx = 0
    while pos < seq_len:
        centers.append(pos)
        if schedule_kind == "selection_linear":
            step = base_stride + selection_idx * int(stride_increment)
        else:
            if stride_alpha <= 0.0:
                step = base_stride
            else:
                step = base_stride + int(float(stride_alpha) * float(max(0, pos - sink)))
        pos += max(1, int(step))
        selection_idx += 1
    return torch.tensor(centers, dtype=torch.long)


def build_schedule_specs(
    alpha_values: list[float],
    sqrt_stride_increments: list[int],
) -> list[dict]:
    specs: list[dict] = []
    for alpha in alpha_values:
        alpha = float(alpha)
        if alpha == 0.0:
            schedule_id = "fixed"
            label = "Fixed Stride"
        else:
            schedule_id = f"position_alpha_{alpha_slug(alpha)}"
            label = f"Dynamic α={alpha:g}"
        specs.append(
            {
                "schedule_id": schedule_id,
                "schedule_kind": "position_linear",
                "schedule_label": label,
                "stride_alpha": alpha,
                "stride_increment": 0,
            }
        )

    for inc in sqrt_stride_increments:
        inc = int(inc)
        specs.append(
            {
                "schedule_id": f"selection_inc_{inc}",
                "schedule_kind": "selection_linear",
                "schedule_label": f"Sqrt +{inc}",
                "stride_alpha": 0.0,
                "stride_increment": inc,
            }
        )

    seen = set()
    deduped = []
    for spec in specs:
        key = spec["schedule_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def mean_best_center_similarity(
    k_norm: torch.Tensor,
    center_indices: torch.Tensor,
    *,
    sink_keep_tokens: int,
    query_block_size: int,
) -> tuple[float, float, float]:
    seq_len = int(k_norm.shape[0])
    sink = min(int(sink_keep_tokens), seq_len)
    if seq_len <= sink:
        raise ValueError(f"seq_len={seq_len} leaves no non-sink tokens for sink={sink}.")

    center_indices = center_indices.to(device=k_norm.device, dtype=torch.long)
    center_states = k_norm[center_indices]
    means = []
    medians = []
    p05s = []

    for q_start in range(sink, seq_len, query_block_size):
        q_end = min(seq_len, q_start + query_block_size)
        query = k_norm[q_start:q_end]
        scores = torch.matmul(query, center_states.t())
        row_pos = torch.arange(q_start, q_end, device=k_norm.device).view(-1, 1)
        valid = center_indices.view(1, -1) < row_pos
        if not bool(torch.all(valid.any(dim=1))):
            raise RuntimeError("At least one query token has no historical reference center.")
        scores = scores.masked_fill(~valid, -1.0)
        vals = scores.max(dim=1).values.detach().float().cpu().numpy()
        means.append(vals)

    vals_np = np.concatenate(means, axis=0)
    medians.append(float(np.median(vals_np)))
    p05s.append(float(np.percentile(vals_np, 5)))
    return float(np.mean(vals_np)), float(np.mean(medians)), float(np.mean(p05s))


def ref_token_count(
    seq_len: int,
    *,
    sink_keep_tokens: int,
    base_stride: int,
    schedule_kind: str,
    stride_alpha: float = 0.0,
    stride_increment: int = 0,
) -> int:
    return int(
        build_center_indices(
            seq_len,
            sink_keep_tokens=sink_keep_tokens,
            base_stride=base_stride,
            schedule_kind=schedule_kind,
            stride_alpha=stride_alpha,
            stride_increment=stride_increment,
        ).numel()
    )


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def aggregate_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for rec in records:
        if rec.get("status") != SUCCESS:
            continue
        key = (int(rec["context_length"]), str(rec["schedule_id"]))
        grouped.setdefault(key, []).append(rec)

    rows: list[dict] = []
    for (ctx_len, _schedule_id), items in sorted(grouped.items()):
        first = items[0]
        mean_values = [float(item["mean_similarity"]) for item in items]
        median_values = [float(item["median_similarity"]) for item in items]
        p05_values = [float(item["p05_similarity"]) for item in items]
        ref_counts = [int(item["reference_token_count"]) for item in items]
        ref_fracs = [float(item["reference_fraction"]) for item in items]
        rows.append(
            {
                "context_length": ctx_len,
                "schedule_id": str(first["schedule_id"]),
                "schedule_kind": str(first["schedule_kind"]),
                "schedule_label": str(first["schedule_label"]),
                "stride_alpha": float(first["stride_alpha"]),
                "stride_increment": int(first["stride_increment"]),
                "num_measurements": len(items),
                "mean_similarity": float(np.mean(mean_values)),
                "sem_similarity": sem(mean_values),
                "median_similarity": float(np.mean(median_values)),
                "p05_similarity": float(np.mean(p05_values)),
                "reference_token_count": float(np.mean(ref_counts)),
                "reference_fraction": float(np.mean(ref_fracs)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_similarity(
    aggregate_rows: list[dict],
    output_path: Path,
    *,
    schedule_specs: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    by_schedule: dict[str, list[dict]] = {}
    for row in aggregate_rows:
        by_schedule.setdefault(str(row["schedule_id"]), []).append(row)

    colors_by_id = {
        "fixed": COLOR_BLACK,
        "position_alpha_0p001": COLOR_PRIMARY,
        "position_alpha_0p02": COLOR_SECONDARY,
        "position_alpha_0p1": COLOR_TERTIARY,
        "selection_inc_1": COLOR_PRIMARY_LIGHT,
    }
    fallback_colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_PRIMARY_LIGHT]

    fig, ax = plt.subplots(figsize=(3.35, 2.51))
    for idx, spec in enumerate(schedule_specs):
        schedule_id = str(spec["schedule_id"])
        rows = sorted(by_schedule.get(schedule_id, []), key=lambda row: row["context_length"])
        if not rows:
            continue
        x = np.asarray([row["context_length"] for row in rows], dtype=np.float64)
        y = np.asarray([row["mean_similarity"] for row in rows], dtype=np.float64)
        yerr = np.asarray([row["sem_similarity"] for row in rows], dtype=np.float64)
        color = colors_by_id.get(schedule_id, fallback_colors[idx % len(fallback_colors)])
        label = str(spec["schedule_label"])
        if schedule_id == "fixed":
            linestyle = "--"
            marker = "o"
        elif spec["schedule_kind"] == "selection_linear":
            linestyle = "-."
            marker = "s"
        else:
            linestyle = "-"
            marker = "o"
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=1.55, markersize=3.0)
        if np.any(yerr > 0):
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["context_length"] for row in aggregate_rows}))
    ax.get_xaxis().set_major_formatter(lambda value, _pos: f"{int(value/1024)}K")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Mean Max Similarity")
    ax.grid(True, linestyle="--", alpha=0.55, color=COLOR_GRID)
    ax.legend(frameon=False, ncol=2, loc="lower left", handlelength=2.4, columnspacing=0.8)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_lengths = parse_csv_ints(args.context_lengths)
    alpha_values = parse_csv_floats(args.stride_alphas)
    sqrt_stride_increments = (
        parse_csv_ints(args.sqrt_stride_increments)
        if args.sqrt_stride_increments.strip()
        else []
    )
    tasks = parse_csv_strings(args.tasks)
    base_stride = max(1, int(round(1.0 / float(args.deltakv_center_ratio))))

    if 0.0 not in alpha_values:
        alpha_values = [0.0] + alpha_values
    schedule_specs = build_schedule_specs(alpha_values, sqrt_stride_increments)

    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    run_config = {
        "started_at": started_at,
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "tasks": tasks,
        "context_lengths": context_lengths,
        "samples_per_length": int(args.samples_per_length),
        "max_rows_per_task": int(args.max_rows_per_task),
        "prompt_mode": args.prompt_mode,
        "target_layers": args.target_layers,
        "state_kind": args.state_kind,
        "sink_keep_tokens": int(args.sink_keep_tokens),
        "deltakv_center_ratio": float(args.deltakv_center_ratio),
        "base_stride": int(base_stride),
        "stride_alphas": alpha_values,
        "sqrt_stride_increments": sqrt_stride_increments,
        "schedule_specs": schedule_specs,
        "query_block_size": int(args.query_block_size),
        "seed": int(args.seed),
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = load_scbench_samples(
        Path(args.data_dir),
        tasks,
        tokenizer,
        max_rows_per_task=int(args.max_rows_per_task),
        prompt_mode=args.prompt_mode,
        min_tokens=min(context_lengths),
        seed=int(args.seed),
    )
    selected = choose_samples_by_context(
        samples,
        context_lengths,
        int(args.samples_per_length),
        int(args.seed),
    )

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": args.device},
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    layers = resolve_layers(args.target_layers, int(model.config.num_hidden_layers))
    handles = patch_model(model, layers)
    run_config["resolved_target_layers"] = layers

    records_path = output_dir / "dynamic_stride_similarity_records.jsonl"
    records: list[dict] = []
    with records_path.open("w", encoding="utf-8") as records_file:
        for ctx_len in context_lengths:
            for sample in selected[ctx_len]:
                input_ids = tokenizer(
                    sample.text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=ctx_len,
                ).input_ids.to(args.device)

                if input_ids.shape[1] < ctx_len:
                    rec = {
                        "status": SKIPPED,
                        "reason": "tokenized prompt shorter than requested context length",
                        "task": sample.task,
                        "row_idx": sample.row_idx,
                        "context_length": ctx_len,
                        "token_count": int(input_ids.shape[1]),
                    }
                    records_file.write(json.dumps(rec) + "\n")
                    records_file.flush()
                    records.append(rec)
                    continue

                collector.clear()
                try:
                    with torch.inference_mode():
                        model(input_ids=input_ids, use_cache=False)
                except Exception as exc:  # noqa: BLE001 - recorded as explicit sample failure.
                    rec = {
                        "status": MODEL_FAILED,
                        "reason": repr(exc),
                        "task": sample.task,
                        "row_idx": sample.row_idx,
                        "context_length": ctx_len,
                    }
                    records_file.write(json.dumps(rec) + "\n")
                    records_file.flush()
                    records.append(rec)
                    torch.cuda.empty_cache()
                    continue

                for layer_idx in layers:
                    if (
                        args.state_kind in {"k", "kv"}
                        and layer_idx not in collector.k_states
                    ) or (
                        args.state_kind in {"v", "kv"}
                        and layer_idx not in collector.v_states
                    ):
                        rec = {
                            "status": INVALID_INPUT,
                            "reason": f"hook did not collect {args.state_kind.upper()} states",
                            "task": sample.task,
                            "row_idx": sample.row_idx,
                            "context_length": ctx_len,
                            "layer_idx": layer_idx,
                            "state_kind": args.state_kind,
                        }
                        records_file.write(json.dumps(rec) + "\n")
                        records_file.flush()
                        records.append(rec)
                        continue

                    try:
                        token_states = build_token_states(layer_idx, args.state_kind, args.device)
                        token_norm = F.normalize(token_states, p=2, dim=-1)
                        seq_len = int(token_norm.shape[0])
                        for spec in schedule_specs:
                            centers = build_center_indices(
                                seq_len,
                                sink_keep_tokens=int(args.sink_keep_tokens),
                                base_stride=base_stride,
                                schedule_kind=str(spec["schedule_kind"]),
                                stride_alpha=float(spec["stride_alpha"]),
                                stride_increment=int(spec["stride_increment"]),
                            )
                            mean_sim, median_sim, p05_sim = mean_best_center_similarity(
                                token_norm,
                                centers,
                                sink_keep_tokens=int(args.sink_keep_tokens),
                                query_block_size=int(args.query_block_size),
                            )
                            rec = {
                                "status": SUCCESS,
                                "task": sample.task,
                                "row_idx": int(sample.row_idx),
                                "source_token_count": int(sample.token_count),
                                "context_length": int(ctx_len),
                                "seq_len": int(seq_len),
                                "layer_idx": int(layer_idx),
                                "state_kind": args.state_kind,
                                "schedule_id": str(spec["schedule_id"]),
                                "schedule_kind": str(spec["schedule_kind"]),
                                "schedule_label": str(spec["schedule_label"]),
                                "sink_keep_tokens": int(args.sink_keep_tokens),
                                "base_stride": int(base_stride),
                                "stride_alpha": float(spec["stride_alpha"]),
                                "stride_increment": int(spec["stride_increment"]),
                                "reference_token_count": int(centers.numel()),
                                "reference_fraction": float(centers.numel() / seq_len),
                                "mean_similarity": mean_sim,
                                "median_similarity": median_sim,
                                "p05_similarity": p05_sim,
                            }
                            records_file.write(json.dumps(rec) + "\n")
                            records_file.flush()
                            records.append(rec)
                        del token_states, token_norm
                    except Exception as exc:  # noqa: BLE001 - recorded as explicit metric failure.
                        rec = {
                            "status": METRIC_FAILED,
                            "reason": repr(exc),
                            "task": sample.task,
                            "row_idx": sample.row_idx,
                            "context_length": ctx_len,
                            "layer_idx": layer_idx,
                            "state_kind": args.state_kind,
                        }
                        records_file.write(json.dumps(rec) + "\n")
                        records_file.flush()
                        records.append(rec)
                    finally:
                        torch.cuda.empty_cache()

                collector.clear()

    for handle in handles:
        handle.remove()

    aggregate_rows = aggregate_records(records)
    aggregate_path = output_dir / "dynamic_stride_similarity_aggregate.csv"
    write_csv(aggregate_path, aggregate_rows)

    summary = {
        "config": run_config,
        "selected_samples": {
            str(ctx_len): [
                {
                    "task": sample.task,
                    "row_idx": sample.row_idx,
                    "token_count": sample.token_count,
                }
                for sample in chosen
            ]
            for ctx_len, chosen in selected.items()
        },
        "status_counts": {
            status: sum(1 for rec in records if rec.get("status") == status)
            for status in sorted({str(rec.get("status")) for rec in records})
        },
        "aggregate": aggregate_rows,
    }
    summary_path = output_dir / "dynamic_stride_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_path = output_dir / "dynamic_stride_similarity_scbench.pdf"
    plot_similarity(aggregate_rows, plot_path, schedule_specs=schedule_specs)

    print(f"Saved records: {records_path}")
    print(f"Saved aggregate: {aggregate_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=default_dataset_path("SCBench-preprocessed"))
    parser.add_argument(
        "--tasks",
        default="scbench_kv,scbench_repoqa,scbench_summary_with_needles,scbench_many_shot",
    )
    parser.add_argument("--context-lengths", default="4096,8192,16384,32768")
    parser.add_argument("--samples-per-length", type=int, default=3)
    parser.add_argument("--max-rows-per-task", type=int, default=24)
    parser.add_argument("--prompt-mode", choices=("first", "join"), default="first")
    parser.add_argument("--target-layers", default="auto")
    parser.add_argument(
        "--state-kind",
        choices=("k", "v", "kv"),
        default="kv",
        help="Token representation used for similarity. 'kv' concatenates key and value projections.",
    )
    parser.add_argument("--sink-keep-tokens", type=int, default=8)
    parser.add_argument("--deltakv-center-ratio", type=float, default=0.1)
    parser.add_argument("--stride-alphas", default="0.0,0.001,0.02,0.1")
    parser.add_argument(
        "--sqrt-stride-increments",
        default="",
        help=(
            "Comma-separated increments for selection-linear sqrt-level schedules. "
            "For increment=1, the selected-reference strides are s0, s0+1, s0+2, ..."
        ),
    )
    parser.add_argument("--query-block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.samples_per_length <= 0:
        parser.error("--samples-per-length must be positive.")
    if args.max_rows_per_task <= 0:
        parser.error("--max-rows-per-task must be positive.")
    if args.query_block_size <= 0:
        parser.error("--query-block-size must be positive.")
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - top-level failure should be visible in logs.
        print(f"ERROR: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
