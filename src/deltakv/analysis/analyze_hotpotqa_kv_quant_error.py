from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from benchmark.long_bench.pred import build_chat
from deltakv.configs.default_paths import longbench_root, model_path, output_path
from deltakv.get_chat_api import get_generate_api
from deltakv.modeling.cache_pipeline import ClusterCachePipeline


DEFAULT_HYPER_PARAMS: dict[str, Any] = {
    "sparse_method": "deltakv-less-memory",
    "use_cluster": True,
    "use_compression": False,
    "chunk_prefill_accel_omnikv": False,
    "full_attention_layers": "0,1,2,4,7,14",
    "sink_keep_tokens": 8,
    "recent_keep_tokens": 128,
    "decode_keep_tokens": 2048,
    "prefill_keep_tokens": 4096,
    "deltakv_center_ratio": 0.03,
    "stride_alpha": 0.02,
    "deltakv_neighbor_count": 4,
    "deltakv_latent_quant_bits": 2,
    "full_layer_cluster_ratio": 0.08,
    "full_layer_stride_alpha": 0.0,
    "full_layer_kv_quant_bits": 4,
    "cluster_metric": "l2",
    "pool_kernel_size": 1,
}

KIVI_DEFAULTS: dict[str, Any] = {
    "k_bits": 2,
    "v_bits": 2,
    "group_size": 32,
    "residual_length": 32,
}


@dataclass
class TensorStats:
    sample_limit: int
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float | None = None
    max: float | None = None
    samples: list[float] = field(default_factory=list)

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        data = tensor.detach().float()
        self.count += int(data.numel())
        self.sum += float(data.sum().item())
        self.sum_sq += float((data * data).sum().item())
        mn = float(data.min().item())
        mx = float(data.max().item())
        self.min = mn if self.min is None else min(self.min, mn)
        self.max = mx if self.max is None else max(self.max, mx)

        if self.sample_limit <= 0:
            return
        remaining = self.sample_limit - len(self.samples)
        if remaining <= 0:
            return
        flat = data.flatten()
        take = min(remaining, int(flat.numel()))
        if take == flat.numel():
            sample = flat
        else:
            idx = torch.randperm(flat.numel(), device=flat.device)[:take]
            sample = flat.index_select(0, idx)
        self.samples.extend(float(x) for x in sample.cpu().tolist())

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        mean = self.sum / self.count
        second = self.sum_sq / self.count
        variance = max(0.0, second - mean * mean)
        out: dict[str, Any] = {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "rms": second**0.5,
            "mse_from_zero": second,
            "min": self.min,
            "max": self.max,
            "abs_max": max(abs(self.min or 0.0), abs(self.max or 0.0)),
        }
        if self.samples:
            arr = np.asarray(self.samples, dtype=np.float64)
            out.update(
                {
                    "sample_count": int(arr.size),
                    "p01": float(np.percentile(arr, 1)),
                    "p05": float(np.percentile(arr, 5)),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                }
            )
        return out


class QuantErrorRecorder:
    def __init__(self, sample_limit_per_stat: int) -> None:
        self.sample_limit_per_stat = int(sample_limit_per_stat)
        self.stats: dict[str, TensorStats] = {}
        self.events: list[dict[str, Any]] = []

    def _stat(self, key: str) -> TensorStats:
        if key not in self.stats:
            self.stats[key] = TensorStats(sample_limit=self.sample_limit_per_stat)
        return self.stats[key]

    def update(self, prefix: str, name: str, tensor: torch.Tensor, *, include_all: bool = True) -> None:
        self._stat(f"{prefix}/{name}").update(tensor)
        if include_all:
            self._stat(f"all/{name}").update(tensor)

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": {key: stat.to_dict() for key, stat in sorted(self.stats.items())},
            "events": self.events,
        }


def load_hyper_params(path_or_json: str | None) -> dict[str, Any]:
    params = dict(DEFAULT_HYPER_PARAMS)
    if not path_or_json:
        return params
    if os.path.exists(path_or_json):
        with open(path_or_json, "r", encoding="utf-8") as f:
            override = json.load(f)
    else:
        override = json.loads(path_or_json)
    params.update(override)
    return params


def load_hotpotqa_prompts(
    *,
    data_root: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    num_samples: int,
    no_chat_template: bool,
    thinking_mode: str,
) -> list[str]:
    data_path = Path(data_root) / "data" / "hotpotqa.jsonl"
    prompt_path = Path("benchmark/long_bench/config/dataset2prompt.json")
    if not data_path.is_file():
        raise FileNotFoundError(f"HotPotQA data file not found: {data_path}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"LongBench prompt config not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_format = json.load(f)["hotpotqa"]

    prompts: list[str] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            obj = json.loads(line)
            prompt = prompt_format.format(**obj)
            tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized) > max_length:
                half = int(max_length / 2)
                prompt = (
                    tokenizer.decode(tokenized[:half], skip_special_tokens=True)
                    + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
                )
            prompts.append(
                build_chat(
                    tokenizer,
                    prompt,
                    "hotpotqa",
                    no_chat_template=no_chat_template,
                    thinking_mode=thinking_mode,
                )
            )
    if not prompts:
        raise ValueError(f"No HotPotQA samples loaded from {data_path}")
    return prompts


def _clone_center_state(cache: ClusterCachePipeline) -> tuple[dict[str, int], dict[str, tuple[tuple[int, int], torch.Tensor]]]:
    next_pos = dict(cache._cluster_next_center_abs_pos_by_scope)
    plan_cache = dict(cache._cluster_center_plan_cache_by_scope)
    return next_pos, plan_cache


def _restore_center_state(
    cache: ClusterCachePipeline,
    state: tuple[dict[str, int], dict[str, tuple[tuple[int, int], torch.Tensor]]],
) -> None:
    cache._cluster_next_center_abs_pos_by_scope = dict(state[0])
    cache._cluster_center_plan_cache_by_scope = dict(state[1])


def _load_kivi_quant_module():
    repo_root = Path(__file__).resolve().parents[3]
    kivi_dir = repo_root / "baselines" / "kivi"
    if not kivi_dir.is_dir():
        raise FileNotFoundError(f"KIVI baseline directory not found: {kivi_dir}")
    kivi_path = str(kivi_dir)
    if kivi_path not in sys.path:
        sys.path.insert(0, kivi_path)
    return importlib.import_module("quant.new_pack")


def _flat_cache_to_heads(states: torch.Tensor, *, num_kv_heads: int) -> torch.Tensor:
    if states.ndim == 4:
        return states
    if states.ndim != 3:
        raise ValueError(f"Expected flat 3D or headed 4D KV states, got shape {tuple(states.shape)}.")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}.")
    bs, seq_len, flat_dim = states.shape
    if flat_dim % num_kv_heads != 0:
        raise ValueError(
            f"Flat KV dim must be divisible by num_kv_heads; got flat_dim={flat_dim}, "
            f"num_kv_heads={num_kv_heads}."
        )
    head_dim = flat_dim // num_kv_heads
    return states.view(bs, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()


def _kivi_quant_dequant(
    states: torch.Tensor,
    *,
    axis: str,
    bits: int,
    group_size: int,
    residual_length: int,
    kivi_quant_module,
) -> tuple[torch.Tensor, int, int]:
    if states.ndim != 4:
        raise ValueError(f"KIVI analysis expects 4D KV states, got shape {tuple(states.shape)}.")
    if bits not in (2, 4, 8):
        raise ValueError(f"KIVI quantization supports bits=2,4,8, got {bits}.")
    if group_size <= 0:
        raise ValueError(f"KIVI group_size must be positive, got {group_size}.")
    if residual_length <= 0:
        raise ValueError(f"KIVI residual_length must be positive, got {residual_length}.")

    if axis == "key":
        seq_len = states.shape[-2]
        tail_len = 0 if seq_len % residual_length == 0 else seq_len % residual_length
        if seq_len < residual_length:
            return states, 0, seq_len
        quant_states = states if tail_len == 0 else states[:, :, :-tail_len, :]
        full_states = None if tail_len == 0 else states[:, :, -tail_len:, :]
        quant_input = quant_states.transpose(2, 3).contiguous()
        if quant_input.shape[-1] % group_size != 0:
            raise ValueError(
                f"KIVI key quantized sequence length must be divisible by group_size; "
                f"got {quant_input.shape[-1]} and group_size={group_size}."
            )
        packed, scale, mn = kivi_quant_module.triton_quantize_and_pack_along_last_dim(quant_input, group_size, bits)
        dequant = kivi_quant_module.unpack_and_dequant_vcache(
            packed,
            scale.unsqueeze(-1),
            mn.unsqueeze(-1),
            group_size,
            bits,
        ).transpose(2, 3)
        if full_states is not None:
            dequant = torch.cat([dequant, full_states], dim=2)
        return dequant, int(quant_states.shape[-2]), int(tail_len)

    if axis == "value":
        seq_len = states.shape[-2]
        if seq_len <= residual_length:
            return states, 0, seq_len
        quant_states = states[:, :, :-residual_length, :].contiguous()
        full_states = states[:, :, -residual_length:, :]
        if quant_states.shape[-1] % group_size != 0:
            raise ValueError(
                f"KIVI value head dimension must be divisible by group_size; "
                f"got {quant_states.shape[-1]} and group_size={group_size}."
            )
        packed, scale, mn = kivi_quant_module.triton_quantize_and_pack_along_last_dim(quant_states, group_size, bits)
        dequant = kivi_quant_module.unpack_and_dequant_vcache(
            packed,
            scale.unsqueeze(-1),
            mn.unsqueeze(-1),
            group_size,
            bits,
        )
        dequant = torch.cat([dequant, full_states], dim=2)
        return dequant, int(quant_states.shape[-2]), int(residual_length)

    raise ValueError(f"Unknown KIVI axis: {axis}")


def _drop_tail_tokens_for_stats(tensor: torch.Tensor, *, tail_tokens: int, name: str, sequence_dim: int) -> torch.Tensor:
    if tail_tokens < 0:
        raise ValueError(f"tail_tokens must be non-negative, got {tail_tokens}.")
    if tail_tokens == 0:
        return tensor
    seq_len = int(tensor.shape[sequence_dim])
    if seq_len <= tail_tokens:
        raise ValueError(
            f"Cannot exclude {tail_tokens} tail tokens from {name}; "
            f"sequence length is only {seq_len}."
        )
    keep = seq_len - tail_tokens
    index = [slice(None)] * tensor.ndim
    index[sequence_dim] = slice(0, keep)
    return tensor[tuple(index)]


@contextmanager
def record_cache_quant_error(
    recorder: QuantErrorRecorder,
    *,
    analysis_mode: str,
    deltakv_quant_group_size_override: int | None = None,
    kivi_params: dict[str, Any] | None = None,
    exclude_tail_tokens_from_stats: int = 0,
):
    original_store_history = ClusterCachePipeline._store_history
    original_apply_full_layer_kivi_roundtrip = ClusterCachePipeline._apply_full_layer_kivi_roundtrip
    include_deltakv = analysis_mode in ("deltakv", "both")
    include_kivi = analysis_mode in ("kivi", "both")
    kivi_quant_module = _load_kivi_quant_module() if include_kivi else None
    kivi_params = dict(KIVI_DEFAULTS if kivi_params is None else kivi_params)

    def wrapped_apply_full_layer_kivi_roundtrip(self, layer_idx):
        if not include_kivi or not self._full_layer_kivi_enabled() or not self._is_full_layer(layer_idx):
            return original_apply_full_layer_kivi_roundtrip(self, layer_idx)

        group_size = int(self.full_layer_kivi_group_size)
        residual_length = int(self.full_layer_kivi_residual_length)
        buffer_len = int(self.buffer_key_cache[layer_idx].shape[1])
        quant_end = buffer_len - residual_length
        if quant_end <= 0:
            return original_apply_full_layer_kivi_roundtrip(self, layer_idx)
        quant_end = (quant_end // group_size) * group_size
        start = int(self._full_layer_kivi_quantized_lens.get(layer_idx, 0))
        start = (start // group_size) * group_size
        if quant_end <= start:
            return original_apply_full_layer_kivi_roundtrip(self, layer_idx)

        before_key = self.buffer_key_cache[layer_idx][:, start:quant_end].detach().clone()
        before_value = self.buffer_value_cache[layer_idx][:, start:quant_end].detach().clone()
        result = original_apply_full_layer_kivi_roundtrip(self, layer_idx)
        after_key = self.buffer_key_cache[layer_idx][:, start:quant_end]
        after_value = self.buffer_value_cache[layer_idx][:, start:quant_end]

        prefix = f"kivi/layer_{layer_idx:02d}"
        recorder.add_event(
            {
                "analysis": "full_layer_kivi",
                "layer_idx": int(layer_idx),
                "layer_kind": "full",
                "tokens": int(buffer_len),
                "quantized_tokens": int(quant_end - start),
                "quant_start": int(start),
                "quant_end": int(quant_end),
                "num_kv_heads": int(getattr(self.config, "num_key_value_heads", 1) or 1),
                "flat_key_dim": int(before_key.shape[-1]),
                "flat_value_dim": int(before_value.shape[-1]),
                "k_bits": int(self.full_layer_kv_quant_bits),
                "v_bits": int(self.full_layer_kv_quant_bits),
                "group_size": group_size,
                "residual_length": residual_length,
                "full_tail_tokens": int(buffer_len - quant_end),
            }
        )
        kivi_tensors = {
            "key_reconstruction_error": after_key - before_key,
            "value_reconstruction_error": after_value - before_value,
            "kv_reconstruction_error": torch.cat([after_key - before_key, after_value - before_value], dim=-1),
            "raw_key": before_key,
            "raw_value": before_value,
        }
        for name, tensor in kivi_tensors.items():
            recorder.update(prefix, name, tensor)
            recorder.update("kivi/full", name, tensor, include_all=False)
            recorder.update("kivi", name, tensor, include_all=False)
        return result

    def wrapped_store_history(self, layer_idx, key, value, pos, compressor_down):
        if key.numel() > 0:
            layer_kind = "full" if self._is_full_layer(layer_idx) else "sparse"
            if include_deltakv:
                kv = torch.cat([key, value], dim=-1)
                existing = self.bases_cache.get(layer_idx)
                if existing is None:
                    existing = self._sink_kv(layer_idx)

                center_state = _clone_center_state(self)
                try:
                    refs, _, _ = self._cluster_refs(
                        kv,
                        existing,
                        abs_start_pos=int(pos[0, 0].item()),
                        cluster_ratio=self._layer_cluster_ratio(layer_idx),
                        stride_alpha=self._layer_stride_alpha(layer_idx),
                        cache_scope=(
                            f"full:{layer_idx}"
                            if self._layer_uses_full_layer_quant(layer_idx)
                            else f"sparse:{layer_idx}"
                        ),
                    )
                finally:
                    _restore_center_state(self, center_state)

                if not self._layer_origin_codec(layer_idx):
                    raise NotImplementedError(
                        "This analysis script is intended for the current no-compressor "
                        "origin-residual DeltaKV config. Set use_compression=false."
                    )

                residual = kv - refs
                quant_bits = self._layer_quant_bits(layer_idx)
                dequant_residual = residual
                group_size = residual.shape[-1]
                if quant_bits in (2, 4):
                    if deltakv_quant_group_size_override is None:
                        group_size = self._layer_quant_group_size(
                            layer_idx,
                            k_dim=key.shape[-1],
                            payload_dim=residual.shape[-1],
                        )
                    else:
                        group_size = int(deltakv_quant_group_size_override)
                        if group_size <= 0 or residual.shape[-1] % group_size != 0:
                            raise ValueError(
                                "DeltaKV analysis group-size override must be positive and divide "
                                f"residual dim; got override={group_size}, residual_dim={residual.shape[-1]}."
                            )
                    packed, scale, mn = self._quantize(residual, quant_bits, group_size=group_size)
                    dequant_residual = self._dequantize(
                        packed,
                        scale,
                        mn,
                        residual.shape[-1],
                        quant_bits,
                        group_size=group_size,
                    )
                reconstructed = dequant_residual + refs
                key_dim = key.shape[-1]
                key_ref, value_ref = refs[..., :key_dim], refs[..., key_dim:]
                key_residual, value_residual = residual[..., :key_dim], residual[..., key_dim:]
                dequant_key_residual = dequant_residual[..., :key_dim]
                dequant_value_residual = dequant_residual[..., key_dim:]
                reconstructed_key = reconstructed[..., :key_dim]
                reconstructed_value = reconstructed[..., key_dim:]

                prefix = f"{layer_kind}/layer_{layer_idx:02d}"
                recorder.add_event(
                    {
                        "analysis": "deltakv",
                        "layer_idx": int(layer_idx),
                        "layer_kind": layer_kind,
                        "tokens": int(kv.shape[1]),
                        "kv_dim": int(kv.shape[-1]),
                        "residual_dim": int(residual.shape[-1]),
                        "quant_bits": int(quant_bits),
                        "quant_group_size": int(group_size),
                        "quant_group_size_override": deltakv_quant_group_size_override,
                        "stats_excluded_tail_tokens": int(exclude_tail_tokens_from_stats),
                        "stats_tokens": int(kv.shape[1] - exclude_tail_tokens_from_stats),
                        "cluster_ratio": float(self._layer_cluster_ratio(layer_idx)),
                        "stride_alpha": float(self._layer_stride_alpha(layer_idx)),
                    }
                )
                tensors = {
                    "raw_kv": kv,
                    "raw_key": key,
                    "raw_value": value,
                    "reference": refs,
                    "key_reference": key_ref,
                    "value_reference": value_ref,
                    "residual_before_quant": residual,
                    "key_residual_before_quant": key_residual,
                    "value_residual_before_quant": value_residual,
                    "residual_after_dequant": dequant_residual,
                    "key_residual_after_dequant": dequant_key_residual,
                    "value_residual_after_dequant": dequant_value_residual,
                    "residual_quant_error": dequant_residual - residual,
                    "key_residual_quant_error": dequant_key_residual - key_residual,
                    "value_residual_quant_error": dequant_value_residual - value_residual,
                    "kv_reconstruction_error": reconstructed - kv,
                    "key_reconstruction_error": reconstructed_key - key,
                    "value_reconstruction_error": reconstructed_value - value,
                    "reference_error_no_residual": refs - kv,
                    "key_reference_error_no_residual": key_ref - key,
                    "value_reference_error_no_residual": value_ref - value,
                }
                for name, tensor in tensors.items():
                    tensor = _drop_tail_tokens_for_stats(
                        tensor,
                        tail_tokens=exclude_tail_tokens_from_stats,
                        name=f"deltakv/{prefix}/{name}",
                        sequence_dim=1,
                    )
                    recorder.update(prefix, name, tensor)
                    recorder.update(layer_kind, name, tensor, include_all=False)

            if include_kivi:
                assert kivi_quant_module is not None
                k_bits = int(kivi_params["k_bits"])
                v_bits = int(kivi_params["v_bits"])
                group_size = int(kivi_params["group_size"])
                residual_length = int(kivi_params["residual_length"])
                num_kv_heads = int(getattr(self.config, "num_key_value_heads", 1) or 1)
                key_heads = _flat_cache_to_heads(key, num_kv_heads=num_kv_heads)
                value_heads = _flat_cache_to_heads(value, num_kv_heads=num_kv_heads)
                dequant_key, quant_key_tokens, full_key_tokens = _kivi_quant_dequant(
                    key_heads,
                    axis="key",
                    bits=k_bits,
                    group_size=group_size,
                    residual_length=residual_length,
                    kivi_quant_module=kivi_quant_module,
                )
                dequant_value, quant_value_tokens, full_value_tokens = _kivi_quant_dequant(
                    value_heads,
                    axis="value",
                    bits=v_bits,
                    group_size=group_size,
                    residual_length=residual_length,
                    kivi_quant_module=kivi_quant_module,
                )
                recorder.add_event(
                    {
                        "analysis": "kivi",
                        "layer_idx": int(layer_idx),
                        "layer_kind": layer_kind,
                        "tokens": int(key_heads.shape[-2]),
                        "num_kv_heads": int(key_heads.shape[1]),
                        "key_head_dim": int(key_heads.shape[-1]),
                        "value_head_dim": int(value_heads.shape[-1]),
                        "k_bits": k_bits,
                        "v_bits": v_bits,
                        "group_size": group_size,
                        "residual_length": residual_length,
                        "quant_key_tokens": quant_key_tokens,
                        "full_key_tokens": full_key_tokens,
                        "quant_value_tokens": quant_value_tokens,
                        "full_value_tokens": full_value_tokens,
                        "stats_excluded_tail_tokens": int(exclude_tail_tokens_from_stats),
                        "stats_tokens": int(key_heads.shape[-2] - exclude_tail_tokens_from_stats),
                    }
                )
                kivi_tensors = {
                    "key_reconstruction_error": dequant_key - key_heads,
                    "value_reconstruction_error": dequant_value - value_heads,
                    "kv_reconstruction_error": torch.cat([dequant_key - key_heads, dequant_value - value_heads], dim=-1),
                    "raw_key": key_heads,
                    "raw_value": value_heads,
                }
                for name, tensor in kivi_tensors.items():
                    tensor = _drop_tail_tokens_for_stats(
                        tensor,
                        tail_tokens=exclude_tail_tokens_from_stats,
                        name=f"kivi/layer_{layer_idx:02d}/{name}",
                        sequence_dim=2,
                    )
                    recorder.update(f"kivi/layer_{layer_idx:02d}", name, tensor)
                    recorder.update(f"kivi/{layer_kind}", name, tensor, include_all=False)
                    recorder.update("kivi", name, tensor, include_all=False)

        return original_store_history(self, layer_idx, key, value, pos, compressor_down)

    ClusterCachePipeline._store_history = wrapped_store_history
    ClusterCachePipeline._apply_full_layer_kivi_roundtrip = wrapped_apply_full_layer_kivi_roundtrip
    try:
        yield
    finally:
        ClusterCachePipeline._store_history = original_store_history
        ClusterCachePipeline._apply_full_layer_kivi_roundtrip = original_apply_full_layer_kivi_roundtrip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze HotPotQA DeltaKV cache residual quantization error using the "
            "current HF cache implementation and hyperparameters."
        )
    )
    parser.add_argument("--model_path", default=model_path("Qwen2.5-7B-Instruct-1M"))
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--data_root", default=longbench_root())
    parser.add_argument("--hyper_param", default=None, help="Optional JSON file or JSON string overriding current defaults.")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--analysis_mode", choices=["deltakv", "kivi", "both"], default="deltakv")
    parser.add_argument("--deltakv_quant_group_size_override", type=int, default=None)
    parser.add_argument("--exclude_tail_tokens_from_stats", type=int, default=0)
    parser.add_argument("--kivi_bits", type=int, default=2)
    parser.add_argument("--kivi_group_size", type=int, default=32)
    parser.add_argument("--kivi_residual_length", type=int, default=32)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=121000)
    parser.add_argument("--sample_limit_per_stat", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--thinking_mode", choices=["off", "on_strip"], default="off")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.exclude_tail_tokens_from_stats < 0:
        raise ValueError("--exclude_tail_tokens_from_stats must be non-negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hyper_params = load_hyper_params(args.hyper_param)
    infer_config = dict(hyper_params)
    infer_config["max_model_len"] = args.max_model_len

    generate, model = get_generate_api(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        infer_config=infer_config,
        sparse_method=str(hyper_params.get("sparse_method", "deltakv-less-memory")),
        backend="hf",
        cuda_device=args.cuda_device,
        return_model=True,
    )

    base_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = load_hotpotqa_prompts(
        data_root=args.data_root,
        tokenizer=tokenizer,
        max_length=args.max_model_len,
        num_samples=args.num_samples,
        no_chat_template=args.no_chat_template,
        thinking_mode=args.thinking_mode,
    )

    recorder = QuantErrorRecorder(sample_limit_per_stat=args.sample_limit_per_stat)
    kivi_params = {
        "k_bits": args.kivi_bits,
        "v_bits": args.kivi_bits,
        "group_size": args.kivi_group_size,
        "residual_length": args.kivi_residual_length,
    }
    with record_cache_quant_error(
        recorder,
        analysis_mode=args.analysis_mode,
        deltakv_quant_group_size_override=args.deltakv_quant_group_size_override,
        kivi_params=kivi_params,
        exclude_tail_tokens_from_stats=args.exclude_tail_tokens_from_stats,
    ):
        for prompt in tqdm(prompts, desc="Analyzing HotPotQA KV quantization"):
            generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=0,
                top_p=1,
                top_k=20,
                eos_token_id=[tokenizer.eos_token_id],
            )
            torch.cuda.empty_cache()

    output = {
        "analysis": "hotpotqa_kv_quant_error",
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "data_root": args.data_root,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "analysis_mode": args.analysis_mode,
        "deltakv_quant_group_size_override": args.deltakv_quant_group_size_override,
        "exclude_tail_tokens_from_stats": args.exclude_tail_tokens_from_stats,
        "kivi_params": kivi_params,
        "model_type": getattr(base_config, "model_type", None),
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers")),
        "num_key_value_heads": int(getattr(model.config, "num_key_value_heads")),
        "hidden_size": int(getattr(model.config, "hidden_size")),
        "hyper_params": hyper_params,
        "recorder": recorder.to_dict(),
    }

    if args.output_path is None:
        output_dir = Path(output_path("deltakv", "analysis"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hotpotqa_kv_quant_error.json"
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Analysis] Wrote KV quantization report to {output_path}")


if __name__ == "__main__":
    main()
