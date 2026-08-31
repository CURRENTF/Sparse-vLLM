from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch

from sparsevllm.configs.cuda_graph import build_decode_cuda_graph_startup_family_plan
from sparsevllm.engine.cache_manager.storage import CacheLayout
from sparsevllm.method_registry import (
    decode_sparse_long_text_threshold,
    normalize_sparse_method,
)


@dataclass(frozen=True)
class StartupMemoryProfile:
    total_bytes: int
    persistent_bytes: int
    prefill_transient_bytes: int
    decode_transient_bytes: int
    cuda_graph_bytes: int

    @property
    def runtime_transient_bytes(self) -> int:
        return max(
            int(self.prefill_transient_bytes),
            int(self.decode_transient_bytes),
        )


@dataclass(frozen=True)
class KVCapacityPlan:
    total_bytes: int
    target_bytes: int
    safety_headroom_bytes: int
    persistent_bytes: int
    runtime_transient_bytes: int
    cuda_graph_bytes: int
    local_kv_budget_bytes: int

    @classmethod
    def from_profile(
        cls,
        profile: StartupMemoryProfile,
        gpu_memory_utilization: float,
    ) -> "KVCapacityPlan":
        utilization = float(gpu_memory_utilization)
        if not 0 < utilization < 1:
            raise ValueError(
                "gpu_memory_utilization must be between 0 and 1, got "
                f"{utilization}."
            )
        target_bytes = int(profile.total_bytes * utilization)
        local_kv_budget_bytes = (
            target_bytes
            - int(profile.persistent_bytes)
            - int(profile.runtime_transient_bytes)
            - int(profile.cuda_graph_bytes)
        )
        if local_kv_budget_bytes <= 0:
            raise RuntimeError(
                "Startup profiling left no memory for KV cache: "
                f"target={target_bytes} persistent={profile.persistent_bytes} "
                f"runtime_transient={profile.runtime_transient_bytes} "
                f"cuda_graph={profile.cuda_graph_bytes}."
            )
        return cls(
            total_bytes=int(profile.total_bytes),
            target_bytes=target_bytes,
            safety_headroom_bytes=int(profile.total_bytes) - target_bytes,
            persistent_bytes=int(profile.persistent_bytes),
            runtime_transient_bytes=int(profile.runtime_transient_bytes),
            cuda_graph_bytes=int(profile.cuda_graph_bytes),
            local_kv_budget_bytes=local_kv_budget_bytes,
        )


def profiling_kv_slots(config) -> int:
    max_prefill_tokens = int(config.max_num_batched_tokens)
    max_prefill_batch = int(config.max_num_seqs_in_batch)
    required = max(
        int(config.max_model_len),
        max_prefill_tokens + 2 * max_prefill_batch,
    )
    if not bool(config.decode_graph_startup_capture):
        return required

    threshold = decode_sparse_long_text_threshold(
        config.sparse_method,
        num_sink_tokens=config.sink_keep_tokens,
        decode_keep_tokens=config.decode_keep_tokens,
        num_recent_tokens=config.recent_keep_tokens,
    )
    for batch_size, _, is_long_text in build_decode_cuda_graph_startup_family_plan(config):
        prompt_tokens = int(threshold) if is_long_text else 1
        required = max(required, int(batch_size) * (prompt_tokens + 2))
    return required


def profiling_kv_budget_bytes(config, num_slots: int) -> int:
    num_slots = int(num_slots)
    if num_slots <= 0:
        raise ValueError(f"Profiling KV slots must be positive, got {num_slots}.")
    dtype_size = torch.empty((), dtype=config.hf_config.torch_dtype).element_size()
    layout = config.runtime_layout
    tp_size = int(config.parallel_topology.attention_tp_size)
    configured_layout = config.attention_cache_layout
    cache_layout = (
        configured_layout
        if isinstance(configured_layout, CacheLayout)
        else CacheLayout(str(configured_layout))
    )

    if cache_layout is CacheLayout.EXPLICIT_KV:
        local_shapes = layout.local_kv_shapes(tp_size)
        if not local_shapes:
            heads = int(config.hf_config.num_key_value_heads)
            local_heads = max(1, heads // tp_size)
            head_dim = int(config.hf_config.hidden_size) // int(
                config.hf_config.num_attention_heads
            )
            local_shapes = tuple(
                (local_heads, head_dim) for _ in range(int(layout.num_kv_layers))
            )
        bytes_per_slot = sum(
            2 * int(heads) * int(head_dim) * dtype_size
            for heads, head_dim in local_shapes
        )
    elif cache_layout is CacheLayout.MLA_LATENT:
        bytes_per_layer = (
            int(config.hf_config.kv_lora_rank)
            + int(config.hf_config.qk_rope_head_dim)
        ) * dtype_size
        bytes_per_slot = int(layout.num_kv_layers) * bytes_per_layer
    else:  # pragma: no cover - CacheLayout currently has no additional values.
        raise AssertionError(f"Unhandled attention cache layout {cache_layout!r}.")

    method = normalize_sparse_method(config.sparse_method)
    if method != "quest":
        multiplier = 1 if method in {"", "vanilla", "omnikv"} else 2
        return int(num_slots * bytes_per_slot * multiplier)

    page_size = int(config.quest_chunk_size)
    pages = ceil(num_slots / page_size)
    token_slots = pages * page_size
    metadata_bytes_per_page = (
        bytes_per_slot
        if cache_layout is CacheLayout.EXPLICIT_KV
        else 2 * bytes_per_slot
    )
    return int(token_slots * bytes_per_slot + pages * metadata_bytes_per_page)


__all__ = [
    "KVCapacityPlan",
    "StartupMemoryProfile",
    "profiling_kv_budget_bytes",
    "profiling_kv_slots",
]
