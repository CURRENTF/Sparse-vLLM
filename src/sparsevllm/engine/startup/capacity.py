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
from sparsevllm.models.layout import resolve_attention_qk_head_dim


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
    method = normalize_sparse_method(config.sparse_method)
    page_size = int(config.quest_chunk_size) if method == "quest" else 1

    def batch_slots(prompt_lengths: tuple[int, ...], output_tokens: int) -> int:
        return sum(
            ceil((int(prompt_len) + int(output_tokens)) / page_size) * page_size
            for prompt_len in prompt_lengths
        )

    prefill_lengths = profiling_prefill_prompt_lengths(config)
    required = max(
        batch_slots(prefill_lengths, 2),
        batch_slots((1,) * int(config.max_decoding_seqs), 2),
    )
    if not bool(config.decode_graph_startup_capture):
        return required

    for batch_size, _, is_long_text in build_decode_cuda_graph_startup_family_plan(config):
        required = max(
            required,
            startup_graph_family_kv_slots(config, batch_size, is_long_text),
        )
    return required


def startup_graph_family_kv_slots(
    config,
    batch_size: int,
    is_long_text: bool,
) -> int:
    method = normalize_sparse_method(config.sparse_method)
    prompt_tokens = (
        decode_sparse_long_text_threshold(
            method,
            num_sink_tokens=config.sink_keep_tokens,
            decode_keep_tokens=config.decode_keep_tokens,
            num_recent_tokens=config.recent_keep_tokens,
        )
        if is_long_text
        else 1
    )
    page_size = int(config.quest_chunk_size) if method == "quest" else 1
    slots_per_sequence = ceil((int(prompt_tokens) + 2) / page_size) * page_size
    return int(batch_size) * slots_per_sequence


def feasible_startup_graph_plan(
    config,
    startup_plan: list[tuple[int, int, bool]],
    available_kv_slots: int,
) -> tuple[list[tuple[int, int, bool]], list[tuple[int, int, bool]]]:
    feasible = []
    skipped = []
    for entry in startup_plan:
        batch_size, _, is_long_text = entry
        destination = (
            feasible
            if startup_graph_family_kv_slots(config, batch_size, is_long_text)
            <= int(available_kv_slots)
            else skipped
        )
        destination.append(entry)
    return feasible, skipped


def profiling_prefill_prompt_lengths(config) -> tuple[int, ...]:
    token_budget = int(config.max_num_batched_tokens)
    batch_size = min(int(config.max_num_seqs_in_batch), token_budget)
    per_prompt_limit = min(
        int(config.engine_prefill_chunk_size),
        int(config.max_model_len) - 1,
    )
    if batch_size <= 0 or per_prompt_limit <= 0:
        raise ValueError(
            "Startup prefill profiling requires positive batch and prompt limits: "
            f"batch_size={batch_size} per_prompt_limit={per_prompt_limit}."
        )
    target_tokens = min(token_budget, batch_size * per_prompt_limit)
    prompt_lengths = [1] * batch_size
    remaining = target_tokens - batch_size
    for index in range(batch_size):
        extra = min(per_prompt_limit - 1, remaining)
        prompt_lengths[index] += extra
        remaining -= extra
    if remaining != 0:
        raise RuntimeError(
            "Startup prefill profiling could not fill its scheduler token budget: "
            f"remaining={remaining}."
        )
    return tuple(prompt_lengths)


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
            head_dim = resolve_attention_qk_head_dim(config.hf_config)
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
        if method in {"", "vanilla", "omnikv"}:
            int32_bytes = torch.empty((), dtype=torch.int32).element_size()
            row_mapping_bytes = (
                int(config.max_num_seqs_in_gpu)
                * int(config.max_model_len)
                * int32_bytes
            )
            return int(
                num_slots * (bytes_per_slot + int32_bytes)
                + row_mapping_bytes
            )
        return int(num_slots * bytes_per_slot * 2)

    page_size = int(config.quest_chunk_size)
    pages = ceil(num_slots / page_size)
    token_slots = pages * page_size
    metadata_bytes_per_page = (
        bytes_per_slot
        if cache_layout is CacheLayout.EXPLICIT_KV
        else 2 * bytes_per_slot
    )
    int32_bytes = torch.empty((), dtype=torch.int32).element_size()
    fixed_metadata_bytes = (
        int(config.max_num_seqs_in_gpu) * int(config.max_model_len) * int32_bytes
        + int(config.max_num_seqs_in_gpu)
        * ceil(int(config.max_model_len) / page_size)
        * int32_bytes
        + page_size
        * (int32_bytes + torch.empty((), dtype=torch.int64).element_size())
    )
    return int(
        token_slots * bytes_per_slot
        + pages * (metadata_bytes_per_page + int32_bytes)
        + fixed_metadata_bytes
    )


__all__ = [
    "KVCapacityPlan",
    "StartupMemoryProfile",
    "profiling_kv_budget_bytes",
    "profiling_kv_slots",
    "profiling_prefill_prompt_lengths",
    "feasible_startup_graph_plan",
    "startup_graph_family_kv_slots",
]
