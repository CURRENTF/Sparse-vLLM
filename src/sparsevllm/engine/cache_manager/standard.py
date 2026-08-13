from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.distributed import ParallelContext
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.prefix_cache import (
    PrefixCacheBlock,
    PrefixTransferKind,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
    select_write_through_candidates,
    usable_prefix_cache_tokens,
)
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler
from sparsevllm.platforms import device_runtime

from .base import (
    AttentionCacheWrite,
    AttentionPayload,
    CacheManager,
    LayerBatchStates,
    SparseSelection,
)
from .prefix_cache_mixin import PrefixCacheMixin
from .prefix_offload import (
    PinnedPrefixKVPool,
    PrefixH2DOperation,
    StandardPrefixOffloadController,
)
from .storage import (
    ExplicitKVStorage,
    create_attention_cache_storage,
)


@dataclass
class StandardPrefixBlockPayload:
    token_slots: torch.Tensor | None
    block_start: int = 0
    block_end: int = 0
    host_block_index: int | None = None


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in ranges if int(e) > int(s)):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _complement_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cur = int(start)
    result: list[tuple[int, int]] = []
    for range_start, range_end in _merge_ranges(ranges):
        if cur < range_start:
            result.append((cur, range_start))
        cur = max(cur, range_end)
    if cur < int(end):
        result.append((cur, int(end)))
    return result


class StandardCacheManager(PrefixCacheMixin, CacheManager):

    def __init__(self, config: Config, parallel_context: ParallelContext):
        super().__init__(config, parallel_context)
        self.attention_cache_storage = create_attention_cache_storage(
            config,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        self.allocate_kv_cache()

        num_slots = config.num_kvcache_slots
        self.free_slots_stack = torch.arange(num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots = num_slots

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device=self.device
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.layer_batch_state = LayerBatchStates()
        self._decode_static_index_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        self.enable_prefix_caching = bool(
            config.enable_prefix_caching and config.vllm_sparse_method in ("", "omnikv")
            and not getattr(getattr(config, "runtime_layout", None), "linear_attention_layer_indices", ())
        )
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        self.prefix_cache: RadixPrefixIndex | None = None
        if self.enable_prefix_caching:
            self.prefix_cache = RadixPrefixIndex(
                block_size=self.prefix_cache_block_size,
                fingerprint=build_prefix_cache_fingerprint(config, self.prefix_cache_block_size),
                max_blocks=config.prefix_cache_max_blocks,
            )
        self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.seq_id_to_cached_ranges: dict[int, list[tuple[int, int]]] = {}
        self._scheduler_capacity_snapshot_depth = 0
        self._scheduler_freeable_block_ids: frozenset[bytes] | None = None
        self._init_prefix_cache_runtime()
        self.prefix_offload_controller: StandardPrefixOffloadController | None = None
        self._prefix_offload_step_h2d_operations: list[PrefixH2DOperation] = []
        self._prefix_write_through_candidates: dict[bytes, PrefixCacheBlock] = {}
        has_linear_layers = bool(
            getattr(getattr(config, "runtime_layout", None), "linear_attention_layer_indices", ())
        )
        if bool(getattr(config, "enable_prefix_cache_offload", False)) and not has_linear_layers:
            self._init_prefix_offload()

    def _init_prefix_offload(self) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            raise RuntimeError("Prefix cache offload requires the Standard prefix cache.")
        if self.tp_size not in (1, 2):
            raise RuntimeError("Prefix cache offload currently supports only TP=1 or TP=2.")
        if not bool(getattr(self.config, "enforce_eager", False)):
            raise RuntimeError("Prefix cache offload currently requires eager execution.")
        if not device_runtime.supports_pin_memory():
            raise RuntimeError("Prefix cache offload requires pinned host memory support.")
        if not device_runtime.supports_streams(self.device):
            raise RuntimeError("Prefix cache offload requires asynchronous device streams.")
        host_size_gb = getattr(self.config, "prefix_cache_host_size_gb", None)
        if host_size_gb is None:
            raise RuntimeError("Prefix cache offload requires prefix_cache_host_size_gb.")
        storage = self._require_explicit_storage("Prefix cache offload")
        kv_cache = storage.cache
        bytes_per_block = int(
            self.prefix_cache_block_size
            * self.num_kv_layers
            * storage.bytes_per_slot_per_layer()
        )
        host_bytes = int(float(host_size_gb) * (1024**3))
        host_capacity_blocks = host_bytes // bytes_per_block
        gpu_capacity_blocks = int(self.config.num_kvcache_slots) // self.prefix_cache_block_size
        required_blocks = gpu_capacity_blocks
        if self.prefix_cache.max_blocks is not None:
            required_blocks = min(required_blocks, int(self.prefix_cache.max_blocks))
        if host_capacity_blocks < required_blocks:
            raise RuntimeError(
                "Prefix host tier is too small for write-through safety: "
                f"host_blocks={host_capacity_blocks} required_blocks={required_blocks} "
                f"bytes_per_block={bytes_per_block} host_size_gb={host_size_gb}."
            )
        host_pool = PinnedPrefixKVPool(
            capacity_blocks=host_capacity_blocks,
            num_layers=self.num_kv_layers,
            block_size=self.prefix_cache_block_size,
            num_kv_heads=storage.num_kv_heads,
            head_dim=storage.head_dim,
            dtype=storage.dtype,
        )
        self.prefix_offload_controller = StandardPrefixOffloadController(
            prefix_cache=self.prefix_cache,
            kv_cache=kv_cache,
            host_pool=host_pool,
            block_size=self.prefix_cache_block_size,
            device=self.device,
        )

    def _prefix_offload_enabled(self) -> bool:
        return getattr(self, "prefix_offload_controller", None) is not None

    def _poll_prefix_offload(self) -> None:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.poll()

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        num_layers = self.num_kv_layers

        slot_bytes = num_layers * slot_bytes_per_layer
        self.config.num_kvcache_slots = available_memory // slot_bytes
        assert self.config.num_kvcache_slots > 0, "可用显存不足以分配 KV Cache"

        logger.info(
            f"Standard Mode: Each layer can accommodate {self.config.num_kvcache_slots} tokens."
        )
        self.attention_cache_storage.allocate(
            num_layers=num_layers,
            num_slots=self.config.num_kvcache_slots,
            device=self.device,
        )
        self.kv_cache = (
            self.attention_cache_storage.kv_cache
            if isinstance(self.attention_cache_storage, ExplicitKVStorage)
            else None
        )

    def attention_cache_bytes_per_slot_per_layer(self) -> int:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is None:
            return super().attention_cache_bytes_per_slot_per_layer()
        return int(storage.bytes_per_slot_per_layer())

    def _require_explicit_storage(self, operation: str) -> ExplicitKVStorage:
        storage = self.attention_cache_storage
        if not isinstance(storage, ExplicitKVStorage):
            raise TypeError(
                f"{operation} requires ExplicitKVStorage, got "
                f"{type(storage).__name__}."
            )
        return storage

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_idx = self.kv_layer_index(layer_idx)
        payload = self._require_explicit_storage("get_layer_kv_cache").layer_payload(kv_idx)
        return payload.k_cache, payload.v_cache

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_state.slot_mapping

    def store_attention_payload(
        self,
        layer_idx: int,
        payload: AttentionCacheWrite,
    ) -> torch.Tensor:
        slot_mapping = self.layer_batch_state.slot_mapping
        if slot_mapping is None:
            raise RuntimeError(
                f"Attention cache store requires slot_mapping at layer={layer_idx}."
            )
        self.attention_cache_storage.store(
            self.kv_layer_index(layer_idx),
            slot_mapping,
            payload,
        )
        return slot_mapping

    def _validate_attention_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        storage = getattr(self, "attention_cache_storage", None)
        if storage is not None:
            storage.validate_slot_mapping(slot_mapping)

    def get_layer_compute_payload(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ) -> tuple[AttentionPayload, torch.Tensor, torch.Tensor, torch.Tensor]:
        del selection
        return (
            self.attention_cache_storage.layer_payload(
                self.kv_layer_index(layer_idx)
            ),
            active_slots,
            req_indices,
            context_lens,
        )

    def get_prefill_compute_payload(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[AttentionPayload, torch.Tensor, torch.Tensor, torch.Tensor]:
        del k_current, v_current, selection
        return self.get_layer_compute_payload(
            layer_idx,
            active_slots,
            req_indices,
            context_lens,
        )

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        self.kv_layer_index(layer_idx)
        return self.buffer_req_to_token_slots

    @property
    def num_free_slots(self) -> int:
        return self._num_free_slots

    @property
    def num_free_rows(self) -> int:
        return len(self.free_rows)

    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        chunk_prefill_size: int,
    ) -> dict[str, int]:
        budgets = super().prompt_admission_budgets(waiting_seqs, chunk_prefill_size)
        budgets["rows"] = int(self.num_free_rows)
        return budgets

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        costs = super().prompt_admission_costs(seq)
        costs["rows"] = 1
        return costs

    def free_slot_stats(self) -> dict[str, int]:
        self._poll_prefix_offload()
        stats = super().free_slot_stats()
        stats["free_rows"] = int(self.num_free_rows)
        if getattr(self, "prefix_cache", None) is not None:
            stats.update(self.prefix_cache.stats())
            stats["prefix_cache_evictable_slots"] = int(self._prefix_evictable_slots())
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            stats.update(controller.stats())
        return stats

    def _require_prefix_cache(self) -> RadixPrefixIndex:
        if getattr(self, "prefix_cache", None) is None:
            raise RuntimeError("prefix cache is not enabled for this cache manager.")
        return self.prefix_cache

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        *,
        include_subtree: bool = False,
    ) -> dict[str, object]:
        self._poll_prefix_offload()
        return self._require_prefix_cache().inspect_prefix(
            [int(token_id) for token_id in token_ids],
            include_subtree=include_subtree,
        )

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        self._poll_prefix_offload()
        if getattr(self, "prefix_cache", None) is None:
            return {
                "supported": True,
                "enabled": False,
                "method": str(getattr(self.config, "vllm_sparse_method", "") or ""),
                "matched_tokens": 0,
                "matched_blocks": 0,
                "match_ratio": 0.0,
                "reason": "prefix cache is not enabled for this cache manager.",
            }
        token_ids = [int(token_id) for token_id in token_ids]
        usable_tokens = usable_prefix_cache_tokens(len(token_ids), self.prefix_cache_block_size)
        hit_len, hit_last_block_id, hit_blocks = self.prefix_cache.match_longest_prefix(
            token_ids,
            max_usable_tokens=usable_tokens,
        )
        return {
            "supported": True,
            "enabled": True,
            "method": str(getattr(self.config, "vllm_sparse_method", "") or ""),
            "block_size": int(self.prefix_cache_block_size),
            "prompt_tokens": int(len(token_ids)),
            "usable_tokens": int(usable_tokens),
            "matched_tokens": int(hit_len),
            "matched_blocks": int(hit_blocks),
            "match_ratio": 0.0 if usable_tokens <= 0 else float(hit_len) / float(usable_tokens),
            "last_block_id": None if hit_last_block_id is None else hit_last_block_id.hex(),
            "live_blocks": int(len(self.prefix_cache)),
        }

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.synchronize_all()
        normalized = [int(token_id) for token_id in token_ids]
        prefix_cache = self._require_prefix_cache()
        plan = prefix_cache.preview_delete_subtree(normalized)
        self.synchronize_prefix_cache_delete_plan(plan.to_dict())
        result = prefix_cache.safe_delete_subtree(normalized)
        self._free_prefix_cache_blocks(result.deleted_blocks)
        return result.to_dict()

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        *,
        priority: int,
    ) -> dict[str, object]:
        return self._require_prefix_cache().set_subtree_eviction_priority(
            [int(token_id) for token_id in token_ids],
            int(priority),
        )

    def _prefix_evictable_slots(self) -> int:
        if getattr(self, "prefix_cache", None) is None:
            return 0
        freeable_blocks = (
            self.prefix_cache.device_freeable_blocks()
            if self._prefix_offload_enabled()
            else len(self._prefix_freeable_block_ids_for_capacity())
        )
        return int(freeable_blocks * self.prefix_cache_block_size)

    def _prefix_freeable_block_ids_for_capacity(self) -> frozenset[bytes]:
        if self.prefix_cache is None:
            return frozenset()
        if self._scheduler_capacity_snapshot_depth <= 0:
            return self.prefix_cache.freeable_block_ids()
        if self._scheduler_freeable_block_ids is None:
            self._scheduler_freeable_block_ids = (
                self.prefix_cache.freeable_block_ids()
            )
        return self._scheduler_freeable_block_ids

    @contextmanager
    def scheduler_capacity_snapshot(self):
        """Reuse one immutable prefix-capacity view within a scheduler pass."""
        self._scheduler_capacity_snapshot_depth += 1
        if self._scheduler_capacity_snapshot_depth == 1:
            self._scheduler_freeable_block_ids = None
        try:
            yield
        finally:
            self._scheduler_capacity_snapshot_depth -= 1
            if self._scheduler_capacity_snapshot_depth == 0:
                self._scheduler_freeable_block_ids = None

    def _prefix_step_reclaimable_slots(self) -> int:
        if getattr(self, "prefix_cache", None) is None:
            return 0
        reclaimable_blocks = (
            self.prefix_cache.device_reclaimable_blocks()
            if self._prefix_offload_enabled()
            else len(self._prefix_freeable_block_ids_for_capacity())
        )
        return int(reclaimable_blocks * self.prefix_cache_block_size)

    def _prefix_immediately_evictable_slots(self) -> int:
        if (
            getattr(self, "prefix_cache", None) is None
            or self._prefix_offload_enabled()
        ):
            return 0
        return int(
            self.prefix_cache.evictable_blocks()
            * self.prefix_cache_block_size
        )

    def prefill_step_free_slots(self) -> int:
        physical_free = int(self.num_free_slots)
        max_step_tokens = int(
            getattr(self.config, "max_num_batched_tokens", 0) or 0
        )
        if max_step_tokens > 0 and physical_free >= max_step_tokens:
            return physical_free
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots())

    def decode_step_free_slots(self) -> int:
        physical_free = int(self.num_free_slots)
        max_step_seqs = int(
            getattr(self.config, "max_num_seqs_in_batch", 0) or 0
        )
        if max_step_seqs > 0 and physical_free >= max_step_seqs:
            return physical_free
        immediately_evictable = self._prefix_immediately_evictable_slots()
        if (
            max_step_seqs > 0
            and physical_free + immediately_evictable >= max_step_seqs
        ):
            return int(physical_free + immediately_evictable)
        return int(self.num_free_slots + self._prefix_step_reclaimable_slots())

    def prompt_admission_free_slots(self) -> int:
        reclaimable_slots = 0
        if getattr(self, "prefix_cache", None) is not None:
            reclaimable_blocks = (
                self.prefix_cache.device_reclaimable_blocks()
                if self._prefix_offload_enabled()
                else len(self._prefix_freeable_block_ids_for_capacity())
            )
            reclaimable_slots = reclaimable_blocks * self.prefix_cache_block_size
        return int(self.num_free_slots + reclaimable_slots)

    def _prefix_hit_evictable_slots(self, seq: Sequence) -> int:
        if getattr(self, "prefix_cache", None) is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return 0
        reclaimable_blocks, _ = self._prefix_hit_capacity_counts(seq)
        return int(reclaimable_blocks * self.prefix_cache_block_size)

    def prompt_admission_cost(self, seq: Sequence) -> int:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        suffix_len = int(seq.num_prompt_tokens - hit_len)
        if hit_len <= 0:
            return suffix_len
        reclaimable_blocks, promotion_blocks = self._prefix_hit_capacity_counts(seq)
        return (
            suffix_len
            + (promotion_blocks + reclaimable_blocks)
            * self.prefix_cache_block_size
        )

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return int(self.prompt_admission_cost(seq))

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self._poll_prefix_offload()
        self.clear_prefix_cache_hit(seq)
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.prefix_cache_block_size)
        if usable_tokens <= 0:
            return
        with profiler.record("prefix_cache_lookup"):
            hit_len, last_block_id, hit_blocks = self._lookup_prefix_cache_hit(
                seq,
                usable_tokens,
            )
        if hit_len <= 0:
            return
        if last_block_id is None or hit_blocks <= 0:
            raise RuntimeError("Prefix cache lookup returned an invalid hit.")
        if hit_len >= seq.num_prompt_tokens or hit_len % self.prefix_cache_block_size != 0:
            raise RuntimeError(
                "Prefix cache lookup returned an unusable hit length: "
                f"seq_id={seq.seq_id} hit_len={hit_len} prompt_len={seq.num_prompt_tokens} "
                f"block_size={self.prefix_cache_block_size}."
            )
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = self.prefix_cache_block_size
        seq.prefix_cache_method = str(self.config.vllm_sparse_method or "")

    def _free_prefix_cache_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        pending = getattr(self, "_prefix_write_through_candidates", None)
        host_blocks: list[PrefixCacheBlock] = []
        for block in blocks:
            if pending is not None:
                pending.pop(block.stable_block_id, None)
            payload = block.payload
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard prefix cache block is missing token slots.")
            if block.residency.device_present:
                self._free_device_prefix_block(block)
            if block.residency.host_present:
                host_blocks.append(block)
        controller = getattr(self, "prefix_offload_controller", None)
        if host_blocks:
            if controller is None:
                raise RuntimeError(
                    "Prefix blocks have host payloads but no offload controller is active."
                )
            controller.free_host_payloads(host_blocks)

    def _free_device_prefix_block(self, block: PrefixCacheBlock) -> None:
        payload = block.payload
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard prefix cache block is missing its device payload.")
        slots = payload.token_slots
        if not isinstance(slots, torch.Tensor) or int(slots.numel()) != self.prefix_cache_block_size:
            raise RuntimeError(
                "Standard prefix cache block has invalid device slots: "
                f"block={block.stable_block_id.hex()[:16]}."
            )
        slots = slots.to(device=self.device, dtype=torch.int32)
        count = int(slots.numel())
        ptr = self._num_free_slots
        self.free_slots_stack[ptr: ptr + count] = slots
        self._num_free_slots += count
        payload.token_slots = None

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> StandardPrefixBlockPayload:
        return StandardPrefixBlockPayload(
            token_slots=slots,
            block_start=0,
            block_end=int(slots.numel()),
        )

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        cached_ranges = self.seq_id_to_cached_ranges.setdefault(seq.seq_id, [])
        start = int(block.logical_block_idx) * self.prefix_cache_block_size
        cached_ranges.append((start, start + self.prefix_cache_block_size))

    def build_prefix_kv_payload(self, seq: Sequence, block_start: int, block_end: int) -> StandardPrefixBlockPayload:
        block_start = int(block_start)
        block_end = int(block_end)
        if block_end <= block_start:
            raise ValueError(f"Invalid prefix KV payload range: {block_start}:{block_end}.")
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is None:
            raise RuntimeError(f"Cannot build prefix KV payload for unknown seq_id={seq.seq_id}.")
        row_len = int(self.row_seq_lens[row_idx])
        if block_end > row_len:
            raise RuntimeError(
                "Cannot build prefix KV payload beyond materialized row length: "
                f"seq_id={seq.seq_id} block={block_start}:{block_end} row_len={row_len}."
            )
        slots = self.buffer_req_to_token_slots[row_idx, block_start:block_end].detach().to(
            dtype=torch.int32,
        ).clone()
        return StandardPrefixBlockPayload(
            token_slots=slots,
            block_start=block_start,
            block_end=block_end,
        )

    def attach_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
        slots = payload.token_slots.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(slots.numel())
        if count <= 0:
            raise RuntimeError("Standard mixed prefix KV payload is empty.")
        if count % int(self.config.prefix_cache_block_size) != 0:
            raise RuntimeError(
                f"Standard mixed prefix KV payload size must be block-aligned, got {count}."
            )
        row_idx = self._get_free_row(int(seq.seq_id))
        cur_len = int(self.row_seq_lens[row_idx])
        if int(payload.block_start) != cur_len:
            raise RuntimeError(
                "Standard mixed prefix KV payload attach must be contiguous: "
                f"seq_id={seq.seq_id} block_start={int(payload.block_start)} row_len={cur_len}."
            )
        start = cur_len
        end = start + count
        if int(payload.block_end) not in {0, end}:
            raise RuntimeError(
                "Standard mixed prefix KV payload has inconsistent block_end: "
                f"payload_end={int(payload.block_end)} expected={end}."
            )
        if end > int(self.max_model_len):
            raise RuntimeError(
                "Attaching mixed prefix KV payload exceeds max_model_len: "
                f"seq_id={seq.seq_id} end={end} max_model_len={self.max_model_len}."
            )
        self.buffer_req_to_token_slots[row_idx, start:end] = slots
        self.row_seq_lens[row_idx] = end
        cached_ranges = self.seq_id_to_cached_ranges.setdefault(int(seq.seq_id), [])
        cached_ranges.append((start, end))

    def validate_prefix_kv_attach(self, seq: Sequence) -> bool:
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is not None and int(self.row_seq_lens[row_idx]) != 0:
            raise RuntimeError(
                "Cannot attach mixed prefix KV to a non-empty row: "
                f"seq_id={seq.seq_id} row_idx={row_idx} "
                f"row_len={int(self.row_seq_lens[row_idx])}."
            )
        if row_idx is None and not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        return row_idx is not None

    def rollback_prefix_kv_attach(
        self,
        seq: Sequence,
        payloads: list[object],
        *,
        row_preexisted: bool,
    ) -> None:
        normalized: list[StandardPrefixBlockPayload] = []
        expected_start = 0
        for payload in payloads:
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard mixed prefix rollback received an invalid payload.")
            if int(payload.block_start) != expected_start or int(payload.block_end) <= expected_start:
                raise RuntimeError(
                    "Standard mixed prefix rollback payloads are not contiguous: "
                    f"expected_start={expected_start} "
                    f"range={int(payload.block_start)}:{int(payload.block_end)}."
                )
            normalized.append(payload)
            expected_start = int(payload.block_end)
        if not normalized:
            return

        seq_id = int(seq.seq_id)
        row_idx = self.seq_id_to_row.get(seq_id)
        if row_idx is None or int(self.row_seq_lens[row_idx]) != expected_start:
            raise RuntimeError(
                "Standard mixed prefix rollback row state is inconsistent: "
                f"seq_id={seq_id} row_idx={row_idx} "
                f"row_len={None if row_idx is None else int(self.row_seq_lens[row_idx])} "
                f"expected={expected_start}."
            )
        expected_ranges = [
            (int(payload.block_start), int(payload.block_end))
            for payload in normalized
        ]
        if self.seq_id_to_cached_ranges.get(seq_id) != expected_ranges:
            raise RuntimeError(
                "Standard mixed prefix rollback cached ranges are inconsistent: "
                f"seq_id={seq_id} expected={expected_ranges} "
                f"got={self.seq_id_to_cached_ranges.get(seq_id)}."
            )

        self.buffer_req_to_token_slots[row_idx, :expected_start] = 0
        self.row_seq_lens[row_idx] = 0
        self.seq_id_to_cached_ranges.pop(seq_id, None)
        if not row_preexisted:
            owner = self.seq_id_to_row.pop(seq_id, None)
            if owner != row_idx:
                raise RuntimeError(
                    "Standard mixed prefix rollback row ownership changed unexpectedly: "
                    f"seq_id={seq_id} expected_row={row_idx} owner={owner}."
                )
            self.free_rows.appendleft(row_idx)

    def free_prefix_kv_payload(self, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
        slots = payload.token_slots.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(slots.numel())
        ptr = self._num_free_slots
        self.free_slots_stack[ptr: ptr + count] = slots
        self._num_free_slots += count

    def allocate_prefix_kv_payload_device(self, payload: object) -> None:
        self.allocate_prefix_kv_payloads_device([payload])

    def allocate_prefix_kv_payloads_device(self, payloads: list[object]) -> None:
        normalized: list[StandardPrefixBlockPayload] = []
        counts: list[int] = []
        for payload in payloads:
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
            if isinstance(payload.token_slots, torch.Tensor):
                raise RuntimeError("Mixed prefix KV payload is already device-resident.")
            count = int(payload.block_end) - int(payload.block_start)
            if count <= 0:
                raise RuntimeError("Mixed prefix KV payload has an invalid token range.")
            normalized.append(payload)
            counts.append(count)
        if not normalized:
            return
        slots = self._take_prefix_device_slots(sum(counts))
        offset = 0
        for payload, count in zip(normalized, counts):
            payload.token_slots = slots[offset : offset + count]
            offset += count

    def free_prefix_kv_payload_device(self, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Mixed prefix KV payload has no device slots to demote.")
        self._return_prefix_device_slots(payload.token_slots)
        payload.token_slots = None

    def prefix_kv_payload_nbytes(self, payload: object) -> int:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        if not isinstance(payload.token_slots, torch.Tensor):
            raise RuntimeError("Standard mixed prefix KV payload has no device slots.")
        dtype_size = self._cache_slot_dtype_size()
        return int(
            payload.token_slots.numel()
            * self.num_kv_layers
            * 2
            * self.num_kv_heads
            * self.head_dim
            * dtype_size
        )

    def mark_materialized_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        row_idx = self.seq_id_to_row.get(int(seq.seq_id))
        if row_idx is None:
            raise RuntimeError(f"Cannot mark mixed prefix payload for unknown seq_id={seq.seq_id}.")
        start = int(payload.block_start)
        end = int(payload.block_end)
        row_len = int(self.row_seq_lens[row_idx])
        if start < 0 or end <= start or end > row_len:
            raise RuntimeError(
                "Cannot mark mixed prefix payload: "
                f"seq_id={seq.seq_id} range={start}:{end} row_len={row_len}."
            )
        self.seq_id_to_cached_ranges.setdefault(int(seq.seq_id), []).append((start, end))

    def rollback_materialized_prefix_kv_payload(
        self,
        seq: Sequence,
        payload: object,
    ) -> None:
        if not isinstance(payload, StandardPrefixBlockPayload):
            raise RuntimeError("Standard mixed prefix KV payload is missing token slots.")
        seq_id = int(seq.seq_id)
        target = (int(payload.block_start), int(payload.block_end))
        cached_ranges = self.seq_id_to_cached_ranges.get(seq_id)
        if not cached_ranges:
            return
        for idx in range(len(cached_ranges) - 1, -1, -1):
            if cached_ranges[idx] == target:
                cached_ranges.pop(idx)
                break
        if not cached_ranges:
            self.seq_id_to_cached_ranges.pop(seq_id, None)

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        if self.seq_id_to_row:
            raise RuntimeError("Cannot reset prefix cache while Standard sequences are active.")
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.reset()
        num_slots = int(self.config.num_kvcache_slots)
        self.free_slots_stack[:num_slots] = torch.arange(num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots = num_slots
        self.seq_id_to_cached_ranges.clear()
        getattr(self, "_prefix_write_through_candidates", {}).clear()

    def _on_prefix_cache_reset(self) -> None:
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None:
            controller.prefix_cache = self._require_prefix_cache()

    def reset_after_warmup(self) -> None:
        if self.enable_prefix_caching and self.prefix_cache is not None:
            self.reset_prefix_cache()
            return
        self._reset_prefix_cache_allocator_after_clear()

    def _evict_prefix_cache_until_free(self, needed_slots: int) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if self._num_free_slots >= needed_slots:
            return
        if self._prefix_offload_enabled():
            controller = self.prefix_offload_controller
            assert controller is not None
            while self._num_free_slots < needed_slots:
                self._poll_prefix_offload()
                missing_slots = needed_slots - int(self._num_free_slots)
                needed_blocks = (
                    missing_slots + self.prefix_cache_block_size - 1
                ) // self.prefix_cache_block_size
                with profiler.record("prefix_cache_device_demote"):
                    demoted = self.prefix_cache.demote_device_until_freeable(needed_blocks)
                for block in demoted:
                    self._free_device_prefix_block(block)
                if self._num_free_slots >= needed_slots:
                    return
                if not controller.wait_oldest_d2h():
                    break
            return

        missing_slots = needed_slots - int(self._num_free_slots)
        needed_blocks = (missing_slots + self.prefix_cache_block_size - 1) // self.prefix_cache_block_size
        with profiler.record("prefix_cache_evict"):
            evicted = self.prefix_cache.evict_until_freeable(needed_blocks)
        self._free_prefix_cache_blocks(evicted)

    def _evict_prefix_cache_for_insert(self, needed_blocks: int = 1) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if self._prefix_offload_enabled():
            max_blocks = self.prefix_cache.max_blocks
            if max_blocks is None:
                return
            over_capacity = len(self.prefix_cache) + int(needed_blocks) - int(max_blocks)
            if over_capacity <= 0:
                return
            controller = self.prefix_offload_controller
            assert controller is not None
            evicted: list[PrefixCacheBlock] = []
            while len(evicted) < over_capacity:
                self._poll_prefix_offload()
                remaining = over_capacity - len(evicted)
                with profiler.record("prefix_cache_host_evict"):
                    host_evicted = self.prefix_cache.evict_host_until_freeable(remaining)
                self._free_prefix_cache_blocks(host_evicted)
                evicted.extend(host_evicted)
                if len(evicted) >= over_capacity:
                    break

                remaining = over_capacity - len(evicted)
                with profiler.record("prefix_cache_device_demote"):
                    demoted = self.prefix_cache.demote_device_until_freeable(remaining)
                for block in demoted:
                    self._free_device_prefix_block(block)
                with profiler.record("prefix_cache_host_evict"):
                    newly_evicted = self.prefix_cache.evict_host_until_freeable(remaining)
                self._free_prefix_cache_blocks(newly_evicted)
                evicted.extend(newly_evicted)
                if len(evicted) >= over_capacity:
                    break

                inflight_before = sum(
                    1
                    for block in self.prefix_cache.blocks.values()
                    if block.residency.transfer == PrefixTransferKind.D2H
                )
                if inflight_before <= 0 or not controller.wait_oldest_d2h():
                    break
                self._poll_prefix_offload()
                inflight_after = sum(
                    1
                    for block in self.prefix_cache.blocks.values()
                    if block.residency.transfer == PrefixTransferKind.D2H
                )
                if inflight_after >= inflight_before:
                    break
            if len(evicted) != over_capacity:
                raise RuntimeError(
                    "Prefix cache logical capacity exceeded and not enough CPU-only leaves "
                    "are evictable: "
                    f"live_blocks={len(self.prefix_cache)} max_blocks={max_blocks} "
                    f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)}."
                )
            return
        with profiler.record("prefix_cache_evict"):
            evicted = self.prefix_cache.ensure_insert_capacity(needed_blocks)
        self._free_prefix_cache_blocks(evicted)

    def _ensure_prefix_host_capacity(self, needed_blocks: int) -> None:
        controller = self.prefix_offload_controller
        if controller is None:
            raise RuntimeError("Prefix host capacity requested without an offload controller.")
        needed_blocks = int(needed_blocks)
        if controller.host_pool.free_blocks >= needed_blocks:
            return
        missing = needed_blocks - controller.host_pool.free_blocks
        with profiler.record("prefix_cache_host_evict"):
            evicted = self._require_prefix_cache().evict_host_until_freeable(missing)
        self._free_prefix_cache_blocks(evicted)
        if controller.host_pool.free_blocks < needed_blocks:
            raise RuntimeError(
                "Prefix host pool cannot preserve write-through residency: "
                f"need={needed_blocks} free={controller.host_pool.free_blocks} "
                f"evicted={len(evicted)} capacity={controller.host_pool.capacity_blocks}."
            )

    def _schedule_write_through_prefix_blocks(
        self,
        newly_unreferenced: list[PrefixCacheBlock] | None = None,
    ) -> None:
        if not self._prefix_offload_enabled():
            return
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Prefix D2H scheduling is forbidden during graph capture.")
        self._poll_prefix_offload()
        prefix_cache = self._require_prefix_cache()
        pending = getattr(self, "_prefix_write_through_candidates", None)
        if pending is None:
            pending = {}
            self._prefix_write_through_candidates = pending
        selected = select_write_through_candidates(
            prefix_cache,
            pending,
            newly_unreferenced,
        )
        if not selected:
            return
        self._ensure_prefix_host_capacity(len(selected))
        controller = self.prefix_offload_controller
        assert controller is not None
        with profiler.record("prefix_cache_d2h_submit"):
            controller.submit_d2h(selected)
        for block in selected:
            pending.pop(block.stable_block_id, None)

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        self._poll_prefix_offload()
        return super().on_forward_end(seqs, is_prefill)

    def _attach_prefix_cache_if_needed(self, seq: Sequence) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0:
            return
        if seq.seq_id in self.seq_id_to_prefix_blocks:
            return
        self._poll_prefix_offload()
        with profiler.record("prefix_cache_attach"):
            if seq.prefix_cache_hit_last_block_id is None:
                raise RuntimeError(f"seq_id={seq.seq_id} has prefix hit length but no last block id.")
            if hit_len % self.prefix_cache_block_size != 0:
                raise RuntimeError(
                    f"seq_id={seq.seq_id} prefix hit length is not block aligned: "
                    f"hit_len={hit_len} block_size={self.prefix_cache_block_size}."
                )
            chain = self.prefix_cache.get_chain(
                seq.prefix_cache_hit_last_block_id,
                int(seq.prefix_cache_hit_block_count),
            )
            if len(chain) * self.prefix_cache_block_size != hit_len:
                raise RuntimeError(
                    "Prefix cache chain length does not match scheduler metadata: "
                    f"seq_id={seq.seq_id} hit_len={hit_len} blocks={len(chain)} "
                    f"block_size={self.prefix_cache_block_size}."
                )
            cpu_only_blocks: list[PrefixCacheBlock] = []
            existing_h2d_operations: list[PrefixH2DOperation] = []
            saw_cpu_only = False
            for block in chain:
                payload = block.payload
                if not isinstance(payload, StandardPrefixBlockPayload):
                    raise RuntimeError(
                        f"Invalid Standard prefix cache payload for seq_id={seq.seq_id}: "
                        f"logical_block_idx={block.logical_block_idx}."
                    )
                residency = block.residency
                residency.validate()
                if not residency.device_present:
                    saw_cpu_only = True
                    if not self._prefix_offload_enabled() or not residency.host_present:
                        raise RuntimeError(
                            "Prefix lookup returned a non-device block that cannot be promoted: "
                            f"seq_id={seq.seq_id} block={block.stable_block_id.hex()[:16]}."
                        )
                    if residency.transfer is not None:
                        raise RuntimeError(
                            "CPU-only prefix block has an unexpected in-flight transfer: "
                            f"seq_id={seq.seq_id} transfer={residency.transfer.value}."
                        )
                    if payload.host_block_index is None:
                        raise RuntimeError(
                            "CPU-only prefix block is missing its host allocation: "
                            f"seq_id={seq.seq_id} block={block.stable_block_id.hex()[:16]}."
                        )
                    cpu_only_blocks.append(block)
                    continue
                if saw_cpu_only:
                    raise RuntimeError(
                        "Prefix device residency is not root-contiguous: "
                        f"seq_id={seq.seq_id} block={block.stable_block_id.hex()[:16]}."
                    )
                if (
                    not isinstance(payload.token_slots, torch.Tensor)
                    or int(payload.token_slots.numel()) != self.prefix_cache_block_size
                ):
                    raise RuntimeError(
                        f"Invalid Standard prefix cache block slots for seq_id={seq.seq_id}: "
                        f"logical_block_idx={block.logical_block_idx}."
                    )
                if residency.transfer == PrefixTransferKind.H2D:
                    controller = self.prefix_offload_controller
                    assert controller is not None
                    operation = controller.h2d_operation_for_block(block)
                    if operation is None:
                        raise RuntimeError(
                            "Prefix block is promoting without a tracked H2D operation: "
                            f"block={block.stable_block_id.hex()[:16]}."
                        )
                    if all(existing is not operation for existing in existing_h2d_operations):
                        existing_h2d_operations.append(operation)

            existing_row_idx = self.seq_id_to_row.get(seq.seq_id)
            if existing_row_idx is not None and int(self.row_seq_lens[existing_row_idx]) != 0:
                raise RuntimeError(
                    f"Cannot attach prefix cache to non-empty row: seq_id={seq.seq_id} "
                    f"row_idx={existing_row_idx} "
                    f"row_len={int(self.row_seq_lens[existing_row_idx])}."
                )
            if existing_row_idx is None and not self.free_rows:
                raise RuntimeError("No free rows in cache manager buffer!")

            for block in chain:
                self.prefix_cache.acquire_block_ref(block)
            allocated_promotion_slots: torch.Tensor | None = None
            submitted_operation: PrefixH2DOperation | None = None
            try:
                if cpu_only_blocks:
                    if not self._prefix_offload_enabled():
                        raise RuntimeError(
                            "CPU prefix hit requires prefix cache offload to be enabled."
                        )
                    if device_runtime.is_stream_capturing():
                        raise RuntimeError("Prefix H2D promotion is forbidden during graph capture.")
                    promotion_slot_count = len(cpu_only_blocks) * self.prefix_cache_block_size
                    allocated_promotion_slots = self._take_prefix_device_slots(
                        promotion_slot_count
                    )
                    for offset, block in enumerate(cpu_only_blocks):
                        payload = block.payload
                        assert isinstance(payload, StandardPrefixBlockPayload)
                        start = offset * self.prefix_cache_block_size
                        end = start + self.prefix_cache_block_size
                        payload.token_slots = allocated_promotion_slots[start:end]
                    controller = self.prefix_offload_controller
                    assert controller is not None
                    with profiler.record("prefix_cache_h2d_submit"):
                        submitted_operation = controller.submit_h2d(cpu_only_blocks)
            except Exception:
                for block in chain:
                    self.prefix_cache.release_block_ref(block)
                if allocated_promotion_slots is not None:
                    self._return_prefix_device_slots(allocated_promotion_slots)
                    for block in cpu_only_blocks:
                        payload = block.payload
                        assert isinstance(payload, StandardPrefixBlockPayload)
                        payload.token_slots = None
                raise

            row_idx = self._get_free_row(seq.seq_id)

            active_operations = list(existing_h2d_operations)
            if submitted_operation is not None:
                active_operations.append(submitted_operation)
            for operation in active_operations:
                if all(
                    existing is not operation
                    for existing in self._prefix_offload_step_h2d_operations
                ):
                    self._prefix_offload_step_h2d_operations.append(operation)

            cached_ranges = self.seq_id_to_cached_ranges.setdefault(seq.seq_id, [])
            for block in chain:
                payload = block.payload
                assert isinstance(payload, StandardPrefixBlockPayload)
                if not isinstance(payload.token_slots, torch.Tensor):
                    raise RuntimeError(
                        "Prefix attach reached a block without device slots after promotion: "
                        f"block={block.stable_block_id.hex()[:16]}."
                    )
                start = int(block.logical_block_idx) * self.prefix_cache_block_size
                end = start + self.prefix_cache_block_size
                self.buffer_req_to_token_slots[row_idx, start:end] = payload.token_slots
                cached_ranges.append((start, end))

            self.row_seq_lens[row_idx] = hit_len
            self.seq_id_to_prefix_blocks[seq.seq_id] = chain
            self.prefix_cache.touch_chain(chain)

    def _take_prefix_device_slots(self, count: int) -> torch.Tensor:
        count = int(count)
        self._evict_prefix_cache_until_free(count)
        if self._num_free_slots < count:
            raise RuntimeError(
                "Out of KV cache slots while promoting a CPU prefix: "
                f"need={count} free={self._num_free_slots}."
            )
        ptr = self._num_free_slots
        slots = self.free_slots_stack[ptr - count:ptr].clone()
        self._num_free_slots -= count
        return slots

    def _return_prefix_device_slots(self, slots: torch.Tensor) -> None:
        slots = slots.to(device=self.device, dtype=torch.int32).reshape(-1)
        count = int(slots.numel())
        ptr = self._num_free_slots
        self.free_slots_stack[ptr:ptr + count] = slots
        self._num_free_slots += count

    def _get_free_row(self, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            self._evict_prefix_cache_until_free(size)
            assert self._num_free_slots >= size, (
                f"Out of KV cache slots: need {size}, free {self._num_free_slots}"
            )

            row_idx = self._get_free_row(seq_id)
            cur_len = self.row_seq_lens[row_idx]

            ptr = self._num_free_slots
            select_index = self.free_slots_stack[ptr - size: ptr]
            self._num_free_slots -= size

            self.buffer_req_to_token_slots[row_idx, cur_len: cur_len + size] = select_index
            self.row_seq_lens[row_idx] += size

            return select_index

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        self._evict_prefix_cache_until_free(batch_size)
        assert self._num_free_slots >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots}"
        )

        row_indices = [self._get_free_row(sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots
        select_indices = self.free_slots_stack[ptr - batch_size: ptr]
        self._num_free_slots -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
        self.buffer_req_to_token_slots[rows_gpu, cols_gpu] = select_indices
        self.row_seq_lens[row_indices] += 1

        return select_indices

    def _get_decode_static_index_buffers(self, graph_batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        graph_batch_size = int(graph_batch_size)
        if not hasattr(self, "_decode_static_index_buffers"):
            self._decode_static_index_buffers = {}
        buffers = self._decode_static_index_buffers.get(graph_batch_size)
        if buffers is None:
            buffers = (
                torch.empty((graph_batch_size,), dtype=torch.long, device=self.device),
                torch.empty((graph_batch_size,), dtype=torch.long, device=self.device),
            )
            self._decode_static_index_buffers[graph_batch_size] = buffers
        return buffers

    @torch.no_grad()
    def _allocate_decode_batch_static(
        self,
        seq_ids: list[int],
        graph_batch_size: int,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        batch_size = len(seq_ids)
        self._evict_prefix_cache_until_free(batch_size)
        if self._num_free_slots < batch_size:
            raise RuntimeError(
                f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots}"
            )

        row_indices = np.asarray([self._get_free_row(sid) for sid in seq_ids], dtype=np.int64)
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots
        select_indices = self.free_slots_stack[ptr - batch_size: ptr]
        self._num_free_slots -= batch_size

        rows_gpu, cols_gpu = self._get_decode_static_index_buffers(graph_batch_size)
        rows_gpu[:batch_size].copy_(torch.from_numpy(row_indices))
        cols_gpu[:batch_size].copy_(torch.from_numpy(cur_lens.astype(np.int64, copy=False)))
        self.buffer_req_to_token_slots[
            rows_gpu[:batch_size],
            cols_gpu[:batch_size],
        ] = select_indices
        self.row_seq_lens[row_indices] += 1

        return select_indices, self.row_seq_lens[row_indices], row_indices

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self._poll_prefix_offload()
            debug_slots = os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1"
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            cur_len = self.row_seq_lens[row_idx]
            cached_ranges = _merge_ranges(self.seq_id_to_cached_ranges.pop(seq_id, []))

            assert cur_len > 0
            before_free = self._num_free_slots
            freed_tokens = 0
            for start, end in _complement_ranges(0, int(cur_len), cached_ranges):
                slots = self.buffer_req_to_token_slots[row_idx, start:end]
                count = int(end - start)
                ptr = self._num_free_slots
                self.free_slots_stack[ptr: ptr + count] = slots
                self._num_free_slots += count
                freed_tokens += count
            released_prefix_blocks = self.seq_id_to_prefix_blocks.pop(seq_id, [])
            released_prefix_blocks.extend(
                self.seq_id_to_materialized_blocks.pop(seq_id, [])
            )
            self._release_prefix_blocks(released_prefix_blocks)
            self._schedule_write_through_prefix_blocks(released_prefix_blocks)
            self.prefix_runtime_states.pop(seq_id, None)
            self.pending_prefix_blocks.pop(seq_id, None)
            after_free = self._num_free_slots

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.row_seq_lens[row_idx] = 0
            self.free_rows.append(row_idx)

            if debug_slots:
                logger.info(
                    "free_seq seq_id={} row_idx={} freed_tokens={} free_slots_before={} free_slots_after={}",
                    seq_id,
                    row_idx,
                    int(freed_tokens),
                    int(before_free),
                    int(after_free),
                )
            if log_level == 'DEBUG': logger.debug(f'free seq {row_idx} with {cur_len} tokens')

    def debug_live_seq_slots(self) -> dict[int, int]:
        return {
            int(seq_id): int(self.row_seq_lens[row_idx])
            for seq_id, row_idx in self.seq_id_to_row.items()
            if int(self.row_seq_lens[row_idx]) > 0
        }

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError('不需要实现该方法')

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = []
            for seq in seqs:
                self._attach_prefix_cache_if_needed(seq)

            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device=self.device)
            context_lens_list = []
            req_indices = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                if seq.seq_id in self.seq_id_to_row:
                    row_idx = self.seq_id_to_row[seq.seq_id]
                    if self.row_seq_lens[row_idx] != start_idx:
                        raise ValueError(
                            "KV cache row length mismatch in prefill: "
                            f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} "
                            f"start_idx={start_idx}"
                        )

                allocated_slots = self._allocate(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                slot_mapping[token_offset: token_offset + chunk_size] = self.buffer_req_to_token_slots[row_idx, start_idx:end_idx]
                context_lens_list.append(end_idx)
                req_indices.append(row_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]
                chunk_tokens = list(chunk_tokens)

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)
                self._record_prefix_materialization(seq, chunk_tokens, allocated_slots)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=self.device)
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device=self.device)

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = max(context_lens_list) if context_lens_list else 0
            self.layer_batch_state.req_indices = req_indices_tensor
            self._validate_attention_slot_mapping(slot_mapping)

            if log_level == 'DEBUG':
                logger.debug(f'{context_lens_list=}   {req_indices=}  {slot_mapping[:10].tolist()=}  {slot_mapping[-10:].tolist()=}')

            input_ids = torch.from_numpy(input_ids_np).to(self.device)
            positions = torch.from_numpy(positions_np).to(self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = []
            batch_size = len(seqs)
            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [seq.decode_input_token], slot.reshape(1))
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device=self.device,
            )
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device=self.device)

            slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device=self.device)
            slot_mapping[:] = new_slots_batch

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = int(max(self.row_seq_lens[row_indices])) if row_indices else 0
            self.layer_batch_state.req_indices = req_indices
            self._validate_attention_slot_mapping(slot_mapping)

            if log_level == 'DEBUG':
                logger.debug(f'{slot_mapping=}   {context_lens.tolist()=}  {slot_mapping[:10]=}  {slot_mapping[-10:]=}')

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            return input_ids, positions, None

    def before_prefill_layer_attention(
        self,
        layer_idx: int,
        selection: SparseSelection,
    ):
        controller = getattr(self, "prefix_offload_controller", None)
        if controller is not None and self._prefix_offload_step_h2d_operations:
            if device_runtime.is_stream_capturing():
                raise RuntimeError("Prefix H2D waits are forbidden during graph capture.")
            kv_layer_index = self.kv_layer_index(layer_idx)
            with profiler.record("prefix_cache_h2d_layer_wait"):
                for operation in self._prefix_offload_step_h2d_operations:
                    controller.wait_for_layer(operation, kv_layer_index)
        return super().before_prefill_layer_attention(layer_idx, selection)

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        """Prepare decode metadata into caller-owned static CUDA buffers.

        Used by CUDA Graph decode replay: tensor addresses must stay stable, so
        this avoids the ordinary per-step metadata tensor allocation path.
        """
        with profiler.record("cache_prepare_decode"):
            self._poll_prefix_offload()
            self._prefix_offload_step_h2d_operations = []
            real_batch_size = len(seqs)
            graph_batch_size = int(input_ids.numel())
            if real_batch_size <= 0:
                raise ValueError("Static decode requires a non-empty real decode batch.")
            if positions.numel() != graph_batch_size:
                raise ValueError("Static decode input buffers must have the same graph batch size.")
            if (
                slot_mapping.numel() != graph_batch_size
                or context_lens.numel() != graph_batch_size
                or req_indices.numel() != graph_batch_size
            ):
                raise ValueError("Static decode metadata buffers must have the same graph batch size.")
            if real_batch_size > graph_batch_size:
                raise ValueError(
                    "Static decode graph batch is smaller than the real decode batch: "
                    f"graph={graph_batch_size}, real={real_batch_size}."
                )

            input_ids_list = [seq.decode_input_token for seq in seqs]
            positions_list = [seq.decode_input_position for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch, real_context_lens, row_indices = self._allocate_decode_batch_static(
                seq_ids,
                graph_batch_size,
            )
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [seq.decode_input_token], slot.reshape(1))

            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64))
            slot_mapping[:real_batch_size].copy_(new_slots_batch)
            context_lens[:real_batch_size].copy_(
                torch.from_numpy(real_context_lens.astype(np.int32, copy=False))
            )
            req_indices[:real_batch_size].copy_(
                torch.from_numpy(row_indices.astype(np.int32, copy=False))
            )

            if graph_batch_size > real_batch_size:
                # CUDA Graph replay is shape-static. Padded rows mirror the first
                # real request for read-only attention work, but use slot -1 so
                # they never write KV or consume persistent cache capacity.
                first_context_len = int(real_context_lens[0])
                first_row_idx = int(row_indices[0])
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))
                slot_mapping[real_batch_size:].fill_(-1)
                context_lens[real_batch_size:].fill_(first_context_len)
                req_indices[real_batch_size:].fill_(first_row_idx)

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = int(real_context_lens.max()) if real_batch_size > 0 else 0
            self.layer_batch_state.req_indices = req_indices
            self.validate_decode_cuda_graph_slot_mappings()

            return input_ids, positions, None
