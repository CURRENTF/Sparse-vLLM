from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.prefix_cache import (
    PrefixCacheBlock,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
    usable_prefix_cache_tokens,
)
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates, SparseSelection
from .prefix_cache_mixin import PrefixCacheMixin


@dataclass
class StandardPrefixBlockPayload:
    token_slots: torch.Tensor


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

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
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

        self.enable_prefix_caching = bool(
            config.enable_prefix_caching and config.vllm_sparse_method in ("", "omnikv")
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
        self._init_prefix_cache_runtime()

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        num_layers = self.num_layers

        slot_bytes = num_layers * slot_bytes_per_layer
        self.config.num_kvcache_slots = available_memory // slot_bytes
        assert self.config.num_kvcache_slots > 0, "可用显存不足以分配 KV Cache"

        logger.info(
            f"Standard Mode: Each layer can accommodate {self.config.num_kvcache_slots} tokens."
        )
        self.kv_cache = torch.empty(
            2,
            num_layers,
            self.config.num_kvcache_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device=self.device,
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx], self.layer_batch_state.slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
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
        stats = super().free_slot_stats()
        stats["free_rows"] = int(self.num_free_rows)
        if getattr(self, "prefix_cache", None) is not None:
            stats.update(self.prefix_cache.stats())
            stats["prefix_cache_evictable_slots"] = int(self._prefix_evictable_slots())
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
        return self._require_prefix_cache().inspect_prefix(
            [int(token_id) for token_id in token_ids],
            include_subtree=include_subtree,
        )

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        result = self._require_prefix_cache().safe_delete_subtree(
            [int(token_id) for token_id in token_ids],
        )
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
        return int(self.prefix_cache.evictable_blocks() * self.prefix_cache_block_size)

    def prefill_step_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_evictable_slots())

    def decode_step_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_evictable_slots())

    def prompt_admission_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_evictable_slots())

    def _prefix_hit_evictable_slots(self, seq: Sequence) -> int:
        if getattr(self, "prefix_cache", None) is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return 0
        if seq.prefix_cache_hit_last_block_id is None:
            raise RuntimeError(f"seq_id={seq.seq_id} has prefix hit length but no last block id.")
        chain = self.prefix_cache.get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )
        return sum(
            self.prefix_cache_block_size
            for block in chain
            if self.prefix_cache.can_evict(block)
        )

    def prompt_admission_cost(self, seq: Sequence) -> int:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        suffix_len = int(seq.num_prompt_tokens - hit_len)
        return suffix_len + self._prefix_hit_evictable_slots(seq)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return int(self.prompt_admission_cost(seq))

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self.clear_prefix_cache_hit(seq)
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.prefix_cache_block_size)
        if usable_tokens <= 0:
            return
        with profiler.record("prefix_cache_lookup"):
            hit_len, last_block_id, hit_blocks = self.prefix_cache.lookup_longest_prefix(
                seq.prompt_token_ids,
                max_usable_tokens=usable_tokens,
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
        for block in blocks:
            payload = block.payload
            if not isinstance(payload, StandardPrefixBlockPayload):
                raise RuntimeError("Standard prefix cache block is missing token slots.")
            slots = payload.token_slots.to(dtype=torch.int32)
            count = int(slots.numel())
            ptr = self._num_free_slots
            self.free_slots_stack[ptr: ptr + count] = slots
            self._num_free_slots += count

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> StandardPrefixBlockPayload:
        return StandardPrefixBlockPayload(token_slots=slots)

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        cached_ranges = self.seq_id_to_cached_ranges.setdefault(seq.seq_id, [])
        start = int(block.logical_block_idx) * self.prefix_cache_block_size
        cached_ranges.append((start, start + self.prefix_cache_block_size))

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        if self.seq_id_to_row:
            raise RuntimeError("Cannot reset prefix cache while Standard sequences are active.")
        num_slots = int(self.config.num_kvcache_slots)
        self.free_slots_stack[:num_slots] = torch.arange(num_slots, dtype=torch.int32, device=self.device)
        self._num_free_slots = num_slots
        self.seq_id_to_cached_ranges.clear()

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
        missing_slots = needed_slots - int(self._num_free_slots)
        needed_blocks = (missing_slots + self.prefix_cache_block_size - 1) // self.prefix_cache_block_size
        with profiler.record("prefix_cache_evict"):
            evicted = self.prefix_cache.evict_until_freeable(needed_blocks)
        self._free_prefix_cache_blocks(evicted)

    def _evict_prefix_cache_for_insert(self, needed_blocks: int = 1) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        with profiler.record("prefix_cache_evict"):
            evicted = self.prefix_cache.ensure_insert_capacity(needed_blocks)
        self._free_prefix_cache_blocks(evicted)

    def _attach_prefix_cache_if_needed(self, seq: Sequence) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        if hit_len <= 0:
            return
        if seq.seq_id in self.seq_id_to_prefix_blocks:
            return
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
            row_idx = self._get_free_row(seq.seq_id)
            if int(self.row_seq_lens[row_idx]) != 0:
                raise RuntimeError(
                    f"Cannot attach prefix cache to non-empty row: seq_id={seq.seq_id} "
                    f"row_idx={row_idx} row_len={int(self.row_seq_lens[row_idx])}."
                )

            cached_ranges = self.seq_id_to_cached_ranges.setdefault(seq.seq_id, [])
            for block in chain:
                payload = block.payload
                if (
                    not isinstance(payload, StandardPrefixBlockPayload)
                    or int(payload.token_slots.numel()) != self.prefix_cache_block_size
                ):
                    raise RuntimeError(
                        f"Invalid Standard prefix cache block slots for seq_id={seq.seq_id}: "
                        f"logical_block_idx={block.logical_block_idx}."
                    )
                start = int(block.logical_block_idx) * self.prefix_cache_block_size
                end = start + self.prefix_cache_block_size
                self.buffer_req_to_token_slots[row_idx, start:end] = payload.token_slots
                block.ref_count += 1
                cached_ranges.append((start, end))

            self.row_seq_lens[row_idx] = hit_len
            self.seq_id_to_prefix_blocks[seq.seq_id] = chain
            self.prefix_cache.touch_chain(chain)

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

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
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
            self._release_prefix_blocks(self.seq_id_to_prefix_blocks.pop(seq_id, []))
            self._release_prefix_blocks(self.seq_id_to_materialized_blocks.pop(seq_id, []))
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

            if log_level == 'DEBUG':
                logger.debug(f'{context_lens_list=}   {req_indices=}  {slot_mapping[:10].tolist()=}  {slot_mapping[-10:].tolist()=}')

            input_ids = torch.from_numpy(input_ids_np).to(self.device)
            positions = torch.from_numpy(positions_np).to(self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [int(seq.last_token)], slot.reshape(1))
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

            logger.debug(f'{slot_mapping=}   {context_lens.tolist()=}  {slot_mapping[:10]=}  {slot_mapping[-10:]=}')

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            return input_ids, positions, None

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

            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            for seq, slot in zip(seqs, new_slots_batch):
                self._record_prefix_materialization(seq, [int(seq.last_token)], slot.reshape(1))
            real_context_lens = self.row_seq_lens[row_indices]

            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64, device=self.device))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
            slot_mapping[:real_batch_size].copy_(new_slots_batch)
            context_lens[:real_batch_size].copy_(torch.tensor(real_context_lens, dtype=torch.int32, device=self.device))
            req_indices[:real_batch_size].copy_(torch.tensor(row_indices, dtype=torch.int32, device=self.device))

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
            self.layer_batch_state.max_context_len = int(max(real_context_lens)) if row_indices else 0
            self.layer_batch_state.req_indices = req_indices

            return input_ids, positions, None
