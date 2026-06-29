from __future__ import annotations

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
from sparsevllm.utils.context import get_context
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates, SparseSelection
from .prefix_cache_mixin import PrefixCacheMixin


@dataclass
class QuestPrefixBlockPayload:
    block_slot: int
    token_slots: torch.Tensor


class QuestCacheManager(PrefixCacheMixin, CacheManager):
    """Paged KV cache + page metadata cache for QuEST."""

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.page_size = int(config.quest_chunk_size)
        self.max_pages_per_row = (self.max_model_len + self.page_size - 1) // self.page_size
        self.page_offsets_i32 = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        self.page_offsets_i64 = self.page_offsets_i32.to(torch.int64)

        self.allocate_kv_cache()

        self.free_pages_stack = torch.arange(self.num_pages, dtype=torch.int32, device=self.device)
        self.free_pages_cpu_stack = np.arange(self.num_pages, dtype=np.int32)
        self._num_free_pages = self.num_pages

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device=self.device
        )
        self.buffer_req_to_page_slots = torch.full(
            (self.max_buffer_rows, self.max_pages_per_row), -1, dtype=torch.int32, device=self.device
        )
        self.buffer_req_to_page_slots_cpu = np.full(
            (self.max_buffer_rows, self.max_pages_per_row), -1, dtype=np.int32
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.layer_batch_state = LayerBatchStates()
        self.enable_prefix_caching = bool(config.enable_prefix_caching and config.vllm_sparse_method == "quest")
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        if self.enable_prefix_caching and self.prefix_cache_block_size != self.page_size:
            raise ValueError(
                "Quest prefix cache requires prefix_cache_block_size == quest_chunk_size: "
                f"prefix_cache_block_size={self.prefix_cache_block_size}, quest_chunk_size={self.page_size}."
            )
        self.prefix_cache: RadixPrefixIndex | None = None
        if self.enable_prefix_caching:
            self.prefix_cache = RadixPrefixIndex(
                block_size=self.prefix_cache_block_size,
                fingerprint=build_prefix_cache_fingerprint(config, self.prefix_cache_block_size),
                max_blocks=config.prefix_cache_max_blocks,
            )
        self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.seq_id_to_cached_pages: dict[int, set[int]] = {}
        self._prefill_metadata_full_pages = False
        self._init_prefix_cache_runtime()

        # [2, L, P, H_kv, D] -> 0:max, 1:min
        self.metadata_cache = torch.empty(
            2,
            self.num_layers,
            self.num_pages,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device=self.device,
        )

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()

        # QuEST keeps one extra min/max page summary per physical page.
        effective_slot_bytes = int(slot_bytes_per_layer * (1.0 + 1.0 / self.page_size))
        total_token_slots = available_memory // (self.num_layers * effective_slot_bytes)
        total_token_slots = (total_token_slots // self.page_size) * self.page_size
        assert total_token_slots > 0, "Available memory is insufficient for QuEST paged KV cache"

        self.config.num_kvcache_slots = total_token_slots
        self.num_pages = total_token_slots // self.page_size

        self.kv_cache = torch.empty(
            2,
            self.num_layers,
            total_token_slots,
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
        return int(self._num_free_pages * self.page_size)

    def _prefix_evictable_slots(self) -> int:
        if self.prefix_cache is None:
            return 0
        return int(self.prefix_cache.freeable_blocks() * self.page_size)

    def _prefix_evictable_pages(self) -> int:
        if self.prefix_cache is None:
            return 0
        return int(self.prefix_cache.freeable_blocks())

    def _partial_page_free_slots(self) -> int:
        total = 0
        for row_idx in self.seq_id_to_row.values():
            row_len = int(self.row_seq_lens[row_idx])
            page_offset = row_len % self.page_size
            if page_offset:
                total += self.page_size - page_offset
        return total

    def prefill_step_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_evictable_slots() + self._partial_page_free_slots())

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        row_idx = self.seq_id_to_row.get(seq.seq_id)
        partial = 0
        if row_idx is not None:
            page_offset = int(self.row_seq_lens[row_idx]) % self.page_size
            if page_offset:
                partial = self.page_size - page_offset
        return int(self.num_free_slots + self._prefix_evictable_slots() + partial)

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        row_idx = self.seq_id_to_row.get(seq.seq_id)
        cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
        remaining = int(scheduled_tokens)
        cost = 0
        page_offset = cur_len % self.page_size
        if page_offset:
            take = min(remaining, self.page_size - page_offset)
            cost += take
            remaining -= take
        if remaining > 0:
            cost += self._ceil_to_page_slots(remaining)
        return int(cost)

    def decode_step_free_slots(self) -> int:
        partial_decode_slots = 0
        for row_idx in self.seq_id_to_row.values():
            if int(self.row_seq_lens[row_idx]) % self.page_size:
                partial_decode_slots += 1
        page_slots = (self._num_free_pages + self._prefix_evictable_pages()) * self.page_size
        return int(page_slots + partial_decode_slots)

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        if self._required_new_pages(seq.seq_id, 1) == 0:
            return 1
        return self.page_size if (self._num_free_pages + self._prefix_evictable_pages()) > 0 else 0

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        if self._required_new_pages(seq.seq_id, 1) == 0:
            return 1
        return self.page_size

    def prompt_admission_free_slots(self) -> int:
        return int(self.num_free_slots + self._prefix_evictable_slots())

    def _prefix_hit_evictable_slots(self, seq: Sequence) -> int:
        if self.prefix_cache is None or int(getattr(seq, "prefix_cache_hit_len", 0) or 0) <= 0:
            return 0
        if seq.prefix_cache_hit_last_block_id is None:
            raise RuntimeError(f"seq_id={seq.seq_id} has prefix hit length but no last block id.")
        chain = self.prefix_cache.get_chain(
            seq.prefix_cache_hit_last_block_id,
            int(seq.prefix_cache_hit_block_count),
        )
        freeable_block_ids = self.prefix_cache.freeable_block_ids()
        return sum(self.page_size for block in chain if block.stable_block_id in freeable_block_ids)

    def _ceil_to_page_slots(self, n_tokens: int) -> int:
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            return 0
        return ((n_tokens + self.page_size - 1) // self.page_size) * self.page_size

    def prompt_admission_cost(self, seq: Sequence) -> int:
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        suffix_len = int(seq.num_prompt_tokens - hit_len)
        return self._ceil_to_page_slots(suffix_len) + self._prefix_hit_evictable_slots(seq)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return int(self.prompt_admission_cost(seq))

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
                reserved += self._ceil_to_page_slots(remaining)
        return reserved

    def free_slot_stats(self) -> dict[str, int]:
        stats = {
            "free_slots": int(self.num_free_slots),
            "quest_free_pages": int(self._num_free_pages),
        }
        if self.prefix_cache is not None:
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

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self.clear_prefix_cache_hit(seq)
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if seq.num_prefilled_tokens != 0 or seq.num_completion_tokens != 0:
            return
        usable_tokens = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.page_size)
        if usable_tokens <= 0:
            return
        with profiler.record("quest_prefix_cache_lookup"):
            hit_len, last_block_id, hit_blocks = self.prefix_cache.lookup_longest_prefix(
                seq.prompt_token_ids,
                max_usable_tokens=usable_tokens,
            )
        if hit_len <= 0:
            return
        if last_block_id is None or hit_blocks <= 0:
            raise RuntimeError("Quest prefix cache lookup returned an invalid hit.")
        if hit_len >= seq.num_prompt_tokens or hit_len % self.page_size != 0:
            raise RuntimeError(
                "Quest prefix cache lookup returned an unusable hit length: "
                f"seq_id={seq.seq_id} hit_len={hit_len} prompt_len={seq.num_prompt_tokens} "
                f"page_size={self.page_size}."
            )
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = self.page_size
        seq.prefix_cache_method = "quest"

    def _free_prefix_cache_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        for block in blocks:
            payload = block.payload
            if not isinstance(payload, QuestPrefixBlockPayload):
                raise RuntimeError("Quest prefix cache block is missing block slot payload.")
            ptr = self._num_free_pages
            self.free_pages_stack[ptr] = int(payload.block_slot)
            self._num_free_pages += 1

    def _prefix_cache_materialization_subject(self) -> str:
        return "Quest prefix materialization"

    def _prefix_cache_negative_refcount_message(self) -> str:
        return "Quest prefix cache block ref_count became negative."

    def _prefix_cache_materialize_profile_name(self) -> str:
        return "quest_prefix_cache_materialize"

    def _validate_page_slots(self, slots: torch.Tensor, page_slot: int | None = None) -> int:
        if int(slots.numel()) != self.page_size:
            raise RuntimeError(
                f"Quest prefix block must contain exactly one full page: "
                f"num_slots={int(slots.numel())} page_size={self.page_size}."
            )
        slots_i32 = slots.to(dtype=torch.int32)
        page_slots = torch.div(slots_i32, self.page_size, rounding_mode="floor")
        page_offsets = torch.remainder(slots_i32, self.page_size)
        first_page_slot = int(page_slots[0].item())
        if page_slot is not None and int(page_slot) != first_page_slot:
            raise RuntimeError(
                f"Quest prefix block page_slot does not match token slots: "
                f"page_slot={page_slot} slots_page={first_page_slot}."
            )
        if hasattr(self, "num_pages") and not (0 <= first_page_slot < int(self.num_pages)):
            raise RuntimeError(
                f"Quest prefix block page_slot out of range: page_slot={first_page_slot} "
                f"num_pages={int(self.num_pages)}."
            )
        if not torch.all(page_slots == first_page_slot).item():
            raise RuntimeError("Quest prefix block token slots span multiple pages.")
        expected_offsets = self.page_offsets_i32.to(device=slots_i32.device)
        if not torch.equal(page_offsets, expected_offsets):
            raise RuntimeError("Quest prefix block token slots are not a contiguous full page.")
        return first_page_slot

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> QuestPrefixBlockPayload:
        return QuestPrefixBlockPayload(
            block_slot=self._validate_page_slots(slots),
            token_slots=slots,
        )

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        cached_pages = self.seq_id_to_cached_pages.setdefault(seq.seq_id, set())
        cached_pages.add(int(block.logical_block_idx))

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        if self.seq_id_to_row:
            raise RuntimeError("Cannot reset prefix cache while QuEST sequences are active.")
        self.free_pages_stack[: self.num_pages] = torch.arange(self.num_pages, dtype=torch.int32, device=self.device)
        self.free_pages_cpu_stack[: self.num_pages] = np.arange(self.num_pages, dtype=np.int32)
        self._num_free_pages = int(self.num_pages)
        self.seq_id_to_cached_pages.clear()

    def reset_after_warmup(self) -> None:
        if self.enable_prefix_caching and self.prefix_cache is not None:
            self.reset_prefix_cache()
            return
        self._reset_prefix_cache_allocator_after_clear()

    def _evict_prefix_cache_until_free(self, needed_slots: int) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        needed_slots = int(needed_slots)
        if self.num_free_slots >= needed_slots:
            return
        missing_slots = needed_slots - int(self.num_free_slots)
        needed_pages = (missing_slots + self.page_size - 1) // self.page_size
        with profiler.record("quest_prefix_cache_evict"):
            evicted = self.prefix_cache.evict_until_freeable(needed_pages)
        self._free_prefix_cache_blocks(evicted)

    def _evict_prefix_cache_for_insert(self, needed_blocks: int = 1) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        with profiler.record("quest_prefix_cache_evict"):
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
        with profiler.record("quest_prefix_cache_attach"):
            if seq.prefix_cache_hit_last_block_id is None:
                raise RuntimeError(f"seq_id={seq.seq_id} has Quest prefix hit length but no last block id.")
            if hit_len % self.page_size != 0:
                raise RuntimeError(
                    f"seq_id={seq.seq_id} Quest prefix hit length is not page aligned: "
                    f"hit_len={hit_len} page_size={self.page_size}."
                )
            chain = self.prefix_cache.get_chain(
                seq.prefix_cache_hit_last_block_id,
                int(seq.prefix_cache_hit_block_count),
            )
            if len(chain) * self.page_size != hit_len:
                raise RuntimeError(
                    "Quest prefix cache chain length does not match scheduler metadata: "
                    f"seq_id={seq.seq_id} hit_len={hit_len} blocks={len(chain)} page_size={self.page_size}."
                )
            row_idx = self._get_free_row(seq.seq_id)
            if int(self.row_seq_lens[row_idx]) != 0:
                raise RuntimeError(
                    f"Cannot attach Quest prefix cache to non-empty row: seq_id={seq.seq_id} "
                    f"row_idx={row_idx} row_len={int(self.row_seq_lens[row_idx])}."
                )

            cached_pages = self.seq_id_to_cached_pages.setdefault(seq.seq_id, set())
            for block in chain:
                payload = block.payload
                if not isinstance(payload, QuestPrefixBlockPayload):
                    raise RuntimeError(
                        f"Invalid Quest prefix cache block page for seq_id={seq.seq_id}: "
                        f"logical_block_idx={block.logical_block_idx}."
                    )
                page_idx = int(block.logical_block_idx)
                start = page_idx * self.page_size
                end = start + self.page_size
                page_slot = int(payload.block_slot)
                self.buffer_req_to_page_slots[row_idx, page_idx] = page_slot
                self._validate_page_slots(payload.token_slots, page_slot)
                slots = payload.token_slots
                self.buffer_req_to_token_slots[row_idx, start:end] = slots
                block.ref_count += 1
                cached_pages.add(page_idx)

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

    def _allocate_new_page(self, row_idx: int, page_idx: int) -> int:
        if self._num_free_pages <= 0:
            raise RuntimeError("Out of QuEST KV pages")
        ptr = self._num_free_pages
        page_slot = int(self.free_pages_stack[ptr - 1].item())
        self._num_free_pages -= 1
        self.buffer_req_to_page_slots[row_idx, page_idx] = page_slot
        return page_slot

    def _required_new_pages(self, seq_id: int, size: int) -> int:
        row_idx = self.seq_id_to_row.get(seq_id)
        cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
        before_pages = (cur_len + self.page_size - 1) // self.page_size
        after_len = cur_len + int(size)
        after_pages = (after_len + self.page_size - 1) // self.page_size
        return max(0, after_pages - before_pages)

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            size = int(size)
            needed_pages = self._required_new_pages(seq_id, size)
            if needed_pages > 0:
                self._evict_prefix_cache_until_free(needed_pages * self.page_size)
            assert self._num_free_pages >= needed_pages, (
                f"Out of QuEST KV pages: need_pages={needed_pages}, free_pages={self._num_free_pages}, "
                f"size={size}, free_slots={self.num_free_slots}"
            )

            row_idx = self._get_free_row(seq_id)
            cur_len = int(self.row_seq_lens[row_idx])
            if cur_len + size > int(self.max_model_len):
                raise RuntimeError(
                    "KV row length exceeds max_model_len in QuEST _allocate: "
                    f"seq_id={seq_id} row={row_idx} cur_len={cur_len} size={size} "
                    f"max_model_len={int(self.max_model_len)}"
                )

            if needed_pages > 0:
                first_new_page = (cur_len + self.page_size - 1) // self.page_size
                ptr = self._num_free_pages
                if self.enable_prefix_caching:
                    new_page_slots = self.free_pages_stack[ptr - needed_pages : ptr].flip(0)
                else:
                    new_page_slots_cpu = self.free_pages_cpu_stack[ptr - needed_pages : ptr][::-1].copy()
                    new_page_slots = torch.from_numpy(new_page_slots_cpu).to(
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.buffer_req_to_page_slots_cpu[
                        row_idx,
                        first_new_page : first_new_page + needed_pages,
                    ] = new_page_slots_cpu
                self._num_free_pages -= needed_pages
                self.buffer_req_to_page_slots[
                    row_idx,
                    first_new_page : first_new_page + needed_pages,
                ] = new_page_slots

            positions = torch.arange(cur_len, cur_len + size, dtype=torch.int64, device=self.device)
            page_indices = torch.div(positions, int(self.page_size), rounding_mode="floor")
            page_offsets = torch.remainder(positions, int(self.page_size)).to(torch.int32)
            page_slots = self.buffer_req_to_page_slots[row_idx, page_indices]
            allocated_slots = page_slots * int(self.page_size) + page_offsets
            self.buffer_req_to_token_slots[row_idx, cur_len: cur_len + size] = allocated_slots
            self.row_seq_lens[row_idx] += size
            return allocated_slots

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        with profiler.record("cache_allocate"):
            batch_size = len(seq_ids)
            row_indices = np.asarray([self._get_free_row(seq_id) for seq_id in seq_ids], dtype=np.int64)
            cur_lens = self.row_seq_lens[row_indices]
            if len(cur_lens) > 0 and int(max(cur_lens)) + 1 > int(self.max_model_len):
                raise RuntimeError(
                    "KV row length exceeds max_model_len in QuEST _allocate_batch: "
                    f"max_cur_len={int(max(cur_lens))} max_model_len={int(self.max_model_len)}"
                )

            page_indices = cur_lens // self.page_size
            page_offsets = cur_lens % self.page_size
            new_page_positions = np.nonzero(page_offsets == 0)[0]
            needed_pages = int(new_page_positions.size)
            if needed_pages > 0:
                self._evict_prefix_cache_until_free(needed_pages * self.page_size)
            assert self._num_free_pages >= needed_pages, (
                f"Out of QuEST KV pages: need_pages={needed_pages}, free_pages={self._num_free_pages}, "
                f"size={batch_size}, free_slots={self.num_free_slots}"
            )

            rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
            page_indices_gpu = torch.tensor(page_indices, dtype=torch.long, device=self.device)
            if needed_pages > 0:
                ptr = self._num_free_pages
                if self.enable_prefix_caching:
                    new_page_slots = self.free_pages_stack[ptr - needed_pages : ptr].flip(0)
                else:
                    new_page_slots_cpu = self.free_pages_cpu_stack[ptr - needed_pages : ptr][::-1].copy()
                    new_page_slots = torch.from_numpy(new_page_slots_cpu).to(
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.buffer_req_to_page_slots_cpu[
                        row_indices[new_page_positions],
                        page_indices[new_page_positions],
                    ] = new_page_slots_cpu
                self._num_free_pages -= needed_pages
                new_pos_gpu = torch.tensor(new_page_positions, dtype=torch.long, device=self.device)
                self.buffer_req_to_page_slots[
                    rows_gpu.index_select(0, new_pos_gpu),
                    page_indices_gpu.index_select(0, new_pos_gpu),
                ] = new_page_slots

            cur_lens_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
            if needed_pages == 0:
                allocated_slots = self.buffer_req_to_token_slots[rows_gpu, cur_lens_gpu - 1] + 1
            elif self.enable_prefix_caching:
                page_offsets_gpu = torch.tensor(page_offsets, dtype=torch.int32, device=self.device)
                page_slots = self.buffer_req_to_page_slots[rows_gpu, page_indices_gpu]
                allocated_slots = page_slots * int(self.page_size) + page_offsets_gpu
            else:
                page_slots_cpu = self.buffer_req_to_page_slots_cpu[row_indices, page_indices]
                allocated_slots_cpu = page_slots_cpu * int(self.page_size) + page_offsets
                allocated_slots = torch.from_numpy(allocated_slots_cpu.astype(np.int32, copy=False)).to(
                    device=self.device,
                    dtype=torch.int32,
                )
            self.buffer_req_to_token_slots[rows_gpu, cur_lens_gpu] = allocated_slots
            self.row_seq_lens[row_indices] += 1
            return allocated_slots.to(torch.int32)

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            cur_len = int(self.row_seq_lens[row_idx])
            num_pages = (cur_len + self.page_size - 1) // self.page_size
            cached_pages = self.seq_id_to_cached_pages.pop(seq_id, set())
            if num_pages > 0:
                free_page_slots = [
                    int(self.buffer_req_to_page_slots_cpu[row_idx, page_idx])
                    if not self.enable_prefix_caching
                    else int(self.buffer_req_to_page_slots[row_idx, page_idx].item())
                    for page_idx in range(num_pages)
                    if page_idx not in cached_pages
                ]
                if free_page_slots:
                    if not self.enable_prefix_caching:
                        self.free_pages_cpu_stack[
                            self._num_free_pages : self._num_free_pages + len(free_page_slots)
                        ] = np.asarray(free_page_slots, dtype=np.int32)
                    page_slots = torch.tensor(free_page_slots, dtype=torch.int32, device=self.free_pages_stack.device)
                    ptr = self._num_free_pages
                    self.free_pages_stack[ptr: ptr + len(free_page_slots)] = page_slots
                    self._num_free_pages += len(free_page_slots)
            self._release_prefix_blocks(self.seq_id_to_prefix_blocks.pop(seq_id, []))
            self._release_prefix_blocks(self.seq_id_to_materialized_blocks.pop(seq_id, []))
            self.prefix_runtime_states.pop(seq_id, None)
            self.pending_prefix_blocks.pop(seq_id, None)

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.buffer_req_to_page_slots[row_idx, :] = -1
            self.buffer_req_to_page_slots_cpu[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.free_rows.append(row_idx)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("QuEST does not physically evict token slots")

    @torch.no_grad()
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
            metadata_full_pages = True

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size
                metadata_full_pages = metadata_full_pages and (
                    int(start_idx) % int(self.page_size) == 0
                    and int(chunk_size) > 0
                    and int(chunk_size) % int(self.page_size) == 0
                )

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
            self.layer_batch_state.req_indices = req_indices_tensor
            self._prefill_metadata_full_pages = bool(metadata_full_pages)

            input_ids = torch.from_numpy(input_ids_np).to(self.device)
            positions = torch.from_numpy(positions_np).to(self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    @torch.no_grad()
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
            self.layer_batch_state.req_indices = req_indices

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            return input_ids, positions, None

    @torch.no_grad()
    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        """Prepare decode metadata into caller-owned static CUDA buffers."""
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
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))
                slot_mapping[real_batch_size:].fill_(-1)
                context_lens[real_batch_size:].fill_(int(real_context_lens[0]))
                req_indices[real_batch_size:].fill_(int(row_indices[0]))

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.max_context_len = int(max(real_context_lens)) if row_indices else 0
            self.layer_batch_state.req_indices = req_indices

            return input_ids, positions, None

    @torch.no_grad()
    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        if not get_context().is_prefill:
            self._on_kv_stored_decode_static(layer_idx, slot_mapping)
            return
        if self._is_stream_capturing():
            self._on_kv_stored_prefill_capture(layer_idx, k, slot_mapping)
            return

        with profiler.record("quest_update_metadata"):
            if self._prefill_metadata_full_pages:
                page_max_cache = self.metadata_cache[0, layer_idx]
                page_min_cache = self.metadata_cache[1, layer_idx]
                full_page_slots = torch.div(
                    slot_mapping[:: self.page_size],
                    self.page_size,
                    rounding_mode="floor",
                ).to(torch.long)
                full_page_k = k.view(
                    -1,
                    self.page_size,
                    self.num_kv_heads,
                    self.head_dim,
                )
                page_min, page_max = torch.aminmax(full_page_k, dim=1)
                page_max_cache.index_copy_(0, full_page_slots, page_max)
                page_min_cache.index_copy_(0, full_page_slots, page_min)
                return

            page_slots = torch.div(slot_mapping, self.page_size, rounding_mode="floor")
            page_offsets = torch.remainder(slot_mapping, self.page_size)
            unique_pages, counts = torch.unique_consecutive(page_slots, return_counts=True)
            page_max_cache = self.metadata_cache[0, layer_idx]
            page_min_cache = self.metadata_cache[1, layer_idx]
            k_cache = self.kv_cache[0, layer_idx]
            run_starts = counts.cumsum(0) - counts
            start_offsets = page_offsets.index_select(0, run_starts)
            end_offsets = start_offsets + counts

            full_page_mask = (start_offsets == 0) & (counts == self.page_size)
            if full_page_mask.any():
                full_run_starts = run_starts[full_page_mask].to(torch.int64)
                full_page_slots = unique_pages[full_page_mask].to(torch.int64)
                full_token_indices = full_run_starts[:, None] + self.page_offsets_i64[None, :]
                full_page_k = k.index_select(0, full_token_indices.reshape(-1)).view(
                    -1,
                    self.page_size,
                    self.num_kv_heads,
                    self.head_dim,
                )
                page_min, page_max = torch.aminmax(full_page_k, dim=1)
                page_max_cache.index_copy_(0, full_page_slots, page_max)
                page_min_cache.index_copy_(0, full_page_slots, page_min)

            completed_page_mask = (end_offsets == self.page_size) & (~full_page_mask)
            if completed_page_mask.any():
                completed_page_slots = unique_pages[completed_page_mask].to(torch.int64)
                page_token_indices = completed_page_slots[:, None] * self.page_size + self.page_offsets_i64[None, :]
                full_page_k = k_cache.index_select(0, page_token_indices.reshape(-1)).view(
                    -1,
                    self.page_size,
                    self.num_kv_heads,
                    self.head_dim,
                )
                page_min, page_max = torch.aminmax(full_page_k, dim=1)
                page_max_cache.index_copy_(0, completed_page_slots, page_max)
                page_min_cache.index_copy_(0, completed_page_slots, page_min)

    @torch.no_grad()
    def _on_kv_stored_prefill_capture(self, layer_idx: int, k: torch.Tensor, slot_mapping: torch.Tensor):
        """Update QuEST full-page metadata without dynamic shape ops during capture.

        Prefill graph capture is currently exercised for first-prefill chunks, so
        touched pages begin at page offset 0. The trailing partial page is left
        without metadata, matching the eager path until that page is completed.
        """
        with profiler.record("quest_update_metadata_capture"):
            full_token_count = (int(slot_mapping.numel()) // self.page_size) * self.page_size
            if full_token_count <= 0:
                return

            page_max_cache = self.metadata_cache[0, layer_idx]
            page_min_cache = self.metadata_cache[1, layer_idx]
            full_page_slots = torch.div(
                slot_mapping[:full_token_count:self.page_size],
                self.page_size,
                rounding_mode="floor",
            ).to(torch.long)
            full_page_k = k[:full_token_count].view(
                -1,
                self.page_size,
                self.num_kv_heads,
                self.head_dim,
            )
            page_min, page_max = torch.aminmax(full_page_k, dim=1)
            page_max_cache.index_copy_(0, full_page_slots, page_max)
            page_min_cache.index_copy_(0, full_page_slots, page_min)

    @torch.no_grad()
    def _on_kv_stored_decode_static(self, layer_idx: int, slot_mapping: torch.Tensor):
        with profiler.record("quest_update_metadata_static"):
            page_max_cache = self.metadata_cache[0, layer_idx]
            page_min_cache = self.metadata_cache[1, layer_idx]
            k_cache = self.kv_cache[0, layer_idx]

            valid = slot_mapping >= 0
            safe_slots = slot_mapping.clamp_min(0)
            page_slots = torch.div(safe_slots, self.page_size, rounding_mode="floor").to(torch.long)
            page_offsets = torch.remainder(safe_slots, self.page_size)
            completed = valid & (page_offsets == self.page_size - 1)

            page_token_indices = page_slots[:, None] * self.page_size + self.page_offsets_i64[None, :]
            full_page_k = k_cache.index_select(0, page_token_indices.reshape(-1)).view(
                slot_mapping.numel(),
                self.page_size,
                self.num_kv_heads,
                self.head_dim,
            )
            page_max = full_page_k.amax(dim=1)
            page_min = full_page_k.amin(dim=1)

            max_src = torch.where(
                completed[:, None, None],
                page_max,
                torch.full_like(page_max, -float("inf")),
            )
            min_src = torch.where(
                completed[:, None, None],
                page_min,
                torch.full_like(page_min, float("inf")),
            )
            dst = page_slots[:, None, None].expand(-1, self.num_kv_heads, self.head_dim)
            page_max_cache.scatter_reduce_(0, dst, max_src, reduce="amax", include_self=True)
            page_min_cache.scatter_reduce_(0, dst, min_src, reduce="amin", include_self=True)

    @staticmethod
    def _score_pages_batched(
        q_heads: torch.Tensor,
        page_max: torch.Tensor,
        page_min: torch.Tensor,
        num_kv_heads: int,
    ) -> torch.Tensor:
        batch_size, num_heads, head_dim = q_heads.shape
        q_dtype = page_max.dtype
        if num_heads == num_kv_heads:
            num_pages = page_max.shape[2]
            q_heads = q_heads.to(q_dtype)
            q_pos = q_heads.clamp_min(0).reshape(batch_size * num_heads, 1, head_dim)
            q_neg = q_heads.clamp_max(0).reshape(batch_size * num_heads, 1, head_dim)
            page_max_t = page_max.reshape(batch_size * num_heads, num_pages, head_dim).transpose(1, 2)
            page_min_t = page_min.reshape(batch_size * num_heads, num_pages, head_dim).transpose(1, 2)
            page_scores = torch.bmm(q_pos, page_max_t).squeeze(1)
            page_scores += torch.bmm(q_neg, page_min_t).squeeze(1)
            return page_scores.view(batch_size, num_heads, num_pages).amax(dim=1)

        group_size = num_heads // num_kv_heads
        num_pages = page_max.shape[2]
        q_grouped = q_heads.view(batch_size, num_kv_heads, group_size, head_dim).to(q_dtype)
        q_pos = q_grouped.clamp_min(0).reshape(batch_size * num_kv_heads, group_size, head_dim)
        q_neg = q_grouped.clamp_max(0).reshape(batch_size * num_kv_heads, group_size, head_dim)
        page_max_t = page_max.reshape(batch_size * num_kv_heads, num_pages, head_dim).transpose(1, 2)
        page_min_t = page_min.reshape(batch_size * num_kv_heads, num_pages, head_dim).transpose(1, 2)
        page_scores = torch.bmm(q_pos, page_max_t)
        page_scores += torch.bmm(q_neg, page_min_t)
        return page_scores.view(batch_size, num_kv_heads, group_size, num_pages).amax(dim=2).amax(dim=1)

    @torch.no_grad()
    def build_decode_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx < self.config.quest_skip_layers:
            return active_slots, req_indices, context_lens

        token_budget = int(self.config.quest_token_budget)
        if token_budget <= 0:
            return active_slots, req_indices, context_lens
        return self._build_decode_view_static(
            layer_idx,
            q,
            active_slots,
            req_indices,
            context_lens,
            token_budget=token_budget,
            num_kv_heads=num_kv_heads,
        )

    @torch.no_grad()
    def _build_decode_view_static(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        token_budget: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with profiler.record("quest_build_decode_view_static"):
            page_budget_base = max(3, int(token_budget) // self.page_size)
            max_keep = max(int(token_budget), page_budget_base * self.page_size, self.page_size)
            max_context_len = self.layer_batch_state.max_context_len
            if max_context_len is None:
                raise RuntimeError("QuEST decode CUDA graph requires max_context_len to be pinned.")
            max_context_len = int(max_context_len)
            if max_context_len <= max_keep:
                return active_slots, req_indices, context_lens

            batch_size = q.shape[0]
            max_pages = min(
                self.max_pages_per_row,
                (max_context_len + self.page_size - 1) // self.page_size,
            )
            prev_budget = min(page_budget_base - 1, max_pages - 1)
            if prev_budget <= 0:
                return active_slots, req_indices, context_lens

            num_pages = torch.div(context_lens + self.page_size - 1, self.page_size, rounding_mode="floor")
            is_long_text = bool(get_context().is_long_text)
            if not is_long_text:
                dense_slots = self.buffer_req_to_token_slots.index_select(0, req_indices.to(torch.long))[:, :max_keep]
                dense_mask = (context_lens <= int(token_budget)) | (num_pages <= page_budget_base)

            row_page_slots = self.buffer_req_to_page_slots.index_select(0, req_indices.to(torch.long))[:, :max_pages]
            prev_page_slots = row_page_slots[:, : max_pages - 1].to(torch.long)
            safe_prev_page_slots = prev_page_slots.clamp_min_(0)

            prev_page_max = self.metadata_cache[0, layer_idx].index_select(
                0,
                safe_prev_page_slots.reshape(-1),
            ).view(batch_size, max_pages - 1, num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            prev_page_min = self.metadata_cache[1, layer_idx].index_select(
                0,
                safe_prev_page_slots.reshape(-1),
            ).view(batch_size, max_pages - 1, num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            page_scores = self._score_pages_batched(q, prev_page_max, prev_page_min, num_kv_heads)
            safe_num_pages = num_pages.to(torch.long).clamp_min(1)
            valid_prev = torch.arange(max_pages - 1, device=q.device)[None, :] < (safe_num_pages - 1)[:, None]
            page_scores.masked_fill_(~valid_prev, -float("inf"))

            top_prev = page_scores.topk(prev_budget, dim=-1, sorted=False).indices
            last_page = (safe_num_pages - 1)[:, None]
            selected_pages = torch.cat((top_prev, last_page), dim=1)
            selected_page_slots = row_page_slots.gather(1, selected_pages)
            sparse_slots = (
                selected_page_slots[:, :, None] * self.page_size + self.page_offsets_i32[None, None, :]
            ).reshape(batch_size, -1)

            sparse_keep = int(sparse_slots.shape[1])

            last_page_len = context_lens - (num_pages - 1) * self.page_size
            sparse_lens = (prev_budget * self.page_size + last_page_len).to(torch.int32)
            if is_long_text:
                packed_slots = sparse_slots
                local_context_lens = sparse_lens
            else:
                packed_slots = torch.empty((batch_size, max_keep), dtype=torch.int32, device=q.device)
                packed_slots[:, :sparse_keep] = torch.where(
                    dense_mask[:, None],
                    dense_slots[:, :sparse_keep],
                    sparse_slots,
                )
                if max_keep > sparse_keep:
                    packed_slots[:, sparse_keep:] = dense_slots[:, sparse_keep:]
                local_context_lens = torch.where(dense_mask, context_lens, sparse_lens)
            local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=q.device)
            return packed_slots, local_req_indices, local_context_lens
