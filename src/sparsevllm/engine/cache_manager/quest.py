from __future__ import annotations

from collections import deque

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates


class QuestCacheManager(CacheManager):
    """Paged KV cache + page metadata cache for QuEST."""

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.page_size = int(config.quest_chunk_size)
        self.max_pages_per_row = (self.max_model_len + self.page_size - 1) // self.page_size
        self.page_offsets_i32 = torch.arange(self.page_size, dtype=torch.int32, device="cuda")
        self.page_offsets_i64 = self.page_offsets_i32.to(torch.int64)

        self.allocate_kv_cache()

        self.free_pages_stack = torch.arange(self.num_pages, dtype=torch.int32, device="cuda")
        self._num_free_pages = self.num_pages

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device="cuda"
        )
        self.buffer_req_to_page_slots = torch.full(
            (self.max_buffer_rows, self.max_pages_per_row), -1, dtype=torch.int32, device="cuda"
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.layer_batch_state = LayerBatchStates()

        # [2, L, P, H_kv, D] -> 0:max, 1:min
        self.metadata_cache = torch.empty(
            2,
            self.num_layers,
            self.num_pages,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
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
            device="cuda",
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx], self.layer_batch_state.slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        return self.buffer_req_to_token_slots

    @property
    def num_free_slots(self) -> int:
        return int(self._num_free_pages * self.page_size)

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

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            assert self.num_free_slots >= size, (
                f"Out of QuEST KV slots: need {size}, free {self.num_free_slots}"
            )

            row_idx = self._get_free_row(seq_id)
            cur_len = int(self.row_seq_lens[row_idx])
            remaining = int(size)
            next_pos = cur_len
            allocated_parts: list[torch.Tensor] = []

            while remaining > 0:
                page_idx = next_pos // self.page_size
                page_offset = next_pos % self.page_size
                if page_offset == 0:
                    page_slot = self._allocate_new_page(row_idx, page_idx)
                else:
                    page_slot = int(self.buffer_req_to_page_slots[row_idx, page_idx].item())

                take = min(remaining, self.page_size - page_offset)
                token_offsets = self.page_offsets_i32[page_offset: page_offset + take]
                allocated_parts.append(page_slot * self.page_size + token_offsets)

                next_pos += take
                remaining -= take

            allocated_slots = torch.cat(allocated_parts, dim=0)
            self.buffer_req_to_token_slots[row_idx, cur_len: cur_len + size] = allocated_slots
            self.row_seq_lens[row_idx] += size
            return allocated_slots

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        slots = [self._allocate(seq_id, 1) for seq_id in seq_ids]
        return torch.cat(slots, dim=0)

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            cur_len = int(self.row_seq_lens[row_idx])
            num_pages = (cur_len + self.page_size - 1) // self.page_size
            if num_pages > 0:
                page_slots = self.buffer_req_to_page_slots[row_idx, :num_pages]
                ptr = self._num_free_pages
                self.free_pages_stack[ptr: ptr + num_pages] = page_slots
                self._num_free_pages += num_pages

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.buffer_req_to_page_slots[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.free_rows.append(row_idx)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("QuEST does not physically evict token slots")

    @torch.no_grad()
    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device="cuda")
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

                self._allocate(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                slot_mapping[token_offset: token_offset + chunk_size] = self.buffer_req_to_token_slots[row_idx, start_idx:end_idx]
                context_lens_list.append(end_idx)
                req_indices.append(row_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device="cuda")

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.req_indices = req_indices_tensor

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
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
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device="cuda",
            )
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device="cuda")

            slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
            slot_mapping[:] = new_slots_batch

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.req_indices = req_indices

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

    @torch.no_grad()
    def on_kv_stored(self, layer_idx: int, k: torch.Tensor, slot_mapping: torch.Tensor):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        with profiler.record("quest_update_metadata"):
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
                page_max_cache.index_copy_(0, full_page_slots, full_page_k.amax(dim=1))
                page_min_cache.index_copy_(0, full_page_slots, full_page_k.amin(dim=1))

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
                page_max_cache.index_copy_(0, completed_page_slots, full_page_k.amax(dim=1))
                page_min_cache.index_copy_(0, completed_page_slots, full_page_k.amin(dim=1))

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
            return page_scores.view(batch_size, num_heads, num_pages).amax(dim=1).float()

        group_size = num_heads // num_kv_heads
        num_pages = page_max.shape[2]
        q_grouped = q_heads.view(batch_size, num_kv_heads, group_size, head_dim).to(q_dtype)
        q_pos = q_grouped.clamp_min(0).reshape(batch_size * num_kv_heads, group_size, head_dim)
        q_neg = q_grouped.clamp_max(0).reshape(batch_size * num_kv_heads, group_size, head_dim)
        page_max_t = page_max.reshape(batch_size * num_kv_heads, num_pages, head_dim).transpose(1, 2)
        page_min_t = page_min.reshape(batch_size * num_kv_heads, num_pages, head_dim).transpose(1, 2)
        page_scores = torch.bmm(q_pos, page_max_t)
        page_scores += torch.bmm(q_neg, page_min_t)
        return page_scores.view(batch_size, num_kv_heads, group_size, num_pages).amax(dim=2).amax(dim=1).float()

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

        with profiler.record("quest_build_decode_view"):
            page_budget_base = max(3, token_budget // self.page_size)
            max_keep = max(token_budget, page_budget_base * self.page_size, self.page_size)
            batch_size = q.shape[0]
            packed_slots = torch.empty((batch_size, max_keep), dtype=torch.int32, device=q.device)
            local_context_lens = torch.empty((batch_size,), dtype=torch.int32, device=q.device)

            num_pages = torch.div(context_lens + self.page_size - 1, self.page_size, rounding_mode="floor")
            dense_mask = (context_lens <= token_budget) | (num_pages <= page_budget_base)

            dense_idx = dense_mask.nonzero(as_tuple=False).squeeze(-1)
            if dense_idx.numel() > 0:
                dense_req = req_indices.index_select(0, dense_idx).to(torch.long)
                dense_lens = context_lens.index_select(0, dense_idx)
                dense_keep = int(dense_lens.max().item())
                packed_slots[dense_idx, :dense_keep] = self.buffer_req_to_token_slots.index_select(0, dense_req)[:, :dense_keep]
                local_context_lens[dense_idx] = dense_lens

            sparse_idx = (~dense_mask).nonzero(as_tuple=False).squeeze(-1)
            if sparse_idx.numel() > 0:
                sparse_num_pages = num_pages.index_select(0, sparse_idx)
                for num_pages_i32 in torch.unique(sparse_num_pages, sorted=True):
                    num_pages_i = int(num_pages_i32.item())
                    group_mask = sparse_num_pages == num_pages_i32
                    group_idx = sparse_idx[group_mask]
                    group_req = req_indices.index_select(0, group_idx).to(torch.long)
                    group_q = q.index_select(0, group_idx)
                    row_page_slots = self.buffer_req_to_page_slots.index_select(0, group_req)[:, :num_pages_i].to(torch.long)
                    prev_page_slots = row_page_slots[:, : num_pages_i - 1]

                    with profiler.record("quest_score_pages"):
                        flat_prev_slots = prev_page_slots.reshape(-1)
                        prev_page_max = self.metadata_cache[0, layer_idx].index_select(0, flat_prev_slots).view(
                            group_idx.numel(),
                            num_pages_i - 1,
                            num_kv_heads,
                            self.head_dim,
                        ).permute(0, 2, 1, 3)
                        prev_page_min = self.metadata_cache[1, layer_idx].index_select(0, flat_prev_slots).view(
                            group_idx.numel(),
                            num_pages_i - 1,
                            num_kv_heads,
                            self.head_dim,
                        ).permute(0, 2, 1, 3)
                        page_scores = self._score_pages_batched(group_q, prev_page_max, prev_page_min, num_kv_heads)

                    prev_budget = min(page_budget_base - 1, num_pages_i - 1)
                    top_prev = page_scores.topk(prev_budget, dim=-1, sorted=False).indices
                    last_page = torch.full((group_idx.numel(), 1), num_pages_i - 1, dtype=torch.long, device=q.device)
                    selected_pages = torch.cat((top_prev, last_page), dim=1)

                    with profiler.record("quest_pack_slots"):
                        selected_page_slots = row_page_slots.gather(1, selected_pages).to(torch.int32)
                        group_slots = (
                            selected_page_slots[:, :, None] * self.page_size + self.page_offsets_i32[None, None, :]
                        ).reshape(group_idx.numel(), -1)
                        keep_len = prev_budget * self.page_size + (
                            context_lens.index_select(0, group_idx) - (num_pages_i - 1) * self.page_size
                        )
                        packed_slots[group_idx, : group_slots.shape[1]] = group_slots
                        local_context_lens[group_idx] = keep_len

            local_req_indices = torch.arange(q.shape[0], dtype=torch.int32, device=q.device)
            return packed_slots, local_req_indices, local_context_lens
