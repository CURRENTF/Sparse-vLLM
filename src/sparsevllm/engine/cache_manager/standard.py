from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates


class StandardCacheManager(CacheManager):

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.allocate_kv_cache()

        num_slots = config.num_kvcache_slots
        self.free_slots_stack = torch.arange(num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots = num_slots

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device="cuda"
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.layer_batch_state = LayerBatchStates()

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
        return self._num_free_slots

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
        assert self._num_free_slots >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots}"
        )

        row_indices = [self._get_free_row(sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots
        select_indices = self.free_slots_stack[ptr - batch_size: ptr]
        self._num_free_slots -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
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
            slots = self.buffer_req_to_token_slots[row_idx, :cur_len]

            assert cur_len > 0
            before_free = self._num_free_slots
            ptr = self._num_free_slots
            self.free_slots_stack[ptr: ptr + cur_len] = slots
            self._num_free_slots += cur_len
            after_free = self._num_free_slots

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.row_seq_lens[row_idx] = 0
            self.free_rows.append(row_idx)

            if debug_slots:
                logger.info(
                    "free_seq seq_id={} row_idx={} freed_tokens={} free_slots_before={} free_slots_after={}",
                    seq_id,
                    row_idx,
                    int(cur_len),
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

            if log_level == 'DEBUG':
                logger.debug(f'{context_lens_list=}   {req_indices=}  {slot_mapping[:10].tolist()=}  {slot_mapping[-10:].tolist()=}')

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

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

            logger.debug(f'{slot_mapping=}   {context_lens.tolist()=}  {slot_mapping[:10]=}  {slot_mapping[-10:]=}')

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None
