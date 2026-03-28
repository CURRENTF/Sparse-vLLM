from __future__ import annotations

from collections import deque

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates


class SnapKVCacheManager(CacheManager):
    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.allocate_kv_cache()

        self.layer_num_slots = []
        self.free_slots_stack = []
        self._num_free_slots = []
        self.buffer_req_to_token_slots = []
        self.seq_id_to_row = []
        self.free_rows = []
        self.row_seq_lens = []
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]

        for layer_id in range(self.num_layers):
            num_slots = (
                config.num_kvcache_slots[layer_id]
                if isinstance(config.num_kvcache_slots, list)
                else config.num_kvcache_slots
            )
            self.layer_num_slots.append(num_slots)
            self.free_slots_stack.append(
                torch.arange(num_slots, dtype=torch.int32, device="cuda")
            )
            self._num_free_slots.append(num_slots)
            self.buffer_req_to_token_slots.append(
                torch.zeros(
                    (self.max_buffer_rows, self.max_model_len),
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            self.seq_id_to_row.append({})
            self.free_rows.append(deque(range(self.max_buffer_rows)))
            self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        num_layers = self.num_layers

        if config.pyramid_layer_ratios is not None:
            # PyramidKV: 根据比例分配每层不同大小的 cache
            total_ratio = sum(config.pyramid_layer_ratios)
            base_slots = available_memory // (slot_bytes_per_layer * total_ratio)
            assert base_slots > 0, "可用显存不足以分配 KV Cache"

            layer_slots = [int(base_slots * ratio) for ratio in config.pyramid_layer_ratios]
            assert layer_slots[0] == max(layer_slots), (
                f"Layer 0 必须是最胖层，但 layer_slots[0]={layer_slots[0]}, max={max(layer_slots)}"
            )

            self.kv_cache = []
            for layer_idx in range(num_layers):
                num_slots = layer_slots[layer_idx]
                k_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.torch_dtype, device="cuda"
                )
                v_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.torch_dtype, device="cuda"
                )
                self.kv_cache.append((k_cache, v_cache))

            config.num_kvcache_slots = layer_slots
            logger.info(f"PyramidKV: Layer slots = {layer_slots}, base_slots = {base_slots}")
        else:
            # 标准模式：所有层使用相同大小
            slot_bytes = num_layers * slot_bytes_per_layer
            config.num_kvcache_slots = available_memory // slot_bytes
            assert config.num_kvcache_slots > 0, "可用显存不足以分配 KV Cache"

            logger.info(
                f"Standard Mode (SnapKV): Each layer can accommodate {config.num_kvcache_slots} tokens."
            )
            self.kv_cache = torch.empty(
                2,
                num_layers,
                config.num_kvcache_slots,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.hf_config.torch_dtype,
                device="cuda",
            )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.kv_cache, list):
            return self.kv_cache[layer_idx]
        elif isinstance(self.kv_cache, torch.Tensor):
            return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]
        else:
            raise ValueError

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        return min(self._num_free_slots)

    def prefill_batched_tokens_margin(self) -> int:
        # Keep headroom for the "window" tokens used by SnapKV/PyramidKV logic.
        return int(self.config.snapkv_window_size)

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        window = int(self.config.snapkv_window_size)
        if window > 0 and remaining > window:
            return remaining - window
        return remaining

    def _get_free_row(self, layer_idx: int, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row[layer_idx]:
            return self.seq_id_to_row[layer_idx][seq_id]
        if not self.free_rows[layer_idx]:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows[layer_idx].popleft()
        self.seq_id_to_row[layer_idx][seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, layer_idx: int, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            assert self._num_free_slots[layer_idx] >= size, (
                f"Out of KV cache slots: need {size}, free {self._num_free_slots[layer_idx]}"
            )

            row_idx = self._get_free_row(layer_idx, seq_id)
            cur_len = self.row_seq_lens[layer_idx][row_idx]

            ptr = self._num_free_slots[layer_idx]
            select_index = self.free_slots_stack[layer_idx][ptr - size: ptr]
            self._num_free_slots[layer_idx] -= size

            self.buffer_req_to_token_slots[layer_idx][row_idx, cur_len: cur_len + size] = select_index
            self.row_seq_lens[layer_idx][row_idx] += size

            return select_index

    @torch.no_grad()
    def _allocate_batch(self, layer_idx: int, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots[layer_idx] >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots[layer_idx]}"
        )

        row_indices = [self._get_free_row(layer_idx, sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[layer_idx][row_indices]

        ptr = self._num_free_slots[layer_idx]
        select_indices = self.free_slots_stack[layer_idx][ptr - batch_size: ptr]
        self._num_free_slots[layer_idx] -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, cols_gpu] = select_indices.to(torch.int32)
        self.row_seq_lens[layer_idx][row_indices] += 1

        return select_indices

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            for layer_idx in range(self.num_layers):
                row_idx = self.seq_id_to_row[layer_idx].pop(seq_id, None)
                if row_idx is None:
                    raise ValueError

                cur_len = self.row_seq_lens[layer_idx][row_idx]
                slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len]

                if cur_len > 0:
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + cur_len] = slots
                    self._num_free_slots[layer_idx] += cur_len

                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
                self.row_seq_lens[layer_idx][row_idx] = 0
                self.free_rows[layer_idx].append(row_idx)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        if keep_indices is None:
            return

        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise ValueError

        cur_len = self.row_seq_lens[layer_idx][row_idx]
        if log_level == 'DEBUG':
            keep_cnt = int(keep_indices.numel())
            logger.debug(
                "[SnapKV] free_part_slots(before): "
                f"layer={layer_idx} seq_id={seq.seq_id} row={row_idx} "
                f"context_len={int(cur_len)} keep={keep_cnt} drop={max(0, int(cur_len) - keep_cnt)}"
            )
        old_slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len].clone()

        new_slots = old_slots[keep_indices]

        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask[keep_indices] = False
        dropped_slots = old_slots[mask]

        if dropped_slots.numel() > 0:
            count = dropped_slots.numel()
            ptr = self._num_free_slots[layer_idx]
            self.free_slots_stack[layer_idx][ptr: ptr + count] = dropped_slots
            self._num_free_slots[layer_idx] += count
        else:
            logger.warning(f"[SnapKV] dropped 0 tokens? layer={layer_idx} seq_id={seq.seq_id} row={row_idx} cur_len={int(cur_len)}")

        self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
        self.buffer_req_to_token_slots[layer_idx][row_idx, :new_slots.numel()] = new_slots
        self.row_seq_lens[layer_idx][row_idx] = new_slots.numel()
        if log_level == 'DEBUG':
            logger.debug(
                "[SnapKV] free_part_slots(after): "
                f"layer={layer_idx} seq_id={seq.seq_id} row={row_idx} "
                f"context_len={int(cur_len)} -> {int(new_slots.numel())}"
            )

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, total_chunk_tokens), dtype=torch.int32, device="cuda"
            )
            context_lens_list = [[] for _ in range(self.num_layers)]

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                for layer_id in range(self.num_layers):
                    if seq.seq_id in self.seq_id_to_row[layer_id]:
                        row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                        if self.row_seq_lens[layer_id][row_idx] != start_idx:
                            raise ValueError(
                                "KV cache row length mismatch in prefill: "
                                f"layer={layer_id} seq_id={seq.seq_id} "
                                f"row_seq_len={self.row_seq_lens[layer_id][row_idx]} "
                                f"start_idx={start_idx}"
                            )
                    self._allocate(layer_id, seq.seq_id, chunk_size)
                    row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                    layers_slot_mapping_cuda[layer_id, token_offset: token_offset + chunk_size] = \
                        self.buffer_req_to_token_slots[layer_id][row_idx, start_idx:end_idx]
                    context_lens_list[layer_id].append(end_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            layers_context_lens_cuda = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device="cuda")

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

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, batch_size), dtype=torch.int32, device="cuda"
            )
            layers_context_lens = []

            for layer_id in range(self.num_layers):
                new_slots_batch = self._allocate_batch(layer_id, seq_ids, 1)
                layers_slot_mapping_cuda[layer_id] = new_slots_batch

                row_indices = [self.seq_id_to_row[layer_id][sid] for sid in seq_ids]
                layers_context_lens.append(self.row_seq_lens[layer_id][row_indices])

            layers_context_lens_cuda = torch.from_numpy(np.array(layers_context_lens)).to(
                device="cuda",
                dtype=torch.int32,
            )

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device="cuda")

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

