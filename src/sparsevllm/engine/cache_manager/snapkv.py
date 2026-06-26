from __future__ import annotations

from collections import deque

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates, PrefillComputeView, SparseSelection


class SnapKVCacheManager(CacheManager):
    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.pyramidkv_prefill_staging_num_slots = 0
        self.pyramidkv_prefill_staging_kv_cache = None
        self._pyramidkv_prefill_staging_active = False
        self._pyramidkv_prefill_staging_was_active = False
        self._pyramidkv_prefill_staging_slot_mapping = None
        self._pyramidkv_prefill_staging_active_slots = None
        self._pyramidkv_prefill_staging_req_indices = None
        self._pyramidkv_prefill_staging_context_lens = None
        self._pyramidkv_prefill_staging_materialized_layers: set[int] = set()
        self.allocate_kv_cache()

        self.layer_num_slots = []
        self.free_slots_stack = []
        self._num_free_slots = []
        self.buffer_req_to_token_slots = []
        self.seq_id_to_row = []
        self.free_rows = []
        self.row_seq_lens = []
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]
        self._decode_static_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._prefill_attn_score_accumulators: dict[tuple[int, int], torch.Tensor] = {}

        for layer_id in range(self.num_layers):
            num_slots = (
                config.num_kvcache_slots[layer_id]
                if isinstance(config.num_kvcache_slots, list)
                else config.num_kvcache_slots
            )
            self.layer_num_slots.append(num_slots)
            self.free_slots_stack.append(
                torch.arange(num_slots, dtype=torch.int32, device=self.device)
            )
            self._num_free_slots.append(num_slots)
            self.buffer_req_to_token_slots.append(
                torch.zeros(
                    (self.max_buffer_rows, self.max_model_len),
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.seq_id_to_row.append({})
            self.free_rows.append(deque(range(self.max_buffer_rows)))
            self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))

    def _pyramidkv_can_use_full_prefill_staging(self) -> bool:
        return (
            self.config.vllm_sparse_method == "pyramidkv"
            and self.config.pyramid_layer_ratios is not None
            and self.config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
        )

    def _pyramidkv_reset_full_prefill_staging(self):
        self._pyramidkv_prefill_staging_active = False
        self._pyramidkv_prefill_staging_was_active = False
        self._pyramidkv_prefill_staging_slot_mapping = None
        self._pyramidkv_prefill_staging_active_slots = None
        self._pyramidkv_prefill_staging_req_indices = None
        self._pyramidkv_prefill_staging_context_lens = None
        self._pyramidkv_prefill_staging_materialized_layers = set()

    def _should_use_pyramidkv_full_prefill_staging(self, seqs: list[Sequence]) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None or len(seqs) != 1:
            return False
        seq = seqs[0]
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        return (
            int(seq.num_prefilled_tokens) == 0
            and int(seq.current_chunk_size) == remaining
            and remaining > int(self.config.chunk_prefill_size)
        )

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        num_layers = self.num_layers

        if config.pyramid_layer_ratios is not None:
            if self._pyramidkv_can_use_full_prefill_staging():
                self.pyramidkv_prefill_staging_num_slots = int(config.max_model_len)
                staging_bytes = int(self.pyramidkv_prefill_staging_num_slots) * int(slot_bytes_per_layer)
                available_memory = int(available_memory) - staging_bytes
                if available_memory <= 0:
                    raise RuntimeError(
                        "Not enough GPU memory for PyramidKV full-prefill staging KV. "
                        f"staging_slots={self.pyramidkv_prefill_staging_num_slots} "
                        f"required={staging_bytes / 1024**3:.2f}GiB."
                    )
                self.pyramidkv_prefill_staging_kv_cache = torch.empty(
                    2,
                    self.pyramidkv_prefill_staging_num_slots,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=self.hf_config.torch_dtype,
                    device=self.device,
                )

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
                    dtype=self.hf_config.torch_dtype, device=self.device
                )
                v_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.torch_dtype, device=self.device
                )
                self.kv_cache.append((k_cache, v_cache))

            config.num_kvcache_slots = layer_slots
            logger.info(
                f"PyramidKV: Layer slots = {layer_slots}, base_slots = {base_slots}, "
                f"prefill_staging_slots={self.pyramidkv_prefill_staging_num_slots}"
            )
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
                device=self.device,
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
        if self.has_prefill_staging_view(layer_idx):
            return (
                self.pyramidkv_prefill_staging_kv_cache[0],
                self.pyramidkv_prefill_staging_kv_cache[1],
                self._pyramidkv_prefill_staging_slot_mapping,
            )
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        if self.has_prefill_staging_view(layer_idx):
            return self.pyramidkv_prefill_staging_kv_cache[0], self.pyramidkv_prefill_staging_kv_cache[1]
        raise NotImplementedError

    def has_prefill_staging_view(self, layer_idx: int) -> bool:
        return bool(
            self._pyramidkv_prefill_staging_active
            and self.config.vllm_sparse_method == "pyramidkv"
            and 0 <= int(layer_idx) < int(self.num_layers)
        )

    def get_prefill_staging_view(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.has_prefill_staging_view(layer_idx):
            raise NotImplementedError("PyramidKV prefill staging view is not active for this layer.")
        return (
            self._pyramidkv_prefill_staging_active_slots,
            self._pyramidkv_prefill_staging_req_indices,
            self._pyramidkv_prefill_staging_context_lens,
            None,
        )

    def prefill_staging_was_active(self) -> bool:
        return bool(self._pyramidkv_prefill_staging_was_active)

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        return min(self._num_free_slots)

    def _pyramidkv_layer_budget(self, layer_idx: int) -> int:
        decode_keep = int(self.config.decode_keep_tokens)
        ratio = float(self.config.pyramid_layer_ratios[layer_idx])
        base_ratio = float(self.config.pyramid_layer_ratios[0])
        scaled_top_tokens = int(decode_keep * ratio / base_ratio)
        return int(self.config.num_sink_tokens) + scaled_top_tokens + int(self.config.num_recent_tokens)

    def _pyramidkv_prompt_admission_cost(self, seq: Sequence) -> int:
        prompt_len = int(seq.num_prompt_tokens)
        if prompt_len <= 0:
            return 0
        return max(
            min(prompt_len, self._pyramidkv_layer_budget(layer_idx))
            for layer_idx in range(self.num_layers)
        )

    def prompt_admission_cost(self, seq: Sequence) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return self._pyramidkv_prompt_admission_cost(seq)
        return super().prompt_admission_cost(seq)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return self._pyramidkv_prompt_admission_cost(seq)
        return super().prompt_logical_reservation_cost(seq)

    def reserved_prefill_slots(self, waiting_seqs, chunk_prefill_size: int) -> int:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return super().reserved_prefill_slots(waiting_seqs, chunk_prefill_size)
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                reserved += self._pyramidkv_prompt_admission_cost(seq)
        return int(reserved)

    def prefill_batched_tokens_margin(self) -> int:
        return 0

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def min_final_prefill_chunk_tokens(self, seq: Sequence) -> int:
        window = int(getattr(self.config, "snapkv_window_size", 0) or 0)
        if window <= 0 or self.config.vllm_sparse_method not in ("snapkv", "pyramidkv"):
            return 0
        prompt_len = int(seq.num_prompt_tokens)
        for layer_idx in range(int(self.num_layers)):
            budget = self._prefill_score_layer_budget(layer_idx)
            if budget is not None and prompt_len > int(budget):
                return min(window, prompt_len)
        return 0

    def _prefill_score_dtype(self) -> torch.dtype:
        score_dtype_name = str(getattr(self.config, "sparse_attn_score_dtype", "float32") or "float32").lower()
        try:
            return {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[score_dtype_name]
        except KeyError as exc:
            raise ValueError(
                "sparse_attn_score_dtype must be 'float32', 'bfloat16', or 'float16', "
                f"got {score_dtype_name!r}."
            ) from exc

    def _prefill_score_layer_budget(self, layer_idx: int) -> int | None:
        if int(layer_idx) < int(getattr(self.config, "snapkv_num_full_layers", 0) or 0):
            return None
        if self.config.vllm_sparse_method == "pyramidkv":
            if self.config.pyramid_layer_ratios is None:
                return None
            return self._pyramidkv_layer_budget(layer_idx)
        if self.config.vllm_sparse_method == "snapkv":
            return (
                int(self.config.num_sink_tokens)
                + int(self.config.decode_keep_tokens)
                + int(self.config.num_recent_tokens)
            )
        return None

    def _prefill_score_rows(
        self,
        layer_idx: int,
        seqs: list[Sequence],
    ) -> list[tuple[int, Sequence, int, int]]:
        budget = self._prefill_score_layer_budget(layer_idx)
        if budget is None:
            return []
        window = int(getattr(self.config, "snapkv_window_size", 0) or 0)
        if window <= 0:
            return []

        rows = []
        for b_idx, seq in enumerate(seqs):
            if seq.current_chunk_size is None:
                raise RuntimeError(
                    "Prefill score collection requires current_chunk_size. "
                    f"layer={layer_idx} seq_id={seq.seq_id}"
                )
            prompt_len = int(seq.num_prompt_tokens)
            if prompt_len <= int(budget):
                continue
            score_end = prompt_len
            score_start = max(0, score_end - window)
            chunk_start = int(seq.num_prefilled_tokens)
            chunk_end = chunk_start + int(seq.current_chunk_size)
            if chunk_start <= score_start and chunk_end >= score_end:
                rows.append((b_idx, seq, score_start, score_end))
            elif seq.is_last_chunk_prefill and chunk_start < score_end and chunk_end > score_start:
                raise RuntimeError(
                    "SnapKV/PyramidKV prefill score requires the score query window to fit in "
                    "the final prefill chunk. "
                    f"layer={layer_idx} seq_id={seq.seq_id} score_range=[{score_start}, {score_end}) "
                    f"chunk_range=[{chunk_start}, {chunk_end})."
                )
        return rows

    def _clear_prefill_attention_scores(self, seq_id: int):
        seq_id = int(seq_id)
        for key in list(self._prefill_attn_score_accumulators):
            if key[1] == seq_id:
                self._prefill_attn_score_accumulators.pop(key, None)

    def _get_prefill_attention_score_accumulator(
        self,
        layer_idx: int,
        seq: Sequence,
        *,
        prompt_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (int(layer_idx), int(seq.seq_id))
        if int(seq.num_prefilled_tokens) == 0:
            self._prefill_attn_score_accumulators.pop(key, None)
        acc = self._prefill_attn_score_accumulators.get(key)
        if acc is None:
            acc = torch.zeros(
                (int(prompt_len),),
                dtype=self._prefill_score_dtype(),
                device=device,
            )
            self._prefill_attn_score_accumulators[key] = acc
        return acc

    @torch.no_grad()
    def collect_prefill_attention_score(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
    ):
        ctx = get_context()
        if not ctx.is_prefill or not ctx.is_long_text:
            return None
        if self.config.vllm_sparse_method not in ("snapkv", "pyramidkv"):
            return None
        seqs = getattr(ctx, "seqs", None)
        if seqs is None:
            raise RuntimeError("Prefill score collection requires current seqs in context.")

        rows = self._prefill_score_rows(layer_idx, seqs)
        if not rows:
            return None

        b_prompt_cache_len = view.context_lens - chunk_lens
        max_query_len = int(chunk_lens.max().item())
        if len(rows) == 1:
            b_idx, seq, score_start, score_end = rows[0]
            acc = self._get_prefill_attention_score_accumulator(
                layer_idx,
                seq,
                prompt_len=int(seq.num_prompt_tokens),
                device=q.device,
            )
            prefill_score_fwd(
                q,
                view.k_cache,
                acc.unsqueeze(0),
                view.req_indices[b_idx : b_idx + 1],
                b_start_loc[b_idx : b_idx + 1],
                view.context_lens[b_idx : b_idx + 1],
                b_prompt_cache_len[b_idx : b_idx + 1],
                int(chunk_lens[b_idx].item()),
                view.active_slots,
                torch.tensor([score_start], dtype=torch.int32, device=q.device),
                torch.tensor([score_end], dtype=torch.int32, device=q.device),
                candidate_start=int(self.config.num_sink_tokens),
                num_recent_tokens=int(self.config.num_recent_tokens),
            )
            return None

        score_starts = torch.zeros((len(seqs),), dtype=torch.int32, device=q.device)
        score_ends = torch.zeros((len(seqs),), dtype=torch.int32, device=q.device)
        for b_idx, _seq, score_start, score_end in rows:
            score_starts[b_idx] = int(score_start)
            score_ends[b_idx] = int(score_end)

        step_score = torch.zeros(
            (len(seqs), int(view.context_lens.max().item())),
            dtype=self._prefill_score_dtype(),
            device=q.device,
        )
        prefill_score_fwd(
            q,
            view.k_cache,
            step_score,
            view.req_indices,
            b_start_loc,
            view.context_lens,
            b_prompt_cache_len,
            max_query_len,
            view.active_slots,
            score_starts,
            score_ends,
            candidate_start=int(self.config.num_sink_tokens),
            num_recent_tokens=int(self.config.num_recent_tokens),
        )
        for b_idx, seq, _score_start, _score_end in rows:
            acc = self._get_prefill_attention_score_accumulator(
                layer_idx,
                seq,
                prompt_len=int(seq.num_prompt_tokens),
                device=q.device,
            )
            context_len = int(view.context_lens[b_idx].item())
            acc[:context_len] = torch.maximum(acc[:context_len], step_score[b_idx, :context_len])
        return None

    def pop_prefill_attention_score(self, layer_idx: int, seq: Sequence) -> torch.Tensor | None:
        return self._prefill_attn_score_accumulators.pop((int(layer_idx), int(seq.seq_id)), None)

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
            if int(cur_len) + int(size) > int(self.max_model_len):
                raise RuntimeError(
                    "KV row length exceeds max_model_len in _allocate: "
                    f"layer={layer_idx} seq_id={seq_id} row={row_idx} "
                    f"cur_len={int(cur_len)} size={int(size)} max_model_len={int(self.max_model_len)}"
                )

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
        if len(cur_lens) > 0 and int(max(cur_lens)) + int(size) > int(self.max_model_len):
            raise RuntimeError(
                "KV row length exceeds max_model_len in _allocate_batch: "
                f"layer={layer_idx} max_cur_len={int(max(cur_lens))} "
                f"size={int(size)} max_model_len={int(self.max_model_len)}"
            )

        ptr = self._num_free_slots[layer_idx]
        select_indices = self.free_slots_stack[layer_idx][ptr - batch_size: ptr]
        self._num_free_slots[layer_idx] -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, cols_gpu] = select_indices.to(torch.int32)
        self.row_seq_lens[layer_idx][row_indices] += 1

        return select_indices

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self._clear_prefill_attention_scores(seq_id)
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

        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        if keep_indices.numel() <= 0:
            raise RuntimeError(
                f"free_part_slots got empty keep_indices: layer={layer_idx} seq_id={seq.seq_id}"
            )
        if bool((keep_indices < 0).any().item()) or bool((keep_indices >= int(cur_len)).any().item()):
            raise RuntimeError(
                "free_part_slots keep_indices out of bounds: "
                f"layer={layer_idx} seq_id={seq.seq_id} cur_len={int(cur_len)} "
                f"keep_min={int(keep_indices.min().item())} "
                f"keep_max={int(keep_indices.max().item())}"
            )
        keep_indices = torch.sort(keep_indices).values
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

    def materialize_prefill_staging_layer(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        if not self.has_prefill_staging_view(layer_idx):
            raise RuntimeError("PyramidKV prefill staging is not active.")
        if layer_idx in self._pyramidkv_prefill_staging_materialized_layers:
            raise RuntimeError(f"PyramidKV prefill staging layer materialized twice: layer={layer_idx}.")

        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise RuntimeError(f"PyramidKV staging row is missing: layer={layer_idx} seq_id={seq.seq_id}.")
        if int(self.row_seq_lens[layer_idx][row_idx]) != 0:
            raise RuntimeError(
                "PyramidKV full-prefill staging expects an empty persistent row before materialization. "
                f"layer={layer_idx} seq_id={seq.seq_id} row_len={int(self.row_seq_lens[layer_idx][row_idx])}."
            )

        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        num_keep = int(keep_indices.numel())
        if num_keep <= 0:
            raise RuntimeError("PyramidKV staging materialization cannot keep zero tokens.")

        slots = self._allocate(layer_idx, seq.seq_id, num_keep).to(torch.long)
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        k_stage = self.pyramidkv_prefill_staging_kv_cache[0]
        v_stage = self.pyramidkv_prefill_staging_kv_cache[1]
        k_cache[slots] = k_stage[keep_indices]
        v_cache[slots] = v_stage[keep_indices]

        self._pyramidkv_prefill_staging_materialized_layers.add(int(layer_idx))
        if len(self._pyramidkv_prefill_staging_materialized_layers) == int(self.num_layers):
            self._pyramidkv_prefill_staging_active = False

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        self._pyramidkv_reset_full_prefill_staging()
        return super().prepare_step(seqs, is_prefill)

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            for seq in seqs:
                if int(seq.num_prefilled_tokens) == 0:
                    self._clear_prefill_attention_scores(seq.seq_id)

            use_full_prefill_staging = self._should_use_pyramidkv_full_prefill_staging(seqs)
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)
            if use_full_prefill_staging and total_chunk_tokens > int(self.pyramidkv_prefill_staging_num_slots):
                raise RuntimeError(
                    "PyramidKV full-prefill staging capacity is too small for this step. "
                    f"tokens={total_chunk_tokens} staging_slots={self.pyramidkv_prefill_staging_num_slots}."
                )

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            if use_full_prefill_staging:
                layers_slot_mapping_cuda = torch.arange(
                    total_chunk_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ).expand(self.num_layers, -1)
            else:
                layers_slot_mapping_cuda = torch.empty(
                    (self.num_layers, total_chunk_tokens), dtype=torch.int32, device=self.device
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
                    if use_full_prefill_staging:
                        if start_idx != 0:
                            raise RuntimeError("PyramidKV full-prefill staging only supports first-prefill prompts.")
                        self._get_free_row(layer_id, seq.seq_id)
                    else:
                        self._allocate(layer_id, seq.seq_id, chunk_size)
                    row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                    if not use_full_prefill_staging:
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

            layers_context_lens_cuda = torch.tensor(context_lens_list, dtype=torch.int32, device=self.device)

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                state.max_context_len = int(max(context_lens_list[layer_id])) if context_lens_list[layer_id] else 0
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device=self.device)

            if use_full_prefill_staging:
                self._pyramidkv_prefill_staging_active = True
                self._pyramidkv_prefill_staging_was_active = True
                self._pyramidkv_prefill_staging_slot_mapping = layers_slot_mapping_cuda[0]
                max_context_len = int(max(max(lens) for lens in context_lens_list if lens))
                active_slots = torch.full(
                    (len(seqs), max_context_len),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                offset = 0
                for b_idx, seq in enumerate(seqs):
                    chunk_size = int(seq.current_chunk_size)
                    active_slots[b_idx, :chunk_size] = torch.arange(
                        offset,
                        offset + chunk_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    offset += chunk_size
                self._pyramidkv_prefill_staging_active_slots = active_slots
                self._pyramidkv_prefill_staging_req_indices = torch.arange(
                    len(seqs),
                    dtype=torch.int32,
                    device=self.device,
                )
                self._pyramidkv_prefill_staging_context_lens = torch.tensor(
                    [int(seq.num_prefilled_tokens + seq.current_chunk_size) for seq in seqs],
                    dtype=torch.int32,
                    device=self.device,
                )

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

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, batch_size), dtype=torch.int32, device=self.device
            )
            layers_context_lens = []

            for layer_id in range(self.num_layers):
                new_slots_batch = self._allocate_batch(layer_id, seq_ids, 1)
                layers_slot_mapping_cuda[layer_id] = new_slots_batch

                row_indices = [self.seq_id_to_row[layer_id][sid] for sid in seq_ids]
                layers_context_lens.append(self.row_seq_lens[layer_id][row_indices])

            layers_context_lens_cuda = torch.from_numpy(np.array(layers_context_lens)).to(
                device=self.device,
                dtype=torch.int32,
            )

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                # row_seq_lens is layer-wise after prefill/decode eviction; attention
                # uses this to size flash decode stage1/stage2 workspaces.
                state.max_context_len = (
                    int(max(layers_context_lens[layer_id]))
                    if len(layers_context_lens[layer_id]) > 0
                    else 0
                )
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device=self.device)

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            return input_ids, positions, None

    def _get_decode_static_buffers(
        self,
        graph_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_batch_size = int(graph_batch_size)
        buffers = self._decode_static_buffers.get(graph_batch_size)
        if buffers is None:
            buffers = (
                torch.empty((self.num_layers, graph_batch_size), dtype=torch.int32, device=self.device),
                torch.empty((self.num_layers, graph_batch_size), dtype=torch.int32, device=self.device),
                torch.empty((self.num_layers, graph_batch_size), dtype=torch.int32, device=self.device),
            )
            self._decode_static_buffers[graph_batch_size] = buffers
        return buffers

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
        """Prepare per-layer decode metadata into graph-stable CUDA buffers."""
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

            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64, device=self.device))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
            if graph_batch_size > real_batch_size:
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))

            layers_slot_mapping, layers_context_lens, layers_req_indices = self._get_decode_static_buffers(
                graph_batch_size
            )

            for layer_id in range(self.num_layers):
                new_slots_batch = self._allocate_batch(layer_id, seq_ids, 1)
                row_indices = [self.seq_id_to_row[layer_id][sid] for sid in seq_ids]
                real_context_lens = self.row_seq_lens[layer_id][row_indices]
                static_cap = getattr(self, "_decode_static_max_context_len", None)
                real_max_context_len = int(max(real_context_lens)) if row_indices else 0
                if static_cap is not None and real_max_context_len > int(static_cap):
                    raise RuntimeError(
                        "static decode context length exceeds captured graph max_context_len: "
                        f"layer={layer_id} real_max_context_len={real_max_context_len} "
                        f"static_cap={int(static_cap)} graph_batch_size={graph_batch_size} "
                        f"real_batch_size={real_batch_size}"
                    )

                layer_slot_mapping = layers_slot_mapping[layer_id]
                layer_context_lens = layers_context_lens[layer_id]
                layer_req_indices = layers_req_indices[layer_id]
                layer_slot_mapping[:real_batch_size].copy_(new_slots_batch)
                layer_context_lens[:real_batch_size].copy_(
                    torch.tensor(real_context_lens, dtype=torch.int32, device=self.device)
                )
                layer_req_indices[:real_batch_size].copy_(
                    torch.tensor(row_indices, dtype=torch.int32, device=self.device)
                )

                if graph_batch_size > real_batch_size:
                    layer_slot_mapping[real_batch_size:].fill_(-1)
                    layer_context_lens[real_batch_size:].fill_(int(real_context_lens[0]))
                    layer_req_indices[real_batch_size:].fill_(int(row_indices[0]))

                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layer_slot_mapping
                state.context_lens = layer_context_lens
                state.max_context_len = real_max_context_len
                state.req_indices = layer_req_indices

            slot_mapping.copy_(layers_slot_mapping[0])
            context_lens.copy_(layers_context_lens[0])
            req_indices.copy_(layers_req_indices[0])

            return input_ids, positions, None
