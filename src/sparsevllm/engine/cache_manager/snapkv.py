from __future__ import annotations

from collections import deque

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
from sparsevllm.platforms import device_runtime
from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates, PrefillComputeView, SparseSelection
from .raw_kv_offload import RawKVOffloadBuffer, resolve_long_prefill_offload_min_tokens


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
        self._pyramidkv_prefill_staging_seq_offsets: dict[int, int] = {}
        self._pyramidkv_prefill_staging_materialized_layers: set[tuple[int, int]] = set()
        self.raw_kv_offload_buffer = RawKVOffloadBuffer(pin_memory=device_runtime.supports_pin_memory())
        self._pyramidkv_long_prefill_offload_step_active = False
        self._pyramidkv_long_prefill_offload_seq_id: int | None = None
        self._pyramidkv_long_prefill_offload_start = 0
        self._pyramidkv_long_prefill_offload_end = 0
        self._pyramidkv_long_prefill_offload_total_len = 0
        self._pyramidkv_long_prefill_offload_is_last_chunk = False
        self._pyramidkv_long_prefill_offload_prefetch_stream = None
        self._pyramidkv_long_prefill_offload_prefetch_states: dict[tuple[int, int, str, int], dict] = {}
        self.allocate_kv_cache()

        self.layer_num_slots = []
        self.free_slots_stack_tensor = None
        self._free_slots_layer_indices = None
        self.free_slots_stack = []
        self._num_free_slots = []
        self.buffer_req_to_token_slots = []
        self.seq_id_to_row = []
        self.free_rows = []
        self.row_seq_lens = []
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]
        self._decode_static_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._decode_static_index_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._decode_static_state_binding_key: tuple[int, int, int, int] | None = None
        self._prefill_attn_score_accumulators: dict[tuple[int, int], torch.Tensor] = {}
        self._prefill_score_bounds: tuple[torch.Tensor, torch.Tensor] | None = None
        self._uniform_decode_metadata = self._sparse_eviction_never_triggers()
        self.buffer_req_to_token_slots_tensor = torch.zeros(
            (self.num_kv_layers, self.max_buffer_rows, self.max_model_len),
            dtype=torch.int32,
            device=self.device,
        )
        if not isinstance(config.num_kvcache_slots, list):
            num_slots = int(config.num_kvcache_slots)
            self.free_slots_stack_tensor = torch.arange(
                num_slots,
                dtype=torch.int32,
                device=self.device,
            ).expand(self.num_kv_layers, -1).clone()
            self._free_slots_layer_indices = torch.arange(
                self.num_kv_layers,
                dtype=torch.long,
                device=self.device,
            )

        for layer_id in range(self.num_layers):
            if not self.is_full_attention_layer(layer_id):
                self.layer_num_slots.append(0)
                self.free_slots_stack.append(None)
                self._num_free_slots.append(0)
                self.buffer_req_to_token_slots.append(None)
                self.seq_id_to_row.append({})
                self.free_rows.append(deque(range(self.max_buffer_rows)))
                self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))
                continue
            kv_idx = self.kv_layer_index(layer_id)
            num_slots = (
                config.num_kvcache_slots[layer_id]
                if isinstance(config.num_kvcache_slots, list)
                else config.num_kvcache_slots
            )
            self.layer_num_slots.append(num_slots)
            if self.free_slots_stack_tensor is not None:
                self.free_slots_stack.append(self.free_slots_stack_tensor[kv_idx])
            else:
                self.free_slots_stack.append(
                    torch.arange(num_slots, dtype=torch.int32, device=self.device)
                )
            self._num_free_slots.append(num_slots)
            self.buffer_req_to_token_slots.append(self.buffer_req_to_token_slots_tensor[kv_idx])
            self.seq_id_to_row.append({})
            self.free_rows.append(deque(range(self.max_buffer_rows)))
            self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))

    def _sparse_eviction_never_triggers(self) -> bool:
        method = str(getattr(self.config, "vllm_sparse_method", "") or "")
        max_model_len = int(getattr(self.config, "max_model_len", 0) or 0)
        sink = int(getattr(self.config, "num_sink_tokens", 0) or 0)
        recent = int(getattr(self.config, "num_recent_tokens", 0) or 0)
        decode_keep = int(getattr(self.config, "decode_keep_tokens", 0) or 0)
        if method in {"snapkv", "rkv", "skipkv"}:
            return max_model_len <= sink + decode_keep + recent
        if method == "pyramidkv" and self.config.pyramid_layer_ratios is None:
            return max_model_len <= sink + decode_keep + recent
        return False

    def _pyramidkv_can_use_full_prefill_staging(self) -> bool:
        return (
            self.config.vllm_sparse_method == "pyramidkv"
            and self.config.pyramid_layer_ratios is not None
            and self.config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
        )

    def _pyramidkv_reset_full_prefill_staging(self):
        self._pyramidkv_clear_long_prefill_offload_prefetch()
        self._pyramidkv_prefill_staging_active = False
        self._pyramidkv_prefill_staging_was_active = False
        self._pyramidkv_prefill_staging_slot_mapping = None
        self._pyramidkv_prefill_staging_active_slots = None
        self._pyramidkv_prefill_staging_req_indices = None
        self._pyramidkv_prefill_staging_context_lens = None
        self._pyramidkv_prefill_staging_seq_offsets = {}
        self._pyramidkv_prefill_staging_materialized_layers = set()
        self._pyramidkv_long_prefill_offload_seq_id = None
        self._pyramidkv_long_prefill_offload_start = 0
        self._pyramidkv_long_prefill_offload_end = 0
        self._pyramidkv_long_prefill_offload_total_len = 0
        self._pyramidkv_long_prefill_offload_is_last_chunk = False

    def _long_prefill_offload_min_tokens(self) -> int:
        return resolve_long_prefill_offload_min_tokens()

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None:
            return False
        prompt_len = int(seq.num_prompt_tokens)
        remaining = prompt_len - int(seq.num_prefilled_tokens)
        return (
            prompt_len > int(self.config.chunk_prefill_size)
            and prompt_len >= self._long_prefill_offload_min_tokens()
            and prompt_len <= int(self.pyramidkv_prefill_staging_num_slots)
            and remaining > 0
        )

    def _should_use_pyramidkv_long_prefill_offload_staging(self, seqs: list[Sequence]) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None or len(seqs) != 1:
            return False
        seq = seqs[0]
        return self.requires_long_prefill_offload(seq) and int(seq.current_chunk_size or 0) > 0

    def _should_use_pyramidkv_full_prefill_staging(self, seqs: list[Sequence]) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None or not seqs:
            return False
        total_chunk_tokens = 0
        for seq in seqs:
            if self.requires_long_prefill_offload(seq):
                return False
            remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
            if int(seq.num_prefilled_tokens) != 0 or int(seq.current_chunk_size) != remaining:
                return False
            total_chunk_tokens += int(seq.current_chunk_size)
        return total_chunk_tokens <= int(self.pyramidkv_prefill_staging_num_slots)

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return False
        if self.pyramidkv_prefill_staging_kv_cache is None:
            return False
        if self.requires_long_prefill_offload(seq):
            return False
        if int(seq.num_prefilled_tokens) != 0:
            return False
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        return 0 < remaining <= int(self.pyramidkv_prefill_staging_num_slots)

    def is_full_prefill_step(self, seqs: list[Sequence]) -> bool:
        return self._should_use_pyramidkv_full_prefill_staging(seqs)

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        num_layers = self.num_kv_layers

        if config.pyramid_layer_ratios is not None:
            if self._pyramidkv_can_use_full_prefill_staging():
                self.pyramidkv_prefill_staging_num_slots = max(
                    int(config.max_model_len),
                    int(config.max_num_batched_tokens),
                )
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
            kv_layer_ids = list(self.runtime_layout.kv_idx_to_layer_idx)
            total_ratio = sum(float(ratio) for ratio in config.pyramid_layer_ratios)
            base_slots = int(available_memory // (slot_bytes_per_layer * total_ratio))
            assert base_slots > 0, "可用显存不足以分配 KV Cache"

            kv_layer_slots = [int(base_slots * ratio) for ratio in config.pyramid_layer_ratios]
            assert kv_layer_slots[0] == max(kv_layer_slots), (
                "The first KV layer must have the largest PyramidKV allocation, but "
                f"first={kv_layer_slots[0]}, max={max(kv_layer_slots)}."
            )
            layer_slots = [0] * self.num_layers

            self.kv_cache = []
            for kv_idx, layer_idx in enumerate(kv_layer_ids):
                num_slots = kv_layer_slots[kv_idx]
                layer_slots[layer_idx] = num_slots
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
                f"PyramidKV: KV layer slots = {list(zip(kv_layer_ids, kv_layer_slots))}, "
                f"base_slots = {base_slots}, "
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
        self.kv_layer_index(layer_idx)
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_idx = self.kv_layer_index(layer_idx)
        if isinstance(self.kv_cache, list):
            return self.kv_cache[kv_idx]
        elif isinstance(self.kv_cache, torch.Tensor):
            return self.kv_cache[0, kv_idx], self.kv_cache[1, kv_idx]
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
            and self.is_full_attention_layer(layer_idx)
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
        self.kv_layer_index(layer_idx)
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        return min(self._num_free_slots[layer_idx] for layer_idx in self.kv_transformer_layer_indices())

    def _pyramidkv_layer_budget(self, layer_idx: int) -> int:
        decode_keep = int(self.config.decode_keep_tokens)
        ratio = float(self.config.pyramid_layer_ratios[self.kv_layer_index(layer_idx)])
        base_ratio = float(self.config.pyramid_layer_ratios[0])
        scaled_top_tokens = int(decode_keep * ratio / base_ratio)
        return int(self.config.num_sink_tokens) + scaled_top_tokens + int(self.config.num_recent_tokens)

    def _pyramidkv_prompt_admission_cost(self, seq: Sequence) -> int:
        prompt_len = int(seq.num_prompt_tokens)
        if prompt_len <= 0:
            return 0
        return max(
            min(prompt_len, self._pyramidkv_layer_budget(layer_idx))
            for layer_idx in self.kv_transformer_layer_indices()
        )

    def prompt_admission_cost(self, seq: Sequence) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return self._pyramidkv_prompt_admission_cost(seq)
        return super().prompt_admission_cost(seq)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return self._pyramidkv_prompt_admission_cost(seq)
        return super().prompt_logical_reservation_cost(seq)

    def prompt_admission_free_slots(self) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return max(
                int(self._num_free_slots[layer_idx])
                for layer_idx in self.kv_transformer_layer_indices()
            )
        return super().prompt_admission_free_slots()

    def prompt_admission_budgets(self, waiting_seqs, chunk_prefill_size: int) -> dict[str, int]:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return super().prompt_admission_budgets(waiting_seqs, chunk_prefill_size)
        return {
            f"layer_{layer_idx}": int(self._num_free_slots[layer_idx])
            for layer_idx in self.kv_transformer_layer_indices()
        }

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        if not self._pyramidkv_can_use_full_prefill_staging():
            return super().prompt_admission_costs(seq)
        prompt_len = int(seq.num_prompt_tokens)
        return {
            f"layer_{layer_idx}": min(prompt_len, self._pyramidkv_layer_budget(layer_idx))
            for layer_idx in self.kv_transformer_layer_indices()
        }

    def prefill_step_free_slots(self) -> int:
        if self._pyramidkv_can_use_full_prefill_staging():
            return int(self.pyramidkv_prefill_staging_num_slots)
        return super().prefill_step_free_slots()

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        if self.requires_long_prefill_offload(seq):
            return max(0, int(self.pyramidkv_prefill_staging_num_slots) - int(seq.num_prefilled_tokens))
        return super().prefill_step_free_slots_for(seq)

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        if self.requires_long_prefill_offload(seq):
            return 0
        return super().prefill_step_reservation_cost(seq, scheduled_tokens)

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
        if self.kv_layer_index(layer_idx) < int(getattr(self.config, "snapkv_num_full_layers", 0) or 0):
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

    def _prefill_score_bound_tensors(
        self,
        *,
        score_start: int,
        score_end: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bounds = self._prefill_score_bounds
        if bounds is None or bounds[0].device != device:
            bounds = (
                torch.empty((1,), dtype=torch.int32, device=device),
                torch.empty((1,), dtype=torch.int32, device=device),
            )
            self._prefill_score_bounds = bounds
        bounds[0].fill_(int(score_start))
        bounds[1].fill_(int(score_end))
        return bounds

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
        if not ctx.is_prefill:
            return None
        if not ctx.is_long_text and not self.has_prefill_staging_view(layer_idx):
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
        max_query_len = max(int(seq.current_chunk_size) for seq in seqs)
        if len(rows) == 1:
            b_idx, seq, score_start, score_end = rows[0]
            query_len = int(seq.current_chunk_size)
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
                query_len,
                view.active_slots,
                *self._prefill_score_bound_tensors(
                    score_start=score_start,
                    score_end=score_end,
                    device=q.device,
                ),
                candidate_start=int(self.config.num_sink_tokens),
                num_recent_tokens=int(self.config.num_recent_tokens),
            )
            return None

        score_starts = torch.zeros((len(seqs),), dtype=torch.int32, device=q.device)
        score_ends = torch.zeros((len(seqs),), dtype=torch.int32, device=q.device)
        for b_idx, _seq, score_start, score_end in rows:
            score_starts[b_idx] = int(score_start)
            score_ends[b_idx] = int(score_end)

        max_context_len = (
            int(view.max_context_len)
            if view.max_context_len is not None
            else max(int(seq.num_prefilled_tokens + seq.current_chunk_size) for seq in seqs)
        )
        step_score = torch.zeros(
            (len(seqs), max_context_len),
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

    @torch.no_grad()
    def _allocate_prefill_batch_same_size_all_layers(
        self,
        seqs: list[Sequence],
        layers_slot_mapping: torch.Tensor,
    ) -> bool:
        if self.free_slots_stack_tensor is None or not seqs:
            return False
        chunk_size = int(seqs[0].current_chunk_size)
        if chunk_size <= 0 or any(int(seq.current_chunk_size) != chunk_size for seq in seqs):
            return False

        batch_size = len(seqs)
        total_size = batch_size * chunk_size
        layer_ids = self.kv_transformer_layer_indices()
        if not layer_ids:
            return False
        with profiler.record("cache_allocate"):
            min_free = min(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)
            if min_free < total_size:
                raise RuntimeError(
                    "Out of KV cache slots in batched prefill allocation: "
                    f"need={total_size} free={min_free}"
                )

            row_indices = np.empty((len(layer_ids), batch_size), dtype=np.int64)
            start_lens = np.empty((len(layer_ids), batch_size), dtype=np.int64)
            for local_layer, layer_id in enumerate(layer_ids):
                for seq_idx, seq in enumerate(seqs):
                    row_idx = self._get_free_row(layer_id, int(seq.seq_id))
                    expected_start = int(seq.num_prefilled_tokens)
                    row_len = int(self.row_seq_lens[layer_id][row_idx])
                    if row_len != expected_start:
                        raise ValueError(
                            "KV cache row length mismatch in batched prefill allocation: "
                            f"layer={layer_id} seq_id={seq.seq_id} row_seq_len={row_len} "
                            f"start_idx={expected_start}"
                        )
                    if row_len + chunk_size > int(self.max_model_len):
                        raise RuntimeError(
                            "KV row length exceeds max_model_len in batched prefill allocation: "
                            f"layer={layer_id} seq_id={seq.seq_id} row={row_idx} "
                            f"cur_len={row_len} size={chunk_size} max_model_len={int(self.max_model_len)}"
                        )
                    row_indices[local_layer, seq_idx] = int(row_idx)
                    start_lens[local_layer, seq_idx] = row_len

            layers_gpu = torch.tensor(layer_ids, dtype=torch.long, device=self.device)
            kv_layers_gpu = torch.tensor(
                [self.kv_layer_index(layer_id) for layer_id in layer_ids],
                dtype=torch.long,
                device=self.device,
            )
            first_layer = int(layer_ids[0])
            if all(int(self._num_free_slots[layer_id]) == int(self._num_free_slots[first_layer]) for layer_id in layer_ids):
                ptr = int(self._num_free_slots[first_layer])
                selected_slots = self.free_slots_stack_tensor[kv_layers_gpu, ptr - total_size: ptr].view(
                    len(layer_ids),
                    batch_size,
                    chunk_size,
                ).flip(1)
            else:
                ptrs = np.asarray([self._num_free_slots[layer_id] for layer_id in layer_ids], dtype=np.int64)
                seq_offsets = (batch_size - 1 - np.arange(batch_size, dtype=np.int64)) * chunk_size
                token_offsets = np.arange(chunk_size, dtype=np.int64)
                slot_offsets = (
                    ptrs[:, None, None]
                    - total_size
                    + seq_offsets[None, :, None]
                    + token_offsets[None, None, :]
                )
                slot_offsets_gpu = torch.from_numpy(slot_offsets).to(device=self.device, dtype=torch.long)
                selected_slots = self.free_slots_stack_tensor[
                    kv_layers_gpu[:, None, None],
                    slot_offsets_gpu,
                ]

            rows_gpu = torch.from_numpy(row_indices).to(device=self.device, dtype=torch.long)
            cols_gpu = (
                torch.from_numpy(start_lens).to(device=self.device, dtype=torch.long)[:, :, None]
                + torch.arange(chunk_size, dtype=torch.long, device=self.device)[None, None, :]
            )
            self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                cols_gpu,
            ] = selected_slots.to(torch.int32)
            layers_slot_mapping[layers_gpu, :total_size] = selected_slots.reshape(len(layer_ids), total_size)
            for local_layer, layer_id in enumerate(layer_ids):
                self._num_free_slots[layer_id] -= total_size
                self.row_seq_lens[layer_id][row_indices[local_layer]] += chunk_size
            return True

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self._clear_prefill_attention_scores(seq_id)
            self._pyramidkv_clear_long_prefill_offload_prefetch()
            for layer_idx in self.kv_transformer_layer_indices():
                row_idx = self.seq_id_to_row[layer_idx].pop(seq_id, None)
                if row_idx is None:
                    raise ValueError
                self.raw_kv_offload_buffer.release_layer(
                    layer_idx=layer_idx,
                    row_idx=int(row_idx),
                    kind=self._pyramidkv_long_prefill_offload_kind(),
                )

                cur_len = self.row_seq_lens[layer_idx][row_idx]
                slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len]

                if cur_len > 0:
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + cur_len] = slots
                    self._num_free_slots[layer_idx] += cur_len

                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
                self.row_seq_lens[layer_idx][row_idx] = 0
                self.free_rows[layer_idx].append(row_idx)

    def decode_kv_lens_for_layer(self, layer_idx: int, seqs: list[Sequence]) -> list[int]:
        self.kv_layer_index(layer_idx)
        kv_lens = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise RuntimeError(
                    f"Missing decode row for seq_id={seq.seq_id} on layer={layer_idx}."
                )
            kv_lens.append(int(self.row_seq_lens[layer_idx][row_idx]))
        return kv_lens

    def free_part_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        if keep_indices is None:
            return

        self.kv_layer_index(layer_idx)
        self._uniform_decode_metadata = False
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
        if not keep_indices_sorted:
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

    def free_part_slots_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        if keep_indices is None:
            return
        if not seqs:
            return
        self.kv_layer_index(layer_idx)
        if len(seqs) == 1:
            self.free_part_slots(
                layer_idx,
                seqs[0],
                keep_indices[0],
                keep_indices_sorted=keep_indices_sorted,
            )
            return

        self._uniform_decode_metadata = False
        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        if keep_indices.dim() != 2 or int(keep_indices.shape[0]) != len(seqs):
            raise RuntimeError(
                "free_part_slots_batch expected keep_indices with shape [batch, keep]: "
                f"batch={len(seqs)} keep_shape={tuple(keep_indices.shape)}"
            )
        if int(keep_indices.shape[1]) <= 0:
            raise RuntimeError(f"free_part_slots_batch got empty keep_indices: layer={layer_idx}")

        row_indices = []
        cur_lens = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise ValueError
            row_indices.append(int(row_idx))
            cur_lens.append(int(self.row_seq_lens[layer_idx][row_idx]))

        first_len = cur_lens[0]
        if any(cur_len != first_len for cur_len in cur_lens):
            for seq, seq_keep_indices in zip(seqs, keep_indices):
                self.free_part_slots(
                    layer_idx,
                    seq,
                    seq_keep_indices,
                    keep_indices_sorted=keep_indices_sorted,
                )
            return
        cur_len = int(first_len)
        bounds_ok = ((keep_indices >= 0) & (keep_indices < cur_len)).all()
        if keep_indices.is_cuda:
            torch._assert_async(bounds_ok)
        elif not bool(bounds_ok.item()):
            raise RuntimeError(
                "free_part_slots_batch keep_indices out of bounds: "
                f"layer={layer_idx} cur_len={cur_len} "
                f"keep_min={int(keep_indices.min().item())} "
                f"keep_max={int(keep_indices.max().item())}"
            )

        if not keep_indices_sorted:
            keep_indices = torch.sort(keep_indices, dim=1).values
        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        old_slots = self.buffer_req_to_token_slots[layer_idx][rows_gpu, :cur_len]
        new_slots = old_slots.gather(1, keep_indices)

        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask.scatter_(1, keep_indices, False)
        dropped_slots = old_slots[mask]
        if dropped_slots.numel() > 0:
            count = int(dropped_slots.numel())
            ptr = self._num_free_slots[layer_idx]
            self.free_slots_stack[layer_idx][ptr: ptr + count] = dropped_slots
            self._num_free_slots[layer_idx] += count
        else:
            logger.warning(
                f"[SnapKV] dropped 0 tokens in batch? layer={layer_idx} "
                f"rows={row_indices} cur_len={cur_len}"
            )

        new_len = int(new_slots.shape[1])
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, :new_len] = new_slots
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, new_len:cur_len] = 0
        self.row_seq_lens[layer_idx][row_indices] = new_len

    def free_part_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        if keep_indices is None:
            return
        if not layer_indices or not seqs:
            return
        for layer_idx in layer_indices:
            self.kv_layer_index(int(layer_idx))
        if len(layer_indices) == 1:
            self.free_part_slots_batch(
                int(layer_indices[0]),
                seqs,
                keep_indices[0],
                keep_indices_sorted=keep_indices_sorted,
            )
            return

        self._uniform_decode_metadata = False
        keep_indices = keep_indices.to(device=self.device, dtype=torch.long).contiguous()
        num_layers = len(layer_indices)
        batch_size = len(seqs)
        if keep_indices.dim() != 3 or tuple(keep_indices.shape[:2]) != (num_layers, batch_size):
            raise RuntimeError(
                "free_part_slots_batch_layers expected keep_indices with shape [layers, batch, keep]: "
                f"layers={num_layers} batch={batch_size} keep_shape={tuple(keep_indices.shape)}"
            )
        if int(keep_indices.shape[2]) <= 0:
            raise RuntimeError("free_part_slots_batch_layers got empty keep_indices.")

        row_indices = np.empty((num_layers, batch_size), dtype=np.int64)
        cur_lens = np.empty((num_layers, batch_size), dtype=np.int64)
        for local_layer, layer_idx in enumerate(layer_indices):
            layer_idx = int(layer_idx)
            for seq_idx, seq in enumerate(seqs):
                row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
                if row_idx is None:
                    raise ValueError
                row_indices[local_layer, seq_idx] = int(row_idx)
                cur_lens[local_layer, seq_idx] = int(self.row_seq_lens[layer_idx][row_idx])

        cur_len = int(cur_lens[0, 0])
        if not np.all(cur_lens == cur_len):
            for local_layer, layer_idx in enumerate(layer_indices):
                self.free_part_slots_batch(
                    int(layer_idx),
                    seqs,
                    keep_indices[local_layer],
                    keep_indices_sorted=keep_indices_sorted,
                )
            return

        bounds_ok = ((keep_indices >= 0) & (keep_indices < cur_len)).all()
        if keep_indices.is_cuda:
            torch._assert_async(bounds_ok)
        elif not bool(bounds_ok.item()):
            raise RuntimeError(
                "free_part_slots_batch_layers keep_indices out of bounds: "
                f"cur_len={cur_len} keep_min={int(keep_indices.min().item())} "
                f"keep_max={int(keep_indices.max().item())}"
            )

        if not keep_indices_sorted:
            keep_indices = torch.sort(keep_indices, dim=2).values
        kv_layers_gpu = torch.tensor(
            [self.kv_layer_index(int(layer_idx)) for layer_idx in layer_indices],
            dtype=torch.long,
            device=self.device,
        )
        rows_gpu = torch.from_numpy(row_indices).to(device=self.device, dtype=torch.long)
        old_slots = self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None],
            rows_gpu,
            :cur_len,
        ]
        new_slots = old_slots.gather(2, keep_indices)

        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask.scatter_(2, keep_indices, False)
        dropped_per_layer = old_slots[mask].view(num_layers, -1)
        drop_count = int(dropped_per_layer.shape[1])
        if drop_count > 0:
            if self.free_slots_stack_tensor is not None:
                ptrs = np.asarray([self._num_free_slots[int(layer_idx)] for layer_idx in layer_indices], dtype=np.int64)
                offsets = ptrs[:, None] + np.arange(drop_count, dtype=np.int64)[None, :]
                self.free_slots_stack_tensor[
                    kv_layers_gpu[:, None],
                    torch.from_numpy(offsets).to(device=self.device, dtype=torch.long),
                ] = dropped_per_layer.to(torch.int32)
            else:
                for local_layer, layer_idx in enumerate(layer_indices):
                    layer_idx = int(layer_idx)
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + drop_count] = dropped_per_layer[local_layer]
            for layer_idx in layer_indices:
                self._num_free_slots[int(layer_idx)] += drop_count
        else:
            logger.warning(
                f"[SnapKV] dropped 0 tokens in layer batch? layers={layer_indices} "
                f"rows={row_indices.tolist()} cur_len={cur_len}"
            )

        new_len = int(new_slots.shape[2])
        new_cols = torch.arange(new_len, dtype=torch.long, device=self.device)
        self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None, None],
            rows_gpu[:, :, None],
            new_cols[None, None, :],
        ] = new_slots
        if new_len < cur_len:
            tail_cols = torch.arange(new_len, cur_len, dtype=torch.long, device=self.device)
            self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                tail_cols[None, None, :],
            ] = 0
        for local_layer, layer_idx in enumerate(layer_indices):
            self.row_seq_lens[int(layer_idx)][row_indices[local_layer]] = new_len

    def free_prefix_recent_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        *,
        kv_len: int,
        num_sink_tokens: int,
        num_recent_tokens: int,
    ):
        if not layer_indices or not seqs:
            return

        for layer_idx in layer_indices:
            self.kv_layer_index(int(layer_idx))
        self._uniform_decode_metadata = False
        kv_len = int(kv_len)
        sink_end = min(int(num_sink_tokens), kv_len)
        recent_start = max(sink_end, kv_len - int(num_recent_tokens))
        new_len = sink_end + (kv_len - recent_start)
        if new_len <= 0:
            raise RuntimeError("prefix/recent compaction cannot keep zero tokens.")
        if new_len >= kv_len:
            return

        num_layers = len(layer_indices)
        batch_size = len(seqs)
        row_indices = np.empty((num_layers, batch_size), dtype=np.int64)
        cur_lens = np.empty((num_layers, batch_size), dtype=np.int64)
        for local_layer, layer_idx in enumerate(layer_indices):
            layer_idx = int(layer_idx)
            for seq_idx, seq in enumerate(seqs):
                row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
                if row_idx is None:
                    raise ValueError
                row_indices[local_layer, seq_idx] = int(row_idx)
                cur_lens[local_layer, seq_idx] = int(self.row_seq_lens[layer_idx][row_idx])
        if not np.all(cur_lens == kv_len):
            raise RuntimeError(
                "prefix/recent compaction expected uniform row lengths: "
                f"kv_len={kv_len} observed={cur_lens.tolist()}"
            )

        kv_layers_gpu = torch.tensor(
            [self.kv_layer_index(int(layer_idx)) for layer_idx in layer_indices],
            dtype=torch.long,
            device=self.device,
        )
        rows_gpu = torch.from_numpy(row_indices).to(device=self.device, dtype=torch.long)
        drop_cols = torch.arange(sink_end, recent_start, dtype=torch.long, device=self.device)
        dropped_per_layer = self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None, None],
            rows_gpu[:, :, None],
            drop_cols[None, None, :],
        ].reshape(num_layers, -1)
        drop_count = int(dropped_per_layer.shape[1])
        if drop_count > 0:
            if self.free_slots_stack_tensor is not None:
                ptrs = np.asarray([self._num_free_slots[int(layer_idx)] for layer_idx in layer_indices], dtype=np.int64)
                offsets = ptrs[:, None] + np.arange(drop_count, dtype=np.int64)[None, :]
                self.free_slots_stack_tensor[
                    kv_layers_gpu[:, None],
                    torch.from_numpy(offsets).to(device=self.device, dtype=torch.long),
                ] = dropped_per_layer.to(torch.int32)
            else:
                for local_layer, layer_idx in enumerate(layer_indices):
                    layer_idx = int(layer_idx)
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + drop_count] = dropped_per_layer[local_layer]
            for layer_idx in layer_indices:
                self._num_free_slots[int(layer_idx)] += drop_count

        if recent_start < kv_len:
            recent_cols = torch.arange(recent_start, kv_len, dtype=torch.long, device=self.device)
            dst_cols = torch.arange(sink_end, new_len, dtype=torch.long, device=self.device)
            recent_slots = self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                recent_cols[None, None, :],
            ]
            self.buffer_req_to_token_slots_tensor[
                kv_layers_gpu[:, None, None],
                rows_gpu[:, :, None],
                dst_cols[None, None, :],
            ] = recent_slots
        tail_cols = torch.arange(new_len, kv_len, dtype=torch.long, device=self.device)
        self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None, None],
            rows_gpu[:, :, None],
            tail_cols[None, None, :],
        ] = 0
        for local_layer, layer_idx in enumerate(layer_indices):
            self.row_seq_lens[int(layer_idx)][row_indices[local_layer]] = new_len

    def materialize_prefill_staging_layer(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        self.kv_layer_index(layer_idx)
        if not self.has_prefill_staging_view(layer_idx):
            raise RuntimeError("PyramidKV prefill staging is not active.")
        materialized_key = (int(layer_idx), int(seq.seq_id))
        if materialized_key in self._pyramidkv_prefill_staging_materialized_layers:
            raise RuntimeError(
                f"PyramidKV prefill staging layer materialized twice: "
                f"layer={layer_idx} seq_id={seq.seq_id}."
            )

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
        staging_offset = int(self._pyramidkv_prefill_staging_seq_offsets[int(seq.seq_id)])
        staging_indices = keep_indices + staging_offset
        k_cache[slots] = k_stage[staging_indices]
        v_cache[slots] = v_stage[staging_indices]

        self._pyramidkv_prefill_staging_materialized_layers.add(materialized_key)
        expected_materializations = int(self.num_kv_layers) * len(self._pyramidkv_prefill_staging_seq_offsets)
        if len(self._pyramidkv_prefill_staging_materialized_layers) == expected_materializations:
            self._pyramidkv_prefill_staging_active = False
            self._release_pyramidkv_long_prefill_offload_rows()

    def materialize_prefill_staging_layer_batch(
        self,
        layer_idx: int,
        seq_keep_indices: list[tuple[Sequence, torch.Tensor]],
    ):
        if not seq_keep_indices:
            return
        if len(seq_keep_indices) == 1:
            seq, keep_indices = seq_keep_indices[0]
            self.materialize_prefill_staging_layer(layer_idx, seq, keep_indices)
            return
        self.kv_layer_index(layer_idx)
        if not self.has_prefill_staging_view(layer_idx):
            raise RuntimeError("PyramidKV prefill staging is not active.")

        seq_ids = []
        keep_tensors = []
        keep_sizes = []
        all_staging_indices = []
        materialized_keys = []
        for seq, keep_indices in seq_keep_indices:
            materialized_key = (int(layer_idx), int(seq.seq_id))
            if materialized_key in self._pyramidkv_prefill_staging_materialized_layers:
                raise RuntimeError(
                    f"PyramidKV prefill staging layer materialized twice: "
                    f"layer={layer_idx} seq_id={seq.seq_id}."
                )

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

            staging_offset = int(self._pyramidkv_prefill_staging_seq_offsets[int(seq.seq_id)])
            seq_ids.append(int(seq.seq_id))
            keep_tensors.append(keep_indices)
            keep_sizes.append(num_keep)
            all_staging_indices.append(keep_indices + staging_offset)
            materialized_keys.append(materialized_key)

        slots = torch.cat(
            [
                self._allocate(layer_idx, seq_id, num_keep).to(torch.long)
                for seq_id, num_keep in zip(seq_ids, keep_sizes)
            ],
            dim=0,
        )
        staging_indices = torch.cat(all_staging_indices, dim=0)
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        k_stage = self.pyramidkv_prefill_staging_kv_cache[0]
        v_stage = self.pyramidkv_prefill_staging_kv_cache[1]
        k_cache[slots] = k_stage[staging_indices]
        v_cache[slots] = v_stage[staging_indices]

        self._pyramidkv_prefill_staging_materialized_layers.update(materialized_keys)
        expected_materializations = int(self.num_kv_layers) * len(self._pyramidkv_prefill_staging_seq_offsets)
        if len(self._pyramidkv_prefill_staging_materialized_layers) == expected_materializations:
            self._pyramidkv_prefill_staging_active = False
            self._release_pyramidkv_long_prefill_offload_rows()

    def _pyramidkv_long_prefill_offload_kind(self) -> str:
        return "pyramidkv_post_rope"

    def _release_pyramidkv_long_prefill_offload_rows(self):
        if getattr(self, "_pyramidkv_long_prefill_offload_seq_id", None) is None:
            return
        self._pyramidkv_clear_long_prefill_offload_prefetch()
        seq_id = int(self._pyramidkv_long_prefill_offload_seq_id)
        seen_rows = set()
        for layer_idx in self.kv_transformer_layer_indices():
            row_idx = self.seq_id_to_row[layer_idx].get(seq_id)
            if row_idx is None:
                continue
            row_idx = int(row_idx)
            if row_idx in seen_rows:
                continue
            self.raw_kv_offload_buffer.release_row(row_idx)
            seen_rows.add(row_idx)

    def _pyramidkv_long_prefill_offload_row(self, layer_idx: int) -> int:
        seq_id = self._pyramidkv_long_prefill_offload_seq_id
        if seq_id is None:
            raise RuntimeError("PyramidKV long-prefill offload has no active seq_id.")
        row_idx = self.seq_id_to_row[int(layer_idx)].get(int(seq_id))
        if row_idx is None:
            raise RuntimeError(
                "PyramidKV long-prefill offload row is missing: "
                f"layer={layer_idx} seq_id={seq_id}."
            )
        return int(row_idx)

    def _pyramidkv_long_prefill_offload_prefetch_enabled(self) -> bool:
        return device_runtime.supports_streams(self.device)

    def _pyramidkv_clear_long_prefill_offload_prefetch(self):
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        for state in list(states.values()):
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
        self._pyramidkv_long_prefill_offload_prefetch_states = {}

    def _pyramidkv_drop_long_prefill_offload_prefetch(self, key: tuple[int, int, str, int]):
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        state = states.pop(key, None)
        if state is not None:
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
        self._pyramidkv_long_prefill_offload_prefetch_states = states

    def _pyramidkv_consume_long_prefill_offload_staged_prefetch(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        end: int,
    ) -> bool:
        kind = self._pyramidkv_long_prefill_offload_kind()
        key = (int(layer_idx), int(row_idx), kind, int(end))
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        state = states.pop(key, None)
        if state is None:
            self._pyramidkv_long_prefill_offload_prefetch_states = states
            return False
        with profiler.record("pyramidkv_long_prefill_offload_prefetch_wait"):
            device_runtime.wait_event(state["event"], device=self.device)
        self._pyramidkv_long_prefill_offload_prefetch_states = states
        return True

    def _pyramidkv_schedule_next_long_prefill_offload_prefetch(self, *, layer_idx: int, end: int):
        if int(end) <= 0 or not self._pyramidkv_long_prefill_offload_prefetch_enabled():
            return
        next_layers = [
            candidate
            for candidate in self.kv_transformer_layer_indices()
            if int(candidate) > int(layer_idx)
        ]
        if not next_layers:
            return
        next_layer = int(next_layers[0])
        row_idx = self._pyramidkv_long_prefill_offload_row(next_layer)
        kind = self._pyramidkv_long_prefill_offload_kind()
        key = (next_layer, int(row_idx), kind, int(end))
        states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        keep_keys = {key}
        for old_key in list(states):
            if old_key not in keep_keys:
                self._pyramidkv_drop_long_prefill_offload_prefetch(old_key)
                states = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_states", None) or {}
        if key in states:
            return

        stream = getattr(self, "_pyramidkv_long_prefill_offload_prefetch_stream", None)
        if stream is None:
            stream = device_runtime.new_stream(device=self.device)
            if stream is None:
                raise RuntimeError(
                    "PyramidKV long-prefill offload prefetch is enabled, but the active platform "
                    f"does not support streams for device={self.device}."
                )
            self._pyramidkv_long_prefill_offload_prefetch_stream = stream

        with profiler.record("pyramidkv_long_prefill_offload_prefetch_schedule"):
            staging_available_event = device_runtime.new_event(device=self.device)
            if staging_available_event is None:
                raise RuntimeError(
                    "PyramidKV long-prefill offload prefetch could not create a staging event "
                    f"for device={self.device}."
                )
            device_runtime.record_event(staging_available_event, device=self.device)
            with device_runtime.stream_context(stream):
                device_runtime.stream_wait_event(stream, staging_available_event)
                self.raw_kv_offload_buffer.copy_prefix_to(
                    layer_idx=next_layer,
                    row_idx=row_idx,
                    kind=kind,
                    end=end,
                    k_out=self.pyramidkv_prefill_staging_kv_cache[0, :end],
                    v_out=self.pyramidkv_prefill_staging_kv_cache[1, :end],
                )
                event = device_runtime.new_event(device=self.device)
                if event is None:
                    raise RuntimeError(
                        "PyramidKV long-prefill offload prefetch could not create a completion event "
                        f"for device={self.device}."
                    )
                device_runtime.record_event(event, device=self.device)
        states[key] = {
            "layer_idx": next_layer,
            "row_idx": int(row_idx),
            "kind": kind,
            "end": int(end),
            "staging_available_event": staging_available_event,
            "event": event,
        }
        self._pyramidkv_long_prefill_offload_prefetch_states = states

    def _pyramidkv_schedule_post_layer_long_prefill_offload_prefetch(self, layer_idx: int):
        if not bool(getattr(self, "_pyramidkv_long_prefill_offload_step_active", False)):
            return
        start = int(getattr(self, "_pyramidkv_long_prefill_offload_start", 0) or 0)
        if start <= 0:
            return
        with profiler.record("pyramidkv_long_prefill_offload_after_attention_prefetch"):
            self._pyramidkv_schedule_next_long_prefill_offload_prefetch(
                layer_idx=layer_idx,
                end=start,
            )

    @torch.no_grad()
    def before_prefill_layer_attention(self, layer_idx: int, selection: SparseSelection):
        del selection
        if not bool(getattr(self, "_pyramidkv_long_prefill_offload_step_active", False)):
            return None
        if not self.has_prefill_staging_view(layer_idx):
            return None
        start = int(getattr(self, "_pyramidkv_long_prefill_offload_start", 0) or 0)
        if start <= 0:
            return None
        row_idx = self._pyramidkv_long_prefill_offload_row(layer_idx)
        with profiler.record("pyramidkv_long_prefill_offload_wait_or_restore"):
            staged = self._pyramidkv_consume_long_prefill_offload_staged_prefetch(
                layer_idx=layer_idx,
                row_idx=row_idx,
                end=start,
            )
            if staged:
                return None
        with profiler.record("pyramidkv_long_prefill_offload_restore_prefix"):
            self.raw_kv_offload_buffer.copy_prefix_to(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=self._pyramidkv_long_prefill_offload_kind(),
                end=start,
                k_out=self.pyramidkv_prefill_staging_kv_cache[0, :start],
                v_out=self.pyramidkv_prefill_staging_kv_cache[1, :start],
            )
        return None

    @torch.no_grad()
    def _offload_pyramidkv_long_prefill_layer(self, layer_idx: int):
        start = int(getattr(self, "_pyramidkv_long_prefill_offload_start", 0) or 0)
        end = int(getattr(self, "_pyramidkv_long_prefill_offload_end", 0) or 0)
        total_len = int(getattr(self, "_pyramidkv_long_prefill_offload_total_len", 0) or 0)
        if end <= start:
            raise RuntimeError(
                "PyramidKV long-prefill offload has invalid range: "
                f"layer={layer_idx} start={start} end={end}."
            )
        row_idx = self._pyramidkv_long_prefill_offload_row(layer_idx)
        k = self.pyramidkv_prefill_staging_kv_cache[0, start:end]
        v = self.pyramidkv_prefill_staging_kv_cache[1, start:end]
        kind = self._pyramidkv_long_prefill_offload_kind()
        with profiler.record("pyramidkv_long_prefill_offload_ensure_entry"):
            self.raw_kv_offload_buffer.ensure_entry(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                total_len=total_len,
                k_shape_tail=tuple(k.shape[1:]),
                v_shape_tail=tuple(v.shape[1:]),
                dtype=k.dtype,
            )
        with profiler.record("pyramidkv_long_prefill_offload_put_range"):
            self.raw_kv_offload_buffer.put_range(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                start=start,
                k=k,
                v=v,
            )

    def on_layer_attention_end(self, layer_idx: int):
        if not self.has_prefill_staging_view(layer_idx):
            return
        if not bool(getattr(self, "_pyramidkv_long_prefill_offload_step_active", False)):
            return
        if bool(getattr(self, "_pyramidkv_long_prefill_offload_is_last_chunk", False)):
            if not bool(getattr(self, "_pyramidkv_prefill_staging_active", False)):
                self._release_pyramidkv_long_prefill_offload_rows()
            return
        self._offload_pyramidkv_long_prefill_layer(layer_idx)
        self._pyramidkv_schedule_post_layer_long_prefill_offload_prefetch(layer_idx)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        self._pyramidkv_reset_full_prefill_staging()
        self._pyramidkv_long_prefill_offload_step_active = bool(
            is_prefill and self._should_use_pyramidkv_long_prefill_offload_staging(seqs)
        )
        return super().prepare_step(seqs, is_prefill)

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            self._decode_static_state_binding_key = None
            layer_ids = self.kv_transformer_layer_indices()
            for seq in seqs:
                if int(seq.num_prefilled_tokens) == 0:
                    self._clear_prefill_attention_scores(seq.seq_id)

            use_long_prefill_offload_staging = self._should_use_pyramidkv_long_prefill_offload_staging(seqs)
            use_full_prefill_staging = (
                self._should_use_pyramidkv_full_prefill_staging(seqs)
                or use_long_prefill_offload_staging
            )
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
                staging_start = int(seqs[0].num_prefilled_tokens) if use_long_prefill_offload_staging else 0
                layers_slot_mapping_cuda = torch.arange(
                    staging_start,
                    staging_start + total_chunk_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ).expand(self.num_layers, -1)
            else:
                layers_slot_mapping_cuda = torch.empty(
                    (self.num_layers, total_chunk_tokens), dtype=torch.int32, device=self.device
                )
            context_lens_list = [[] for _ in range(self.num_layers)]

            use_batched_prefill_alloc = (
                not use_full_prefill_staging
                and self._allocate_prefill_batch_same_size_all_layers(seqs, layers_slot_mapping_cuda)
            )
            if use_batched_prefill_alloc:
                for layer_id in layer_ids:
                    context_lens_list[layer_id] = [
                        int(seq.num_prefilled_tokens + seq.current_chunk_size) for seq in seqs
                    ]

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                if not use_batched_prefill_alloc:
                    for layer_id in layer_ids:
                        if seq.seq_id in self.seq_id_to_row[layer_id]:
                            row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                            expected_row_len = 0 if use_long_prefill_offload_staging else start_idx
                            if self.row_seq_lens[layer_id][row_idx] != expected_row_len:
                                raise ValueError(
                                    "KV cache row length mismatch in prefill: "
                                    f"layer={layer_id} seq_id={seq.seq_id} "
                                    f"row_seq_len={self.row_seq_lens[layer_id][row_idx]} "
                                    f"expected={expected_row_len} start_idx={start_idx}"
                                )
                        if use_full_prefill_staging:
                            if start_idx != 0 and not use_long_prefill_offload_staging:
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

            layers_context_lens_np = np.zeros((self.num_layers, len(seqs)), dtype=np.int32)
            for layer_id in layer_ids:
                layers_context_lens_np[layer_id] = context_lens_list[layer_id]
            layers_context_lens_cuda = torch.from_numpy(layers_context_lens_np).to(
                device=self.device,
                dtype=torch.int32,
            )

            for layer_id in layer_ids:
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
                    visible_len = int(seq.num_prefilled_tokens) + chunk_size if use_long_prefill_offload_staging else chunk_size
                    slot_start = 0 if use_long_prefill_offload_staging else offset
                    self._pyramidkv_prefill_staging_seq_offsets[int(seq.seq_id)] = int(slot_start)
                    active_slots[b_idx, :visible_len] = torch.arange(
                        slot_start,
                        slot_start + visible_len,
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
                if use_long_prefill_offload_staging:
                    seq = seqs[0]
                    self._pyramidkv_long_prefill_offload_seq_id = int(seq.seq_id)
                    self._pyramidkv_long_prefill_offload_start = int(seq.num_prefilled_tokens)
                    self._pyramidkv_long_prefill_offload_end = int(seq.num_prefilled_tokens + seq.current_chunk_size)
                    self._pyramidkv_long_prefill_offload_total_len = int(seq.num_prompt_tokens)
                    self._pyramidkv_long_prefill_offload_is_last_chunk = bool(seq.is_last_chunk_prefill)

            input_ids = torch.from_numpy(input_ids_np).to(self.device)
            positions = torch.from_numpy(positions_np).to(self.device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            self._decode_static_state_binding_key = None
            layer_ids = self.kv_transformer_layer_indices()
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, batch_size), dtype=torch.int32, device=self.device
            )
            layers_context_lens = []

            for layer_id in layer_ids:
                new_slots_batch = self._allocate_batch(layer_id, seq_ids, 1)
                layers_slot_mapping_cuda[layer_id] = new_slots_batch

                row_indices = [self.seq_id_to_row[layer_id][sid] for sid in seq_ids]
                layers_context_lens.append(self.row_seq_lens[layer_id][row_indices])

            layers_context_lens_np = np.zeros((self.num_layers, batch_size), dtype=np.int32)
            for local_layer, layer_id in enumerate(layer_ids):
                layers_context_lens_np[layer_id] = layers_context_lens[local_layer]
            layers_context_lens_cuda = torch.from_numpy(layers_context_lens_np).to(
                device=self.device,
                dtype=torch.int32,
            )

            for layer_id in layer_ids:
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                # row_seq_lens is layer-wise after prefill/decode eviction; attention
                # uses this to size flash decode stage1/stage2 workspaces.
                state.max_context_len = (
                    int(max(layers_context_lens_np[layer_id]))
                    if len(layers_context_lens_np[layer_id]) > 0
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

    def _get_decode_static_index_buffers(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size)
        buffers = self._decode_static_index_buffers.get(batch_size)
        if buffers is None:
            buffers = (
                torch.empty((self.num_layers, batch_size), dtype=torch.long, device=self.device),
                torch.empty((self.num_layers, batch_size), dtype=torch.long, device=self.device),
                torch.empty((self.num_layers, batch_size), dtype=torch.long, device=self.device),
            )
            self._decode_static_index_buffers[batch_size] = buffers
        return buffers

    def _bind_decode_static_layer_states(
        self,
        graph_batch_size: int,
        layers_slot_mapping: torch.Tensor,
        layers_context_lens: torch.Tensor,
        layers_req_indices: torch.Tensor,
        max_context_lens: np.ndarray,
    ) -> None:
        binding_key = (
            int(graph_batch_size),
            int(layers_slot_mapping.data_ptr()),
            int(layers_context_lens.data_ptr()),
            int(layers_req_indices.data_ptr()),
        )
        if self._decode_static_state_binding_key != binding_key:
            for layer_id in self.kv_transformer_layer_indices():
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping[layer_id]
                state.context_lens = layers_context_lens[layer_id]
                state.req_indices = layers_req_indices[layer_id]
            self._decode_static_state_binding_key = binding_key
        for layer_id in self.kv_transformer_layer_indices():
            self.layer_batch_states[layer_id].max_context_len = int(max_context_lens[layer_id])

    def _prepare_decode_static_uniform(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        real_batch_size = len(seqs)
        graph_batch_size = int(input_ids.numel())
        input_ids_list = [seq.last_token for seq in seqs]
        positions_list = [seq.num_tokens - 1 for seq in seqs]
        seq_ids = [seq.seq_id for seq in seqs]
        layer_ids = self.kv_transformer_layer_indices()
        first_layer = int(layer_ids[0])
        row_indices = [self.seq_id_to_row[first_layer][sid] for sid in seq_ids]
        if any(
            self.seq_id_to_row[layer_id].get(sid) != row_idx
            for layer_id in layer_ids[1:]
            for sid, row_idx in zip(seq_ids, row_indices)
        ):
            self._uniform_decode_metadata = False
            return None

        cur_lens = self.row_seq_lens[first_layer][row_indices]
        if len(cur_lens) > 0 and int(max(cur_lens)) + 1 > int(self.max_model_len):
            raise RuntimeError(
                "KV row length exceeds max_model_len in uniform prepare_decode_static: "
                f"max_cur_len={int(max(cur_lens))} max_model_len={int(self.max_model_len)}"
            )
        if any(int(self._num_free_slots[layer_id]) < real_batch_size for layer_id in layer_ids):
            raise RuntimeError(
                "Out of KV cache slots in uniform prepare_decode_static: "
                f"need={real_batch_size} free={min(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)}"
            )

        ptr = int(self._num_free_slots[first_layer])
        new_slots_batch = self.free_slots_stack[first_layer][ptr - real_batch_size : ptr]
        for layer_id in layer_ids:
            self._num_free_slots[layer_id] -= real_batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device=self.device)
        kv_layers_gpu = torch.tensor(
            [self.kv_layer_index(layer_id) for layer_id in layer_ids],
            dtype=torch.long,
            device=self.device,
        )
        self.buffer_req_to_token_slots_tensor[
            kv_layers_gpu[:, None],
            rows_gpu[None, :],
            cols_gpu[None, :],
        ] = new_slots_batch.to(torch.int32).unsqueeze(0)
        for layer_id in layer_ids:
            self.row_seq_lens[layer_id][row_indices] += 1
        real_context_lens = self.row_seq_lens[first_layer][row_indices]
        real_max_context_len = int(max(real_context_lens)) if row_indices else 0

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

        for layer_id in layer_ids:
            state = self.layer_batch_states[layer_id]
            state.slot_mapping = slot_mapping
            state.context_lens = context_lens
            state.max_context_len = real_max_context_len
            state.req_indices = req_indices
        self._decode_static_state_binding_key = None
        return input_ids, positions, None

    def _allocate_decode_batch_all_layers(
        self,
        seq_ids: list[int],
        *,
        static_cap: int | None = None,
        slot_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, bool]:
        batch_size = len(seq_ids)
        layer_ids = self.kv_transformer_layer_indices()
        row_indices = np.zeros((self.num_layers, batch_size), dtype=np.int64)
        cur_lens = np.zeros((self.num_layers, batch_size), dtype=np.int64)
        for layer_id in layer_ids:
            rows = [self._get_free_row(layer_id, sid) for sid in seq_ids]
            row_indices[layer_id] = rows
            cur_lens[layer_id] = self.row_seq_lens[layer_id][rows]

        active_cur_lens = cur_lens[list(layer_ids)] if layer_ids else cur_lens
        max_cur_len = int(active_cur_lens.max()) if active_cur_lens.size else 0
        if max_cur_len + 1 > int(self.max_model_len):
            raise RuntimeError(
                "KV row length exceeds max_model_len in batched static decode allocation: "
                f"max_cur_len={max_cur_len} max_model_len={int(self.max_model_len)}"
            )
        next_lens = cur_lens + 1
        if static_cap is not None:
            max_context_lens = next_lens.max(axis=1) if next_lens.size else np.zeros(self.num_layers)
            too_long = np.nonzero(max_context_lens > int(static_cap))[0]
            too_long = np.asarray([layer_id for layer_id in too_long if layer_id in set(layer_ids)], dtype=np.int64)
            if too_long.size > 0:
                layer_id = int(too_long[0])
                raise RuntimeError(
                    "static decode context length exceeds captured graph max_context_len: "
                    f"layer={layer_id} real_max_context_len={int(max_context_lens[layer_id])} "
                    f"static_cap={int(static_cap)}"
                )
        min_free = min(int(self._num_free_slots[layer_id]) for layer_id in layer_ids)
        if min_free < batch_size:
            raise RuntimeError(
                "Out of KV cache slots in batched static decode allocation: "
                f"need={batch_size} free={min_free}"
            )

        layer_index = torch.tensor(layer_ids, dtype=torch.long, device=self.device)
        kv_layer_indices = torch.tensor(
            [self.kv_layer_index(layer_id) for layer_id in layer_ids],
            dtype=torch.long,
            device=self.device,
        )
        if self.free_slots_stack_tensor is not None:
            ptrs = np.asarray([self._num_free_slots[layer_id] for layer_id in layer_ids], dtype=np.int64)
            slot_offsets = ptrs[:, None] - batch_size + np.arange(batch_size, dtype=np.int64)[None, :]
            _slot_offsets_gpu, _rows_gpu, _cols_gpu = self._get_decode_static_index_buffers(batch_size)
            slot_offsets_active = torch.from_numpy(slot_offsets).to(device=self.device, dtype=torch.long)
            selected_active = torch.gather(
                self.free_slots_stack_tensor.index_select(0, kv_layer_indices),
                1,
                slot_offsets_active,
            )
            if slot_output is not None:
                selected_slots = slot_output
                active_rows = selected_slots.index_select(0, layer_index)
                active_rows[:, :batch_size] = selected_active
                selected_slots.index_copy_(0, layer_index, active_rows)
                wrote_slot_output = True
            else:
                selected_slots = torch.empty((self.num_layers, batch_size), dtype=torch.int32, device=self.device)
                selected_slots.fill_(-1)
                selected_slots.index_copy_(0, layer_index, selected_active)
                wrote_slot_output = False
            for layer_id in layer_ids:
                self._num_free_slots[layer_id] -= batch_size
        else:
            _slot_offsets_gpu, rows_gpu, cols_gpu = self._get_decode_static_index_buffers(batch_size)
            if slot_output is not None:
                selected_slots = slot_output
                wrote_slot_output = True
            else:
                selected_slots = torch.empty((self.num_layers, batch_size), dtype=torch.int32, device=self.device)
                selected_slots.fill_(-1)
                wrote_slot_output = False
            for layer_id in layer_ids:
                ptr = int(self._num_free_slots[layer_id])
                selected_slots[layer_id, :batch_size].copy_(self.free_slots_stack[layer_id][ptr - batch_size: ptr])
                self._num_free_slots[layer_id] -= batch_size

        rows_active = torch.from_numpy(row_indices[list(layer_ids)]).to(device=self.device, dtype=torch.long)
        cols_active = torch.from_numpy(cur_lens[list(layer_ids)]).to(device=self.device, dtype=torch.long)
        selected_active = selected_slots.index_select(0, layer_index)[:, :batch_size]
        self.buffer_req_to_token_slots_tensor[
            kv_layer_indices[:, None],
            rows_active,
            cols_active,
        ] = selected_active

        for layer_id in layer_ids:
            self.row_seq_lens[layer_id][row_indices[layer_id]] += 1
        return selected_slots, next_lens, row_indices, wrote_slot_output

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

            if self._uniform_decode_metadata:
                result = self._prepare_decode_static_uniform(
                    seqs,
                    input_ids,
                    positions,
                    slot_mapping,
                    context_lens,
                    req_indices,
                )
                if result is not None:
                    return result

            input_ids[:real_batch_size].copy_(torch.tensor(input_ids_list, dtype=torch.int64, device=self.device))
            positions[:real_batch_size].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
            if graph_batch_size > real_batch_size:
                input_ids[real_batch_size:].fill_(int(input_ids_list[0]))
                positions[real_batch_size:].fill_(int(positions_list[0]))

            layers_slot_mapping, layers_context_lens, layers_req_indices = self._get_decode_static_buffers(
                graph_batch_size
            )

            static_cap = getattr(self, "_decode_static_max_context_len", None)
            new_slots, context_lens_np, req_indices_np, wrote_slot_output = self._allocate_decode_batch_all_layers(
                seq_ids,
                static_cap=None if static_cap is None else int(static_cap),
                slot_output=layers_slot_mapping,
            )
            max_context_lens = context_lens_np.max(axis=1) if context_lens_np.size else np.zeros(self.num_layers)

            if not wrote_slot_output:
                layers_slot_mapping[:, :real_batch_size].copy_(new_slots)
            layers_context_lens[:, :real_batch_size].copy_(
                torch.from_numpy(context_lens_np).to(device=self.device, dtype=torch.int32)
            )
            layers_req_indices[:, :real_batch_size].copy_(
                torch.from_numpy(req_indices_np).to(device=self.device, dtype=torch.int32)
            )
            if graph_batch_size > real_batch_size:
                layers_slot_mapping[:, real_batch_size:].fill_(-1)
                layers_context_lens[:, real_batch_size:].copy_(
                    torch.from_numpy(context_lens_np[:, :1]).to(device=self.device, dtype=torch.int32)
                )
                layers_req_indices[:, real_batch_size:].copy_(
                    torch.from_numpy(req_indices_np[:, :1]).to(device=self.device, dtype=torch.int32)
                )

            self._bind_decode_static_layer_states(
                graph_batch_size,
                layers_slot_mapping,
                layers_context_lens,
                layers_req_indices,
                max_context_lens,
            )

            first_layer = int(self.kv_transformer_layer_indices()[0])
            slot_mapping.copy_(layers_slot_mapping[first_layer])
            context_lens.copy_(layers_context_lens[first_layer])
            req_indices.copy_(layers_req_indices[first_layer])

            return input_ids, positions, None
