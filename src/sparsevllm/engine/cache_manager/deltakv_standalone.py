from __future__ import annotations

from collections import deque
import os

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.layers.rotary_embedding import get_rope
from sparsevllm.utils.log import logger
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates
from .deltakv import DeltaKVCacheTritonManagerV4


class DeltaKVStandaloneCacheManager(DeltaKVCacheTritonManagerV4):
    """Standalone DeltaKV with per-layer persistent cache + one global temp reconstruct pool.

    All layers use DeltaKV compression. There is no notion of full-attention layers and no
    OmniKV-style top-k decode selection. At decode/prefill-time attention, each layer rebuilds
    the full visible context into a single shared scratch KV pool and attends over that pool.
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        CacheManager.__init__(self, config, rank, world_size)
        assert world_size == 1, "DeltaKVStandaloneCacheManager currently only supports world_size=1."

        # Standalone mode: every layer uses DeltaKV.
        self.config.full_attn_layers = []
        self.config.obs_layer_ids = []
        self.full_attn_layers = []
        self.deltakv_layer_ids = list(range(self.num_layers))
        self.full_layer_ids = []
        self.deltakv_layer_to_idx = {l: l for l in self.deltakv_layer_ids}
        self.full_layer_to_idx = {}

        self.full_num_slots = 0
        self.deltakv_latent_num_slots = 0
        self.deltakv_full_num_slots = 0
        self.deltakv_temp_num_slots = 0
        self.full_kv_cache = None
        self.deltakv_full_kv_cache = None
        self.deltakv_temp_kv_cache = None
        self.deltakv_latent_cache = None
        self.deltakv_latent_to_full_slots = None
        self.deltakv_slot_to_pos = None
        self._deltakv_temp_full_reserve = 0
        self._deltakv_centers_capacity = 0
        self._deltakv_centers_reserved_total = 0
        self._deltakv_centers_reserved_by_seq: dict[int, int] = {}

        self.allocate_kv_cache()

        self.free_slots_stack_deltakv_full = torch.arange(self.deltakv_full_num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots_deltakv_full = self.deltakv_full_num_slots

        self.free_slots_stack_deltakv_latent = torch.arange(self.deltakv_latent_num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots_deltakv_latent = self.deltakv_latent_num_slots

        self.free_slots_stack_deltakv_temp = torch.arange(self.deltakv_temp_num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots_deltakv_temp = self.deltakv_temp_num_slots
        self.deltakv_temp_slot_ids = torch.arange(self.deltakv_temp_num_slots, dtype=torch.int32, device="cuda")

        self.sparse_layer_raw_slots_map = torch.full(
            (self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda"
        )
        self.sparse_layer_latent_slots_map = torch.full(
            (self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda"
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_deltakv_compressed_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_deltakv_center_slots = [[None for _ in range(self.num_layers)] for _ in range(self.max_buffer_rows)]

        self.deltakv_layer_batch_states = LayerBatchStates()
        self._standalone_view_cache_key: tuple[int, int] | None = None
        self._standalone_view_cache_value: dict[str, torch.Tensor | int | dict[int, torch.Tensor]] | None = None

        from sparsevllm.utils.compressor import create_compressor

        self.compress_down = []
        self.compress_up = []
        for _ in range(self.num_layers):
            self.compress_down.append(create_compressor(is_down=True, config=config).cuda())
            self.compress_up.append(create_compressor(is_down=False, config=config).cuda())

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_model_len,
            base=self.hf_config.rope_theta,
            rope_scaling=None,
        ).cuda()
        self.cos_sin_cache = self.rotary_emb.cos_sin_cache
        self.cos_sin_dense = self.cos_sin_cache[:, 0, :]

    def _reset_standalone_view_cache(self):
        self._standalone_view_cache_key = None
        self._standalone_view_cache_value = None

    def _deltakv_reconstruct_writeback_srcdst(
        self,
        *,
        kv_delta: torch.Tensor,
        father_slots: torch.Tensor,
        slot_to_pos: torch.Tensor,
        out_slots: torch.Tensor,
        out_pos: torch.Tensor,
        src_k_cache: torch.Tensor,
        src_v_cache: torch.Tensor,
        dst_k_cache: torch.Tensor,
        dst_v_cache: torch.Tensor,
    ):
        from sparsevllm.triton_kernel.deltakv_kernels import (
            deltakv_reconstruct_writeback_grouped_heads_srcdst,
        )

        hp = int(getattr(self.config, "deltakv_triton_reconstruct_heads_per_program", 4) or 1)
        hp = max(1, min(hp, int(self.num_kv_heads)))
        return deltakv_reconstruct_writeback_grouped_heads_srcdst(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=self.cos_sin_dense,
            src_k_cache=src_k_cache,
            src_v_cache=src_v_cache,
            dst_k_cache=dst_k_cache,
            dst_v_cache=dst_v_cache,
            heads_per_program=hp,
        )

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        dtype_size = torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()

        num_layers = self.num_layers
        latent_bytes = config.kv_compressed_size * dtype_size
        cluster_ratio = max(0.0, float(config.cluster_ratio))

        sink = int(config.num_sink_tokens)
        recent = int(config.num_recent_tokens)
        max_seqs = int(config.max_num_seqs_in_batch)
        max_total_len = int(config.max_model_len)
        max_step_chunk = int(min(int(config.max_num_batched_tokens), max_seqs * int(config.chunk_prefill_size)))
        overhead_slots = max_seqs * (sink + 2 * recent) + max_step_chunk
        cluster_step = max(1, int(1.0 / max(1e-6, cluster_ratio))) if cluster_ratio > 0 else 0
        centers_per_seq = self._estimate_centers_for_total_len(max_total_len)
        center_slack = int(np.ceil(max_step_chunk / max(1, cluster_step))) if cluster_ratio > 0 else 0
        desired_centers = max(1, int(max_seqs * centers_per_seq + center_slack))
        temp_decode_upper = max_seqs * max_total_len
        reserve_ratio = float(config.deltakv_full_pool_reserve_ratio)
        reserve_ratio = max(0.0, min(0.5, reserve_ratio))

        temp_slots_override = os.getenv("SPARSEVLLM_DELTAKV_STANDALONE_TEMP_SLOTS", "").strip()

        temp_override_i = int(temp_slots_override) if temp_slots_override else None
        centers_capacity = desired_centers
        full_slots = int(overhead_slots + centers_capacity)
        temp_slots_target = int(temp_override_i) if temp_override_i is not None else int(temp_decode_upper)
        const_bytes = temp_slots_target * slot_bytes_per_layer + full_slots * num_layers * slot_bytes_per_layer
        bytes_budget = available_memory - const_bytes
        if bytes_budget <= 0:
            raise RuntimeError(
                "Not enough GPU memory left for standalone DeltaKV after reserving persistent overhead/temp margin. "
                "Reduce chunk_prefill_size/max_num_batched_tokens or increase gpu_memory_utilization."
            )

        latent_slots = max(1, int(bytes_budget // (num_layers * latent_bytes)))
        if reserve_ratio > 0:
            latent_slots = max(1, int(latent_slots * (1.0 - reserve_ratio)))

        total_bytes = (
            temp_slots_target * slot_bytes_per_layer
            + full_slots * num_layers * slot_bytes_per_layer
            + latent_slots * num_layers * latent_bytes
        )
        if total_bytes > available_memory:
            raise RuntimeError(
                "Not enough GPU memory for standalone DeltaKV caches after exact capacity check. "
                f"required={total_bytes / 1024**3:.2f}GiB available={available_memory / 1024**3:.2f}GiB."
            )

        self.full_num_slots = 0
        self.deltakv_latent_num_slots = int(latent_slots)
        self.deltakv_full_num_slots = int(full_slots)
        self._deltakv_centers_capacity = int(centers_capacity)
        self.deltakv_temp_num_slots = int(temp_slots_target)
        self._deltakv_temp_full_reserve = 0

        logger.info(
            f"DeltaKV standalone allocation: persistent_slots={self.deltakv_full_num_slots} "
            f"(overhead={overhead_slots}, centers={centers_capacity}, centers_per_seq={centers_per_seq}); "
            f"latent_slots={self.deltakv_latent_num_slots}; temp_slots={self.deltakv_temp_num_slots} "
            f"(layers={num_layers}, deltakv_full_pool_reserve_ratio={reserve_ratio:.3f})."
        )

        self.deltakv_full_kv_cache = torch.empty(
            2,
            num_layers,
            self.deltakv_full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )
        self.deltakv_temp_kv_cache = torch.empty(
            2,
            self.deltakv_temp_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )
        self.deltakv_latent_cache = torch.empty(
            num_layers,
            self.deltakv_latent_num_slots,
            config.kv_compressed_size,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )
        self.deltakv_latent_to_full_slots = torch.full(
            (num_layers, self.deltakv_latent_num_slots, config.deltakv_k_neighbors),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self.deltakv_slot_to_pos = torch.full(
            (self.deltakv_full_num_slots,),
            -1,
            dtype=torch.int32,
            device="cuda",
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.deltakv_layer_batch_states

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.deltakv_layer_to_idx[layer_idx]
        return self.deltakv_full_kv_cache[0, idx], self.deltakv_full_kv_cache[1, idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        state = self.get_layer_batch_states(layer_idx)
        return k_cache, v_cache, state.slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        return self.deltakv_temp_kv_cache[0], self.deltakv_temp_kv_cache[1]

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError("Standalone DeltaKV always materializes a temp full-context view.")

    @property
    def num_free_slots(self) -> int:
        return int(self._num_free_slots_deltakv_full)

    def num_free_slots_full_layers(self) -> int:
        return int(self._num_free_slots_deltakv_full)

    def prompt_admission_free_slots(self) -> int:
        return int(self._num_free_slots_deltakv_full)

    def _standalone_persistent_slots_for_prompt(
        self,
        prompt_len: int,
        *,
        chunk_prefill_size: int | None = None,
    ) -> int:
        """Upper bound for long-lived full/raw slots of one standalone sequence.

        Standalone DeltaKV does not keep the whole prompt in the full/raw pool.
        For long prompts, the persistent full/raw occupancy is dominated by:
        - sink tokens
        - at most ~2 * recent tokens in the uncompressed rolling buffer
        - the current prefill chunk before it is compressed/compacted

        Centers are budgeted separately via ``deltakv_centers`` and therefore
        intentionally excluded here.
        """
        prompt_len = max(0, int(prompt_len))
        if prompt_len <= 0:
            return 0

        sink = int(self.config.num_sink_tokens or 0)
        recent = int(self.config.num_recent_tokens or 0)
        chunk = int(
            chunk_prefill_size
            if chunk_prefill_size is not None
            else (self.config.chunk_prefill_size or 0)
        )
        # Long prompts stabilize around sink + buffer + current chunk.
        persistent_upper = sink + 2 * recent + max(0, chunk)
        return min(prompt_len, persistent_upper)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        """Reserve only the extra raw/full growth of already-started prefills.

        Base CacheManager reserves the whole remaining prompt, which is correct
        for dense full-attention caches but far too conservative for standalone
        DeltaKV. Ongoing prefills can only grow the persistent full/raw pool by
        roughly one more chunk before the next compression/compaction pass.
        """
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
                reserved += min(remaining, int(chunk_prefill_size))
        return reserved

    def prompt_admission_cost(self, seq: Sequence) -> int:
        return int(
            self._standalone_persistent_slots_for_prompt(
                seq.num_prompt_tokens,
                chunk_prefill_size=int(self.config.chunk_prefill_size or 0),
            )
        )

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        return 0

    def prompt_admission_budgets(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> dict[str, int]:
        reserved = int(self.reserved_prefill_slots(waiting_seqs, chunk_prefill_size))
        centers_free = max(0, int(self._deltakv_centers_capacity) - int(self._deltakv_centers_reserved_total))
        return {
            "persistent": max(0, int(self._num_free_slots_deltakv_full) - reserved),
            "deltakv_centers": centers_free,
        }

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        prompt_len = int(seq.num_prompt_tokens)
        total_len = int(seq.num_prompt_tokens + (getattr(seq, "max_tokens", 0) or 0))
        return {
            "persistent": self._standalone_persistent_slots_for_prompt(
                prompt_len,
                chunk_prefill_size=int(self.config.chunk_prefill_size or 0),
            ),
            "deltakv_centers": self._estimate_centers_for_total_len(total_len),
        }

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        seq_id = int(seq.seq_id)
        if seq_id in self._deltakv_centers_reserved_by_seq:
            return
        centers = int(costs.get("deltakv_centers", 0) or 0)
        self._deltakv_centers_reserved_by_seq[seq_id] = centers
        self._deltakv_centers_reserved_total += centers

    @torch.no_grad()
    def _allocate_temp_deltakv_full(self, size: int) -> torch.Tensor:
        if self.deltakv_temp_num_slots < size:
            raise RuntimeError(
                "Out of standalone DeltaKV temp slots. "
                f"need={size} free={self.deltakv_temp_num_slots} temp_capacity={self.deltakv_temp_num_slots}"
            )
        return self.deltakv_temp_slot_ids[:size]

    @torch.no_grad()
    def free_temp_deltakv_full(self, slots: torch.Tensor | None):
        return

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            self._reset_standalone_view_cache()
            reserved = self._deltakv_centers_reserved_by_seq.pop(seq_id, 0)
            if reserved:
                self._deltakv_centers_reserved_total -= int(reserved)
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            cur_len = self.row_seq_lens[row_idx]
            assert cur_len > 0

            deltakv_slots = self.sparse_layer_raw_slots_map[row_idx, :cur_len]
            mask = deltakv_slots >= 0
            slots = deltakv_slots[mask]
            ptr = self._num_free_slots_deltakv_full
            self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
            self._num_free_slots_deltakv_full += slots.numel()
            self.deltakv_slot_to_pos[slots] = -1

            latent_slots = self.sparse_layer_latent_slots_map[row_idx, :cur_len]
            mask_latent = latent_slots >= 0
            if mask_latent.any():
                slots = latent_slots[mask_latent]
                ptr = self._num_free_slots_deltakv_latent
                self.free_slots_stack_deltakv_latent[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_deltakv_latent += slots.numel()

            self.sparse_layer_raw_slots_map[row_idx, :] = -1
            self.sparse_layer_latent_slots_map[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.row_deltakv_compressed_lens[row_idx] = 0
            self.row_deltakv_center_slots[row_idx] = [None for _ in range(self.num_layers)]
            self.free_rows.append(row_idx)

    def free_slot_stats(self) -> dict[str, int]:
        centers_cap = int(getattr(self, "_deltakv_centers_capacity", 0) or 0)
        centers_reserved = int(getattr(self, "_deltakv_centers_reserved_total", 0) or 0)
        centers_free = max(0, centers_cap - centers_reserved)
        return {
            "free_slots": int(self.num_free_slots),
            "deltakv_persistent_free": int(self._num_free_slots_deltakv_full),
            "deltakv_latent_free": int(self._num_free_slots_deltakv_latent),
            "deltakv_temp_free": int(self._num_free_slots_deltakv_temp),
            "deltakv_temp_capacity": int(self.deltakv_temp_num_slots),
            "centers_capacity": centers_cap,
            "centers_reserved": centers_reserved,
            "centers_free": centers_free,
            "active_seqs": int(len(self.seq_id_to_row)),
        }

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("Standalone DeltaKV does not support partial slot freeing via this method.")

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            self._reset_standalone_view_cache()
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            deltakv_slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device="cuda")
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

                self._allocate_deltakv_full(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                deltakv_slot_mapping[token_offset: token_offset + chunk_size] = \
                    self.sparse_layer_raw_slots_map[row_idx, start_idx:end_idx]

                self.row_seq_lens[row_idx] += chunk_size
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

            state = self.deltakv_layer_batch_states
            state.slot_mapping = deltakv_slot_mapping
            state.context_lens = context_lens
            state.req_indices = req_indices_tensor

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            self._reset_standalone_view_cache()
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            deltakv_slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
            deltakv_slots = self._allocate_batch_deltakv_full(seq_ids, 1)
            deltakv_slot_mapping[:] = deltakv_slots

            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            self.row_seq_lens[row_indices] += 1
            context_lens = torch.tensor(self.row_seq_lens[row_indices], dtype=torch.int32, device="cuda")
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device="cuda")

            state = self.deltakv_layer_batch_states
            state.slot_mapping = deltakv_slot_mapping
            state.context_lens = context_lens
            state.req_indices = req_indices

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

    def _get_standalone_reconstruct_plan(
        self,
        req_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor | int | dict[int, torch.Tensor]]:
        req_ptr = int(req_indices.data_ptr()) if req_indices.numel() > 0 else 0
        key = (req_ptr, int(req_indices.numel()))
        if self._standalone_view_cache_key == key and self._standalone_view_cache_value is not None:
            return self._standalone_view_cache_value

        rows = req_indices.to(torch.long)
        bsz = int(rows.numel())
        sink = int(self.config.num_sink_tokens)

        plans = []
        max_s = 0
        total_temp = 0
        new_context_lens_list = []
        for b in range(bsz):
            row = int(rows[b].item())
            total_len = int(self.row_seq_lens[row])
            comp_len = int(self.row_deltakv_compressed_lens[row])
            sink_len = min(sink, total_len)
            buffer_start = (sink + comp_len) if total_len > sink else sink_len
            buffer_len = total_len - buffer_start
            plans.append((row, total_len, sink_len, comp_len, buffer_start, buffer_len))
            max_s = max(max_s, total_len)
            total_temp += total_len
            new_context_lens_list.append(total_len)

        active_local_slots = torch.full((bsz, max_s), -1, device="cuda", dtype=torch.int32)
        raw_src = []
        raw_dst = []
        recon_pos = []
        recon_latent = []
        recon_out = []

        offset = 0
        for b, (row, total_len, sink_len, comp_len, buffer_start, buffer_len) in enumerate(plans):
            local_seq = torch.arange(offset, offset + total_len, device="cuda", dtype=torch.int32)
            offset += total_len
            active_local_slots[b, :total_len] = local_seq

            if sink_len > 0:
                sink_slots = self.sparse_layer_raw_slots_map[row, :sink_len].to(torch.int32)
                if (sink_slots < 0).any():
                    raise RuntimeError("Standalone DeltaKV: missing raw slots in sink window.")
                raw_src.append(sink_slots)
                raw_dst.append(local_seq[:sink_len])

            if comp_len > 0:
                comp_pos = torch.arange(sink, sink + comp_len, device="cuda", dtype=torch.int32)
                comp_out = local_seq[sink_len : sink_len + comp_len]
                comp_raw_slots = self.sparse_layer_raw_slots_map[row, comp_pos.to(torch.long)].to(torch.int32)
                raw_mask = comp_raw_slots >= 0
                if raw_mask.any():
                    raw_src.append(comp_raw_slots[raw_mask])
                    raw_dst.append(comp_out[raw_mask])
                need_mask = ~raw_mask
                if need_mask.any():
                    latent_slots = self.sparse_layer_latent_slots_map[row, comp_pos[need_mask].to(torch.long)].to(torch.int32)
                    if (latent_slots < 0).any():
                        raise RuntimeError("Standalone DeltaKV: compressed token has neither raw nor latent slot.")
                    recon_pos.append(comp_pos[need_mask])
                    recon_latent.append(latent_slots)
                    recon_out.append(comp_out[need_mask])

            if buffer_len > 0:
                buf_slots = self.sparse_layer_raw_slots_map[row, buffer_start : buffer_start + buffer_len].to(torch.int32)
                if (buf_slots < 0).any():
                    raise RuntimeError("Standalone DeltaKV: missing raw slots in buffer window.")
                raw_src.append(buf_slots)
                raw_dst.append(local_seq[sink_len + comp_len : sink_len + comp_len + buffer_len])

        raw_src_all_i32 = (
            torch.cat(raw_src, dim=0).to(torch.int32)
            if raw_src
            else torch.empty((0,), device="cuda", dtype=torch.int32)
        )
        raw_dst_all_i32 = (
            torch.cat(raw_dst, dim=0).to(torch.int32)
            if raw_dst
            else torch.empty((0,), device="cuda", dtype=torch.int32)
        )
        recon_pos_all_i32 = (
            torch.cat(recon_pos, dim=0).to(torch.int32)
            if recon_pos
            else torch.empty((0,), device="cuda", dtype=torch.int32)
        )
        recon_latent_all_i32 = (
            torch.cat(recon_latent, dim=0).to(torch.int32)
            if recon_latent
            else torch.empty((0,), device="cuda", dtype=torch.int32)
        )
        recon_out_all_i32 = (
            torch.cat(recon_out, dim=0).to(torch.int32)
            if recon_out
            else torch.empty((0,), device="cuda", dtype=torch.int32)
        )

        value: dict[str, torch.Tensor | int | dict[int, torch.Tensor]] = {
            "active_local_slots": active_local_slots,
            "new_context_lens": torch.tensor(new_context_lens_list, device="cuda", dtype=torch.int32),
            "local_req": torch.arange(bsz, device="cuda", dtype=torch.int32),
            "total_temp": total_temp,
            "raw_src_all": raw_src_all_i32,
            "raw_src_all_i64": raw_src_all_i32.to(torch.long),
            "raw_dst_all": raw_dst_all_i32,
            "raw_dst_all_i64": raw_dst_all_i32.to(torch.long),
            "recon_pos_all": recon_pos_all_i32,
            "recon_latent_all": recon_latent_all_i32,
            "recon_out_all": recon_out_all_i32,
            "recon_out_all_i64": recon_out_all_i32.to(torch.long),
            "father_slots_by_layer": {},
        }
        self._standalone_view_cache_key = key
        self._standalone_view_cache_value = value
        return value

    @torch.no_grad()
    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del active_compressed_indices, context_lens, chunk_lens

        with profiler.record("deltakv_standalone_reconstruct_total"):
            with profiler.record("deltakv_standalone_reconstruct_plan"):
                plan = self._get_standalone_reconstruct_plan(req_indices)
                bsz = int(req_indices.numel())
                max_s = int(plan["active_local_slots"].shape[1])
                total_temp = int(plan["total_temp"])
                active_local_slots = plan["active_local_slots"]
                new_context_lens = plan["new_context_lens"]
                local_req = plan["local_req"]
                raw_src_all = plan["raw_src_all"]
                raw_src_all_i64 = plan["raw_src_all_i64"]
                raw_dst_all_i64 = plan["raw_dst_all_i64"]
                recon_pos_all = plan["recon_pos_all"]
                recon_latent_all = plan["recon_latent_all"]
                recon_out_all = plan["recon_out_all_i64"]
                father_slots_by_layer = plan["father_slots_by_layer"]

            with profiler.record("deltakv_standalone_reconstruct_alloc_temp"):
                temp_slots = self._allocate_temp_deltakv_full(total_temp).to(torch.int32)
                active_slots = active_local_slots

            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_persist = self.deltakv_full_kv_cache[0, l_idx]
            v_persist = self.deltakv_full_kv_cache[1, l_idx]
            k_temp = self.deltakv_temp_kv_cache[0]
            v_temp = self.deltakv_temp_kv_cache[1]

            with profiler.record("deltakv_standalone_reconstruct_build_views"):
                pass

            if raw_src_all.numel() > 0:
                with profiler.record("deltakv_standalone_reconstruct_raw_copy"):
                    k_temp[raw_dst_all_i64] = k_persist[raw_src_all_i64]
                    v_temp[raw_dst_all_i64] = v_persist[raw_src_all_i64]

            if recon_latent_all.numel() > 0:
                with profiler.record("deltakv_standalone_reconstruct_concat"):
                    pass

                total_recon = int(recon_latent_all.numel())
                decompress_chunk_tokens = int(
                    os.getenv("SPARSEVLLM_DELTAKV_STANDALONE_DECOMPRESS_CHUNK_TOKENS", "524288")
                )
                kernel_chunk_tokens = int(
                    os.getenv(
                        "SPARSEVLLM_DELTAKV_STANDALONE_KERNEL_CHUNK_TOKENS",
                        str(decompress_chunk_tokens),
                    )
                )
                decompress_chunk_tokens = max(1, decompress_chunk_tokens)
                kernel_chunk_tokens = max(1, kernel_chunk_tokens)

                father_slots_all = father_slots_by_layer.get(l_idx, None)
                if father_slots_all is None:
                    with profiler.record("deltakv_standalone_reconstruct_father"):
                        father_slots_all = self.deltakv_latent_to_full_slots[l_idx, recon_latent_all].to(torch.int32).contiguous()
                        if (father_slots_all < 0).any():
                            raise RuntimeError("Standalone DeltaKV: missing father slots for reconstruction.")
                        if (self.deltakv_slot_to_pos[father_slots_all] < 0).any():
                            raise RuntimeError("Standalone DeltaKV: father center slot has unknown position.")
                        father_slots_by_layer[l_idx] = father_slots_all

                for start in range(0, total_recon, decompress_chunk_tokens):
                    end = min(start + decompress_chunk_tokens, total_recon)
                    latent_idx = recon_latent_all[start:end]

                    with profiler.record("deltakv_standalone_reconstruct_decompress"):
                        latent = self.deltakv_latent_cache[l_idx, latent_idx]
                        kv_delta = self.compress_up[l_idx](latent)

                    for local_start in range(0, int(kv_delta.shape[0]), kernel_chunk_tokens):
                        local_end = min(local_start + kernel_chunk_tokens, int(kv_delta.shape[0]))
                        abs_start = start + local_start
                        abs_end = start + local_end
                        with profiler.record("deltakv_standalone_reconstruct_triton_kernel"):
                            self._deltakv_reconstruct_writeback_srcdst(
                                kv_delta=kv_delta[local_start:local_end],
                                father_slots=father_slots_all[abs_start:abs_end],
                                slot_to_pos=self.deltakv_slot_to_pos,
                                out_slots=recon_out_all[abs_start:abs_end].to(torch.int32),
                                out_pos=recon_pos_all[abs_start:abs_end],
                                src_k_cache=k_persist,
                                src_v_cache=v_persist,
                                dst_k_cache=k_temp,
                                dst_v_cache=v_temp,
                            )

            return active_slots, local_req, new_context_lens, temp_slots
