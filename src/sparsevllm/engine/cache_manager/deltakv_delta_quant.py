from __future__ import annotations

import torch

from sparsevllm.engine.sequence import Sequence
from sparsevllm.triton_kernel.quant import triton_quantize_and_pack_along_last_dim, unpack_4bit_to_16bit
from sparsevllm.utils.log import logger
from sparsevllm.utils.profiler import profiler

from .deltakv import DeltaKVCacheTritonManagerV4


class DeltaKVDeltaQuantCacheManager(DeltaKVCacheTritonManagerV4):
    """DeltaKV clustering with direct token-space residual storage.

    This no-checkpoint method keeps DeltaKV's center selection and sparse view
    construction, but stores `KV - mean(ref KV)` directly instead of passing the
    residual through a learned compressor. With `kv_quant_bits=4`, the residual
    is packed as int4 plus per-token min/max scale metadata.
    """

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        dtype_size = torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()

        num_full_layers = len(self.full_layer_ids)
        num_deltakv_layers = len(self.deltakv_layer_ids)
        assert num_full_layers > 0, "DeltaKV delta-quant requires at least one full-attention layer."
        assert num_deltakv_layers > 0, "DeltaKV delta-quant requires at least one sparse layer."

        kv_dim = 2 * self.num_kv_heads * self.head_dim
        quant_bits = int(config.kv_quant_bits or 0)
        if quant_bits not in (0, 4):
            raise ValueError(f"deltakv-delta-quant supports kv_quant_bits=0 or 4, got {quant_bits}.")

        if quant_bits == 4:
            if kv_dim % 8 != 0:
                raise ValueError(f"int4 residual packing requires kv_dim divisible by 8, got {kv_dim}.")
            latent_bytes = (kv_dim // 8) * 4 + 2 * dtype_size
        else:
            latent_bytes = kv_dim * dtype_size

        cluster_ratio = max(0.0, float(config.cluster_ratio))
        per_token_bytes = (
            num_full_layers * slot_bytes_per_layer
            + num_deltakv_layers * (cluster_ratio * slot_bytes_per_layer + latent_bytes)
        )
        if per_token_bytes <= 0:
            raise ValueError("Invalid DeltaKV delta-quant allocation configuration.")

        max_tokens = max(1, int(available_memory / per_token_bytes))
        reserve_ratio = float(config.deltakv_full_pool_reserve_ratio)
        if reserve_ratio > 0:
            reserve_ratio = max(0.0, min(0.5, reserve_ratio))
            max_tokens = max(1, int(max_tokens * (1.0 - reserve_ratio)))
        self.full_num_slots = max_tokens
        self.deltakv_latent_num_slots = max_tokens

        bytes_full_layers = self.full_num_slots * num_full_layers * slot_bytes_per_layer
        bytes_latent = self.deltakv_latent_num_slots * num_deltakv_layers * latent_bytes
        bytes_left = available_memory - bytes_full_layers - bytes_latent
        if bytes_left <= 0:
            raise RuntimeError(
                "Not enough GPU memory left for DeltaKV delta-quant full-KV pool after "
                "allocating full layers + residual cache."
            )
        max_deltakv_full_slots = max(1, int(bytes_left // (num_deltakv_layers * slot_bytes_per_layer)))

        sink = int(config.num_sink_tokens)
        recent = int(config.num_recent_tokens)
        max_seqs = int(config.max_num_seqs_in_batch)
        top_decode = int(config.num_top_tokens)
        top_prefill = int(config.num_top_tokens_in_prefill)
        max_prefill_seqs_by_tokens = (int(config.max_num_batched_tokens) + int(config.chunk_prefill_size) - 1) // int(
            config.chunk_prefill_size
        )
        max_prefill_seqs = min(max_seqs, max_prefill_seqs_by_tokens)
        total_top_slots = max(max_seqs * top_decode, max_prefill_seqs * top_prefill)
        max_step_chunk = int(min(int(config.max_num_batched_tokens), max_seqs * int(config.chunk_prefill_size)))
        overhead_slots = max_seqs * (sink + 2 * recent) + total_top_slots + max_step_chunk
        if max_deltakv_full_slots <= overhead_slots:
            raise RuntimeError(
                f"DeltaKV delta-quant full-KV pool too small: max={max_deltakv_full_slots}, "
                f"required>={overhead_slots + 1}."
            )

        desired_centers = max(1, int(cluster_ratio * self.full_num_slots * 1.5))
        centers_capacity = min(desired_centers, max_deltakv_full_slots - overhead_slots)
        self.deltakv_full_num_slots = overhead_slots + centers_capacity
        self._deltakv_centers_capacity = int(centers_capacity)
        self._deltakv_temp_full_reserve = min(self.deltakv_full_num_slots, int(total_top_slots))

        logger.info(
            f"DeltaKV delta-quant allocation: full_layers_slots={self.full_num_slots}; "
            f"deltakv_full_slots={self.deltakv_full_num_slots} (overhead={overhead_slots}, centers={centers_capacity}); "
            f"residual_slots={self.deltakv_latent_num_slots}; kv_dim={kv_dim}; kv_quant_bits={quant_bits}."
        )

        self.full_kv_cache = torch.empty(
            2,
            num_full_layers,
            self.full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )
        self.deltakv_full_kv_cache = torch.empty(
            2,
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

        latent_width = kv_dim // 8 if quant_bits == 4 else kv_dim
        latent_dtype = torch.int32 if quant_bits == 4 else self.hf_config.torch_dtype
        self.deltakv_latent_cache = torch.empty(
            num_deltakv_layers,
            self.deltakv_latent_num_slots,
            latent_width,
            dtype=latent_dtype,
            device="cuda",
        )
        if quant_bits == 4:
            self.deltakv_latent_scales = torch.empty(
                num_deltakv_layers,
                self.deltakv_latent_num_slots,
                1,
                dtype=self.hf_config.torch_dtype,
                device="cuda",
            )
            self.deltakv_latent_mins = torch.empty_like(self.deltakv_latent_scales)
        else:
            self.deltakv_latent_scales = None
            self.deltakv_latent_mins = None

        self.deltakv_latent_to_full_slots = torch.full(
            (num_deltakv_layers, self.deltakv_latent_num_slots, config.deltakv_k_neighbors),
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

    def _init_compressor_modules(self, config, num_deltakv_layers: int):
        del config, num_deltakv_layers
        # This method reconstructs from token-space residuals, not learned latents.
        self.compress_down = []
        self.compress_up = []

    def _quantize_residual(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed, scale, mn = triton_quantize_and_pack_along_last_dim(
            residual.unsqueeze(0).unsqueeze(0),
            residual.shape[-1],
            4,
        )
        return packed.squeeze(0).squeeze(0), scale.squeeze(0).squeeze(0), mn.squeeze(0).squeeze(0)

    def _dequantize_residual(self, packed: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, kv_dim: int) -> torch.Tensor:
        return unpack_4bit_to_16bit(
            packed.unsqueeze(0).unsqueeze(0),
            scale.unsqueeze(0).unsqueeze(0),
            mn.unsqueeze(0).unsqueeze(0),
            kv_dim,
        ).squeeze(0).squeeze(0)

    def _store_residual(self, l_idx: int, latent_slots: torch.Tensor, residual: torch.Tensor):
        if int(self.config.kv_quant_bits or 0) == 4:
            packed, scale, mn = self._quantize_residual(residual)
            self.deltakv_latent_cache[l_idx, latent_slots] = packed.to(self.deltakv_latent_cache.dtype)
            self.deltakv_latent_scales[l_idx, latent_slots] = scale.to(self.deltakv_latent_scales.dtype)
            self.deltakv_latent_mins[l_idx, latent_slots] = mn.to(self.deltakv_latent_mins.dtype)
        else:
            self.deltakv_latent_cache[l_idx, latent_slots] = residual.to(self.deltakv_latent_cache.dtype)

    def _load_residual(self, l_idx: int, recon_latent: torch.Tensor, kv_dim: int) -> torch.Tensor:
        residual = self.deltakv_latent_cache[l_idx, recon_latent]
        if int(self.config.kv_quant_bits or 0) == 4:
            scales = self.deltakv_latent_scales[l_idx, recon_latent]
            mins = self.deltakv_latent_mins[l_idx, recon_latent]
            residual = self._dequantize_residual(residual, scales, mins, kv_dim)
        return residual

    def _reconstruct_writeback_int4(
        self,
        *,
        l_idx: int,
        latent_slots: torch.Tensor,
        father_slots: torch.Tensor,
        out_slots: torch.Tensor,
        out_pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_delta_quant_reconstruct_writeback_int4

        hp = int(getattr(self.config, "deltakv_triton_reconstruct_heads_per_program", 4) or 1)
        hp = max(1, min(hp, int(self.num_kv_heads)))
        return deltakv_delta_quant_reconstruct_writeback_int4(
            packed_delta_cache=self.deltakv_latent_cache[l_idx],
            scale_cache=self.deltakv_latent_scales[l_idx],
            min_cache=self.deltakv_latent_mins[l_idx],
            latent_slots=latent_slots,
            father_slots=father_slots,
            slot_to_pos=self.deltakv_slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
            heads_per_program=hp,
        )

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        with profiler.record("deltakv_delta_quant_evict_total"):
            if not self.deltakv_layer_ids:
                return
            sink = int(self.config.num_sink_tokens)
            recent = int(self.config.num_recent_tokens)
            cluster_step = max(1, int(1.0 / max(1e-6, float(self.config.cluster_ratio))))
            cos_sin = self.cos_sin_cache[:, 0, :]

            for seq in seqs:
                row_idx = self.seq_id_to_row.get(seq.seq_id, None)
                if row_idx is None:
                    continue

                total_len = int(self.row_seq_lens[row_idx])
                compressed_len = int(self.row_deltakv_compressed_lens[row_idx])
                buffer_start = sink + compressed_len
                buffer_len = total_len - buffer_start
                if buffer_len <= recent:
                    continue

                evict_len = ((buffer_len - recent) // recent) * recent
                if evict_len <= 0:
                    continue

                evict_start = buffer_start
                evict_end = evict_start + evict_len
                raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
                if (raw_slots_block < 0).any():
                    raise RuntimeError("DeltaKV delta-quant eviction expects raw slots for the buffer block.")

                center_rel = torch.arange(0, evict_len, cluster_step, device="cuda", dtype=torch.long)
                new_center_slots = raw_slots_block[center_rel].to(torch.int32)
                sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
                prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
                for layer_idx in self.deltakv_layer_ids:
                    existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                    prev_center_slots_by_layer[layer_idx] = sink_slots if existing is None else existing.to(torch.int32)

                is_center = torch.zeros((evict_len,), device="cuda", dtype=torch.bool)
                is_center[center_rel] = True
                to_compress_mask = ~is_center
                num_to_compress = int(to_compress_mask.sum().item())
                if num_to_compress <= 0:
                    for layer_idx in self.deltakv_layer_ids:
                        self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                            [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                        )
                    self.row_deltakv_compressed_lens[row_idx] += evict_len
                    continue

                latent_slots = self._allocate_deltakv_latent(num_to_compress).to(torch.int32)
                pos_all = torch.arange(evict_start, evict_end, device="cuda", dtype=torch.int32)
                pos_to_compress = pos_all[to_compress_mask]
                self.sparse_layer_latent_slots_map[row_idx, pos_to_compress.to(torch.long)] = latent_slots
                raw_slots_block_i32 = raw_slots_block.to(torch.int32)

                for layer_idx in self.deltakv_layer_ids:
                    l_idx = self.deltakv_layer_to_idx[layer_idx]
                    k_cache = self.deltakv_full_kv_cache[0, l_idx]
                    v_cache = self.deltakv_full_kv_cache[1, l_idx]
                    kv_block = self._deltakv_gather_kv_unrope(
                        slots=raw_slots_block_i32,
                        pos=pos_all,
                        cos_sin=cos_sin,
                        k_cache=k_cache,
                        v_cache=v_cache,
                    ).unsqueeze(0)

                    existing_center_slots = prev_center_slots_by_layer[layer_idx]
                    topk_center_indices, base_kv = self._cluster_compress(
                        layer_idx=layer_idx,
                        kv_states=kv_block,
                        existing_center_slots=existing_center_slots,
                        cluster_step=cluster_step,
                    )

                    all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)
                    father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]
                    father_slots = father_slots_full[to_compress_mask]
                    K = self.deltakv_latent_to_full_slots.shape[-1]
                    k_eff = father_slots.shape[1]
                    if k_eff < K:
                        pad = father_slots[:, :1].expand(-1, K - k_eff)
                        father_slots = torch.cat([father_slots, pad], dim=1)
                    elif k_eff > K:
                        father_slots = father_slots[:, :K]
                    self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

                    residual = (kv_block - base_kv).squeeze(0)[to_compress_mask]
                    self._store_residual(l_idx, latent_slots, residual)

                for layer_idx in self.deltakv_layer_ids:
                    self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                        [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                    )

                free_slots = raw_slots_block_i32[to_compress_mask]
                ptr = self._num_free_slots_deltakv_full
                self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
                self._num_free_slots_deltakv_full += free_slots.numel()
                self.deltakv_slot_to_pos[free_slots] = -1

                self.sparse_layer_raw_slots_map[row_idx, pos_to_compress.to(torch.long)] = -1
                self.row_deltakv_compressed_lens[row_idx] += evict_len

    @torch.no_grad()
    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del context_lens, chunk_lens
        with profiler.record("deltakv_delta_quant_reconstruct_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
            if temp_slots.numel() == 0:
                return active_slots, local_req, new_context_lens, temp_slots

            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]
            kv_dim = 2 * self.num_kv_heads * self.head_dim

            father_slots = self.deltakv_latent_to_full_slots[l_idx, recon_latent].to(torch.int32)
            if (father_slots < 0).any():
                raise RuntimeError("DeltaKV delta-quant: missing father slots for reconstruction.")

            cos_sin = self.cos_sin_cache[:, 0, :]
            if int(self.config.kv_quant_bits or 0) == 4:
                with profiler.record("deltakv_delta_quant_reconstruct_int4_kernel"):
                    self._reconstruct_writeback_int4(
                        l_idx=l_idx,
                        latent_slots=recon_latent,
                        father_slots=father_slots,
                        out_slots=recon_out_slot,
                        out_pos=recon_pos,
                        cos_sin=cos_sin,
                        k_cache=k_cache,
                        v_cache=v_cache,
                    )
            else:
                kv_delta = self._load_residual(l_idx, recon_latent, kv_dim)
                self._deltakv_reconstruct_writeback(
                    kv_delta=kv_delta,
                    father_slots=father_slots,
                    slot_to_pos=self.deltakv_slot_to_pos,
                    out_slots=recon_out_slot,
                    out_pos=recon_pos,
                    cos_sin=cos_sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                )

            return active_slots, local_req, new_context_lens, temp_slots
