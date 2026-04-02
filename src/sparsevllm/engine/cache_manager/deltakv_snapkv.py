from __future__ import annotations

import os

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.profiler import profiler

from .deltakv_standalone import DeltaKVStandaloneCacheManager


class DeltaKVSnapKVCacheManager(DeltaKVStandaloneCacheManager):
    """Standalone DeltaKV with one-shot SnapKV-style static prune at prefill end.

    This variant keeps the standalone cache layout:
    - persistent per-layer DeltaKV cache
    - one global temp reconstruct pool

    The difference is that after the last prefill chunk we:
    1. keep a larger protected suffix (recent + SnapKV window)
    2. statically prune the compressed middle using attention scores

    To support sparse kept positions, compressed logical positions map to true
    absolute token positions via ``row_deltakv_comp_abs_pos``.
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.snapkv_window_size = int(config.snapkv_window_size or 0)
        self.row_deltakv_comp_abs_pos = torch.full(
            (self.max_buffer_rows, self.max_model_len),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self._snapkv_finalized_seq_ids: set[int] = set()
        self._snapkv_view_cache_key: tuple[int, int] | None = None
        self._snapkv_view_cache_value: dict[str, torch.Tensor | int] | None = None

    def _reset_snapkv_view_cache(self):
        self._snapkv_view_cache_key = None
        self._snapkv_view_cache_value = None

    @property
    def _protected_recent_tokens(self) -> int:
        return int(self.config.num_recent_tokens or 0) + int(self.snapkv_window_size)

    def _standalone_persistent_slots_for_prompt(
        self,
        prompt_len: int,
        *,
        chunk_prefill_size: int | None = None,
    ) -> int:
        prompt_len = max(0, int(prompt_len))
        if prompt_len <= 0:
            return 0

        sink = int(self.config.num_sink_tokens or 0)
        recent = int(self._protected_recent_tokens)
        chunk = int(
            chunk_prefill_size
            if chunk_prefill_size is not None
            else (self.config.chunk_prefill_size or 0)
        )
        persistent_upper = sink + 2 * recent + max(0, chunk)
        return min(prompt_len, persistent_upper)

    def free_seq(self, seq_id: int):
        row_idx = self.seq_id_to_row.get(seq_id, None)
        self._reset_snapkv_view_cache()
        super().free_seq(seq_id)
        self._snapkv_finalized_seq_ids.discard(int(seq_id))
        if row_idx is not None:
            self.row_deltakv_comp_abs_pos[row_idx, :] = -1

    def _prepare_prefill(self, seqs: list[Sequence]):
        self._reset_snapkv_view_cache()
        return super()._prepare_prefill(seqs)

    def _prepare_decode(self, seqs: list[Sequence]):
        self._reset_snapkv_view_cache()
        return super()._prepare_decode(seqs)

    def _resolve_comp_abs_pos(self, row: int, logical_pos: torch.Tensor) -> torch.Tensor:
        abs_pos = self.row_deltakv_comp_abs_pos[row, logical_pos.to(torch.long)].to(torch.int32)
        if (abs_pos < 0).any():
            # Before static pruning, compressed middle still matches true positions.
            fallback = logical_pos.to(torch.int32)
            abs_pos = torch.where(abs_pos >= 0, abs_pos, fallback)
        return abs_pos

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        self._reset_snapkv_view_cache()
        sink = int(self.config.num_sink_tokens)
        protected_recent = int(self._protected_recent_tokens)
        append_infos: list[tuple[int, int, int, torch.Tensor]] = []

        for seq in seqs:
            row_idx = self.seq_id_to_row.get(seq.seq_id, None)
            if row_idx is None:
                continue
            total_len = int(self.row_seq_lens[row_idx])
            compressed_len = int(self.row_deltakv_compressed_lens[row_idx])
            buffer_start = sink + compressed_len
            buffer_len = total_len - buffer_start
            if buffer_len <= protected_recent:
                continue

            evict_len = ((buffer_len - protected_recent) // protected_recent) * protected_recent
            if evict_len <= 0:
                continue

            evict_start = buffer_start
            evict_end = evict_start + evict_len
            raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
            if (raw_slots_block < 0).any():
                raise RuntimeError("DeltaKV+SnapKV eviction expects raw slots for the buffer block.")
            abs_pos_all = self.deltakv_slot_to_pos[raw_slots_block.to(torch.int32)].to(torch.int32)
            append_infos.append((row_idx, evict_start, evict_end, abs_pos_all))

        old_recent = int(self.config.num_recent_tokens)
        self.config.num_recent_tokens = protected_recent
        try:
            super().deltakv_evict(seqs)
        finally:
            self.config.num_recent_tokens = old_recent

        for row_idx, evict_start, evict_end, abs_pos_all in append_infos:
            self.row_deltakv_comp_abs_pos[row_idx, evict_start:evict_end] = abs_pos_all

    def _get_snapkv_reconstruct_plan(self, req_indices: torch.Tensor) -> dict[str, torch.Tensor | int]:
        req_ptr = int(req_indices.data_ptr()) if req_indices.numel() > 0 else 0
        key = (req_ptr, int(req_indices.numel()))
        if self._snapkv_view_cache_key == key and self._snapkv_view_cache_value is not None:
            return self._snapkv_view_cache_value

        rows = req_indices.to(torch.long)
        bsz = int(rows.numel())
        sink = int(self.config.num_sink_tokens)

        max_s = 0
        total_temp = 0
        new_context_lens_list = []
        plans: list[tuple[int, int, int, int, int, int, int]] = []
        for b in range(bsz):
            row = int(rows[b].item())
            total_len = int(self.row_seq_lens[row])
            comp_len = int(self.row_deltakv_compressed_lens[row])
            sink_len = min(sink, total_len)
            buffer_start = (sink + comp_len) if total_len > sink else sink_len
            buffer_len = total_len - buffer_start
            plans.append((b, row, total_len, sink_len, comp_len, buffer_start, buffer_len))
            max_s = max(max_s, total_len)
            total_temp += total_len
            new_context_lens_list.append(total_len)

        active_local_slots = torch.full((bsz, max_s), -1, device="cuda", dtype=torch.int32)
        raw_src = []
        raw_dst_local = []
        recon_pos = []
        recon_latent = []
        recon_out_local = []

        offset = 0
        for b, row, total_len, sink_len, comp_len, buffer_start, buffer_len in plans:
            local_seq = torch.arange(offset, offset + total_len, device="cuda", dtype=torch.int32)
            offset += total_len
            active_local_slots[b, :total_len] = local_seq

            if sink_len > 0:
                sink_slots = self.sparse_layer_raw_slots_map[row, :sink_len].to(torch.int32)
                if (sink_slots < 0).any():
                    raise RuntimeError("DeltaKV+SnapKV: missing raw slots in sink window.")
                raw_src.append(sink_slots)
                raw_dst_local.append(local_seq[:sink_len])

            if comp_len > 0:
                comp_logical_pos = torch.arange(sink, sink + comp_len, device="cuda", dtype=torch.int32)
                comp_local_out = local_seq[sink_len: sink_len + comp_len]
                comp_raw_slots = self.sparse_layer_raw_slots_map[row, comp_logical_pos.to(torch.long)].to(torch.int32)
                raw_mask = comp_raw_slots >= 0
                if raw_mask.any():
                    raw_src.append(comp_raw_slots[raw_mask])
                    raw_dst_local.append(comp_local_out[raw_mask])
                need_mask = ~raw_mask
                if need_mask.any():
                    latent_slots = self.sparse_layer_latent_slots_map[row, comp_logical_pos[need_mask].to(torch.long)].to(torch.int32)
                    if (latent_slots < 0).any():
                        raise RuntimeError("DeltaKV+SnapKV: compressed token has neither raw nor latent slot.")
                    comp_abs_pos = self._resolve_comp_abs_pos(row, comp_logical_pos)
                    recon_pos.append(comp_abs_pos[need_mask])
                    recon_latent.append(latent_slots)
                    recon_out_local.append(comp_local_out[need_mask])

            if buffer_len > 0:
                buf_slots = self.sparse_layer_raw_slots_map[row, buffer_start: buffer_start + buffer_len].to(torch.int32)
                if (buf_slots < 0).any():
                    raise RuntimeError("DeltaKV+SnapKV: missing raw slots in buffer window.")
                raw_src.append(buf_slots)
                raw_dst_local.append(local_seq[sink_len + comp_len: sink_len + comp_len + buffer_len])

        raw_src_all_i32 = torch.cat(raw_src, dim=0).to(torch.int32) if raw_src else torch.empty((0,), device="cuda", dtype=torch.int32)
        raw_dst_local_all_i32 = torch.cat(raw_dst_local, dim=0).to(torch.int32) if raw_dst_local else torch.empty((0,), device="cuda", dtype=torch.int32)
        recon_pos_all_i32 = torch.cat(recon_pos, dim=0).to(torch.int32) if recon_pos else torch.empty((0,), device="cuda", dtype=torch.int32)
        recon_latent_all_i32 = torch.cat(recon_latent, dim=0).to(torch.int32) if recon_latent else torch.empty((0,), device="cuda", dtype=torch.int32)
        recon_out_local_all_i32 = torch.cat(recon_out_local, dim=0).to(torch.int32) if recon_out_local else torch.empty((0,), device="cuda", dtype=torch.int32)

        value: dict[str, torch.Tensor | int | dict[int, torch.Tensor]] = {
            "active_local_slots": active_local_slots,
            "new_context_lens": torch.tensor(new_context_lens_list, device="cuda", dtype=torch.int32),
            "local_req": torch.arange(bsz, device="cuda", dtype=torch.int32),
            "total_temp": total_temp,
            "bsz": bsz,
            "max_s": max_s,
            "raw_src_all": raw_src_all_i32,
            "raw_src_all_i64": raw_src_all_i32.to(torch.long),
            "raw_dst_local_all": raw_dst_local_all_i32,
            "raw_dst_local_all_i64": raw_dst_local_all_i32.to(torch.long),
            "recon_pos_all": recon_pos_all_i32,
            "recon_latent_all": recon_latent_all_i32,
            "recon_out_local_all": recon_out_local_all_i32,
            "recon_out_local_all_i64": recon_out_local_all_i32.to(torch.long),
            "father_slots_by_layer": {},
        }
        self._snapkv_view_cache_key = key
        self._snapkv_view_cache_value = value
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

        with profiler.record("deltakv_snapkv_reconstruct_total"):
            with profiler.record("deltakv_snapkv_reconstruct_plan"):
                plan = self._get_snapkv_reconstruct_plan(req_indices)
                bsz = int(plan["bsz"])
                max_s = int(plan["max_s"])
                total_temp = int(plan["total_temp"])
                active_local_slots = plan["active_local_slots"]
                new_context_lens = plan["new_context_lens"]
                local_req = plan["local_req"]
                raw_src_all = plan["raw_src_all"]
                raw_src_all_i64 = plan["raw_src_all_i64"]
                raw_dst_local_all = plan["raw_dst_local_all"]
                raw_dst_local_all_i64 = plan["raw_dst_local_all_i64"]
                recon_pos_all = plan["recon_pos_all"]
                recon_latent_all = plan["recon_latent_all"]
                recon_out_local_all = plan["recon_out_local_all"]
                recon_out_local_all_i64 = plan["recon_out_local_all_i64"]
                father_slots_by_layer = plan["father_slots_by_layer"]

            with profiler.record("deltakv_snapkv_reconstruct_alloc_temp"):
                temp_slots = self._allocate_temp_deltakv_full(total_temp).to(torch.int32)
                active_slots = active_local_slots

            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_persist = self.deltakv_full_kv_cache[0, l_idx]
            v_persist = self.deltakv_full_kv_cache[1, l_idx]
            k_temp = self.deltakv_temp_kv_cache[0]
            v_temp = self.deltakv_temp_kv_cache[1]

            with profiler.record("deltakv_snapkv_reconstruct_build_views"):
                pass

            if raw_src_all.numel() > 0:
                with profiler.record("deltakv_snapkv_reconstruct_raw_copy"):
                    k_temp[raw_dst_local_all_i64] = k_persist[raw_src_all_i64]
                    v_temp[raw_dst_local_all_i64] = v_persist[raw_src_all_i64]

            if recon_latent_all.numel() > 0:
                with profiler.record("deltakv_snapkv_reconstruct_concat"):
                    recon_out_all = recon_out_local_all_i64

                father_slots_all = father_slots_by_layer.get(l_idx, None)
                if father_slots_all is None:
                    with profiler.record("deltakv_snapkv_reconstruct_father"):
                        father_slots_all = self.deltakv_latent_to_full_slots[l_idx, recon_latent_all].to(torch.int32).contiguous()
                        if (father_slots_all < 0).any():
                            raise RuntimeError("DeltaKV+SnapKV: missing father slots for reconstruction.")
                        if (self.deltakv_slot_to_pos[father_slots_all] < 0).any():
                            raise RuntimeError("DeltaKV+SnapKV: father center slot has unknown position.")
                        father_slots_by_layer[l_idx] = father_slots_all

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

                for start in range(0, total_recon, decompress_chunk_tokens):
                    end = min(start + decompress_chunk_tokens, total_recon)
                    latent_idx = recon_latent_all[start:end]

                    with profiler.record("deltakv_snapkv_reconstruct_decompress"):
                        latent = self.deltakv_latent_cache[l_idx, latent_idx]
                        kv_delta = self.compress_up[l_idx](latent)

                    for local_start in range(0, int(kv_delta.shape[0]), kernel_chunk_tokens):
                        local_end = min(local_start + kernel_chunk_tokens, int(kv_delta.shape[0]))
                        abs_start = start + local_start
                        abs_end = start + local_end
                        with profiler.record("deltakv_snapkv_reconstruct_triton_kernel"):
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

    def _prepare_decode(self, seqs: list[Sequence]):
        input_ids, positions, _ = super()._prepare_decode(seqs)
        state = self.deltakv_layer_batch_states
        if state.slot_mapping is not None and state.slot_mapping.numel() > 0:
            self.deltakv_slot_to_pos[state.slot_mapping.to(torch.long)] = positions.to(torch.int32)
        return input_ids, positions, None

    @torch.no_grad()
    def deltakv_snapkv_finalize_static_prune(
        self,
        seqs: list[Sequence],
        combined_scores: torch.Tensor,
    ):
        self._reset_snapkv_view_cache()
        sink = int(self.config.num_sink_tokens)
        keep_budget = int(self.config.num_top_tokens_in_prefill or self.config.num_top_tokens or 0)

        for b_idx, seq in enumerate(seqs):
            seq_id = int(seq.seq_id)
            if seq_id in self._snapkv_finalized_seq_ids:
                continue
            row_idx = self.seq_id_to_row.get(seq_id, None)
            if row_idx is None:
                continue

            total_len = int(self.row_seq_lens[row_idx])
            comp_len = int(self.row_deltakv_compressed_lens[row_idx])
            if comp_len <= 0:
                self._snapkv_finalized_seq_ids.add(seq_id)
                continue

            comp_logical = torch.arange(sink, sink + comp_len, device="cuda", dtype=torch.int32)
            comp_abs = self._resolve_comp_abs_pos(row_idx, comp_logical)
            comp_raw = self.sparse_layer_raw_slots_map[row_idx, comp_logical.to(torch.long)].to(torch.int32)
            comp_latent = self.sparse_layer_latent_slots_map[row_idx, comp_logical.to(torch.long)].to(torch.int32)

            ref_mask = comp_raw >= 0
            cand_mask = comp_latent >= 0

            keep_mask = ref_mask.clone()
            cand_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(-1)
            if cand_idx.numel() > 0 and keep_budget > 0:
                score_row = combined_scores[b_idx]
                cand_abs = comp_abs[cand_idx].to(torch.long)
                cand_scores = score_row.index_select(0, cand_abs)
                k = min(int(keep_budget), int(cand_scores.numel()))
                top_rel = cand_scores.topk(k, dim=0).indices
                keep_cand_idx = cand_idx[top_rel]
                keep_mask[keep_cand_idx] = True

            keep_abs = comp_abs[keep_mask]
            keep_raw = comp_raw[keep_mask]
            keep_latent = comp_latent[keep_mask]
            if keep_abs.numel() > 0:
                order = torch.argsort(keep_abs)
                keep_abs = keep_abs[order]
                keep_raw = keep_raw[order]
                keep_latent = keep_latent[order]

            drop_latent = comp_latent[~keep_mask]
            drop_latent = drop_latent[drop_latent >= 0]
            if drop_latent.numel() > 0:
                ptr = self._num_free_slots_deltakv_latent
                self.free_slots_stack_deltakv_latent[ptr: ptr + drop_latent.numel()] = drop_latent.to(torch.int32)
                self._num_free_slots_deltakv_latent += int(drop_latent.numel())

            old_buffer_start = sink + comp_len
            buffer_len = total_len - old_buffer_start
            buffer_raw = self.sparse_layer_raw_slots_map[row_idx, old_buffer_start:total_len].clone()
            if (buffer_raw < 0).any():
                raise RuntimeError("DeltaKV+SnapKV finalize expects raw slots in the protected suffix.")

            new_comp_len = int(keep_abs.numel())
            new_total_len = sink + new_comp_len + buffer_len
            new_buffer_start = sink + new_comp_len

            self.sparse_layer_raw_slots_map[row_idx, sink:total_len] = -1
            self.sparse_layer_latent_slots_map[row_idx, sink:total_len] = -1
            self.row_deltakv_comp_abs_pos[row_idx, sink:total_len] = -1

            if new_comp_len > 0:
                comp_dst = torch.arange(sink, sink + new_comp_len, device="cuda", dtype=torch.long)
                keep_raw_mask = keep_raw >= 0
                if keep_raw_mask.any():
                    self.sparse_layer_raw_slots_map[row_idx, comp_dst[keep_raw_mask]] = keep_raw[keep_raw_mask]
                keep_latent_mask = keep_latent >= 0
                if keep_latent_mask.any():
                    self.sparse_layer_latent_slots_map[row_idx, comp_dst[keep_latent_mask]] = keep_latent[keep_latent_mask]
                self.row_deltakv_comp_abs_pos[row_idx, comp_dst] = keep_abs.to(torch.int32)

            if buffer_len > 0:
                self.sparse_layer_raw_slots_map[row_idx, new_buffer_start:new_total_len] = buffer_raw

            self.row_seq_lens[row_idx] = new_total_len
            self.row_deltakv_compressed_lens[row_idx] = new_comp_len
            self._snapkv_finalized_seq_ids.add(seq_id)
