from __future__ import annotations

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd

from .base import PrefillComputeView
from .snapkv import SnapKVCacheManager


class RKVCacheManager(SnapKVCacheManager):
    """SnapKV-style physical cache with R-KV decode-time joint eviction scoring."""

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self._rkv_observation_tokens = int(config.rkv_observation_tokens)
        self._rkv_query_cache_enabled = self._query_cache_needed_for_config(config)
        self._rkv_vectorized_prefill_query_cache = True
        self._rkv_batch_clear_query_cache_rows = True
        self._rkv_query_cache = []
        self._rkv_query_positions = []
        self._rkv_query_score_static_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        if self._rkv_query_cache_enabled:
            self._rkv_query_cache = [
                torch.empty(
                    (
                        self.max_buffer_rows,
                        self._rkv_observation_tokens,
                        self._rkv_num_query_heads(),
                        self.head_dim,
                    ),
                    dtype=self._rkv_query_cache_dtype(),
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]
            self._rkv_query_positions = [
                torch.full(
                    (self.max_buffer_rows, self._rkv_observation_tokens),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]

    @staticmethod
    def _query_cache_needed_for_config(config: Config) -> bool:
        obs = int(getattr(config, "rkv_observation_tokens", 0) or 0)
        if obs <= 0:
            return False
        budget = (
            int(getattr(config, "num_sink_tokens", 0) or 0)
            + int(getattr(config, "decode_keep_tokens", 0) or 0)
            + int(getattr(config, "num_recent_tokens", 0) or 0)
        )
        trigger_len = budget + int(getattr(config, "rkv_compression_interval", 0) or 0)
        return int(getattr(config, "max_model_len", 0) or 0) >= trigger_len

    def _is_rkv_query_cache_enabled(self) -> bool:
        return bool(getattr(self, "_rkv_query_cache_enabled", True))

    def _rkv_query_cache_dtype(self) -> torch.dtype:
        dtype = getattr(self.hf_config, "torch_dtype", torch.float16)
        return dtype if isinstance(dtype, torch.dtype) else torch.float16

    def _rkv_num_query_heads(self) -> int:
        return int(self.hf_config.num_attention_heads) // int(self.world_size)

    def _rkv_query_cache_bytes(self) -> int:
        if not self._is_rkv_query_cache_enabled():
            return 0
        obs = int(getattr(self.config, "rkv_observation_tokens", 0) or 0)
        if obs <= 0:
            return 0
        dtype_size = torch.tensor([], dtype=self._rkv_query_cache_dtype()).element_size()
        query_elems = (
            int(self.num_layers)
            * int(self.max_buffer_rows)
            * obs
            * self._rkv_num_query_heads()
            * int(self.head_dim)
        )
        position_elems = int(self.num_layers) * int(self.max_buffer_rows) * obs
        position_dtype_size = torch.tensor([], dtype=torch.int32).element_size()
        return int(query_elems * dtype_size + position_elems * position_dtype_size)

    def _get_available_slots_info(self) -> tuple[int, int]:
        available_memory, slot_bytes_per_layer = super()._get_available_slots_info()
        query_cache_bytes = self._rkv_query_cache_bytes()
        if query_cache_bytes >= available_memory:
            raise RuntimeError(
                "Not enough GPU memory for R-KV query cache. "
                f"query_cache={query_cache_bytes / 1024**3:.2f}GiB "
                f"available={available_memory / 1024**3:.2f}GiB. "
                "Reduce rkv_observation_tokens or max_num_seqs_in_batch."
            )
        return int(available_memory - query_cache_bytes), int(slot_bytes_per_layer)

    def _get_rkv_query_score_static_buffers(
        self,
        batch_size: int,
        max_score_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(batch_size), int(max_score_len))
        if not hasattr(self, "_rkv_query_score_static_buffers"):
            self._rkv_query_score_static_buffers = {}
        buffers = self._rkv_query_score_static_buffers.get(key)
        if buffers is None:
            offsets = torch.arange(int(max_score_len), dtype=torch.long, device=self.device)
            b_start_loc = (
                torch.arange(int(batch_size), dtype=torch.int32, device=self.device)
                * int(max_score_len)
            )
            buffers = (offsets, b_start_loc)
            self._rkv_query_score_static_buffers[key] = buffers
        return buffers

    def _clear_rkv_query_cache_row(self, layer_idx: int, row_idx: int):
        if not self._is_rkv_query_cache_enabled():
            return
        self._rkv_query_positions[int(layer_idx)][int(row_idx)].fill_(-1)

    def _clear_rkv_query_cache_rows(self, layer_idx: int, row_indices: list[int | None]):
        if not self._is_rkv_query_cache_enabled():
            return
        rows = [int(row_idx) for row_idx in row_indices if row_idx is not None]
        if not rows:
            return
        rows_tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
        self._rkv_query_positions[int(layer_idx)][rows_tensor].fill_(-1)

    def free_seq(self, seq_id: int):
        row_by_layer = [
            self.seq_id_to_row[layer_idx].get(int(seq_id))
            for layer_idx in range(self.num_layers)
        ]
        for layer_idx, row_idx in enumerate(row_by_layer):
            if row_idx is not None:
                self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return super().free_seq(seq_id)

    def free_part_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        row_idx = self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
        result = super().free_part_slots(
            layer_idx,
            seq,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
        )
        if row_idx is not None:
            self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return result

    def free_part_slots_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        row_indices = [
            self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
            for seq in seqs
        ]
        result = super().free_part_slots_batch(
            layer_idx,
            seqs,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
        )
        if bool(getattr(self, "_rkv_batch_clear_query_cache_rows", True)):
            self._clear_rkv_query_cache_rows(layer_idx, row_indices)
        else:
            for row_idx in row_indices:
                if row_idx is not None:
                    self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return result

    def free_part_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
        *,
        keep_indices_sorted: bool = False,
    ):
        row_indices_by_layer = [
            [
                self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
                for seq in seqs
            ]
            for layer_idx in layer_indices
        ]
        result = super().free_part_slots_batch_layers(
            layer_indices,
            seqs,
            keep_indices,
            keep_indices_sorted=keep_indices_sorted,
        )
        for layer_idx, row_indices in zip(layer_indices, row_indices_by_layer):
            if bool(getattr(self, "_rkv_batch_clear_query_cache_rows", True)):
                self._clear_rkv_query_cache_rows(int(layer_idx), row_indices)
            else:
                for row_idx in row_indices:
                    if row_idx is not None:
                        self._clear_rkv_query_cache_row(int(layer_idx), row_idx)
        return result

    def decode_cuda_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        if not self._is_rkv_query_cache_enabled():
            return super().decode_cuda_graph_keepalive_tensors()
        return super().decode_cuda_graph_keepalive_tensors() + list(self._rkv_query_cache) + list(self._rkv_query_positions)

    @torch.no_grad()
    def record_prefill_query(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
    ):
        obs = int(self._rkv_observation_tokens)
        if not self._is_rkv_query_cache_enabled() or obs <= 0 or q.numel() == 0:
            return None

        layer_idx = int(layer_idx)
        cache = self._rkv_query_cache[layer_idx]
        positions_cache = self._rkv_query_positions[layer_idx]
        if not bool(getattr(self, "_rkv_vectorized_prefill_query_cache", True)):
            batch = int(view.context_lens.numel())
            for b_idx in range(batch):
                context_len = int(view.context_lens[b_idx].item())
                chunk_len = int(chunk_lens[b_idx].item())
                if context_len <= 0 or chunk_len <= 0:
                    continue
                chunk_start = context_len - chunk_len
                record_start = max(chunk_start, context_len - obs)
                record_len = context_len - record_start
                if record_len <= 0:
                    continue

                row_idx = int(view.req_indices[b_idx].item())
                q_start = int(b_start_loc[b_idx].item()) + (record_start - chunk_start)
                token_positions = torch.arange(
                    record_start,
                    context_len,
                    dtype=torch.long,
                    device=q.device,
                )
                cols = token_positions.remainder(obs)
                cache[row_idx, cols] = q[q_start : q_start + record_len]
                positions_cache[row_idx, cols] = token_positions.to(torch.int32)
            return None

        context_lens = view.context_lens.to(device=q.device, dtype=torch.long)
        chunk_lens = chunk_lens.to(device=q.device, dtype=torch.long)
        req_indices = view.req_indices.to(device=q.device, dtype=torch.long)
        b_start_loc = b_start_loc.to(device=q.device, dtype=torch.long)

        offsets = torch.arange(obs, dtype=torch.long, device=q.device)
        record_lens = torch.minimum(chunk_lens.clamp_min(0), torch.full_like(chunk_lens, obs))
        record_starts = context_lens - record_lens
        valid = offsets.unsqueeze(0) < record_lens.unsqueeze(1)
        positions = record_starts.unsqueeze(1) + offsets.unsqueeze(0)
        cols = positions.remainder(obs)
        q_starts = b_start_loc + (chunk_lens - record_lens)
        q_indices = q_starts.unsqueeze(1) + offsets.unsqueeze(0)

        rows = req_indices.unsqueeze(1).expand_as(cols)
        cache[rows[valid], cols[valid]] = q[q_indices[valid]]
        positions_cache[rows[valid], cols[valid]] = positions[valid].to(torch.int32)
        return None

    @torch.no_grad()
    def record_decode_query(self, layer_idx: int, q: torch.Tensor):
        obs = int(self._rkv_observation_tokens)
        if not self._is_rkv_query_cache_enabled() or obs <= 0 or q.numel() == 0:
            return None

        layer_idx = int(layer_idx)
        batch_state = self.get_layer_batch_states(layer_idx)
        if batch_state.req_indices is None or batch_state.context_lens is None:
            raise RuntimeError(
                f"R-KV decode query cache requires decode metadata at layer={layer_idx}."
            )
        rows = batch_state.req_indices.to(device=q.device, dtype=torch.long)
        positions = batch_state.context_lens.to(device=q.device, dtype=torch.long) - 1
        cols = positions.remainder(obs)
        self._rkv_query_cache[layer_idx][rows, cols] = q
        self._rkv_query_positions[layer_idx][rows, cols] = positions.to(torch.int32)
        return None

    @torch.no_grad()
    def rkv_query_attention_scores(
        self,
        layer_idx: int,
        seq: Sequence,
        kv_len: int,
        *,
        candidate_start: int,
        num_recent_tokens: int,
    ) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if not self._is_rkv_query_cache_enabled():
            raise RuntimeError(
                "R-KV query cache is disabled because max_model_len is below the "
                "decode-eviction trigger; query attention scores should not be requested."
            )
        kv_len = int(kv_len)
        obs = int(self._rkv_observation_tokens)
        score_end = kv_len
        score_start = max(0, score_end - obs)
        score_len = score_end - score_start
        if score_len <= 0:
            return torch.zeros((kv_len,), dtype=self._prefill_score_dtype(), device=self.device)

        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise RuntimeError(f"Missing R-KV row: layer={layer_idx} seq_id={seq.seq_id}.")

        positions = torch.arange(score_start, score_end, dtype=torch.long, device=self.device)
        cols = positions.remainder(obs)
        stored_positions = self._rkv_query_positions[layer_idx][row_idx, cols].to(torch.long)
        if bool((stored_positions != positions).any().item()):
            raise RuntimeError(
                "R-KV query cache missing observation positions: "
                f"layer={layer_idx} seq_id={seq.seq_id} "
                f"needed=[{score_start}, {score_end}) "
                f"stored_min={int(stored_positions.min().item())} "
                f"stored_max={int(stored_positions.max().item())}."
            )

        q_window = self._rkv_query_cache[layer_idx][row_idx, cols].contiguous()
        k_cache, _ = self.get_layer_kv_cache(layer_idx)
        attn_score = torch.zeros(
            (1, kv_len),
            dtype=self._prefill_score_dtype(),
            device=self.device,
        )
        b_req_idx = torch.tensor([row_idx], dtype=torch.int32, device=self.device)
        b_start_loc = torch.zeros((1,), dtype=torch.int32, device=self.device)
        b_seq_len = torch.tensor([kv_len], dtype=torch.int32, device=self.device)
        b_prompt_cache_len = torch.tensor([score_start], dtype=torch.int32, device=self.device)
        score_q_start = torch.tensor([score_start], dtype=torch.int32, device=self.device)
        score_q_end = torch.tensor([score_end], dtype=torch.int32, device=self.device)

        prefill_score_fwd(
            q_window,
            k_cache,
            attn_score,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_prompt_cache_len,
            score_len,
            self.buffer_req_to_token_slots[layer_idx],
            score_q_start,
            score_q_end,
            candidate_start=int(candidate_start),
            num_recent_tokens=int(num_recent_tokens),
        )
        return attn_score[0]

    @torch.no_grad()
    def rkv_query_attention_scores_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        kv_lens: list[int],
        *,
        candidate_start: int,
        num_recent_tokens: int,
    ) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if not self._is_rkv_query_cache_enabled():
            raise RuntimeError(
                "R-KV query cache is disabled because max_model_len is below the "
                "decode-eviction trigger; query attention scores should not be requested."
            )
        if not seqs:
            return torch.empty((0, 0), dtype=self._prefill_score_dtype(), device=self.device)
        if len(seqs) != len(kv_lens):
            raise RuntimeError(
                "rkv_query_attention_scores_batch expected one kv_len per sequence: "
                f"seqs={len(seqs)} kv_lens={len(kv_lens)}"
            )

        obs = int(self._rkv_observation_tokens)
        if obs <= 0:
            return torch.zeros((len(seqs), max(int(x) for x in kv_lens)), dtype=self._prefill_score_dtype(), device=self.device)

        row_indices = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise RuntimeError(f"Missing R-KV row: layer={layer_idx} seq_id={seq.seq_id}.")
            row_indices.append(int(row_idx))

        kv_lens_cpu = [int(kv_len) for kv_len in kv_lens]
        score_ends_cpu = kv_lens_cpu
        score_starts_cpu = [max(0, int(kv_len) - obs) for kv_len in score_ends_cpu]
        score_lens_cpu = [
            int(score_end) - int(score_start)
            for score_start, score_end in zip(score_starts_cpu, score_ends_cpu)
        ]
        max_score_len = max(score_lens_cpu) if score_lens_cpu else 0
        max_kv_len = max(kv_lens_cpu) if kv_lens_cpu else 0

        rows = torch.tensor(row_indices, dtype=torch.long, device=self.device)
        kv_lens_tensor = torch.tensor(kv_lens_cpu, dtype=torch.long, device=self.device)
        score_ends = kv_lens_tensor
        score_starts = torch.tensor(score_starts_cpu, dtype=torch.long, device=self.device)
        score_lens = torch.tensor(score_lens_cpu, dtype=torch.long, device=self.device)
        if max_score_len <= 0:
            return torch.zeros((len(seqs), max_kv_len), dtype=self._prefill_score_dtype(), device=self.device)

        offsets, b_start_loc = self._get_rkv_query_score_static_buffers(len(seqs), max_score_len)
        positions = score_starts[:, None] + offsets[None, :]
        valid_positions = offsets[None, :] < score_lens[:, None]
        cols = positions.remainder(obs)
        stored_positions = self._rkv_query_positions[layer_idx][rows[:, None], cols].to(torch.long)
        positions_ok = ((stored_positions == positions) | ~valid_positions).all()
        if positions_ok.is_cuda:
            torch._assert_async(positions_ok)
        elif not bool(positions_ok.item()):
            raise RuntimeError(
                "R-KV query cache missing observation positions in batch: "
                f"layer={layer_idx} kv_lens={[int(x) for x in kv_lens]}."
            )

        q_window = self._rkv_query_cache[layer_idx][rows[:, None], cols].contiguous()
        q_window = q_window.view(len(seqs) * max_score_len, q_window.shape[2], q_window.shape[3])
        k_cache, _ = self.get_layer_kv_cache(layer_idx)
        attn_score = torch.zeros(
            (len(seqs), max_kv_len),
            dtype=self._prefill_score_dtype(),
            device=self.device,
        )
        prefill_score_fwd(
            q_window,
            k_cache,
            attn_score,
            rows.to(torch.int32),
            b_start_loc,
            kv_lens_tensor.to(torch.int32),
            score_starts.to(torch.int32),
            int(max_score_len),
            self.buffer_req_to_token_slots[layer_idx],
            score_starts.to(torch.int32),
            score_ends.to(torch.int32),
            candidate_start=int(candidate_start),
            num_recent_tokens=int(num_recent_tokens),
        )
        return attn_score

    @staticmethod
    def redundancy_scores_from_keys(
        keys: torch.Tensor,
        *,
        similarity_threshold: float,
        recent_similar_keep: int,
        max_tokens: int,
    ) -> torch.Tensor:
        token_count = int(keys.shape[0])
        if token_count == 0:
            return torch.empty((0,), dtype=torch.float32, device=keys.device)
        if token_count > int(max_tokens):
            raise RuntimeError(
                "R-KV redundancy scoring is quadratic in candidate tokens. "
                f"candidate_tokens={token_count} exceeds rkv_max_redundancy_tokens={int(max_tokens)}. "
                "Reduce decode_keep_tokens/rkv_compression_interval or raise the explicit limit."
            )

        flat_keys = keys.float().reshape(token_count, -1)
        flat_keys = torch.nn.functional.normalize(flat_keys, p=2, dim=-1, eps=1.0e-6)
        sim = flat_keys @ flat_keys.transpose(0, 1)
        sim.diagonal().zero_()

        threshold = float(similarity_threshold)
        if threshold > 0.0:
            sim = torch.where(sim >= threshold, sim, torch.zeros_like(sim))

        keep = int(recent_similar_keep)
        if keep > 0 and token_count > 1:
            upper = torch.triu(torch.ones((token_count, token_count), dtype=torch.bool, device=keys.device), diagonal=1)
            high_future = (sim > 0) & upper
            # For each token, ignore up to the most recent similar future tokens so
            # later reasoning tokens are not penalized just because older tokens match them.
            future_rank_from_right = high_future.flip(1).to(torch.int32).cumsum(1).flip(1)
            keep_recent_links = high_future & (future_rank_from_right <= keep)
            sim = sim.masked_fill(keep_recent_links, 0.0)

        avg_sim = sim.mean(dim=1)
        return torch.softmax(avg_sim, dim=0)

    @staticmethod
    def redundancy_scores_from_keys_batch(
        keys: torch.Tensor,
        *,
        similarity_threshold: float,
        recent_similar_keep: int,
        max_tokens: int,
    ) -> torch.Tensor:
        batch_size = int(keys.shape[0])
        token_count = int(keys.shape[1])
        if token_count == 0:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=keys.device)
        if token_count > int(max_tokens):
            raise RuntimeError(
                "R-KV redundancy scoring is quadratic in candidate tokens. "
                f"candidate_tokens={token_count} exceeds rkv_max_redundancy_tokens={int(max_tokens)}. "
                "Reduce decode_keep_tokens/rkv_compression_interval or raise the explicit limit."
            )

        flat_keys = keys.float().reshape(batch_size, token_count, -1)
        flat_keys = torch.nn.functional.normalize(flat_keys, p=2, dim=-1, eps=1.0e-6)
        sim = torch.bmm(flat_keys, flat_keys.transpose(1, 2))
        diag = torch.arange(token_count, device=keys.device)
        sim[:, diag, diag] = 0.0

        threshold = float(similarity_threshold)
        if threshold > 0.0:
            sim = torch.where(sim >= threshold, sim, torch.zeros_like(sim))

        keep = int(recent_similar_keep)
        if keep > 0 and token_count > 1:
            upper = torch.triu(
                torch.ones((token_count, token_count), dtype=torch.bool, device=keys.device),
                diagonal=1,
            )
            high_future = (sim > 0) & upper.unsqueeze(0)
            future_rank_from_right = high_future.flip(2).to(torch.int32).cumsum(2).flip(2)
            keep_recent_links = high_future & (future_rank_from_right <= keep)
            sim = sim.masked_fill(keep_recent_links, 0.0)

        avg_sim = sim.mean(dim=2)
        return torch.softmax(avg_sim, dim=1)

    @staticmethod
    def joint_retention_scores(
        importance: torch.Tensor,
        redundancy: torch.Tensor,
        *,
        alpha: float,
    ) -> torch.Tensor:
        """Paper-style R-KV score: alpha * importance - (1 - alpha) * redundancy."""
        alpha = float(alpha)
        return alpha * importance.float() - (1.0 - alpha) * redundancy.float()

    def select_rkv_indices(
        self,
        layer_idx: int,
        seq: Sequence,
        importance_scores: torch.Tensor,
        kv_len: int,
        budget: int,
    ) -> torch.Tensor:
        kv_len = int(kv_len)
        budget = int(budget)
        if kv_len <= budget:
            return torch.arange(kv_len, dtype=torch.long, device=importance_scores.device)

        num_sink = min(int(self.config.num_sink_tokens), kv_len)
        num_recent = min(int(self.config.num_recent_tokens), max(0, kv_len - num_sink))
        recent_start = kv_len - num_recent
        candidate_start = num_sink
        candidate_end = max(candidate_start, recent_start)
        num_top = max(0, budget - num_sink - num_recent)

        sink_indices = torch.arange(0, num_sink, dtype=torch.long, device=importance_scores.device)
        recent_indices = torch.arange(recent_start, kv_len, dtype=torch.long, device=importance_scores.device)
        if num_top <= 0 or candidate_end <= candidate_start:
            return torch.cat((sink_indices, recent_indices), dim=0)

        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise RuntimeError(f"Missing R-KV row: layer={layer_idx} seq_id={seq.seq_id}.")
        slots = self.buffer_req_to_token_slots[layer_idx][row_idx, candidate_start:candidate_end].to(torch.long)
        candidate_importance = importance_scores[candidate_start:candidate_end].float()
        final_scores = candidate_importance.clone()

        configured_window = int(self.config.rkv_redundancy_window)
        window = int(slots.numel()) if configured_window == 0 else min(configured_window, int(slots.numel()))
        if window > 0:
            k_cache, _ = self.get_layer_kv_cache(layer_idx)
            window_slots = slots[-window:]
            window_keys = k_cache.index_select(0, window_slots)
            redundancy = self.redundancy_scores_from_keys(
                window_keys,
                similarity_threshold=float(self.config.rkv_similarity_threshold),
                recent_similar_keep=int(self.config.rkv_recent_similar_keep),
                max_tokens=int(self.config.rkv_max_redundancy_tokens),
            )
            final_scores[-window:] = self.joint_retention_scores(
                candidate_importance[-window:],
                redundancy,
                alpha=float(self.config.rkv_alpha),
            )

        keep_count = min(int(num_top), int(final_scores.numel()))
        top_rel = final_scores.topk(keep_count, dim=0, sorted=False).indices
        top_indices = top_rel + int(candidate_start)
        return torch.cat((sink_indices, top_indices, recent_indices), dim=0)

    def select_rkv_indices_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        importance_scores: torch.Tensor,
        kv_lens: list[int],
        budget: int,
    ) -> torch.Tensor:
        if not seqs:
            return torch.empty((0, 0), dtype=torch.long, device=importance_scores.device)
        if len(seqs) != len(kv_lens):
            raise RuntimeError(
                "select_rkv_indices_batch expected one kv_len per sequence: "
                f"seqs={len(seqs)} kv_lens={len(kv_lens)}"
            )
        kv_len = int(kv_lens[0])
        if any(int(x) != kv_len for x in kv_lens):
            raise RuntimeError(
                "select_rkv_indices_batch requires uniform kv_lens. "
                f"kv_lens={[int(x) for x in kv_lens]}"
            )
        budget = int(budget)
        if kv_len <= budget:
            keep = torch.arange(kv_len, dtype=torch.long, device=importance_scores.device)
            return keep.unsqueeze(0).expand(len(seqs), -1).contiguous()

        num_sink = min(int(self.config.num_sink_tokens), kv_len)
        num_recent = min(int(self.config.num_recent_tokens), max(0, kv_len - num_sink))
        recent_start = kv_len - num_recent
        candidate_start = num_sink
        candidate_end = max(candidate_start, recent_start)
        num_top = max(0, budget - num_sink - num_recent)

        sink_indices = torch.arange(0, num_sink, dtype=torch.long, device=importance_scores.device)
        recent_indices = torch.arange(recent_start, kv_len, dtype=torch.long, device=importance_scores.device)
        if num_top <= 0 or candidate_end <= candidate_start:
            keep = torch.cat((sink_indices, recent_indices), dim=0)
            return keep.unsqueeze(0).expand(len(seqs), -1).contiguous()

        row_indices = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            if row_idx is None:
                raise RuntimeError(f"Missing R-KV row: layer={layer_idx} seq_id={seq.seq_id}.")
            row_indices.append(int(row_idx))

        rows = torch.tensor(row_indices, dtype=torch.long, device=importance_scores.device)
        slots = self.buffer_req_to_token_slots[layer_idx][rows, candidate_start:candidate_end].to(torch.long)
        final_scores = importance_scores[:, candidate_start:candidate_end].float().clone()

        configured_window = int(self.config.rkv_redundancy_window)
        window = int(slots.shape[1]) if configured_window == 0 else min(configured_window, int(slots.shape[1]))
        if window > 0:
            k_cache, _ = self.get_layer_kv_cache(layer_idx)
            window_slots = slots[:, -window:]
            window_keys = k_cache.index_select(0, window_slots.reshape(-1)).view(
                len(seqs),
                window,
                k_cache.shape[1],
                k_cache.shape[2],
            )
            redundancy = self.redundancy_scores_from_keys_batch(
                window_keys,
                similarity_threshold=float(self.config.rkv_similarity_threshold),
                recent_similar_keep=int(self.config.rkv_recent_similar_keep),
                max_tokens=int(self.config.rkv_max_redundancy_tokens),
            )
            final_scores[:, -window:] = self.joint_retention_scores(
                importance_scores[:, candidate_start:candidate_end][:, -window:],
                redundancy,
                alpha=float(self.config.rkv_alpha),
            )

        keep_count = min(int(num_top), int(final_scores.shape[1]))
        top_rel = final_scores.topk(keep_count, dim=1, sorted=False).indices
        top_indices = top_rel + int(candidate_start)
        sink_batch = sink_indices.unsqueeze(0).expand(len(seqs), -1)
        recent_batch = recent_indices.unsqueeze(0).expand(len(seqs), -1)
        return torch.cat((sink_batch, top_indices, recent_batch), dim=1)
