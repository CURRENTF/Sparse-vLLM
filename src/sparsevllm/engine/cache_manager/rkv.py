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

    def _rkv_query_cache_dtype(self) -> torch.dtype:
        dtype = getattr(self.hf_config, "torch_dtype", torch.float16)
        return dtype if isinstance(dtype, torch.dtype) else torch.float16

    def _rkv_num_query_heads(self) -> int:
        return int(self.hf_config.num_attention_heads) // int(self.world_size)

    def _rkv_query_cache_bytes(self) -> int:
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

    def _clear_rkv_query_cache_row(self, layer_idx: int, row_idx: int):
        self._rkv_query_positions[int(layer_idx)][int(row_idx)].fill_(-1)

    def free_seq(self, seq_id: int):
        row_by_layer = [
            self.seq_id_to_row[layer_idx].get(int(seq_id))
            for layer_idx in range(self.num_layers)
        ]
        for layer_idx, row_idx in enumerate(row_by_layer):
            if row_idx is not None:
                self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return super().free_seq(seq_id)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        row_idx = self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
        result = super().free_part_slots(layer_idx, seq, keep_indices)
        if row_idx is not None:
            self._clear_rkv_query_cache_row(layer_idx, row_idx)
        return result

    def decode_cuda_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return (
            super().decode_cuda_graph_keepalive_tensors()
            + list(self._rkv_query_cache)
            + list(self._rkv_query_positions)
        )

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
        if obs <= 0 or q.numel() == 0:
            return None

        layer_idx = int(layer_idx)
        cache = self._rkv_query_cache[layer_idx]
        positions_cache = self._rkv_query_positions[layer_idx]
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

    @torch.no_grad()
    def record_decode_query(self, layer_idx: int, q: torch.Tensor):
        obs = int(self._rkv_observation_tokens)
        if obs <= 0 or q.numel() == 0:
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
        logical_indices = torch.arange(candidate_start, candidate_end, dtype=torch.long, device=importance_scores.device)
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
        top_indices = logical_indices.index_select(0, top_rel)
        return torch.cat((sink_indices, top_indices, recent_indices), dim=0)
