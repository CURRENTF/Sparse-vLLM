from __future__ import annotations

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence

from .snapkv import SnapKVCacheManager


class RKVCacheManager(SnapKVCacheManager):
    """SnapKV-style physical cache with R-KV decode-time joint eviction scoring."""

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

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
