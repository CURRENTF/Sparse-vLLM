from __future__ import annotations

from dataclasses import dataclass

import torch


H2O_PREFILL_QUERY_TILE = 128


@dataclass(frozen=True)
class H2ORetention:
    """One layer/request selection in the current packed cache coordinates."""

    layer_idx: int
    seq_id: int
    source_length: int
    keep: torch.Tensor  # [native KV heads, budget], or [1, budget] for MLA
    final_prefill: bool


class H2OPrefillRetentionMixin:
    """Cache-owned per-head state and overlap-safe physical retention."""

    @staticmethod
    def _assert_retention_tensor(condition: torch.Tensor, message: str) -> None:
        if condition.is_cuda:
            torch._assert_async(condition)
        elif not bool(condition.item()):
            raise RuntimeError(message)

    @property
    def h2o_selection_groups(self) -> int:
        from .storage import MlaLatentStorage

        return (
            1 if isinstance(self.attention_cache_storage, MlaLatentStorage)
            else self.num_kv_heads
        )

    def commit_h2o_retention(self, requests: list[H2ORetention]) -> None:
        from .storage import ExplicitKVStorage

        prepared = []
        seen = set()
        seen_rows = set()
        release_counts = {}
        storage = self.attention_cache_storage
        for request in requests:
            layer, seq_id, length = request.layer_idx, request.seq_id, request.source_length
            key = (layer, seq_id)
            if key in seen:
                raise ValueError("Duplicate H2O retention request.")
            seen.add(key)
            row = self.seq_id_to_row[layer][seq_id]
            if (layer, row) in seen_rows:
                raise ValueError("H2O retention requests share a physical row.")
            seen_rows.add((layer, row))
            if int(self.row_seq_lens[layer][row]) != length:
                raise ValueError("H2O retention refers to a stale physical row.")
            keep = request.keep
            groups = self.h2o_selection_groups
            if (
                keep.ndim != 2 or keep.shape[0] != groups
                or keep.dtype != torch.long or keep.device != self.device
                or not 0 < keep.shape[-1] < length
            ):
                raise ValueError("H2O retention requires [selection_groups, budget] int64 indices.")
            budget = int(keep.shape[-1])
            self._assert_retention_tensor(
                ((keep >= 0) & (keep < length)).all()
                & (keep[:, 1:] > keep[:, :-1]).all(),
                "H2O retention indices must be in bounds and strictly increasing.",
            )
            score = self._h2o_scores[key]
            positions = self._h2o_positions[key]
            if (
                score.ndim != 2 or score.shape[-1] != length
                or score.shape[0] % groups or tuple(positions.shape) != (groups, length)
            ):
                raise ValueError("H2O retention score/position metadata is not aligned.")
            slots = self.buffer_req_to_token_slots[layer][row, :length].long().clone()
            ordered_slots = slots.sort().values
            self._assert_retention_tensor(
                ((slots >= 0) & (slots < storage.slot_capacity())).all()
                & (ordered_slots[1:] > ordered_slots[:-1]).all(),
                "H2O retention physical slots must be valid and unique.",
            )
            release_counts[layer] = release_counts.get(layer, 0) + length - budget
            pointer = int(self._num_free_slots[layer])
            end = pointer + release_counts[layer]
            if pointer < 0 or end > self.free_slots_stack[layer].numel():
                raise RuntimeError(f"H2O retention would overflow the free-slot stack: layer={layer}.")
            # Every query head keeps its own history, including heads that did
            # not supply the group's maximum on this step.
            score_keep = keep.repeat_interleave(score.shape[0] // groups, dim=0)
            kept_score = score.gather(1, score_keep).contiguous()
            kept_positions = positions.gather(1, keep).contiguous()
            prepared.append((request, row, slots, ordered_slots, kept_score, kept_positions))

        # Validate the complete submission before publishing any row mutation.
        for request, row, slots, ordered_slots, score, positions in prepared:
            layer, seq_id, length = request.layer_idx, request.seq_id, request.source_length
            keep = request.keep
            budget = int(keep.shape[-1])
            destination = ordered_slots[:budget]
            released = ordered_slots[budget:]
            kv_layer = self.kv_layer_index(layer)
            selected = slots[keep]
            if isinstance(storage, ExplicitKVStorage):
                storage.copy_head_slots(kv_layer, selected, destination)
            else:
                storage.copy_slots(kv_layer, selected[0], destination)
            ptr = int(self._num_free_slots[layer])
            self.free_slots_stack[layer][ptr:ptr + released.numel()] = released
            self._num_free_slots[layer] = ptr + released.numel()
            self.buffer_req_to_token_slots[layer][row, :budget] = destination
            self.buffer_req_to_token_slots[layer][row, budget:length] = 0
            self.row_seq_lens[layer][row] = budget
            self._h2o_scores[(layer, seq_id)] = score
            self._h2o_positions[(layer, seq_id)] = positions
            counter = "final_prefill_evictions" if request.final_prefill else "intermediate_prefill_evictions"
            self._h2o_counters[counter] += 1
            self._h2o_counters["dropped_tokens"] += length - budget
        if prepared:
            self._uniform_decode_metadata = False
            self._decode_static_state_binding_key = None
            self._invalidate_h2o_decode_score_workspace()
