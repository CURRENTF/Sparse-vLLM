from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence

from .rkv import RKVCacheManager


@dataclass
class SkipKVSentence:
    start_gen: int
    end_gen: int
    embedding: torch.Tensor
    redundancy: float = 0.0
    cache_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)


@dataclass
class SkipKVSequenceState:
    num_prompt_tokens: int
    open_start_gen: int | None = None
    open_embedding_sum: torch.Tensor | None = None
    open_embedding_count: int = 0
    open_has_non_execution_marker: bool = False
    sentences: list[SkipKVSentence] = field(default_factory=list)
    non_execution_count: int = 0
    redundant_sentence_count: int = 0


class SkipKVCacheManager(RKVCacheManager):
    """SkipKV sentence-aware KV storage skipping."""

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self._rkv_vectorized_prefill_query_cache = False
        self._rkv_batch_clear_query_cache_rows = False
        self._skipkv_delimiter_token_ids: set[int] = set()
        self._skipkv_non_execution_token_ids: set[int] = set()
        self._skipkv_seq_states: dict[int, SkipKVSequenceState] = {}
        self._skipkv_row_gen_indices: list[dict[int, list[int]]] = [
            {} for _ in range(self.num_layers)
        ]

    def set_skipkv_delimiter_token_ids(self, token_ids):
        self._skipkv_delimiter_token_ids = {int(x) for x in token_ids}

    def set_skipkv_non_execution_token_ids(self, token_ids):
        self._skipkv_non_execution_token_ids = {int(x) for x in token_ids}

    def skipkv_non_execution_count(self, seq_id: int) -> int:
        state = self._skipkv_seq_states.get(int(seq_id))
        return 0 if state is None else int(state.non_execution_count)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        result = super().prepare_step(seqs, is_prefill)
        if is_prefill:
            self._append_prefill_gen_indices(seqs)
        else:
            self._append_decode_gen_indices(seqs)
        return result

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
    ):
        result = super().prepare_decode_static(
            seqs,
            input_ids,
            positions,
            slot_mapping,
            context_lens,
            req_indices,
        )
        self._append_decode_gen_indices(seqs)
        return result

    def free_seq(self, seq_id: int):
        seq_id = int(seq_id)
        rows = [
            self.seq_id_to_row[layer_idx].get(seq_id)
            for layer_idx in range(self.num_layers)
        ]
        super().free_seq(seq_id)
        self._skipkv_seq_states.pop(seq_id, None)
        for layer_idx, row_idx in enumerate(rows):
            if row_idx is not None:
                self._skipkv_row_gen_indices[layer_idx].pop(int(row_idx), None)

    def _append_prefill_gen_indices(self, seqs: list[Sequence]):
        seq_chunks: list[tuple[int, list[int]]] = []
        for seq in seqs:
            start = int(seq.num_prefilled_tokens)
            size = int(seq.current_chunk_size or 0)
            if size <= 0:
                continue
            seq_chunks.append((int(seq.seq_id), list(range(start, start + size))))
        if not seq_chunks:
            return
        for layer_idx in range(self.num_layers):
            for seq_id, gen_indices in seq_chunks:
                row_idx = self.seq_id_to_row[layer_idx].get(seq_id)
                if row_idx is None:
                    continue
                self._skipkv_row_gen_indices[layer_idx].setdefault(int(row_idx), []).extend(
                    gen_indices
                )

    def _append_decode_gen_indices(self, seqs: list[Sequence]):
        for layer_idx in range(self.num_layers):
            for seq in seqs:
                row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
                if row_idx is None:
                    continue
                self._skipkv_row_gen_indices[layer_idx].setdefault(int(row_idx), []).append(
                    int(seq.num_tokens) - 1
                )

    def _get_seq_state(self, seq: Sequence) -> SkipKVSequenceState:
        state = self._skipkv_seq_states.get(int(seq.seq_id))
        if state is None:
            state = SkipKVSequenceState(num_prompt_tokens=int(seq.num_prompt_tokens))
            self._skipkv_seq_states[int(seq.seq_id)] = state
        return state

    def record_skipkv_decode_hidden_states(self, seqs: list[Sequence], hidden_states: torch.Tensor):
        if not bool(getattr(self.config, "skipkv_enable_sentence_scoring", True)):
            return
        if not self._skipkv_delimiter_token_ids:
            return
        for b_idx, seq in enumerate(seqs):
            token_pos = int(seq.num_tokens) - 1
            if token_pos < int(seq.num_prompt_tokens):
                continue
            token_id = int(seq.last_token) if seq.last_token is not None else -1
            state = self._get_seq_state(seq)
            if state.open_start_gen is None:
                state.open_start_gen = token_pos
                state.open_embedding_sum = torch.zeros_like(hidden_states[b_idx].float())
                state.open_embedding_count = 0
            if state.open_embedding_sum is None:
                state.open_embedding_sum = torch.zeros_like(hidden_states[b_idx].float())
            state.open_embedding_sum.add_(hidden_states[b_idx].detach().float())
            state.open_embedding_count += 1
            if token_id in self._skipkv_non_execution_token_ids:
                state.open_has_non_execution_marker = True

            sentence_len = token_pos - int(state.open_start_gen) + 1
            hit_delimiter = token_id in self._skipkv_delimiter_token_ids
            hit_max_len = sentence_len >= int(self.config.skipkv_sentence_max_tokens)
            if hit_delimiter or hit_max_len:
                self._finalize_sentence(seq, state, token_pos + 1)

    def _finalize_sentence(self, seq: Sequence, state: SkipKVSequenceState, end_gen: int):
        start_gen = int(state.open_start_gen) if state.open_start_gen is not None else int(end_gen)
        count = int(state.open_embedding_count)
        if count < int(self.config.skipkv_sentence_min_tokens) or state.open_embedding_sum is None:
            state.open_start_gen = None
            state.open_embedding_sum = None
            state.open_embedding_count = 0
            state.open_has_non_execution_marker = False
            return

        embedding = state.open_embedding_sum / float(max(1, count))
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0, eps=1.0e-6).detach()
        sentence = SkipKVSentence(start_gen=start_gen, end_gen=int(end_gen), embedding=embedding)
        if state.open_has_non_execution_marker:
            state.non_execution_count += 1
        self._score_new_sentence(state, sentence)
        state.sentences.append(sentence)

        max_sentences = int(self.config.skipkv_max_tracked_sentences)
        if len(state.sentences) > max_sentences:
            del state.sentences[: len(state.sentences) - max_sentences]

        state.open_start_gen = None
        state.open_embedding_sum = None
        state.open_embedding_count = 0
        state.open_has_non_execution_marker = False

    def _score_new_sentence(self, state: SkipKVSequenceState, sentence: SkipKVSentence):
        if not state.sentences:
            return
        prev_embeddings = torch.stack([s.embedding.to(sentence.embedding.device) for s in state.sentences], dim=0)
        sims = prev_embeddings.float() @ sentence.embedding.float()
        threshold = float(self.config.skipkv_similarity_threshold)
        redundant = sims >= threshold
        if not bool(redundant.any().item()):
            return
        sim_values = sims.detach().cpu().tolist()
        for idx, is_redundant in enumerate(redundant.detach().cpu().tolist()):
            if not is_redundant:
                continue
            score = float(sim_values[idx])
            state.sentences[idx].redundancy = max(float(state.sentences[idx].redundancy), score)
        state.redundant_sentence_count += int(bool(redundant.any().item()))

    def _update_sentence_cache_ranges(self, layer_idx: int, seq: Sequence, row_idx: int):
        state = self._skipkv_seq_states.get(int(seq.seq_id))
        if state is None or not state.sentences:
            return
        gen_indices = self._skipkv_row_gen_indices[layer_idx].get(int(row_idx), [])
        for sentence in state.sentences:
            positions = [
                pos
                for pos, gen_idx in enumerate(gen_indices)
                if sentence.start_gen <= int(gen_idx) < sentence.end_gen
            ]
            if not positions:
                sentence.cache_ranges.pop(int(layer_idx), None)
                continue
            sentence.cache_ranges[int(layer_idx)] = (min(positions), max(positions) + 1)

    def _sentence_redundancy_penalty(
        self,
        layer_idx: int,
        seq: Sequence,
        row_idx: int,
        *,
        candidate_start: int,
        candidate_end: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        state = self._skipkv_seq_states.get(int(seq.seq_id))
        if state is None or not state.sentences:
            return None
        self._update_sentence_cache_ranges(layer_idx, seq, row_idx)
        penalty = torch.zeros((candidate_end - candidate_start,), dtype=torch.float32, device=device)
        touched = False
        for sentence in state.sentences:
            redundancy = float(sentence.redundancy)
            if redundancy <= 0.0:
                continue
            cache_range = sentence.cache_ranges.get(int(layer_idx))
            if cache_range is None:
                continue
            start = max(int(cache_range[0]), int(candidate_start))
            end = min(int(cache_range[1]), int(candidate_end))
            if end <= start:
                continue
            rel_start = start - int(candidate_start)
            rel_end = end - int(candidate_start)
            penalty[rel_start:rel_end] = torch.maximum(
                penalty[rel_start:rel_end],
                torch.full((rel_end - rel_start,), redundancy, dtype=penalty.dtype, device=device),
            )
            touched = True
        return penalty if touched else None

    @staticmethod
    def segment_redundancy_penalty(
        keys: torch.Tensor,
        *,
        segment_size: int,
        similarity_threshold: float,
    ) -> torch.Tensor:
        token_count = int(keys.shape[0])
        if token_count == 0:
            return torch.empty((0,), dtype=torch.float32, device=keys.device)
        segment_size = int(segment_size)
        if segment_size <= 0:
            raise ValueError(f"segment_size must be > 0, got {segment_size}.")

        flat_keys = keys.float().reshape(token_count, -1)
        num_segments = (token_count + segment_size - 1) // segment_size
        if token_count == num_segments * segment_size:
            segments = flat_keys.view(num_segments, segment_size, flat_keys.shape[1])
            segment_emb = segments.mean(dim=1)
        else:
            padded = torch.zeros(
                (num_segments * segment_size, flat_keys.shape[1]),
                dtype=flat_keys.dtype,
                device=flat_keys.device,
            )
            padded[:token_count] = flat_keys
            segments = padded.view(num_segments, segment_size, flat_keys.shape[1])

            lengths = torch.full((num_segments,), segment_size, dtype=flat_keys.dtype, device=flat_keys.device)
            tail = token_count - (num_segments - 1) * segment_size
            lengths[-1] = float(tail)
            segment_emb = segments.sum(dim=1) / lengths[:, None].clamp_min(1.0)
        segment_emb = torch.nn.functional.normalize(segment_emb, p=2, dim=-1, eps=1.0e-6)

        sim = segment_emb @ segment_emb.transpose(0, 1)
        sim.diagonal().zero_()
        future = torch.triu(sim, diagonal=1)
        redundant_future = torch.where(
            future >= float(similarity_threshold),
            future,
            torch.zeros_like(future),
        )
        segment_penalty = redundant_future.max(dim=1).values
        token_penalty = torch.repeat_interleave(segment_penalty, segment_size)[:token_count]
        return token_penalty

    @staticmethod
    def segment_redundancy_penalty_batch(
        keys: torch.Tensor,
        *,
        segment_size: int,
        similarity_threshold: float,
    ) -> torch.Tensor:
        batch_size = int(keys.shape[0])
        token_count = int(keys.shape[1])
        if token_count == 0:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=keys.device)
        segment_size = int(segment_size)
        if segment_size <= 0:
            raise ValueError(f"segment_size must be > 0, got {segment_size}.")

        flat_keys = keys.float().reshape(batch_size, token_count, -1)
        num_segments = (token_count + segment_size - 1) // segment_size
        if token_count == num_segments * segment_size:
            segments = flat_keys.view(batch_size, num_segments, segment_size, flat_keys.shape[2])
            segment_emb = segments.mean(dim=2)
        else:
            padded = torch.zeros(
                (batch_size, num_segments * segment_size, flat_keys.shape[2]),
                dtype=flat_keys.dtype,
                device=flat_keys.device,
            )
            padded[:, :token_count] = flat_keys
            segments = padded.view(batch_size, num_segments, segment_size, flat_keys.shape[2])

            lengths = torch.full((num_segments,), segment_size, dtype=flat_keys.dtype, device=flat_keys.device)
            tail = token_count - (num_segments - 1) * segment_size
            lengths[-1] = float(tail)
            segment_emb = segments.sum(dim=2) / lengths[None, :, None].clamp_min(1.0)
        segment_emb = torch.nn.functional.normalize(segment_emb, p=2, dim=-1, eps=1.0e-6)

        sim = torch.bmm(segment_emb, segment_emb.transpose(1, 2))
        diag = torch.arange(num_segments, device=keys.device)
        sim[:, diag, diag] = 0.0
        future = torch.triu(sim, diagonal=1)
        redundant_future = torch.where(
            future >= float(similarity_threshold),
            future,
            torch.zeros_like(future),
        )
        segment_penalty = redundant_future.max(dim=2).values
        return torch.repeat_interleave(segment_penalty, segment_size, dim=1)[:, :token_count]

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        old_gen_indices = None
        if row_idx is not None:
            old_gen_indices = self._skipkv_row_gen_indices[layer_idx].get(int(row_idx), [])
        keep_sorted = torch.sort(keep_indices.to(device=self.device, dtype=torch.long)).values
        keep_cpu = keep_sorted.detach().to(device="cpu").tolist()
        super().free_part_slots(
            layer_idx,
            seq,
            keep_sorted,
            keep_indices_sorted=True,
        )
        if row_idx is None or old_gen_indices is None:
            return
        self._skipkv_row_gen_indices[layer_idx][int(row_idx)] = [
            old_gen_indices[int(i)]
            for i in keep_cpu
            if 0 <= int(i) < len(old_gen_indices)
        ]
        self._update_sentence_cache_ranges(layer_idx, seq, int(row_idx))

    def free_part_slots_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
    ):
        if not seqs:
            return
        if len(seqs) == 1:
            self.free_part_slots(layer_idx, seqs[0], keep_indices[0])
            return

        row_indices = []
        old_gen_indices = []
        for seq in seqs:
            row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
            row_indices.append(row_idx)
            if row_idx is None:
                old_gen_indices.append(None)
            else:
                old_gen_indices.append(self._skipkv_row_gen_indices[layer_idx].get(int(row_idx), []))

        keep_sorted = torch.sort(keep_indices.to(device=self.device, dtype=torch.long), dim=1).values
        keep_cpu = keep_sorted.detach().to(device="cpu").tolist()
        result = super().free_part_slots_batch(
            layer_idx,
            seqs,
            keep_sorted,
            keep_indices_sorted=True,
        )
        for seq, row_idx, seq_old_gen_indices, seq_keep_cpu in zip(
            seqs,
            row_indices,
            old_gen_indices,
            keep_cpu,
        ):
            if row_idx is None or seq_old_gen_indices is None:
                continue
            self._skipkv_row_gen_indices[layer_idx][int(row_idx)] = [
                seq_old_gen_indices[int(i)]
                for i in seq_keep_cpu
                if 0 <= int(i) < len(seq_old_gen_indices)
            ]
            self._update_sentence_cache_ranges(layer_idx, seq, int(row_idx))
        return result

    def free_part_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        keep_indices: torch.Tensor,
    ):
        if not layer_indices or not seqs:
            return
        if len(layer_indices) == 1:
            self.free_part_slots_batch(int(layer_indices[0]), seqs, keep_indices[0])
            return

        row_indices_by_layer = []
        old_gen_indices_by_layer = []
        for layer_idx in layer_indices:
            row_indices = []
            old_gen_indices = []
            for seq in seqs:
                row_idx = self.seq_id_to_row[int(layer_idx)].get(seq.seq_id)
                row_indices.append(row_idx)
                if row_idx is None:
                    old_gen_indices.append(None)
                else:
                    old_gen_indices.append(
                        self._skipkv_row_gen_indices[int(layer_idx)].get(int(row_idx), [])
                    )
            row_indices_by_layer.append(row_indices)
            old_gen_indices_by_layer.append(old_gen_indices)

        keep_sorted = torch.sort(keep_indices.to(device=self.device, dtype=torch.long), dim=2).values
        keep_cpu = keep_sorted.detach().to(device="cpu").tolist()
        result = super().free_part_slots_batch_layers(
            layer_indices,
            seqs,
            keep_sorted,
            keep_indices_sorted=True,
        )
        for local_layer, layer_idx in enumerate(layer_indices):
            layer_idx = int(layer_idx)
            for seq_idx, seq in enumerate(seqs):
                row_idx = row_indices_by_layer[local_layer][seq_idx]
                seq_old_gen_indices = old_gen_indices_by_layer[local_layer][seq_idx]
                if row_idx is None or seq_old_gen_indices is None:
                    continue
                self._skipkv_row_gen_indices[layer_idx][int(row_idx)] = [
                    seq_old_gen_indices[int(i)]
                    for i in keep_cpu[local_layer][seq_idx]
                    if 0 <= int(i) < len(seq_old_gen_indices)
                ]
                self._update_sentence_cache_ranges(layer_idx, seq, int(row_idx))
        return result

    def select_skipkv_indices(
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
            raise RuntimeError(f"Missing SkipKV row: layer={layer_idx} seq_id={seq.seq_id}.")
        slots = self.buffer_req_to_token_slots[layer_idx][row_idx, candidate_start:candidate_end].to(torch.long)
        candidate_importance = importance_scores[candidate_start:candidate_end].float()
        final_scores = candidate_importance.clone()

        window = min(int(self.config.skipkv_redundancy_window), int(slots.numel()))
        if window > 0:
            k_cache, _ = self.get_layer_kv_cache(layer_idx)
            window_slots = slots[-window:]
            window_keys = k_cache.index_select(0, window_slots)
            if float(self.config.skipkv_alpha) > 0.0:
                token_redundancy = self.redundancy_scores_from_keys(
                    window_keys,
                    similarity_threshold=float(self.config.rkv_similarity_threshold),
                    recent_similar_keep=int(self.config.rkv_recent_similar_keep),
                    max_tokens=int(self.config.skipkv_max_redundancy_tokens),
                )
                final_scores[-window:] -= float(self.config.skipkv_alpha) * token_redundancy
            sentence_penalty = self._sentence_redundancy_penalty(
                layer_idx,
                seq,
                int(row_idx),
                candidate_start=candidate_start,
                candidate_end=candidate_end,
                device=importance_scores.device,
            )
            if sentence_penalty is not None and bool(getattr(self.config, "skipkv_enable_sentence_scoring", True)):
                final_scores -= float(self.config.skipkv_sentence_score_weight) * sentence_penalty
            else:
                segment_penalty = self.segment_redundancy_penalty(
                    window_keys,
                    segment_size=int(self.config.skipkv_segment_size),
                    similarity_threshold=float(self.config.skipkv_similarity_threshold),
                )
                final_scores[-window:] -= segment_penalty

        keep_count = min(int(num_top), int(final_scores.numel()))
        top_rel = final_scores.topk(keep_count, dim=0, sorted=False).indices
        top_indices = top_rel + int(candidate_start)
        return torch.cat((sink_indices, top_indices, recent_indices), dim=0)

    def select_skipkv_indices_batch(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        importance_scores: torch.Tensor,
        kv_lens: list[int],
        budget: int,
    ) -> torch.Tensor | None:
        if not seqs:
            return torch.empty((0, 0), dtype=torch.long, device=importance_scores.device)
        if len(seqs) != len(kv_lens):
            raise RuntimeError(
                "select_skipkv_indices_batch expected one kv_len per sequence: "
                f"seqs={len(seqs)} kv_lens={len(kv_lens)}"
            )
        kv_len = int(kv_lens[0])
        if any(int(x) != kv_len for x in kv_lens):
            raise RuntimeError(
                "select_skipkv_indices_batch requires uniform kv_lens. "
                f"kv_lens={[int(x) for x in kv_lens]}"
            )
        for seq in seqs:
            state = self._skipkv_seq_states.get(int(seq.seq_id))
            if state is not None and state.sentences:
                return None

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
                raise RuntimeError(f"Missing SkipKV row: layer={layer_idx} seq_id={seq.seq_id}.")
            row_indices.append(int(row_idx))

        rows = torch.tensor(row_indices, dtype=torch.long, device=importance_scores.device)
        slots = self.buffer_req_to_token_slots[layer_idx][rows, candidate_start:candidate_end].to(torch.long)
        final_scores = importance_scores[:, candidate_start:candidate_end].float().clone()

        window = min(int(self.config.skipkv_redundancy_window), int(slots.shape[1]))
        if window > 0:
            k_cache, _ = self.get_layer_kv_cache(layer_idx)
            window_slots = slots[:, -window:]
            window_keys = k_cache.index_select(0, window_slots.reshape(-1)).view(
                len(seqs),
                window,
                k_cache.shape[1],
                k_cache.shape[2],
            )
            if float(self.config.skipkv_alpha) > 0.0:
                token_redundancy = self.redundancy_scores_from_keys_batch(
                    window_keys,
                    similarity_threshold=float(self.config.rkv_similarity_threshold),
                    recent_similar_keep=int(self.config.rkv_recent_similar_keep),
                    max_tokens=int(self.config.skipkv_max_redundancy_tokens),
                )
                final_scores[:, -window:] -= float(self.config.skipkv_alpha) * token_redundancy
            segment_penalty = self.segment_redundancy_penalty_batch(
                window_keys,
                segment_size=int(self.config.skipkv_segment_size),
                similarity_threshold=float(self.config.skipkv_similarity_threshold),
            )
            final_scores[:, -window:] -= segment_penalty

        keep_count = min(int(num_top), int(final_scores.shape[1]))
        top_rel = final_scores.topk(keep_count, dim=1, sorted=False).indices
        top_indices = top_rel + int(candidate_start)
        sink_batch = sink_indices.unsqueeze(0).expand(len(seqs), -1)
        recent_batch = recent_indices.unsqueeze(0).expand(len(seqs), -1)
        return torch.cat((sink_batch, top_indices, recent_batch), dim=1)
