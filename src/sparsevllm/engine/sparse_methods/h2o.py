from __future__ import annotations

import torch

from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import (
    normalize_sparse_method,
    resolve_prefill_sparse_method,
)
from sparsevllm.engine.cache_manager.h2o_retention import H2ORetention, H2O_PREFILL_QUERY_TILE
from sparsevllm.utils.profiler import profiler
from sparsevllm.utils.context import get_context
from sparsevllm.engine.cache_manager.base import ExplicitKVPayload
from sparsevllm.kernels.triton.prefill_score import PrefillScoreWorkspace

from .base import SparseStepContext, PrefillScoreEvent
from .passthrough import PassThroughRuntime
from .h2o_selection import select_h2o_heads


class H2ORuntime(PassThroughRuntime):
    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        self._prefill_score_workspace = PrefillScoreWorkspace()
        self._prefill_head_score_buffer: torch.Tensor | None = None
        self._prefill_head_score_total: torch.Tensor | None = None
        self._h2o_decode_attn_score_buffers: dict[
            tuple[int, ...],
            torch.Tensor,
        ] = {}
        self._active_h2o_decode_score_view: (
            tuple[list[int], torch.Tensor] | None
        ) = None

    def clear_decode_attn_score_buffers(self) -> None:
        super().clear_decode_attn_score_buffers()
        self._h2o_decode_attn_score_buffers.clear()
        self._active_h2o_decode_score_view = None

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return list(self._h2o_decode_attn_score_buffers.values())

    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        del layer_idx, step
        return False

    def prefill_score_shape(
        self,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> tuple[int, ...]:
        return batch_size, num_heads, max_len

    def prefill_score_fill_value(self) -> float:
        return 0.0

    @torch.no_grad()
    def collect_prefill_attention_score(self, event: PrefillScoreEvent) -> None:
        manager = self.cache_manager
        layer_idx, q, view = event.layer_idx, event.query, event.view
        b_start_loc, chunk_lens = event.b_start_loc, event.chunk_lens
        attention_lse = event.attention_lse
        ctx = get_context()
        if not ctx.is_prefill:
            raise RuntimeError("H2O prefill score collection was called outside prefill.")
        seqs = getattr(ctx, "seqs", None)
        if seqs is None:
            raise RuntimeError("H2O prefill score collection requires current seqs in context.")
        if int(chunk_lens.ndim) != 1 or int(chunk_lens.shape[0]) != len(seqs):
            raise RuntimeError(
                "H2O prefill scoring chunk-length batch mismatch: "
                f"shape={tuple(chunk_lens.shape)} seqs={len(seqs)}."
            )
        ranges = manager.prefill_score_ranges(layer_idx, seqs)
        if not ranges:
            return None
        if not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError(
                "H2O prefill scoring requires ExplicitKVPayload, got "
                f"{type(view.payload).__name__}."
            )
        meta = view.meta
        payload = view.payload

        context_lens = tuple(int(item[4]) for item in ranges)
        prepared_context_lens = getattr(
            manager,
            "_prefill_context_lens_cpu_by_layer",
            {},
        ).get(int(layer_idx))
        if prepared_context_lens is None and meta.context_lens.device.type == "cpu":
            prepared_context_lens = tuple(
                int(value) for value in meta.context_lens.tolist()
            )
        if prepared_context_lens is None:
            raise RuntimeError(
                "H2O prefill scoring requires CPU context lengths prepared "
                f"for layer={layer_idx}."
            )
        if tuple(prepared_context_lens) != context_lens:
            raise RuntimeError(
                "H2O prefill score view is not in compressed physical coordinates: "
                f"layer={layer_idx} view={tuple(prepared_context_lens)} "
                f"physical={context_lens}."
            )
        prompt_cache_lens_cpu = tuple(int(item[2]) for item in ranges)
        score_starts_cpu = tuple(int(item[3]) for item in ranges)
        score_ends_cpu = tuple(int(item[4]) for item in ranges)
        (
            prompt_cache_lens,
            _batch_indices,
            score_starts,
            score_ends,
        ) = manager._cached_prefill_score_metadata_tensors(
            device=q.device,
            context_lens=context_lens,
            prompt_cache_lens=prompt_cache_lens_cpu,
            batch_indices=tuple(range(len(ranges))),
            score_starts=score_starts_cpu,
            score_ends=score_ends_cpu,
        )
        max_context_len = max(context_lens)
        if meta.attn_score is not None:
            raise ValueError("H2O requires per-head probability scores after attention.")
        from sparsevllm.kernels.triton.prefill_score import (
            prefill_score_fwd, prefill_score_from_lse_fwd,
        )

        shape = (len(seqs), int(q.shape[1]), max_context_len)
        buffer = self._prefill_head_score_buffer
        if buffer is None or any(old < new for old, new in zip(buffer.shape, shape)):
            buffer = torch.empty(shape, dtype=torch.float32, device=q.device)
            self._prefill_head_score_buffer = buffer
            self._prefill_head_score_total = torch.empty_like(buffer)
        step_score = buffer[:shape[0], :shape[1], :shape[2]]
        total = self._prefill_head_score_total[:shape[0], :shape[1], :shape[2]]
        total.zero_()
        score_kwargs = dict(
            workspace=self._prefill_score_workspace, per_head=True,
            softmax_scale=event.softmax_scale,
        )
        max_queries = max(item[4] - item[3] for item in ranges)
        # Bound QK statistics workspace independently of prompt chunk size.
        # Tiles contribute probability sums; no head reduction occurs here.
        for offset in range(0, max_queries, H2O_PREFILL_QUERY_TILE):
            tile_start = torch.minimum(score_starts + offset, score_ends)
            tile_end = torch.minimum(tile_start + H2O_PREFILL_QUERY_TILE, score_ends)
            score_args = (
                step_score, meta.req_indices, b_start_loc, meta.context_lens,
                prompt_cache_lens, min(H2O_PREFILL_QUERY_TILE, max_queries - offset),
                meta.active_slots, tile_start, tile_end,
            )
            if attention_lse is None:
                prefill_score_fwd(q, payload.k_cache, *score_args, **score_kwargs)
            else:
                prefill_score_from_lse_fwd(
                    q, payload.k_cache, attention_lse, *score_args, **score_kwargs,
                )
            total.add_(step_score)
        manager.accumulate_prefill_scores(layer_idx, seqs, total)

    def finish_step(self, step: SparseStepContext) -> None:
        if not step.is_prefill:
            return
        manager = self.cache_manager
        requests = []
        for layer_idx in manager.kv_transformer_layer_indices():
            for seq in step.seqs:
                final = bool(seq.is_last_chunk_prefill)
                if final:
                    if not manager.h2o_decode_enabled:
                        continue
                elif not manager.h2o_prefill_enabled:
                    continue
                budget = manager.h2o_decode_budget if final else manager.h2o_prefill_budget
                length = manager._physical_row_len(layer_idx, seq)
                scores = manager._require_score_length(layer_idx, seq, length)
                if length <= budget:
                    continue
                keep = select_h2o_heads(
                    scores,
                    selection_groups=manager.h2o_selection_groups,
                    budget=budget,
                    recent_ratio=float(self.config.h2o_recent_ratio),
                    reduction=self.config.h2o_head_reduction,
                )
                requests.append(H2ORetention(layer_idx, int(seq.seq_id), length, keep, final))
        manager.commit_h2o_retention(requests)

    def _h2o_kv_layer_indices(self) -> list[int]:
        return [
            layer_idx
            for layer_idx in range(self.num_layers)
            if self._is_kv_layer(layer_idx)
        ]

    def _h2o_decode_score_width(self, layer_indices: list[int]) -> int:
        max_len = max(
            self._state_max_context_len(
                self.layer_batch_sparse_states[layer_idx]
            )
            for layer_idx in layer_indices
        )
        required_width = max_len
        if bool(getattr(self.config, "decode_graph", False)):
            graph_capacity = getattr(
                self.cache_manager,
                "_decode_static_max_context_len",
                None,
            )
            if graph_capacity is None or int(graph_capacity) < max_len:
                raise RuntimeError(
                    "H2O decode CUDA graph requires a score capacity covering the "
                    f"current context: graph_capacity={graph_capacity} "
                    f"current={max_len}."
                )
            required_width = int(graph_capacity)
        return int(required_width)

    def _get_h2o_decode_score_buffer(
        self,
        num_kv_layers: int,
        batch_size: int,
        width: int,
    ) -> torch.Tensor:
        if min(num_kv_layers, batch_size, width) <= 0:
            raise RuntimeError(
                "H2O decode score buffer requires positive dimensions: "
                f"shape={(num_kv_layers, batch_size, width)}."
            )
        if bool(getattr(self.config, "decode_graph", False)):
            key = (num_kv_layers, batch_size, width)
        else:
            key = (num_kv_layers,)
        buffer = self._h2o_decode_attn_score_buffers.get(key)
        needs_alloc = (
            buffer is None
            or buffer.dtype != self.snapkv_decode_score_dtype
            or buffer.device != self.device
            or int(buffer.shape[0]) < num_kv_layers
            or int(buffer.shape[1]) < batch_size
            or int(buffer.shape[2]) < width
        )
        if needs_alloc:
            buffer = torch.empty(
                (num_kv_layers, batch_size, width),
                dtype=self.snapkv_decode_score_dtype,
                device=self.device,
            )
            self._h2o_decode_attn_score_buffers[key] = buffer
        view = buffer[:num_kv_layers, :batch_size, :width]
        if needs_alloc or not bool(getattr(self.config, "decode_graph", False)):
            view.fill_(-1e20)
        return view

    def _prepare_h2o_decode_attn_score_buffer(self, seqs: list[Sequence]):
        del seqs
        layer_indices = self._h2o_kv_layer_indices()
        if not layer_indices:
            self._active_h2o_decode_score_view = None
            return
        batch_sizes = []
        kv_indices = []
        for layer_idx in layer_indices:
            state = self.layer_batch_sparse_states[layer_idx]
            if state.context_lens is None:
                raise RuntimeError(
                    "H2O decode state is missing context lengths: "
                    f"layer={layer_idx}."
                )
            batch_sizes.append(int(state.context_lens.numel()))
            kv_indices.append(self._kv_layer_index(layer_idx))
        if any(batch_size != batch_sizes[0] for batch_size in batch_sizes[1:]):
            raise RuntimeError(
                f"H2O decode KV layers disagree on batch size: {batch_sizes}."
            )
        if sorted(kv_indices) != list(range(len(layer_indices))):
            raise RuntimeError(
                "H2O decode KV-layer indices must densely cover the continuous "
                f"buffer: indices={kv_indices}."
            )
        width = self._h2o_decode_score_width(layer_indices)
        reduced_scores = self._get_h2o_decode_score_buffer(
            len(layer_indices),
            batch_sizes[0],
            width,
        )
        for layer_idx, kv_idx in zip(layer_indices, kv_indices):
            self.layer_batch_sparse_states[layer_idx].attn_score = reduced_scores[
                kv_idx
            ]
        self._active_h2o_decode_score_view = (layer_indices, reduced_scores)

    def _resolve_h2o_decode_attn_score_buffer(
        self,
        layer_tensors: dict[int, torch.Tensor],
    ) -> tuple[list[int], torch.Tensor]:
        layer_indices = self._h2o_kv_layer_indices()
        if set(layer_tensors) != set(layer_indices):
            raise RuntimeError(
                "H2O decode score slices do not cover every KV layer: "
                f"expected={layer_indices} got={sorted(layer_tensors)}."
            )
        first = layer_tensors[layer_indices[0]]
        if first.dim() != 2:
            raise RuntimeError(
                "H2O decode score slice must be [B, W], got "
                f"{tuple(first.shape)}."
            )
        batch_size, width = map(int, first.shape)
        for buffer in self._h2o_decode_attn_score_buffers.values():
            if (
                buffer.dtype != first.dtype
                or buffer.device != first.device
                or int(buffer.shape[0]) < len(layer_indices)
                or int(buffer.shape[1]) < batch_size
                or int(buffer.shape[2]) < width
            ):
                continue
            view = buffer[: len(layer_indices), :batch_size, :width]
            matches = True
            for layer_idx in layer_indices:
                kv_idx = self._kv_layer_index(layer_idx)
                layer_tensor = layer_tensors[layer_idx]
                if (
                    tuple(layer_tensor.shape) != (batch_size, width)
                    or layer_tensor.data_ptr() != view[kv_idx].data_ptr()
                ):
                    matches = False
                    break
            if matches:
                return layer_indices, view
        raise RuntimeError(
            "H2O decode layer score slices do not share a known contiguous buffer."
        )

    def reset_decode_attn_scores_for_graph(
        self,
        refs: dict[int, dict[str, object]],
    ) -> bool:
        layer_tensors = {
            layer_idx: layer_refs["attn_score"]
            for layer_idx, layer_refs in refs.items()
            if self._is_kv_layer(layer_idx)
            and isinstance(layer_refs.get("attn_score"), torch.Tensor)
        }
        if not layer_tensors:
            return False
        _layer_indices, reduced_scores = (
            self._resolve_h2o_decode_attn_score_buffer(layer_tensors)
        )
        reduced_scores.fill_(-1e20)
        return True

    @torch.no_grad()
    def _h2o_decode_eviction(self, seqs: list[Sequence]):
        with profiler.record("h2o_decode_eviction"):
            if self.validate_runtime_invariants:
                layer_tensors = {}
                layer_context_lens = []
                for layer_idx in self._h2o_kv_layer_indices():
                    state = self.layer_batch_sparse_states[layer_idx]
                    if state.attn_score is None or state.context_lens is None:
                        raise RuntimeError(
                            "H2O decode requires reduced probability scores for "
                            f"every KV layer: layer={layer_idx}."
                        )
                    if state.attn_score.dim() != 2:
                        raise RuntimeError(
                            "H2O decode must use the SnapKV-style [B, L] score "
                            f"path, got layer={layer_idx} "
                            f"shape={tuple(state.attn_score.shape)}."
                        )
                    if int(state.context_lens.numel()) != int(
                        state.attn_score.shape[0]
                    ):
                        raise RuntimeError(
                            "H2O decode context lengths do not match score batch: "
                            f"layer={layer_idx} "
                            f"contexts={int(state.context_lens.numel())} "
                            f"score_batch={int(state.attn_score.shape[0])}."
                        )
                    layer_tensors[layer_idx] = state.attn_score
                    layer_context_lens.append(state.context_lens)
                layer_indices, probability_scores = (
                    self._resolve_h2o_decode_attn_score_buffer(layer_tensors)
                )
                context_lens = torch.stack(layer_context_lens, dim=0)
                bounds_ok = (
                    (context_lens >= 0)
                    & (context_lens <= int(probability_scores.shape[2]))
                ).all()
                if context_lens.is_cuda:
                    torch._assert_async(bounds_ok)
                elif not bool(bounds_ok.item()):
                    raise RuntimeError(
                        "H2O decode context lengths exceed the reduced score width: "
                        f"width={int(probability_scores.shape[2])} "
                        f"contexts={context_lens.tolist()}."
                    )
            else:
                active = self._active_h2o_decode_score_view
                if active is None:
                    layer_tensors = {}
                    for layer_idx in self._h2o_kv_layer_indices():
                        score = self.layer_batch_sparse_states[layer_idx].attn_score
                        if score is not None:
                            layer_tensors[layer_idx] = score
                    layer_indices, probability_scores = (
                        self._resolve_h2o_decode_attn_score_buffer(layer_tensors)
                    )
                else:
                    layer_indices, probability_scores = active
            if int(probability_scores.shape[1]) < len(seqs):
                raise RuntimeError(
                    "H2O decode score batch does not cover current sequences: "
                    f"score_batch={int(probability_scores.shape[1])} "
                    f"seqs={len(seqs)}."
                )
            with profiler.record("h2o_decode_score_update"):
                self.cache_manager.update_decode_attention_scores_all_layers(
                    layer_indices,
                    seqs,
                    probability_scores[:, : len(seqs)],
                    normalize_logits=False,
                )
            with profiler.record("h2o_decode_compact_total"):
                self.cache_manager.evict_after_decode(seqs)
