from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import torch

from sparsevllm.engine.sequence import Sequence
import sparsevllm.platforms as platforms
from sparsevllm.utils.context import get_context, set_context
from sparsevllm.utils.profiler import profiler


def _default_context_buckets(max_context_len: int) -> list[int]:
    max_context_len = int(max_context_len)
    if max_context_len <= 0:
        max_context_len = 1024
    size = min(1024, max_context_len)
    buckets: list[int] = []
    while size < max_context_len:
        buckets.append(size)
        size *= 2
    buckets.append(size)
    return sorted(set(buckets))


def _normalize_context_buckets(value) -> list[int]:
    if value is None:
        return _default_context_buckets(1024)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            return _default_context_buckets(1024)
        buckets = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        buckets = [int(value)]
    else:
        buckets = [int(item) for item in value]
    buckets = sorted(set(buckets))
    if not buckets or any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"decode_cuda_graph_context_sizes must contain positive integers, got {buckets}.")
    return buckets


@dataclass(frozen=True)
class DecodeCudaGraphKey:
    method: str
    batch_size: int
    context_capacity: int
    is_long_text: bool
    capture_sampling: bool


@dataclass
class DecodeCudaGraphState:
    key: DecodeCudaGraphKey
    graph: torch.cuda.CUDAGraph | None = None
    input_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    req_indices: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    keepalive: list[object] = field(default_factory=list)
    sparse_state_refs: dict[int, dict[str, object]] = field(default_factory=dict)


class DecodeCudaGraphRunner:
    """Fixed-shape decode runner, optionally backed by CUDA Graph replay.

    The runner owns graph-stable decode metadata tensors. Cache managers still
    allocate real KV slots every step, but write the per-step metadata into these
    stable buffers before the model forward. CUDA Graph is an execution mode on
    top of the same static-compatible decode path; eager decode uses the same
    preparation and view-building route without capture/replay.
    """

    def __init__(
        self,
        *,
        runtime_state,
        cache_manager,
        recurrent_state_manager,
        sparse_controller,
        run_model: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        is_long_text_batch: Callable[[list[Sequence], bool], bool],
        method: str,
        rank: int,
        capture_sizes: list[int],
        context_sizes: list[int] | tuple[int, ...] | str | int | None = None,
        graph_pool=None,
    ):
        self.runtime_state = runtime_state
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.sparse_controller = sparse_controller
        self.run_model = run_model
        self.is_long_text_batch = is_long_text_batch
        self.method = str(method or "")
        self.rank = int(rank)
        self.platform = platforms.current_platform
        self.capture_sizes = sorted(set(int(size) for size in capture_sizes))
        if not self.capture_sizes or any(size <= 0 for size in self.capture_sizes):
            raise ValueError(f"decode_cuda_graph capture_sizes must be positive, got {capture_sizes}.")
        self.context_sizes = _normalize_context_buckets(context_sizes)
        self.max_context_len_override: int | None = None
        self._graphs: OrderedDict[DecodeCudaGraphKey, DecodeCudaGraphState] = OrderedDict()
        self.max_cached_graphs = self._resolve_max_cached_graphs()
        self.last_state_key: DecodeCudaGraphKey | None = None
        self.last_real_batch_size: int | None = None
        self.graph_pool = graph_pool

    def _resolve_max_cached_graphs(self) -> int | None:
        resolver = getattr(self.cache_manager, "decode_cuda_graph_max_cached_graphs", None)
        if resolver is None:
            return None
        max_cached_graphs = resolver()
        if max_cached_graphs is None:
            return None
        max_cached_graphs = int(max_cached_graphs)
        if max_cached_graphs <= 0:
            raise ValueError(f"decode_cuda_graph_max_cached_graphs must be positive, got {max_cached_graphs}.")
        return max_cached_graphs

    def set_max_context_len_override(self, max_context_len: int | None):
        self.max_context_len_override = None if max_context_len is None else int(max_context_len)

    def clear_captured_graphs(self):
        for state in list(self._graphs.values()):
            self._release_graph_state(state)
        self._graphs.clear()
        self.last_state_key = None
        self.last_real_batch_size = None

    @staticmethod
    def _release_graph_state(state: DecodeCudaGraphState):
        state.graph = None
        state.input_ids = None
        state.positions = None
        state.slot_mapping = None
        state.context_lens = None
        state.req_indices = None
        state.logits = None
        state.token_ids = None
        state.keepalive.clear()
        state.sparse_state_refs.clear()

    def _touch_graph_state(self, key: DecodeCudaGraphKey):
        move_to_end = getattr(self._graphs, "move_to_end", None)
        if move_to_end is not None:
            move_to_end(key)

    def _evict_cached_graphs(self, protected_key: DecodeCudaGraphKey):
        max_cached_graphs = getattr(self, "max_cached_graphs", None)
        if max_cached_graphs is None:
            return
        while len(self._graphs) > int(max_cached_graphs):
            for key in list(self._graphs.keys()):
                if key == protected_key:
                    continue
                state = self._graphs.pop(key)
                self._release_graph_state(state)
                break
            else:
                break

    def _context_capacity_bucket(self, context_len: int) -> int:
        """Map a real/requested context length to the configured graph bucket.

        Default buckets are 1k, 2k, 4k, 8k, ... . We intentionally do not
        match an arbitrary larger cached graph here; bucket selection should
        decide the exact graph family so a later 4k request does not silently
        replay a previously captured 128k graph.
        """
        context_len = max(1, int(context_len))
        buckets = getattr(self, "context_sizes", None)
        if not buckets:
            buckets = _default_context_buckets(max(1024, context_len))
            self.context_sizes = buckets
        for bucket in buckets:
            if int(bucket) >= context_len:
                return int(bucket)
        raise ValueError(
            "decode_cuda_graph_context_sizes do not cover current context length: "
            f"context_len={context_len}, context_sizes={list(buckets)}."
        )

    def _requested_context_capacity(self, seqs: list[Sequence]) -> int:
        max_context_len = max(int(seq.num_prompt_tokens) + int(seq.max_tokens) for seq in seqs)
        if self.max_context_len_override is not None:
            max_context_len = max(max_context_len, int(self.max_context_len_override))
        return self._context_capacity_bucket(max_context_len)

    def _current_context_capacity(self, seqs: list[Sequence]) -> int:
        max_context_len = max(int(seq.num_tokens) for seq in seqs)
        if self.max_context_len_override is not None:
            max_context_len = max(max_context_len, int(self.max_context_len_override))
        return self._context_capacity_bucket(max_context_len)

    def _select_graph_batch_size(self, real_batch_size: int) -> int:
        selector = getattr(self.cache_manager, "select_decode_cuda_graph_batch_size", None)
        if selector is not None:
            selected = selector(int(real_batch_size), list(self.capture_sizes))
            if selected is not None:
                return int(selected)

        for size in self.capture_sizes:
            if size >= real_batch_size:
                return int(size)
        raise ValueError(
            "decode_cuda_graph capture sizes do not cover current decode batch: "
            f"batch_size={real_batch_size}, capture_sizes={self.capture_sizes}."
        )

    def _select_state(
        self,
        *,
        method: str,
        batch_size: int,
        context_capacity: int,
        is_long_text: bool,
        capture_sampling: bool,
        allow_larger_context_capacity: bool = True,
    ) -> DecodeCudaGraphState:
        candidates = [
            state
            for key, state in self._graphs.items()
            if key.method == method
            and key.batch_size == batch_size
            and key.is_long_text == is_long_text
            and key.capture_sampling == capture_sampling
            and (
                key.context_capacity == context_capacity
                or (allow_larger_context_capacity and key.context_capacity >= context_capacity)
            )
        ]
        if candidates:
            state = min(candidates, key=lambda state: state.key.context_capacity)
            self._touch_graph_state(state.key)
            return state

        key = DecodeCudaGraphKey(
            method=method,
            batch_size=batch_size,
            context_capacity=context_capacity,
            is_long_text=bool(is_long_text),
            capture_sampling=capture_sampling,
        )
        state = DecodeCudaGraphState(key=key)
        device = getattr(
            self.cache_manager,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        state.input_ids = torch.empty((batch_size,), dtype=torch.int64, device=device)
        state.positions = torch.empty((batch_size,), dtype=torch.int64, device=device)
        state.slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device=device)
        state.context_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
        state.req_indices = torch.empty((batch_size,), dtype=torch.int32, device=device)
        self._graphs[key] = state
        self._evict_cached_graphs(key)
        return state

    def _prepare_static_step(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
        is_long_text: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepare_decode_static = getattr(self.runtime_state, "prepare_decode_static", None)
        if prepare_decode_static is None:
            raise TypeError("decode_cuda_graph requires runtime_state.prepare_decode_static().")

        assert state.input_ids is not None
        assert state.positions is not None
        assert state.slot_mapping is not None
        assert state.context_lens is not None
        assert state.req_indices is not None

        self.cache_manager.set_decode_static_max_context_len(int(state.key.context_capacity))
        input_ids, positions, _ = prepare_decode_static(
            seqs,
            state.input_ids,
            state.positions,
            state.slot_mapping,
            state.context_lens,
            state.req_indices,
        )

        set_context(
            False,
            cu_seqlens_q=None,
            cache_manager=self.cache_manager,
            is_long_text=bool(is_long_text),
            seqs=seqs,
            recurrent_state_manager=self.recurrent_state_manager,
        )
        self.cache_manager.set_decode_static_max_context_len(int(state.key.context_capacity))

        return input_ids, positions

    def _static_context_capacity_policy(self, seqs: list[Sequence]) -> tuple[int, bool]:
        """Return the static decode context bucket and whether larger cached buckets may match."""
        custom = self._cache_manager_graph_context_capacity(seqs)
        if custom is not None:
            return custom
        return self._current_context_capacity(seqs), False

    def _graph_context_capacity_policy(self, seqs: list[Sequence]) -> tuple[int, bool]:
        """Return the graph context bucket and whether larger cached buckets may match."""
        custom = self._cache_manager_graph_context_capacity(seqs)
        if custom is not None:
            return custom
        policy = str(
            getattr(getattr(self.cache_manager, "config", None), "decode_cuda_graph_context_policy", "current")
            or "current"
        ).strip().lower()
        if policy in {"requested", "request", "final"}:
            return self._requested_context_capacity(seqs), False
        if policy not in {"current", "cur", "now"}:
            raise ValueError(
                "decode_cuda_graph_context_policy must be 'current' or 'requested', "
                f"got {policy!r}."
            )
        return self._current_context_capacity(seqs), False

    def bucket_plan(self) -> dict[str, object]:
        return {
            "batch_sizes": list(self.capture_sizes),
            "context_sizes": list(self.context_sizes),
            "context_policy": str(
                getattr(getattr(self.cache_manager, "config", None), "decode_cuda_graph_context_policy", "current")
                or "current"
            ),
            "max_cached_graphs": self.max_cached_graphs,
        }

    def _cache_manager_graph_context_capacity(self, seqs: list[Sequence]) -> tuple[int, bool] | None:
        resolver = getattr(self.cache_manager, "decode_cuda_graph_context_capacity", None)
        if resolver is None:
            return None
        result = resolver(
            seqs,
            requested_context_capacity=self._requested_context_capacity(seqs),
            current_context_capacity=self._current_context_capacity(seqs),
        )
        if result is None:
            return None
        context_capacity, allow_larger_context_capacity = result
        return int(context_capacity), bool(allow_larger_context_capacity)

    def _snapshot_sparse_state_refs(self) -> dict[int, dict[str, object]]:
        refs: dict[int, dict[str, object]] = {}
        for layer_idx, sparse_state in self.sparse_controller.layer_batch_sparse_states.items():
            refs[int(layer_idx)] = {
                "attn_score": sparse_state.attn_score,
                "active_indices": sparse_state.active_indices,
                "active_slots": sparse_state.active_slots,
                "req_indices": sparse_state.req_indices,
                "context_lens": sparse_state.context_lens,
                "max_context_len": sparse_state.max_context_len,
                "active_compressed_indices": sparse_state.active_compressed_indices,
                "global_req_indices": sparse_state.global_req_indices,
                "deltakv_free_temp_slots": sparse_state.deltakv_free_temp_slots,
            }
        return refs

    def _restore_sparse_state_refs(self, state: DecodeCudaGraphState):
        """Restore Python sparse-state pointers captured by this graph.

        CUDA Graph replay uses the tensor addresses captured during warmup. A
        real request's prefill can overwrite SparseController Python fields
        before decode; restoring here keeps post-forward sparse eviction reading
        the same stable tensors that prepare_decode_static updates in place.
        """
        for layer_idx, refs in state.sparse_state_refs.items():
            sparse_state = self.sparse_controller.layer_batch_sparse_states[layer_idx]
            for name, value in refs.items():
                setattr(sparse_state, name, value)

    def _reset_graph_input_attn_scores(self, refs: dict[int, dict[str, object]]):
        """Reset graph-input score buffers inside capture/replay.

        prepare_forward() allocates and initializes decode attn_score tensors
        before graph capture. Replay does not re-run that Python setup, so score
        buffers used by captured observation-layer kernels must be reset by a
        captured fill before each replay.
        """
        for layer_refs in refs.values():
            attn_score = layer_refs.get("attn_score")
            if isinstance(attn_score, torch.Tensor):
                attn_score.fill_(-1e20)

    def _capture(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> DecodeCudaGraphState:
        if not self.platform.supports_graph_capture():
            raise RuntimeError(f"Platform {self.platform.name!r} does not support decode CUDA graph capture.")
        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller

        with profiler.record("decode_cuda_graph_warmup"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            logits = self.run_model(input_ids, positions, is_prefill=False)
            if state.key.capture_sampling:
                if logits is None:
                    raise RuntimeError("decode_cuda_graph capture_sampling requires rank-0 logits.")
                _ = logits.argmax(dim=-1)
        self.platform.synchronize()

        with profiler.record("decode_cuda_graph_capture"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            # OmniKV observation layers pass a 3D attn_score tensor into the
            # captured decode kernel, then replace the Python field with a 2D
            # head-reduced tensor in on_layer_end(). Keep the original refs so
            # graph replay cannot point at allocator-reused storage.
            graph_input_sparse_state_refs = self._snapshot_sparse_state_refs()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self.graph_pool):
                    self._reset_graph_input_attn_scores(graph_input_sparse_state_refs)
                    logits = self.run_model(input_ids, positions, is_prefill=False)
                    if state.key.capture_sampling:
                        if logits is None:
                            raise RuntimeError("decode_cuda_graph capture_sampling requires rank-0 logits.")
                        token_ids = logits.argmax(dim=-1)
                    else:
                        token_ids = None
            except Exception as exc:
                raise RuntimeError(f"decode_cuda_graph capture failed: {exc!r}") from exc

        state.graph = graph
        state.logits = logits
        state.token_ids = token_ids
        state.sparse_state_refs = self._snapshot_sparse_state_refs()

        keepalive: list[object] = [
            ctx,
            logits,
            ctx.decode_mid_o,
            ctx.decode_mid_o_logexpsum,
            state.input_ids,
            state.positions,
            state.slot_mapping,
            state.context_lens,
            state.req_indices,
        ]
        if token_ids is not None:
            keepalive.append(token_ids)
        for sparse_refs_by_layer in (graph_input_sparse_state_refs, state.sparse_state_refs):
            for refs in sparse_refs_by_layer.values():
                for value in refs.values():
                    if isinstance(value, torch.Tensor):
                        keepalive.append(value)
        keepalive.extend(self.cache_manager.decode_cuda_graph_keepalive_tensors())
        sparse_keepalive = getattr(self.sparse_controller, "decode_cuda_graph_keepalive_tensors", None)
        if sparse_keepalive is not None:
            keepalive.extend(sparse_keepalive())
        state.keepalive = keepalive
        return state

    def run(
        self,
        seqs: list[Sequence],
        *,
        capture_sampling: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not seqs:
            raise ValueError("decode_cuda_graph requires a non-empty decode batch.")
        if capture_sampling and any(seq.temperature > 1e-10 for seq in seqs):
            raise ValueError("decode_cuda_graph capture_sampling currently supports greedy decode only.")

        real_batch_size = len(seqs)
        force_eager = getattr(self.cache_manager, "decode_cuda_graph_force_eager", None)
        if force_eager is not None and force_eager():
            return self.run_eager_static(seqs), None

        graph_batch_size = self._select_graph_batch_size(real_batch_size)
        is_long_text = self.is_long_text_batch(seqs, False)
        context_capacity, allow_larger_context_capacity = self._graph_context_capacity_policy(seqs)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            context_capacity=context_capacity,
            is_long_text=is_long_text,
            capture_sampling=bool(capture_sampling),
            allow_larger_context_capacity=allow_larger_context_capacity,
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs, is_long_text)

        if state.graph is None:
            state = self._capture(state, seqs, input_ids, positions)
            self._restore_sparse_state_refs(state)
            with profiler.record("decode_cuda_graph_replay_after_capture"):
                state.graph.replay()
            logits = state.logits[:real_batch_size] if state.logits is not None else None
            token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
            return logits, token_ids

        self._restore_sparse_state_refs(state)
        with profiler.record("decode_cuda_graph_replay"):
            state.graph.replay()
        logits = state.logits[:real_batch_size] if state.logits is not None else None
        token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
        return logits, token_ids

    def run_eager_static(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """Run decode eagerly through the same static-compatible path used by graphs."""
        if not seqs:
            raise ValueError("static decode requires a non-empty decode batch.")

        real_batch_size = len(seqs)
        graph_batch_size = self._select_graph_batch_size(real_batch_size)
        is_long_text = self.is_long_text_batch(seqs, False)
        context_capacity, allow_larger_context_capacity = self._static_context_capacity_policy(seqs)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            context_capacity=context_capacity,
            is_long_text=is_long_text,
            capture_sampling=False,
            allow_larger_context_capacity=allow_larger_context_capacity,
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs, is_long_text)

        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller
        with profiler.record("model_sparse_prepare"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
        logits = self.run_model(input_ids, positions, is_prefill=False)
        if logits is None:
            return None
        return logits[:real_batch_size]
