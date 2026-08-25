from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import torch

import sparsevllm.platforms as platforms
from sparsevllm.configs.cuda_graph import _select_decode_cuda_graph_batch_size
from sparsevllm.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
    DecodeGraphState,
)
from sparsevllm.engine.sequence import Sequence
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
        raise ValueError(f"decode_graph_context_sizes must contain positive integers, got {buckets}.")
    return buckets


@dataclass(frozen=True)
class DecodeCudaGraphKey:
    method: str
    batch_size: int
    context_capacity: int
    is_long_text: bool
    capture_sampling: bool
    graph_path_id: str = ""
    shape_policy: str = "bucketed"


@dataclass
class DecodeCudaGraphState:
    key: DecodeCudaGraphKey
    capture_context_capacity: int = 0
    decode_state: DecodeGraphState | None = None
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    keepalive: list[object] = field(default_factory=list)
    sparse_state_refs: dict[int, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capture_context_capacity <= 0 and self.key.context_capacity > 0:
            self.capture_context_capacity = int(self.key.context_capacity)


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
        capture_sizes: list[int],
        context_sizes: list[int] | tuple[int, ...] | str | int | None = None,
        shape_policy: str = "bucketed",
        graph_pool=None,
    ):
        self.runtime_state = runtime_state
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.sparse_controller = sparse_controller
        self.run_model = run_model
        self.is_long_text_batch = is_long_text_batch
        self.method = str(method or "")
        self.platform = platforms.current_platform
        self.capture_sizes = sorted(set(int(size) for size in capture_sizes))
        if not self.capture_sizes or any(size <= 0 for size in self.capture_sizes):
            raise ValueError(f"decode_graph capture_sizes must be positive, got {capture_sizes}.")
        self.context_sizes = _normalize_context_buckets(context_sizes)
        self.shape_policy = str(shape_policy).strip().lower()
        if self.shape_policy not in {"bucketed", "batch_only"}:
            raise ValueError(f"Unsupported decode graph shape policy {self.shape_policy!r}.")
        self.max_context_len_override: int | None = None
        self._graphs: OrderedDict[DecodeCudaGraphKey, DecodeCudaGraphState] = OrderedDict()
        self.max_cached_graphs = self._resolve_max_cached_graphs()
        self.last_state_key: DecodeCudaGraphKey | None = None
        self.last_real_batch_size: int | None = None
        self.graph_pool = graph_pool
        self.capture_count = 0
        self.replay_count = 0
        self.eager_static_count = 0
        self.force_eager_count = 0
        self.eviction_count = 0
        self.recapture_count = 0
        self._captured_keys: set[DecodeCudaGraphKey] = set()
        self.reuse_larger_context_graphs = False
        self.startup_plan_sealed = False

    def _resolve_max_cached_graphs(self) -> int | None:
        resolver = getattr(self.cache_manager, "decode_graph_max_cached_graphs", None)
        if resolver is None:
            return None
        max_cached_graphs = resolver()
        if max_cached_graphs is None:
            return None
        max_cached_graphs = int(max_cached_graphs)
        if max_cached_graphs <= 0:
            raise ValueError(f"decode_graph_max_cached_graphs must be positive, got {max_cached_graphs}.")
        return max_cached_graphs

    def set_max_context_len_override(self, max_context_len: int | None):
        self.max_context_len_override = None if max_context_len is None else int(max_context_len)

    def set_reuse_larger_context_graphs(self, enabled: bool):
        self.reuse_larger_context_graphs = bool(enabled)

    def seal_startup_plan(self):
        if getattr(self, "shape_policy", "bucketed") == "batch_only":
            self.startup_plan_sealed = True

    def clear_captured_graphs(self):
        for state in list(self._graphs.values()):
            self._release_graph_state(state)
        self._graphs.clear()
        self.last_state_key = None
        self.last_real_batch_size = None

    @staticmethod
    def _release_graph_state(state: DecodeCudaGraphState):
        state.graph = None
        if state.decode_state is not None:
            state.decode_state.close()
        state.decode_state = None
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
                self.eviction_count = int(getattr(self, "eviction_count", 0)) + 1
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
            "decode_graph_context_sizes do not cover current context length: "
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

        return _select_decode_cuda_graph_batch_size(
            real_batch_size,
            self.capture_sizes,
        )

    def _select_state(
        self,
        *,
        method: str,
        batch_size: int,
        context_capacity: int,
        is_long_text: bool,
        capture_sampling: bool,
        graph_path_id: str = "",
        allow_larger_context_capacity: bool = True,
    ) -> DecodeCudaGraphState:
        shape_policy = getattr(self, "shape_policy", "bucketed")
        graph_path_id = str(graph_path_id) or (
            "dense" if not method else ("long" if is_long_text else "short")
        )
        candidates = [
            state
            for key, state in self._graphs.items()
            if key.method == method
            and key.batch_size == batch_size
            and key.capture_sampling == capture_sampling
            and key.graph_path_id == graph_path_id
            and key.shape_policy == shape_policy
            and (shape_policy == "batch_only" or key.is_long_text == is_long_text)
            and (
                shape_policy == "batch_only"
                or key.context_capacity == context_capacity
                or (allow_larger_context_capacity and key.context_capacity >= context_capacity)
            )
        ]
        if candidates:
            state = min(candidates, key=lambda state: state.capture_context_capacity)
            if (
                shape_policy == "batch_only"
                and context_capacity > state.capture_context_capacity
            ):
                raise RuntimeError(
                    "batch-only decode CUDA Graph request exceeded captured path "
                    f"capacity: requested={context_capacity}, "
                    f"captured={state.capture_context_capacity}."
                )
            self._touch_graph_state(state.key)
            return state

        if shape_policy == "batch_only" and getattr(self, "startup_plan_sealed", False):
            raise RuntimeError(
                "batch-only decode CUDA Graph has no startup-captured graph for "
                f"batch_size={batch_size}, path={graph_path_id!r}."
            )

        key = DecodeCudaGraphKey(
            method=method,
            batch_size=batch_size,
            context_capacity=0 if shape_policy == "batch_only" else context_capacity,
            is_long_text=bool(is_long_text),
            capture_sampling=capture_sampling,
            graph_path_id=graph_path_id,
            shape_policy=shape_policy,
        )
        state = DecodeCudaGraphState(
            key=key,
            capture_context_capacity=int(context_capacity),
        )
        device = getattr(
            self.cache_manager,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        contract = DecodeGraphContract(
            method=str(method),
            shape_policy=shape_policy,
            topology_path_id=graph_path_id,
            batch_capacity=int(batch_size),
            context_capacity=int(context_capacity),
            capture_sampling=bool(capture_sampling),
        )
        platform = getattr(self, "platform", None)
        pin_memory = bool(
            device.type != "cpu"
            and platform is not None
            and platform.supports_pin_memory()
        )
        inputs = DecodeGraphInputs.allocate(
            contract,
            device=device,
            pin_memory=pin_memory,
        )
        state.decode_state = DecodeGraphState(contract=contract, inputs=inputs)
        self._graphs[key] = state
        self._evict_cached_graphs(key)
        return state

    def _prepare_static_step(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
        is_long_text: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepare_decode_graph_step = getattr(
            self.runtime_state,
            "prepare_decode_graph_step",
            None,
        )
        if prepare_decode_graph_step is None:
            raise TypeError(
                "decode_graph requires runtime_state.prepare_decode_graph_step()."
            )
        graph_state = state.decode_state
        if graph_state is None:
            raise RuntimeError("Decode graph state was released before preparation.")
        graph_state.inputs.validate(graph_state.contract)

        self.cache_manager.set_decode_static_max_context_len(
            int(state.capture_context_capacity)
        )
        input_ids, positions, _ = prepare_decode_graph_step(
            seqs,
            graph_state,
        )

        set_context(
            False,
            cu_seqlens_q=None,
            cache_manager=self.cache_manager,
            is_long_text=bool(is_long_text),
            seqs=seqs,
            recurrent_state_manager=self.recurrent_state_manager,
        )
        self.cache_manager.set_decode_static_max_context_len(
            int(state.capture_context_capacity)
        )

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
            getattr(getattr(self.cache_manager, "config", None), "decode_graph_context_policy", "current")
            or "current"
        ).strip().lower()
        if policy in {"requested", "request", "final"}:
            return self._requested_context_capacity(seqs), bool(
                getattr(self, "reuse_larger_context_graphs", False)
            )
        if policy not in {"current", "cur", "now"}:
            raise ValueError(
                "decode_graph_context_policy must be 'current' or 'requested', "
                f"got {policy!r}."
            )
        return self._current_context_capacity(seqs), bool(
            getattr(self, "reuse_larger_context_graphs", False)
        )

    def bucket_plan(self) -> dict[str, object]:
        return {
            "shape_policy": getattr(self, "shape_policy", "bucketed"),
            "batch_sizes": list(self.capture_sizes),
            "context_sizes": list(self.context_sizes),
            "context_policy": str(
                getattr(getattr(self.cache_manager, "config", None), "decode_graph_context_policy", "current")
                or "current"
            ),
            "max_cached_graphs": self.max_cached_graphs,
            "cached_graph_keys": [
                {
                    "method": key.method,
                    "batch_size": key.batch_size,
                    "context_capacity": key.context_capacity,
                    "capture_context_capacity": state.capture_context_capacity,
                    "is_long_text": key.is_long_text,
                    "graph_path_id": key.graph_path_id,
                    "capture_sampling": key.capture_sampling,
                }
                for key, state in self._graphs.items()
                if state.graph is not None
            ],
        }

    def _graph_path_id(self, is_long_text: bool) -> str:
        resolver = getattr(self.cache_manager, "decode_graph_path_id", None)
        if callable(resolver):
            return str(resolver(bool(is_long_text)))
        return "dense" if not self.method else ("long" if is_long_text else "short")

    def _batch_only_context_capacity(
        self, seqs: list[Sequence], *, is_long_text: bool
    ) -> int:
        if self.max_context_len_override is not None:
            capacity = int(self.max_context_len_override)
        else:
            resolver = getattr(
                self.cache_manager,
                "decode_graph_context_independent_capacity",
                None,
            )
            if not callable(resolver):
                raise TypeError(
                    "batch-only decode CUDA Graph requires a context-independent "
                    "capacity resolver."
                )
            capacity = int(resolver(bool(is_long_text)))
        validator = getattr(
            self.cache_manager,
            "validate_decode_graph_context_independent_capacity",
            None,
        )
        if callable(validator):
            validator(seqs, capacity=capacity, is_long_text=bool(is_long_text))
        return capacity

    def _cache_manager_graph_context_capacity(self, seqs: list[Sequence]) -> tuple[int, bool] | None:
        resolver = getattr(self.cache_manager, "decode_graph_context_capacity", None)
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
        reset_contiguous = getattr(
            self.sparse_controller,
            "reset_decode_attn_scores_for_graph",
            None,
        )
        if callable(reset_contiguous) and reset_contiguous(refs):
            return
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
        graph_state = state.decode_state
        if graph_state is None:
            raise RuntimeError("Decode graph state was released before capture.")
        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller

        with profiler.record("decode_graph_warmup"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            participant = graph_state.runtime_state
            if participant is not None:
                participant.prepare_in_graph()
            logits = self.run_model(input_ids, positions, is_prefill=False)
            if state.key.capture_sampling:
                if logits is None:
                    raise RuntimeError("decode_graph capture_sampling requires rank-0 logits.")
                _ = logits.argmax(dim=-1)
        self.platform.synchronize()

        # In runtime-invariant mode, eager warmup consumes the prevalidated
        # storage scope. Establish a fresh scope for capture.
        self.cache_manager.validate_decode_cuda_graph_slot_mappings()

        with profiler.record("decode_graph_capture"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            # Dynamic score paths can replace Python state fields during the
            # captured forward. Keep both input and post-forward refs alive;
            # SnapKV-family fused 2D score buffers are retained by the controller.
            graph_input_sparse_state_refs = self._snapshot_sparse_state_refs()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self.graph_pool):
                    participant = graph_state.runtime_state
                    if participant is not None:
                        participant.prepare_in_graph()
                    self._reset_graph_input_attn_scores(graph_input_sparse_state_refs)
                    logits = self.run_model(input_ids, positions, is_prefill=False)
                    if state.key.capture_sampling:
                        if logits is None:
                            raise RuntimeError("decode_graph capture_sampling requires rank-0 logits.")
                        token_ids = logits.argmax(dim=-1)
                    else:
                        token_ids = None
            except Exception as exc:
                raise RuntimeError(f"decode_graph capture failed: {exc!r}") from exc

        state.graph = graph
        state.logits = logits
        state.token_ids = token_ids
        state.sparse_state_refs = self._snapshot_sparse_state_refs()

        keepalive: list[object] = [
            ctx,
            logits,
            ctx.decode_mid_o,
            ctx.decode_mid_o_logexpsum,
        ]
        keepalive.extend(graph_state.keepalive_tensors())
        if token_ids is not None:
            keepalive.append(token_ids)
        for sparse_refs_by_layer in (graph_input_sparse_state_refs, state.sparse_state_refs):
            for refs in sparse_refs_by_layer.values():
                for value in refs.values():
                    if isinstance(value, torch.Tensor):
                        keepalive.append(value)
        sparse_keepalive = getattr(self.sparse_controller, "decode_graph_keepalive_tensors", None)
        if sparse_keepalive is not None:
            keepalive.extend(sparse_keepalive())
        state.keepalive = keepalive
        captured_keys = getattr(self, "_captured_keys", None)
        if captured_keys is None:
            captured_keys = set()
            self._captured_keys = captured_keys
        if state.key in captured_keys:
            self.recapture_count = int(getattr(self, "recapture_count", 0)) + 1
        else:
            captured_keys.add(state.key)
        self.capture_count += 1
        return state

    def run(
        self,
        seqs: list[Sequence],
        *,
        capture_sampling: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not seqs:
            raise ValueError("decode_graph requires a non-empty decode batch.")
        if capture_sampling and any(seq.temperature > 1e-10 for seq in seqs):
            raise ValueError("decode_graph capture_sampling currently supports greedy decode only.")
        if capture_sampling and any(
            float(getattr(seq, "presence_penalty", 0.0)) != 0.0
            or float(getattr(seq, "repetition_penalty", 1.0)) != 1.0
            for seq in seqs
        ):
            raise ValueError(
                "decode_graph capture_sampling cannot apply sampling penalties."
            )

        real_batch_size = len(seqs)
        force_eager = getattr(self.cache_manager, "decode_graph_force_eager", None)
        if force_eager is not None and force_eager():
            self.force_eager_count += 1
            return self.run_eager_static(seqs), None

        graph_batch_size = self._select_graph_batch_size(real_batch_size)
        is_long_text = self.is_long_text_batch(seqs, False)
        graph_path_id = self._graph_path_id(is_long_text)
        if getattr(self, "shape_policy", "bucketed") == "batch_only":
            context_capacity = self._batch_only_context_capacity(
                seqs, is_long_text=is_long_text
            )
            allow_larger_context_capacity = False
        else:
            context_capacity, allow_larger_context_capacity = self._graph_context_capacity_policy(seqs)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            context_capacity=context_capacity,
            is_long_text=is_long_text,
            capture_sampling=bool(capture_sampling),
            graph_path_id=graph_path_id,
            allow_larger_context_capacity=allow_larger_context_capacity,
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs, is_long_text)

        if state.graph is None:
            state = self._capture(state, seqs, input_ids, positions)
            self._restore_sparse_state_refs(state)
            with profiler.record("decode_graph_replay_after_capture"):
                state.graph.replay()
            self.replay_count += 1
            logits = state.logits[:real_batch_size] if state.logits is not None else None
            token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
            return logits, token_ids

        self._restore_sparse_state_refs(state)
        with profiler.record("decode_graph_replay"):
            state.graph.replay()
        self.replay_count += 1
        logits = state.logits[:real_batch_size] if state.logits is not None else None
        token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
        return logits, token_ids

    def run_eager_static(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """Run decode eagerly through the same static-compatible path used by graphs."""
        if not seqs:
            raise ValueError("static decode requires a non-empty decode batch.")
        self.eager_static_count += 1

        real_batch_size = len(seqs)
        graph_batch_size = self._select_graph_batch_size(real_batch_size)
        is_long_text = self.is_long_text_batch(seqs, False)
        graph_path_id = self._graph_path_id(is_long_text)
        if getattr(self, "shape_policy", "bucketed") == "batch_only":
            context_capacity = self._batch_only_context_capacity(
                seqs, is_long_text=is_long_text
            )
            allow_larger_context_capacity = False
        else:
            context_capacity, allow_larger_context_capacity = self._static_context_capacity_policy(seqs)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            context_capacity=context_capacity,
            is_long_text=is_long_text,
            capture_sampling=False,
            graph_path_id=graph_path_id,
            allow_larger_context_capacity=allow_larger_context_capacity,
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs, is_long_text)
        graph_state = state.decode_state
        if graph_state is None:
            raise RuntimeError("Decode graph state was released before static execution.")

        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller
        with profiler.record("model_sparse_prepare"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
        participant = graph_state.runtime_state
        if participant is not None:
            participant.prepare_in_graph()
        logits = self.run_model(input_ids, positions, is_prefill=False)
        if logits is None:
            return None
        return logits[:real_batch_size]
