from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.context import get_context, set_context
from sparsevllm.utils.profiler import profiler


@dataclass(frozen=True)
class DecodeCudaGraphKey:
    method: str
    batch_size: int
    max_context_len: int
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


class DecodeCudaGraphRunner:
    """Fixed-shape CUDA Graph runner for decode forward.

    The runner owns graph-stable decode metadata tensors. Cache managers still
    allocate real KV slots every step, but write the per-step metadata into these
    stable buffers before replay.
    """

    def __init__(
        self,
        *,
        cache_manager,
        sparse_controller,
        run_model: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        is_long_text_batch: Callable[[list[Sequence], bool], bool],
        method: str,
        rank: int,
        capture_sizes: list[int],
    ):
        self.cache_manager = cache_manager
        self.sparse_controller = sparse_controller
        self.run_model = run_model
        self.is_long_text_batch = is_long_text_batch
        self.method = str(method or "")
        self.rank = int(rank)
        self.capture_sizes = sorted(set(int(size) for size in capture_sizes))
        if not self.capture_sizes or any(size <= 0 for size in self.capture_sizes):
            raise ValueError(f"decode_cuda_graph capture_sizes must be positive, got {capture_sizes}.")
        self.max_context_len_override: int | None = None
        self._graphs: dict[DecodeCudaGraphKey, DecodeCudaGraphState] = {}
        self.last_state_key: DecodeCudaGraphKey | None = None
        self.last_real_batch_size: int | None = None

    def set_max_context_len_override(self, max_context_len: int | None):
        self.max_context_len_override = None if max_context_len is None else int(max_context_len)

    def _requested_max_context_len(self, seqs: list[Sequence]) -> int:
        max_context_len = max(int(seq.num_prompt_tokens) + int(seq.max_tokens) for seq in seqs)
        if self.max_context_len_override is not None:
            max_context_len = max(max_context_len, int(self.max_context_len_override))
        return int(max_context_len)

    def _select_graph_batch_size(self, real_batch_size: int) -> int:
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
        max_context_len: int,
        is_long_text: bool,
        capture_sampling: bool,
    ) -> DecodeCudaGraphState:
        candidates = [
            state
            for key, state in self._graphs.items()
            if key.method == method
            and key.batch_size == batch_size
            and key.is_long_text == is_long_text
            and key.capture_sampling == capture_sampling
            and key.max_context_len >= max_context_len
        ]
        if candidates:
            return min(candidates, key=lambda state: state.key.max_context_len)

        key = DecodeCudaGraphKey(
            method=method,
            batch_size=batch_size,
            max_context_len=max_context_len,
            is_long_text=bool(is_long_text),
            capture_sampling=capture_sampling,
        )
        state = DecodeCudaGraphState(key=key)
        state.input_ids = torch.empty((batch_size,), dtype=torch.int64, device="cuda")
        state.positions = torch.empty((batch_size,), dtype=torch.int64, device="cuda")
        state.slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
        state.context_lens = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
        state.req_indices = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
        self._graphs[key] = state
        return state

    def _prepare_static_step(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
        is_long_text: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepare_decode_static = getattr(self.cache_manager, "prepare_decode_static", None)
        if prepare_decode_static is None:
            raise TypeError("decode_cuda_graph requires cache_manager.prepare_decode_static().")

        assert state.input_ids is not None
        assert state.positions is not None
        assert state.slot_mapping is not None
        assert state.context_lens is not None
        assert state.req_indices is not None

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
        )

        layer_batch_state = getattr(self.cache_manager, "layer_batch_state", None)
        if layer_batch_state is not None:
            layer_batch_state.slot_mapping = state.slot_mapping
            layer_batch_state.context_lens = state.context_lens
            layer_batch_state.max_context_len = int(state.key.max_context_len)
            layer_batch_state.req_indices = state.req_indices

        return input_ids, positions

    def _capture(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> DecodeCudaGraphState:
        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller
        ctx.decode_cuda_graph_static = True

        with profiler.record("decode_cuda_graph_warmup"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            logits = self.run_model(input_ids, positions, is_prefill=False)
            if state.key.capture_sampling:
                _ = logits.argmax(dim=-1)
        torch.cuda.synchronize()

        with profiler.record("decode_cuda_graph_capture"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    logits = self.run_model(input_ids, positions, is_prefill=False)
                    if state.key.capture_sampling:
                        token_ids = logits.argmax(dim=-1)
                    else:
                        token_ids = None
            except Exception as exc:
                raise RuntimeError(f"decode_cuda_graph capture failed: {exc!r}") from exc

        state.graph = graph
        state.logits = logits
        state.token_ids = token_ids

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
        for sparse_state in self.sparse_controller.layer_batch_sparse_states.values():
            for value in (
                sparse_state.attn_score,
                sparse_state.active_indices,
                sparse_state.active_slots,
                sparse_state.req_indices,
                sparse_state.context_lens,
                sparse_state.active_compressed_indices,
                sparse_state.global_req_indices,
            ):
                if isinstance(value, torch.Tensor):
                    keepalive.append(value)
        state.keepalive = keepalive
        return state

    def run(
        self,
        seqs: list[Sequence],
        *,
        capture_sampling: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.rank != 0:
            raise ValueError("decode_cuda_graph currently supports rank 0 / TP=1 only.")
        if not seqs:
            raise ValueError("decode_cuda_graph requires a non-empty decode batch.")
        if capture_sampling and any(seq.temperature > 1e-10 for seq in seqs):
            raise ValueError("decode_cuda_graph capture_sampling currently supports greedy decode only.")

        real_batch_size = len(seqs)
        graph_batch_size = self._select_graph_batch_size(real_batch_size)
        is_long_text = self.is_long_text_batch(seqs, False)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            max_context_len=self._requested_max_context_len(seqs),
            is_long_text=is_long_text,
            capture_sampling=bool(capture_sampling),
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs, is_long_text)

        if state.graph is None:
            state = self._capture(state, seqs, input_ids, positions)
            assert state.logits is not None
            logits = state.logits[:real_batch_size]
            token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
            return logits, token_ids

        assert state.logits is not None
        with profiler.record("decode_cuda_graph_replay"):
            state.graph.replay()
        logits = state.logits[:real_batch_size]
        token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
        return logits, token_ids
