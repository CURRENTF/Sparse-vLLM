from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm.engine.cache_manager.base import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparsevllm.layers.attention_backend import TritonAttentionBackend
from sparsevllm.operators.mla_attention import (
    MlaAttentionOpSpec,
    MlaAttentionProvider,
    resolve_mla_attention_provider,
)
from sparsevllm.triton_kernel.mla import (
    gather_latent_history,
    validate_gather_metadata,
)
from sparsevllm.utils.context import get_context


@dataclass(frozen=True, slots=True)
class MlaPrefillHistory:
    """Gathered full history and its packed logical coordinates."""

    gathered_latent: torch.Tensor
    gathered_rope: torch.Tensor
    packed_offsets: torch.Tensor
    packed_slots: torch.Tensor
    local_req_indices: torch.Tensor
    context_lens: torch.Tensor
    context_lengths: tuple[int, ...]
    max_context_len: int
    required_workspace_bytes: int

    @property
    def visible_tokens(self) -> int:
        return int(self.gathered_latent.shape[0])


@dataclass(frozen=True, slots=True)
class MlaPrefillWorkset:
    """Full-history MLA buffers ready for ordinary 256-wide attention."""

    history: MlaPrefillHistory
    expanded_k: torch.Tensor
    expanded_v: torch.Tensor


@dataclass(frozen=True, slots=True)
class _MlaPrefillPlan:
    """Step-local packing metadata shared by every MLA layer."""

    validation_scope: object
    source_active_slots: torch.Tensor
    source_req_indices: torch.Tensor
    source_context_lens: torch.Tensor
    source_max_context_len: int | None
    cache_slot_count: int
    packed_offsets: torch.Tensor
    packed_slots: torch.Tensor
    local_req_indices: torch.Tensor
    context_lengths: tuple[int, ...]
    total_visible_tokens: int
    max_context_len: int
    required_workspace_bytes: int

    def matches(
        self,
        validation_scope: object,
        meta: AttentionViewMeta,
        cache_slot_count: int,
    ) -> bool:
        return (
            self.validation_scope is validation_scope
            and self.source_active_slots is meta.active_slots
            and self.source_req_indices is meta.req_indices
            and self.source_context_lens is meta.context_lens
            and self.source_max_context_len == meta.max_context_len
            and self.cache_slot_count == int(cache_slot_count)
        )


@dataclass(frozen=True, slots=True)
class _MlaPrefillQueryPlan:
    """Step-local query-packing validation shared by every MLA layer."""

    validation_scope: object
    source_context_lens: torch.Tensor
    query_tokens: int

    def matches(
        self,
        validation_scope: object,
        context_lens: torch.Tensor,
        query_tokens: int,
    ) -> bool:
        return (
            self.validation_scope is validation_scope
            and self.source_context_lens is context_lens
            and self.query_tokens == int(query_tokens)
        )


def _host_int_values(tensor: torch.Tensor) -> tuple[int, ...]:
    """Synchronize an integer tensor once inside a validation scope."""

    return tuple(int(value) for value in tensor.tolist())


def estimate_mla_prefill_workspace_bytes(
    *,
    total_visible_tokens: int,
    batch_size: int,
    max_context_len: int,
    local_q_heads: int,
    activation_dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> int:
    """Conservatively bound full-history gather and K/V expansion storage."""

    values = {
        "total_visible_tokens": total_visible_tokens,
        "batch_size": batch_size,
        "max_context_len": max_context_len,
        "local_q_heads": local_q_heads,
    }
    for name, value in values.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if batch_size == 0 or local_q_heads == 0:
        raise ValueError("batch_size and local_q_heads must be positive.")

    cache_element_size = torch.empty((), dtype=cache_dtype).element_size()
    activation_element_size = torch.empty(
        (),
        dtype=activation_dtype,
    ).element_size()
    gathered_values = int(total_visible_tokens) * (512 + 64)
    # kv_b projection materializes 192 K-noPE + 256 V values per head. The
    # final K concatenation additionally materializes 192+64 values per head;
    # expanded_v may remain a view of the projection output.
    expanded_values = (
        int(total_visible_tokens)
        * int(local_q_heads)
        * ((192 + 256) + 256)
    )
    metadata_values = (
        int(batch_size) * int(max_context_len)
        + 2 * int(batch_size)
        + int(max_context_len)
    )
    return int(
        gathered_values * cache_element_size
        + expanded_values * activation_element_size
        + metadata_values * torch.empty((), dtype=torch.int32).element_size()
    )


class MLAAttention:
    """Semantic MLA execution over tagged cache views.

    Model code owns projection weights, query absorption, and V reconstruction.
    This object owns provider binding, decode workspace, full-history gathering,
    and reuse of the existing 256-wide prefill attention backend.
    """

    def __init__(
        self,
        *,
        spec: MlaAttentionOpSpec,
        provider: MlaAttentionProvider,
        prefill_workspace_bytes: int,
    ) -> None:
        self.spec = spec
        self.provider = provider
        provider_spec = getattr(provider, "spec", None)
        if provider_spec is not None and provider_spec != spec:
            raise ValueError(
                "MLA semantic layer and provider specs must match: "
                f"layer={spec!r} provider={provider_spec!r}."
            )
        self.max_batch_size = int(getattr(provider, "max_batch_size", 0))
        if self.max_batch_size <= 0:
            raise ValueError("MLA provider must expose a positive max_batch_size.")
        self.prefill_workspace_bytes = int(prefill_workspace_bytes)
        if self.prefill_workspace_bytes <= 0:
            raise ValueError(
                "MLA prefill_workspace_bytes must be positive, got "
                f"{self.prefill_workspace_bytes}."
            )
        if self.spec.qk_head_dim != self.spec.value_head_dim:
            raise ValueError(
                "The existing prefill backend requires equal QK/value widths, "
                f"got {self.spec.qk_head_dim}/{self.spec.value_head_dim}."
            )
        self.prefill_backend = TritonAttentionBackend()
        self._prefill_plan: _MlaPrefillPlan | None = None
        self._prefill_query_plan: _MlaPrefillQueryPlan | None = None

    @classmethod
    def bind(
        cls,
        *,
        spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        prefill_workspace_bytes: int,
    ) -> "MLAAttention":
        provider = resolve_mla_attention_provider(
            spec,
            device=device,
            max_batch_size=max_batch_size,
        )
        return cls(
            spec=spec,
            provider=provider,
            prefill_workspace_bytes=prefill_workspace_bytes,
        )

    @property
    def device(self) -> torch.device:
        return torch.device(getattr(self.provider, "device"))

    def _require_mla_payload(
        self,
        view: PrefillComputeView | DecodeComputeView,
        *,
        operation: str,
    ) -> MlaLatentPayload:
        payload = view.payload
        if not isinstance(payload, MlaLatentPayload):
            raise TypeError(
                f"{operation} requires MlaLatentPayload, got "
                f"{type(payload).__name__}."
            )
        for name, tensor, width in (
            ("latent_cache", payload.latent_cache, self.spec.kv_lora_rank),
            ("rope_cache", payload.rope_cache, self.spec.rope_dim),
        ):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            if tensor.dtype != self.spec.cache_dtype:
                raise TypeError(
                    f"{name} must use {self.spec.cache_dtype}, got {tensor.dtype}."
                )
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (1, width):
                raise ValueError(
                    f"{name} must have shape [slots, 1, {width}], got "
                    f"{tuple(tensor.shape)}."
                )
        if payload.latent_cache.shape[0] != payload.rope_cache.shape[0]:
            raise ValueError("MLA latent and RoPE caches must have equal slots.")
        return payload

    def _get_prefill_plan(
        self,
        meta: AttentionViewMeta,
        *,
        cache_slot_count: int,
    ) -> _MlaPrefillPlan:
        cached = self._prefill_plan
        validation_scope = get_context().attention_validation_scope
        if cached is not None and cached.matches(
            validation_scope,
            meta,
            cache_slot_count,
        ):
            return cached

        metadata = {
            "active_slots": meta.active_slots,
            "req_indices": meta.req_indices,
            "context_lens": meta.context_lens,
        }
        for name, tensor in metadata.items():
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            if tensor.dtype != torch.int32:
                raise TypeError(
                    f"{name} must use {torch.int32}, got {tensor.dtype}."
                )
        if meta.context_lens.ndim != 1 or meta.context_lens.numel() == 0:
            raise ValueError(
                "MLA prefill context_lens must be a non-empty 1D tensor."
            )
        batch_size = int(meta.context_lens.numel())
        if meta.active_slots.ndim != 2:
            raise ValueError(
                "MLA prefill active_slots must have shape "
                "[rows, max_context_len]."
            )
        if meta.req_indices.shape != (batch_size,):
            raise ValueError(
                f"MLA prefill req_indices must have shape ({batch_size},), "
                f"got {tuple(meta.req_indices.shape)}."
            )
        if batch_size > self.max_batch_size:
            raise ValueError(
                "MLA prefill batch exceeds the bound operator capacity: "
                f"batch={batch_size} max_batch_size={self.max_batch_size}."
            )

        lengths = _host_int_values(meta.context_lens)
        if any(length < 0 for length in lengths):
            raise ValueError(
                f"MLA prefill context lengths must be non-negative: {lengths}."
            )
        total_visible_tokens = int(sum(lengths))
        if total_visible_tokens <= 0:
            raise ValueError("MLA prefill requires at least one visible token.")
        max_context_len = int(max(lengths))
        if (
            meta.max_context_len is not None
            and int(meta.max_context_len) < max_context_len
        ):
            raise ValueError(
                "MLA prefill max_context_len is smaller than an actual context: "
                f"declared={meta.max_context_len} actual={max_context_len}."
            )
        required_bytes = estimate_mla_prefill_workspace_bytes(
            total_visible_tokens=total_visible_tokens,
            batch_size=batch_size,
            max_context_len=max_context_len,
            local_q_heads=self.spec.local_q_heads,
            activation_dtype=self.spec.activation_dtype,
            cache_dtype=self.spec.cache_dtype,
        )
        if required_bytes > self.prefill_workspace_bytes:
            raise MemoryError(
                "MLA full-history prefill workspace exceeds its configured "
                f"budget: required={required_bytes} bytes budget="
                f"{self.prefill_workspace_bytes} bytes visible_tokens="
                f"{total_visible_tokens} local_heads={self.spec.local_q_heads}."
            )

        packed_starts: list[int] = []
        cursor = 0
        for length in lengths:
            packed_starts.append(cursor)
            cursor += length
        packed_offsets = torch.tensor(
            packed_starts,
            dtype=torch.int32,
            device=self.device,
        )
        local_req_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.arange(
            max_context_len,
            dtype=torch.int32,
            device=self.device,
        )
        packed_slots = packed_offsets[:, None] + positions[None, :]
        validate_gather_metadata(
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            packed_offsets,
            cache_slot_count=cache_slot_count,
            output_capacity=total_visible_tokens,
            max_context_len=max_context_len,
        )
        plan = _MlaPrefillPlan(
            validation_scope=validation_scope,
            source_active_slots=meta.active_slots,
            source_req_indices=meta.req_indices,
            source_context_lens=meta.context_lens,
            source_max_context_len=meta.max_context_len,
            cache_slot_count=int(cache_slot_count),
            packed_offsets=packed_offsets,
            packed_slots=packed_slots,
            local_req_indices=local_req_indices,
            context_lengths=lengths,
            total_visible_tokens=total_visible_tokens,
            max_context_len=max_context_len,
            required_workspace_bytes=required_bytes,
        )
        self._prefill_plan = plan
        return plan

    def prepare_prefill_history(
        self,
        view: PrefillComputeView,
    ) -> MlaPrefillHistory:
        if not isinstance(view, PrefillComputeView):
            raise TypeError(
                "MLA prefill requires PrefillComputeView, got "
                f"{type(view).__name__}."
            )
        payload = self._require_mla_payload(view, operation="MLA prefill")
        meta = view.meta
        plan = self._get_prefill_plan(
            meta,
            cache_slot_count=int(payload.latent_cache.shape[0]),
        )
        gathered_latent = torch.empty(
            plan.total_visible_tokens,
            self.spec.kv_lora_rank,
            dtype=self.spec.cache_dtype,
            device=self.device,
        )
        gathered_rope = torch.empty(
            plan.total_visible_tokens,
            self.spec.rope_dim,
            dtype=self.spec.cache_dtype,
            device=self.device,
        )
        gather_latent_history(
            payload.latent_cache,
            payload.rope_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            plan.packed_offsets,
            gathered_latent,
            gathered_rope,
            max_context_len=plan.max_context_len,
            validate_metadata=False,
        )
        return MlaPrefillHistory(
            gathered_latent=gathered_latent,
            gathered_rope=gathered_rope,
            packed_offsets=plan.packed_offsets,
            packed_slots=plan.packed_slots,
            local_req_indices=plan.local_req_indices,
            context_lens=meta.context_lens,
            context_lengths=plan.context_lengths,
            max_context_len=plan.max_context_len,
            required_workspace_bytes=plan.required_workspace_bytes,
        )

    def bind_prefill_kv(
        self,
        history: MlaPrefillHistory,
        *,
        expanded_k: torch.Tensor,
        expanded_v: torch.Tensor,
    ) -> MlaPrefillWorkset:
        if not isinstance(history, MlaPrefillHistory):
            raise TypeError(
                "bind_prefill_kv requires MlaPrefillHistory, got "
                f"{type(history).__name__}."
            )
        if (
            history.gathered_latent.device != self.device
            or history.gathered_rope.device != self.device
        ):
            raise ValueError("MLA prefill history is on the wrong device.")
        if (
            history.gathered_latent.dtype != self.spec.cache_dtype
            or history.gathered_rope.dtype != self.spec.cache_dtype
        ):
            raise TypeError("MLA prefill history uses the wrong cache dtype.")
        expected_k_shape = (
            history.visible_tokens,
            self.spec.local_q_heads,
            self.spec.qk_head_dim,
        )
        expected_v_shape = (
            history.visible_tokens,
            self.spec.local_q_heads,
            self.spec.value_head_dim,
        )
        for name, tensor, expected_shape in (
            ("expanded_k", expanded_k, expected_k_shape),
            ("expanded_v", expanded_v, expected_v_shape),
        ):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got "
                    f"{tuple(tensor.shape)}."
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            if tensor.dtype != self.spec.activation_dtype:
                raise TypeError(
                    f"{name} must use {self.spec.activation_dtype}, got "
                    f"{tensor.dtype}."
                )
            if tensor.stride(-1) != 1:
                raise ValueError(f"{name} must be contiguous in its last dimension.")
        return MlaPrefillWorkset(
            history=history,
            expanded_k=expanded_k,
            expanded_v=expanded_v,
        )

    def run_prefill(
        self,
        q: torch.Tensor,
        workset: MlaPrefillWorkset,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
    ) -> torch.Tensor:
        history = workset.history
        if q.ndim != 3:
            raise ValueError(
                "MLA prefill q must have shape [tokens, local_heads, 256], "
                f"got {tuple(q.shape)}."
            )
        expected_q_shape = (
            int(q.shape[0]),
            self.spec.local_q_heads,
            self.spec.qk_head_dim,
        )
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(
                f"MLA prefill q must have shape {expected_q_shape}, got "
                f"{tuple(q.shape)}."
            )
        if q.device != self.device or q.dtype != self.spec.activation_dtype:
            raise TypeError(
                "MLA prefill q must match the operator device/dtype: "
                f"q={q.device}/{q.dtype} expected="
                f"{self.device}/{self.spec.activation_dtype}."
            )
        batch_size = int(history.context_lens.numel())
        for name, tensor in (
            ("b_start_loc", b_start_loc),
            ("chunk_lens", chunk_lens),
        ):
            if tensor.shape != (batch_size,):
                raise ValueError(
                    f"{name} must have shape ({batch_size},), got "
                    f"{tuple(tensor.shape)}."
                )
            if tensor.device != self.device or tensor.dtype != torch.int32:
                raise TypeError(
                    f"{name} must be int32 on {self.device}, got "
                    f"{tensor.device}/{tensor.dtype}."
                )
        validation_scope = get_context().attention_validation_scope
        query_tokens = int(q.shape[0])
        cached_query_plan = self._prefill_query_plan
        if cached_query_plan is None or not cached_query_plan.matches(
            validation_scope,
            history.context_lens,
            query_tokens,
        ):
            chunks = _host_int_values(chunk_lens)
            starts = _host_int_values(b_start_loc)
            expected_starts: list[int] = []
            cursor = 0
            for chunk in chunks:
                expected_starts.append(cursor)
                cursor += chunk
            if starts != tuple(expected_starts) or cursor != query_tokens:
                raise ValueError(
                    "MLA prefill query packing is inconsistent: "
                    f"starts={starts} expected_starts={expected_starts} "
                    f"chunk_tokens={cursor} q_tokens={query_tokens}."
                )
            contexts = history.context_lengths
            if any(
                chunk <= 0 or chunk > context
                for chunk, context in zip(chunks, contexts)
            ):
                raise ValueError(
                    "MLA prefill chunk lengths must be positive and no larger "
                    f"than their contexts: chunks={chunks} contexts={contexts}."
                )
            self._prefill_query_plan = _MlaPrefillQueryPlan(
                validation_scope=validation_scope,
                source_context_lens=history.context_lens,
                query_tokens=query_tokens,
            )

        explicit_view = PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=history.packed_slots,
                req_indices=history.local_req_indices,
                context_lens=history.context_lens,
                max_context_len=history.max_context_len,
            ),
            payload=ExplicitKVPayload(
                k_cache=workset.expanded_k,
                v_cache=workset.expanded_v,
            ),
        )
        return self.prefill_backend.run_prefill(
            q,
            explicit_view,
            b_start_loc=b_start_loc,
            chunk_lens=chunk_lens,
            max_input_len=history.max_context_len,
        )

    def run_decode(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
    ) -> torch.Tensor:
        if not isinstance(view, DecodeComputeView):
            raise TypeError(
                "MLA decode requires DecodeComputeView, got "
                f"{type(view).__name__}."
            )
        self._require_mla_payload(view, operation="MLA decode")
        output = torch.empty_like(q_nope_absorbed)
        return self.provider.run(
            q_nope_absorbed,
            q_rope,
            view,
            output,
            validation_scope=get_context().attention_validation_scope,
        )


__all__ = [
    "MLAAttention",
    "MlaPrefillHistory",
    "MlaPrefillWorkset",
    "estimate_mla_prefill_workspace_bytes",
]
