from __future__ import annotations

import math
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

import torch

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    runtime_version_at_least,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class PrefillAttentionOpSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    softmax_scale: float
    causal: bool = True
    page_size: int = 1
    requires_attention_scores: bool = False
    layer_invariant_page_table: bool = False

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Prefill attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0:
            raise ValueError("Prefill attention dimensions must be positive.")
        if self.softmax_scale <= 0:
            raise ValueError("Prefill attention softmax_scale must be positive.")


class PrefillAttentionProvider:
    name = ""
    priority = 0

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index

    def close(self) -> None:
        pass

    def run(
        self,
        spec: PrefillAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        *,
        qo_indptr: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_context_len: int,
        layer_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError


def _validate_token_page_table(view: Any) -> None:
    if view.active_slots.dtype != torch.int32:
        raise TypeError(
            "Paged prefill requires an int32 physical-slot page table, got "
            f"{view.active_slots.dtype}."
        )
    if view.active_slots.ndim != 2:
        raise ValueError(
            "Paged prefill expects a 2D physical-slot page table, got "
            f"shape={tuple(view.active_slots.shape)}."
        )


PREFILL_ATTENTION_REGISTRY: OpRegistry[
    PrefillAttentionOpSpec, PrefillAttentionProvider
] = OpRegistry("paged prefill attention")


def _view_parts(view: Any) -> tuple[Any, Any]:
    """Return the physical payload and logical metadata for a prefill view."""

    return getattr(view, "payload", view), getattr(view, "meta", view)


@PREFILL_ATTENTION_REGISTRY.register
class FlashInferPagedPrefillAttentionProvider(PrefillAttentionProvider):
    name = "flashinfer_paged_prefill_fa3_sm90"
    priority = 100

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if caps.device_name != "NVIDIA H100 80GB HBM3":
            return SupportResult.no(
                "requires profiled NVIDIA H100 80GB HBM3 hardware, "
                f"got {caps.device_name}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.no(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 Q/K/V, got {spec.activation_dtype}"
            )
        expected_shape = (12, 2, 128)
        actual_shape = (
            spec.num_query_heads,
            spec.num_kv_heads,
            spec.head_dim,
        )
        if actual_shape != expected_shape:
            return SupportResult.no(
                f"requires profiled local Q/KV/head shape {expected_shape}, got {actual_shape}"
            )
        if not spec.causal:
            return SupportResult.no("requires causal attention")
        if spec.page_size != 1:
            return SupportResult.no(
                f"requires token-page KV storage (page_size=1), got {spec.page_size}"
            )
        if spec.requires_attention_scores:
            return SupportResult.no("does not produce per-token attention scores")
        if not spec.layer_invariant_page_table:
            return SupportResult.no("requires one page table shared across model layers")
        if find_spec("flashinfer") is None:
            return SupportResult.no("flashinfer is not installed")
        try:
            installed = version("flashinfer-python")
        except PackageNotFoundError:
            return SupportResult.no("flashinfer-python package metadata is unavailable")
        numeric = tuple(int(part) for part in re.findall(r"\d+", installed)[:3])
        if numeric < (0, 6, 15):
            return SupportResult.no(
                f"requires flashinfer-python >= 0.6.15, got {installed}"
            )
        return SupportResult.yes()

    def __init__(self) -> None:
        self._state: _FlashInferPagedPrefillState | None = None

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self._state is not None:
            return
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashInfer prefill must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        device = torch.device("cuda", int(device_index))
        self._state = _FlashInferPagedPrefillState(device)

    def close(self) -> None:
        self._state = None

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del chunk_lens
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if self._state is None:
            self.prepare(spec, device_index=q.device.index)
        state = self._state
        assert state is not None
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"FlashInfer paged prefill expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "FlashInfer paged prefill requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if meta.attn_score is not None:
            raise RuntimeError(
                "FlashInfer paged prefill was selected for a view that requires "
                "per-token attention scores."
            )
        if layer_idx == 0:
            state.plan(
                spec,
                qo_indptr=qo_indptr,
                active_slots=meta.active_slots,
                req_indices=meta.req_indices,
                context_lens=meta.context_lens,
                max_context_len=max_context_len,
            )
        elif not state.planned:
            raise RuntimeError(
                "FlashInfer paged prefill reached a nonzero layer before layer-0 planning."
            )
        output = torch.empty_like(q)
        state.wrapper.run(
            q,
            (
                payload.k_cache.unsqueeze(1),
                payload.v_cache.unsqueeze(1),
            ),
            out=output,
        )
        return output


class _FlashInferPagedPrefillState:
    def __init__(self, device: torch.device) -> None:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="fa3",
        )
        self.planned = False

    def plan(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        qo_indptr: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> None:
        if active_slots.dim() != 2:
            raise ValueError(
                "FlashInfer paged prefill expects a 2D active slot table, got "
                f"{tuple(active_slots.shape)}."
            )
        batch_size = int(context_lens.numel())
        if batch_size <= 0 or int(req_indices.numel()) != batch_size:
            raise ValueError("FlashInfer paged prefill requires matched non-empty metadata.")
        max_context_len = int(max_context_len)
        if max_context_len <= 0 or max_context_len > int(active_slots.shape[1]):
            raise ValueError(
                "FlashInfer paged prefill max context is outside the active slot table: "
                f"max_context_len={max_context_len} width={int(active_slots.shape[1])}."
            )
        rows = active_slots.index_select(0, req_indices.to(torch.long))[
            :, :max_context_len
        ]
        positions = torch.arange(
            max_context_len,
            device=context_lens.device,
            dtype=context_lens.dtype,
        )
        valid = positions.unsqueeze(0) < context_lens.unsqueeze(1)
        paged_kv_indices = rows.masked_select(valid).to(torch.int32).contiguous()
        zero = torch.zeros(1, device=context_lens.device, dtype=torch.int32)
        paged_kv_indptr = torch.cat(
            (
                zero,
                context_lens.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        last_page_len = torch.ones(
            batch_size,
            device=context_lens.device,
            dtype=torch.int32,
        )
        self.wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            last_page_len,
            num_qo_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim_qk=spec.head_dim,
            page_size=spec.page_size,
            causal=spec.causal,
            sm_scale=spec.softmax_scale,
            q_data_type=spec.activation_dtype,
            kv_data_type=spec.activation_dtype,
            non_blocking=True,
        )
        self.planned = True


@PREFILL_ATTENTION_REGISTRY.register
class TritonPagedPrefillAttentionProvider(PrefillAttentionProvider):
    name = "triton_paged_prefill"
    priority = 10

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        if caps.platform not in {PlatformEnum.CUDA, PlatformEnum.ROCM}:
            return SupportResult.no(f"requires a GPU platform, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.no(
                f"requires BF16 or FP16 Q/K/V, got {spec.activation_dtype}"
            )
        if spec.head_dim not in {16, 32, 64, 128, 256}:
            return SupportResult.no(f"unsupported head_dim={spec.head_dim}")
        if not spec.causal:
            return SupportResult.no("legacy Triton prefill requires causal attention")
        if spec.page_size != 1:
            return SupportResult.no(
                f"legacy Triton prefill requires page_size=1, got {spec.page_size}"
            )
        expected_scale = spec.head_dim**-0.5
        if not math.isclose(
            spec.softmax_scale,
            expected_scale,
            rel_tol=1.0e-6,
            abs_tol=0.0,
        ):
            return SupportResult.no(
                "legacy Triton prefill requires the default head-dimension scale "
                f"{expected_scale}, got {spec.softmax_scale}"
            )
        return SupportResult.yes()

    def run(
        self,
        spec,
        q,
        view,
        *,
        qo_indptr,
        chunk_lens,
        max_context_len,
        layer_idx,
    ):
        del spec, layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        from sparsevllm.triton_kernel.context_flashattention_nopad import (
            context_attention_fwd,
        )

        output = torch.empty_like(q)
        context_attention_fwd(
            q,
            payload.k_cache,
            payload.v_cache,
            output,
            meta.req_indices,
            qo_indptr[:-1],
            meta.context_lens,
            meta.context_lens - chunk_lens,
            max_context_len,
            meta.active_slots,
            attn_score=meta.attn_score,
        )
        return output


def resolve_prefill_attention_provider(
    spec: PrefillAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> PrefillAttentionProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(spec, caps).provider


class PreparedPrefillAttentionOp:
    """One prepared provider shared by all layers in one model runtime."""

    def __init__(
        self,
        spec: PrefillAttentionOpSpec,
        provider: PrefillAttentionProvider,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self._closed = False

    @property
    def name(self) -> str:
        return self.provider.name

    def run(self, q, view, **kwargs):
        if self._closed:
            raise RuntimeError("Prefill attention operator is closed.")
        return self.provider.run(self.spec, q, view, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        self._closed = True


def prepare_prefill_attention_op(
    spec: PrefillAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> PreparedPrefillAttentionOp:
    provider = resolve_prefill_attention_provider(spec, device_index=device_index)
    provider.prepare(spec, device_index=device_index)
    return PreparedPrefillAttentionOp(spec, provider)
