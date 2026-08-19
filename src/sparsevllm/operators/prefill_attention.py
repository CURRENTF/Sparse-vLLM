from __future__ import annotations

import math
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

import torch

import sparsevllm.platforms as platforms
from sparsevllm.kernels.external.sgl.fa3 import (
    SglFa3DecodeKernel,
    sgl_fa3_device_support,
)
from sparsevllm.operators.attention_capabilities import (
    AttentionKernelCapabilities,
    AttentionKernelRequest,
    AttentionScoreKind,
    match_attention_capabilities,
)
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum
from sparsevllm.utils.log import logger


@dataclass(frozen=True)
class PrefillAttentionOpSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    softmax_scale: float
    causal: bool = True
    page_size: int = 1
    score_output: AttentionScoreKind = AttentionScoreKind.NONE
    layer_varying_page_table: bool = False
    varlen: bool = True
    cuda_graph: bool = False

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Prefill attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0:
            raise ValueError("Prefill attention dimensions must be positive.")
        if self.softmax_scale <= 0:
            raise ValueError("Prefill attention softmax_scale must be positive.")

    @property
    def kernel_request(self) -> AttentionKernelRequest:
        return AttentionKernelRequest(
            activation_dtype=self.activation_dtype,
            head_dim=self.head_dim,
            page_size=self.page_size,
            score_output=self.score_output,
            layer_varying_page_table=self.layer_varying_page_table,
            varlen=self.varlen,
            cuda_graph=self.cuda_graph,
        )


class PrefillAttentionProvider:
    name = ""
    priority = 0
    capabilities: AttentionKernelCapabilities

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


def _view_score_kind(view: Any) -> AttentionScoreKind:
    _, meta = _view_parts(view)
    score = meta.attn_score
    if score is None:
        return AttentionScoreKind.NONE
    if score.ndim == 3:
        return AttentionScoreKind.RAW_QK_PER_HEAD
    if score.ndim == 2:
        return AttentionScoreKind.RAW_QK_REDUCED
    raise ValueError(
        "Prefill attention score must have shape [batch, heads, tokens] or "
        f"[batch, tokens], got {tuple(score.shape)}."
    )


@PREFILL_ATTENTION_REGISTRY.register
class FlashInferPagedPrefillAttentionProvider(PrefillAttentionProvider):
    name = "flashinfer_paged_prefill_fa3_sm90"
    priority = 180
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        layer_varying_page_table=False,
        varlen=True,
        minimum_runtime_version=(12, 8),
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if caps.device_name != "NVIDIA H100 80GB HBM3":
            return SupportResult.no(
                "requires profiled NVIDIA H100 80GB HBM3 hardware, "
                f"got {caps.device_name}"
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
            raise RuntimeError("FlashInfer prefill provider was not prepared.")
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
class SglFa3PagedPrefillAttentionProvider(PrefillAttentionProvider):
    """SGL FA3 over Sparse-vLLM's page-size-one physical KV table."""

    name = "sgl_fa3_paged_prefill_sm90"
    priority = 200
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        layer_varying_page_table=True,
        varlen=True,
        minimum_runtime_version=(12, 3),
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.no("requires causal attention")
        supported, reason = sgl_fa3_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.no(reason)

    def __init__(self) -> None:
        self._kernel: SglFa3DecodeKernel | None = None
        self._query_plan_scope: object | None = None
        self._max_query_len = 0

    def prepare(
        self,
        spec: PrefillAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self._kernel is not None:
            return
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "SGL FA3 prefill must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        self._kernel = SglFa3DecodeKernel(
            device=torch.device("cuda", int(device_index)),
            # Explicit prefill supplies its own cu_seqlens_q. The decode-only
            # buffer capacity is therefore intentionally unused by this provider.
            max_batch_size=1,
            softmax_scale=spec.softmax_scale,
        )

    def close(self) -> None:
        self._kernel = None
        self._query_plan_scope = None
        self._max_query_len = 0

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
        del max_context_len, layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if self._kernel is None:
            raise RuntimeError("SGL FA3 prefill provider was not prepared.")
        kernel = self._kernel
        assert kernel is not None
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"SGL FA3 paged prefill expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "SGL FA3 paged prefill requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if qo_indptr.dtype != torch.int32 or chunk_lens.dtype != torch.int32:
            raise TypeError("SGL FA3 paged prefill requires int32 sequence metadata.")

        from sparsevllm.utils.context import get_context

        validation_scope = get_context().attention_validation_scope
        if self._query_plan_scope is not validation_scope:
            self._max_query_len = int(chunk_lens.max().item())
            self._query_plan_scope = validation_scope
        output = torch.empty_like(q)
        return kernel.run_explicit_varlen(
            q,
            payload.k_cache,
            payload.v_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            output,
            cu_seqlens_q=qo_indptr,
            max_seqlen_q=self._max_query_len,
            validation_scope=validation_scope,
        )


@PREFILL_ATTENTION_REGISTRY.register
class TilelangGqaPagedPrefillAttentionProvider(PrefillAttentionProvider):
    """TileLang GQA paged prefill with optional fused score extraction on SM90."""

    name = "tilelang_gqa_paged_prefill_sm90"
    priority = 150
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({
            AttentionScoreKind.NONE,
            AttentionScoreKind.RAW_QK_REDUCED,
        }),
        layer_varying_page_table=True,
        varlen=True,
        minimum_runtime_version=(12, 3),
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.no("requires causal attention")
        from sparsevllm.kernels.tilelang.gqa.runtime import (
            tilelang_gqa_device_support,
        )

        supported, reason = tilelang_gqa_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.no(reason)

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
        del max_context_len, layer_idx
        payload, meta = _view_parts(view)
        _validate_token_page_table(meta)
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"TileLang GQA paged prefill expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "TileLang GQA paged prefill requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if qo_indptr.dtype != torch.int32 or chunk_lens.dtype != torch.int32:
            raise TypeError("TileLang GQA paged prefill requires int32 sequence metadata.")

        from sparsevllm.kernels.tilelang.gqa.prefill import (
            gqa_paged_prefill_attention_tilelang,
        )

        prompt_cache_lens = meta.context_lens - chunk_lens
        output = torch.empty_like(q)
        return gqa_paged_prefill_attention_tilelang(
            q,
            payload.k_cache,
            payload.v_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            prompt_cache_lens,
            qo_indptr,
            output,
            attn_score=meta.attn_score,
            sm_scale=spec.softmax_scale,
        )


@PREFILL_ATTENTION_REGISTRY.register
class TritonPagedPrefillAttentionProvider(PrefillAttentionProvider):
    name = "triton_paged_prefill"
    priority = 10
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA, PlatformEnum.ROCM}),
        activation_dtypes=frozenset({torch.bfloat16, torch.float16}),
        head_dims=frozenset({16, 32, 64, 128, 256}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset(AttentionScoreKind),
        layer_varying_page_table=True,
        varlen=True,
        requires_triton=True,
    )

    @classmethod
    def supports(
        cls, spec: PrefillAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        common = match_attention_capabilities(
            spec.kernel_request, caps, cls.capabilities
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.no("Triton prefill requires causal attention")
        expected_scale = spec.head_dim**-0.5
        if not math.isclose(
            spec.softmax_scale,
            expected_scale,
            rel_tol=1.0e-6,
            abs_tol=0.0,
        ):
            return SupportResult.no(
                "Triton prefill requires the default head-dimension scale "
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
        from sparsevllm.kernels.triton.context_flashattention_nopad import (
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
    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(spec, caps)
    logger.info(
        "Resolved MHA prefill provider={} rejected={}",
        resolved.provider.name,
        dict(resolved.rejected),
    )
    return resolved.provider


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
        actual_score = _view_score_kind(view)
        if actual_score is not self.spec.score_output:
            raise RuntimeError(
                "Prefill attention view violates the resolved score contract: "
                f"resolved={self.spec.score_output.name} actual={actual_score.name}."
            )
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
