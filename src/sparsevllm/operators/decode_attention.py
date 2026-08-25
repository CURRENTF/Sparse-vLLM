from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

import sparsevllm.platforms as platforms
from sparsevllm.kernels.external.flashinfer.decode import (
    flashinfer_paged_decode_support,
    make_flashinfer_paged_decode_wrapper,
)
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
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger


def get_decode_workspace(
    context,
    batch_size: int,
    num_heads: int,
    num_blocks: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_o = (batch_size, num_heads, num_blocks, head_dim)
    shape_lse = (batch_size, num_heads, num_blocks)
    mid_o = context.decode_mid_o
    if (
        mid_o is None
        or mid_o.device != device
        or mid_o.shape[0] < batch_size
        or mid_o.shape[1] < num_heads
        or mid_o.shape[2] < num_blocks
        or mid_o.shape[3] < head_dim
    ):
        mid_o = torch.empty(shape_o, dtype=torch.float32, device=device)
        context.decode_mid_o = mid_o

    mid_lse = context.decode_mid_o_logexpsum
    if (
        mid_lse is None
        or mid_lse.device != device
        or mid_lse.shape[0] < batch_size
        or mid_lse.shape[1] < num_heads
        or mid_lse.shape[2] < num_blocks
    ):
        mid_lse = torch.empty(shape_lse, dtype=torch.float32, device=device)
        context.decode_mid_o_logexpsum = mid_lse

    return (
        mid_o[:batch_size, :num_heads, :num_blocks, :head_dim],
        mid_lse[:batch_size, :num_heads, :num_blocks],
    )


@dataclass(frozen=True)
class DecodeAttentionOpSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    softmax_scale: float
    max_batch_size: int
    causal: bool = True
    page_size: int = 1
    may_require_attention_scores: bool = False
    layer_varying_page_table: bool = False
    cuda_graph: bool = True
    h2o_layerwise_probability_scores: bool = False
    context_independent_cuda_graph: bool = False
    context_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Decode attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Decode query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0 or self.max_batch_size <= 0:
            raise ValueError("Decode attention dimensions and capacity must be positive.")
        if self.softmax_scale <= 0:
            raise ValueError("Decode attention softmax_scale must be positive.")
        if (
            self.h2o_layerwise_probability_scores
            and not self.may_require_attention_scores
        ):
            raise ValueError(
                "H2O layer-wise probability scoring requires decode score output."
            )
        if self.context_capacity is not None and self.context_capacity <= 0:
            raise ValueError("Decode attention context_capacity must be positive.")

    @property
    def kernel_request(self) -> AttentionKernelRequest:
        return AttentionKernelRequest(
            activation_dtype=self.activation_dtype,
            head_dim=self.head_dim,
            page_size=self.page_size,
            score_output=(
                AttentionScoreKind.RAW_QK_PER_HEAD
                if self.may_require_attention_scores
                and not self.h2o_layerwise_probability_scores
                else AttentionScoreKind.NONE
            ),
            requires_softmax_lse=self.h2o_layerwise_probability_scores,
            layer_varying_page_table=self.layer_varying_page_table,
            varlen=True,
            cuda_graph=self.cuda_graph,
        )


class DecodeAttentionProvider:
    name = ""
    capabilities: AttentionKernelCapabilities

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index

    def close(self) -> None:
        pass

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class DecodeAttentionRunResult:
    output: torch.Tensor
    softmax_lse: torch.Tensor


@dataclass(frozen=True)
class GraphStableDecodeLaunchPlan:
    """Capture-time launch envelope for context-independent MHA/GQA decode."""

    plan_id: str
    context_capacity: int
    max_kv_splits: int
    target_tokens_per_split: int
    block_n: int
    stage1_num_warps: int
    stage1_num_stages: int
    stage2_num_warps: int
    stage2_num_stages: int

    def __post_init__(self) -> None:
        positive = (
            self.context_capacity,
            self.max_kv_splits,
            self.target_tokens_per_split,
            self.block_n,
            self.stage1_num_warps,
            self.stage1_num_stages,
            self.stage2_num_warps,
            self.stage2_num_stages,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"Decode launch plan values must be positive: {self}.")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "plan_id": self.plan_id,
            "context_capacity": self.context_capacity,
            "max_kv_splits": self.max_kv_splits,
            "target_tokens_per_split": self.target_tokens_per_split,
            "block_n": self.block_n,
            "stage1_num_warps": self.stage1_num_warps,
            "stage1_num_stages": self.stage1_num_stages,
            "stage2_num_warps": self.stage2_num_warps,
            "stage2_num_stages": self.stage2_num_stages,
        }


def build_graph_stable_decode_launch_plan(
    spec: DecodeAttentionOpSpec,
    caps: DeviceCaps,
) -> GraphStableDecodeLaunchPlan:
    """Resolve one context-invariant portable plan before provider preparation."""
    del caps
    if spec.context_capacity is None:
        raise ValueError(
            "Context-independent decode requires a static context_capacity."
        )
    if spec.head_dim == 256:
        block_n, stage1_warps, stage2_warps = 128, 4, 8
    elif spec.head_dim in {64, 128}:
        block_n, stage1_warps, stage2_warps = 64, 2, 4
    else:
        raise ValueError(
            f"No context-independent decode launch plan for head_dim={spec.head_dim}."
        )

    # The grid is derived from the configured capacity, never the current
    # request length. Capping the envelope bounds workspace and empty programs;
    # each replay derives its effective split count from device context_lens.
    max_kv_splits = min(
        64,
        max(16, math.ceil(int(spec.context_capacity) / 4096)),
    )
    return GraphStableDecodeLaunchPlan(
        plan_id="portable_context_independent_v1",
        context_capacity=int(spec.context_capacity),
        max_kv_splits=max_kv_splits,
        target_tokens_per_split=256,
        block_n=block_n,
        stage1_num_warps=stage1_warps,
        stage1_num_stages=2,
        stage2_num_warps=stage2_warps,
        stage2_num_stages=2,
    )


DECODE_ATTENTION_REGISTRY: OpRegistry[
    DecodeAttentionOpSpec, DecodeAttentionProvider
] = OpRegistry(
    "paged decode attention",
    portfolio=PortfolioPolicy(
        upstream_standard=(
            "sgl_fa3_paged_decode_sm90",
            "flashinfer_paged_decode",
        ),
        repo_portable=("triton_paged_decode",),
        repo_nonstandard=("triton_context_independent",),
    ),
)


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SglFa3PagedDecodeAttentionProvider(DecodeAttentionProvider):
    name = "sgl_fa3_paged_decode_sm90"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        compute_capabilities=frozenset({(9, 0)}),
        activation_dtypes=frozenset({torch.bfloat16}),
        head_dims=frozenset({128, 256}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        returns_softmax_lse=True,
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=True,
        minimum_runtime_version=(12, 3),
    )

    def __init__(self) -> None:
        self._kernel: SglFa3DecodeKernel | None = None

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.context_independent_cuda_graph:
            return SupportResult.unsupported("launch topology depends on context length")
        common = match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = sgl_fa3_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "SGL FA3 decode must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        self._kernel = SglFa3DecodeKernel(
            device=torch.device("cuda", int(device_index)),
            max_batch_size=spec.max_batch_size,
            softmax_scale=spec.softmax_scale,
        )

    def close(self) -> None:
        self._kernel = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "sglang-kernel",
            "kernel_path": "sgl_kernel.fa3.fwd",
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        return_softmax_lse = spec.kernel_request.requires_softmax_lse
        if view.meta.attn_score is not None and not return_softmax_lse:
            raise RuntimeError("SGL FA3 decode does not produce attention scores.")
        result = self._run_sgl(
            spec,
            q,
            view,
            return_softmax_lse=return_softmax_lse,
            **kwargs,
        )
        if not return_softmax_lse:
            return result
        if not isinstance(result, tuple):
            raise RuntimeError("SGL FA3 decode did not return the requested softmax LSE.")
        output, softmax_lse = result
        return DecodeAttentionRunResult(output=output, softmax_lse=softmax_lse)

    def _run_sgl(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        return_softmax_lse: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "SGL FA3 decode received unsupported runtime arguments: "
                f"{sorted(kwargs)}."
            )
        if self._kernel is None:
            raise RuntimeError("SGL FA3 decode provider was not prepared.")
        payload = view.payload
        meta = view.meta
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"SGL FA3 decode expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "SGL FA3 decode requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        if meta.active_slots.dtype != torch.int32 or meta.active_slots.ndim != 2:
            raise TypeError(
                "SGL FA3 decode requires a rank-2 int32 physical-slot page table."
            )
        if meta.req_indices.dtype != torch.int32:
            raise TypeError("SGL FA3 decode requires int32 request indices.")
        if meta.context_lens.dtype != torch.int32:
            raise TypeError("SGL FA3 decode requires int32 context lengths.")
        output = torch.empty_like(q)
        return self._kernel.run_explicit(
            q,
            payload.k_cache,
            payload.v_cache,
            meta.active_slots,
            meta.req_indices,
            meta.context_lens,
            output,
            validation_scope=get_context().attention_validation_scope,
            return_softmax_lse=return_softmax_lse,
        )


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferPagedDecodeAttentionProvider(DecodeAttentionProvider):
    name = "flashinfer_paged_decode"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        activation_dtypes=frozenset({torch.bfloat16, torch.float16}),
        page_sizes=frozenset({1}),
        score_outputs=frozenset({AttentionScoreKind.NONE}),
        returns_softmax_lse=True,
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=False,
    )

    def __init__(self) -> None:
        self._state: _FlashInferPagedDecodeState | None = None

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.context_independent_cuda_graph:
            return SupportResult.unsupported("planning depends on context length")
        common = match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )
        if not common.supported:
            return common
        if not spec.causal:
            return SupportResult.unsupported("requires causal attention")
        supported, reason = flashinfer_paged_decode_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec
        if self._state is not None:
            return
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashInfer decode must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        self._state = _FlashInferPagedDecodeState(
            torch.device("cuda", int(device_index))
        )

    def close(self) -> None:
        self._state = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "flashinfer-python",
            "kernel_path": "flashinfer.BatchDecodeWithPagedKVCacheWrapper",
            "kv_layout": "NHD",
            "page_size": 1,
            "cuda_graph": False,
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor | DecodeAttentionRunResult:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "FlashInfer decode received unsupported runtime arguments: "
                f"{sorted(kwargs)}."
            )
        if self._state is None:
            raise RuntimeError("FlashInfer decode provider was not prepared.")
        payload = view.payload
        meta = view.meta
        if q.dtype != spec.activation_dtype:
            raise TypeError(
                f"FlashInfer decode expected {spec.activation_dtype} Q, got {q.dtype}."
            )
        if payload.k_cache.dtype != q.dtype or payload.v_cache.dtype != q.dtype:
            raise TypeError(
                "FlashInfer decode requires Q/K/V with the same dtype, got "
                f"{q.dtype}/{payload.k_cache.dtype}/{payload.v_cache.dtype}."
            )
        max_context_len = getattr(meta, "max_context_len", None)
        if max_context_len is None:
            raise RuntimeError(
                "FlashInfer decode requires host-side max_context_len metadata."
            )
        context = get_context()
        plan_key = (
            context.attention_validation_scope,
            meta.active_slots.data_ptr(),
            meta.req_indices.data_ptr(),
            meta.context_lens.data_ptr(),
            int(max_context_len),
        )
        if getattr(self._state, "plan_key", None) != plan_key:
            self._state.plan(
                spec,
                active_slots=meta.active_slots,
                req_indices=meta.req_indices,
                context_lens=meta.context_lens,
                max_context_len=int(max_context_len),
            )
            self._state.plan_key = plan_key
        output = torch.empty_like(q)
        return_softmax_lse = spec.kernel_request.requires_softmax_lse
        result = self._state.wrapper.run(
            q,
            (
                payload.k_cache.unsqueeze(1),
                payload.v_cache.unsqueeze(1),
            ),
            out=output,
            return_lse=return_softmax_lse,
        )
        if not return_softmax_lse:
            if not isinstance(result, torch.Tensor) or (
                result.data_ptr() != output.data_ptr()
            ):
                raise RuntimeError(
                    "FlashInfer decode did not write to the supplied output."
                )
            return output
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "FlashInfer decode did not return the requested softmax LSE."
            )
        returned_output, softmax_lse_log2 = result
        if returned_output.data_ptr() != output.data_ptr():
            raise RuntimeError("FlashInfer decode did not write to the supplied output.")
        expected_shape = (int(q.shape[0]), spec.num_query_heads)
        if (
            softmax_lse_log2.dtype != torch.float32
            or tuple(softmax_lse_log2.shape) != expected_shape
        ):
            raise RuntimeError(
                "FlashInfer decode returned an unexpected softmax LSE: "
                f"shape={tuple(softmax_lse_log2.shape)} "
                f"dtype={softmax_lse_log2.dtype} "
                f"expected={expected_shape}/torch.float32."
            )
        softmax_lse = softmax_lse_log2.mul(math.log(2.0)).transpose(0, 1)
        return DecodeAttentionRunResult(output=output, softmax_lse=softmax_lse)


class _FlashInferPagedDecodeState:
    def __init__(self, device: torch.device) -> None:
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.wrapper = make_flashinfer_paged_decode_wrapper(self.workspace)
        self.plan_key: tuple[object, ...] | None = None

    def plan(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> None:
        if active_slots.dtype != torch.int32 or active_slots.ndim != 2:
            raise TypeError(
                "FlashInfer decode requires a rank-2 int32 physical-slot page table."
            )
        if req_indices.dtype != torch.int32 or context_lens.dtype != torch.int32:
            raise TypeError("FlashInfer decode requires int32 request metadata.")
        batch_size = int(context_lens.numel())
        if batch_size <= 0 or int(req_indices.numel()) != batch_size:
            raise ValueError("FlashInfer decode requires matched non-empty metadata.")
        max_context_len = int(max_context_len)
        if max_context_len <= 0 or max_context_len > int(active_slots.shape[1]):
            raise ValueError(
                "FlashInfer decode context is outside the active slot table: "
                f"max_context_len={max_context_len} "
                f"width={int(active_slots.shape[1])}."
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
        indices = rows.masked_select(valid).to(torch.int32).contiguous()
        indptr = torch.cat(
            (
                torch.zeros(1, device=context_lens.device, dtype=torch.int32),
                context_lens.cumsum(0, dtype=torch.int32),
            )
        )
        last_page_len = torch.ones(
            batch_size,
            device=context_lens.device,
            dtype=torch.int32,
        )
        self.wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            page_size=spec.page_size,
            sm_scale=spec.softmax_scale,
            q_data_type=spec.activation_dtype,
            kv_data_type=spec.activation_dtype,
            non_blocking=True,
        )


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonPagedDecodeAttentionProvider(DecodeAttentionProvider):
    name = "triton_paged_decode"
    capabilities = AttentionKernelCapabilities(
        platforms=frozenset({PlatformEnum.CUDA}),
        activation_dtypes=frozenset(
            {torch.bfloat16, torch.float16, torch.float32}
        ),
        page_sizes=frozenset({1}),
        score_outputs=frozenset(
            {
                AttentionScoreKind.NONE,
                AttentionScoreKind.RAW_QK_PER_HEAD,
                AttentionScoreKind.RAW_QK_REDUCED,
            }
        ),
        layer_varying_page_table=True,
        varlen=True,
        cuda_graph=True,
        requires_triton=True,
    )

    def __init__(self) -> None:
        self._backend = None

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.context_independent_cuda_graph:
            return SupportResult.unsupported("split count depends on context length")
        return match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index
        from sparsevllm.layers.attention_backend import TritonAttentionBackend

        self._backend = TritonAttentionBackend()

    def close(self) -> None:
        self._backend = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": "triton_flash_decode_stage1_stage2",
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("Triton decode provider was not prepared.")
        decode_launch_op = kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "Triton decode received unsupported runtime arguments: "
                f"{sorted(kwargs)}."
            )

        context = get_context()
        cache_manager = context.cache_manager
        layer_idx = int(context.now_layer_idx)
        meta = view.meta
        max_context_len = meta.max_context_len
        static_cap = getattr(cache_manager, "_decode_static_max_context_len", None)
        if static_cap is not None:
            max_context_len = max(
                int(max_context_len) if max_context_len is not None else 0,
                int(static_cap),
            )
        if max_context_len is None:
            raise RuntimeError(
                "static decode requires max_context_len, got None at "
                f"layer={layer_idx}"
            )
        max_len_in_batch = int(max_context_len)
        if meta.active_slots.dim() == 2:
            slot_table_len = int(meta.active_slots.shape[1])
            if max_len_in_batch > slot_table_len:
                max_len_in_batch = slot_table_len
            if max_len_in_batch <= 0:
                raise RuntimeError(
                    "decode requires a positive context length, got "
                    f"{max_len_in_batch} at layer={layer_idx}"
                )

        block_seq = cache_manager.get_decode_block_seq(layer_idx, 256)
        if decode_launch_op is None:
            gqa_block_n, gqa_num_warps = 16, 2
        else:
            block_seq, gqa_block_n, gqa_num_warps = (
                decode_launch_op.launch_config(
                    block_seq=block_seq,
                    max_context_len=max_len_in_batch,
                    requires_attention_scores=meta.attn_score is not None,
                )
            )
        num_seq_blocks = (max_len_in_batch + block_seq - 1) // block_seq
        mid_o, mid_o_logexpsum = get_decode_workspace(
            context,
            int(q.shape[0]),
            spec.num_query_heads,
            num_seq_blocks,
            spec.head_dim,
            q.device,
        )
        return self._backend.run_decode(
            q,
            view,
            mid_o=mid_o,
            mid_o_logexpsum=mid_o_logexpsum,
            max_len_in_batch=max_len_in_batch,
            block_seq=block_seq,
            num_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            gqa_block_n=gqa_block_n,
            gqa_num_warps=gqa_num_warps,
        )


@DECODE_ATTENTION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class ContextIndependentTritonDecodeAttentionProvider(DecodeAttentionProvider):
    """Fixed-grid Triton MHA/GQA decode provider for batch-only graphs."""

    name = "triton_context_independent"
    context_independent_cuda_graph = True
    capabilities = replace(
        TritonPagedDecodeAttentionProvider.capabilities,
        activation_dtypes=frozenset({torch.bfloat16, torch.float16}),
        head_dims=frozenset({64, 128, 256}),
        returns_softmax_lse=True,
    )

    def __init__(
        self,
        *,
        launch_plan: GraphStableDecodeLaunchPlan,
    ) -> None:
        self.launch_plan = launch_plan
        self._mid_o: torch.Tensor | None = None
        self._mid_lse: torch.Tensor | None = None
        self._softmax_lse: torch.Tensor | None = None

    @classmethod
    def supports(
        cls, spec: DecodeAttentionOpSpec, caps: DeviceCaps
    ) -> SupportResult:
        if not spec.context_independent_cuda_graph:
            return SupportResult.unsupported("reserved for batch-only CUDA Graph")
        if spec.context_capacity is None:
            return SupportResult.unsupported("requires a static context capacity")
        return match_attention_capabilities(
            spec.kernel_request,
            caps,
            cls.capabilities,
        )

    def prepare(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        if self.launch_plan.context_capacity != spec.context_capacity:
            raise RuntimeError(
                "Context-independent decode launch plan does not match the operator "
                f"capacity: plan={self.launch_plan.context_capacity} "
                f"spec={spec.context_capacity}."
            )
        if device_index is None:
            device_index = torch.cuda.current_device()
        device = torch.device("cuda", int(device_index))
        self._mid_o = torch.empty(
            (
                spec.max_batch_size,
                spec.num_query_heads,
                self.launch_plan.max_kv_splits,
                spec.head_dim,
            ),
            dtype=torch.float32,
            device=device,
        )
        self._mid_lse = torch.empty(
            (
                spec.max_batch_size,
                spec.num_query_heads,
                self.launch_plan.max_kv_splits,
            ),
            dtype=torch.float32,
            device=device,
        )
        self._softmax_lse = torch.empty(
            (spec.num_query_heads, spec.max_batch_size),
            dtype=torch.float32,
            device=device,
        )

    def close(self) -> None:
        self._mid_o = None
        self._mid_lse = None
        self._softmax_lse = None

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": "context_independent_flash_decode",
            "cuda_graph_shape_policy": "batch_only",
            "launch_plan": self.launch_plan.as_dict(),
            "workspace_owner": "provider",
        }

    def run(
        self,
        spec: DecodeAttentionOpSpec,
        q: torch.Tensor,
        view: Any,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("decode_launch_op", None)
        if kwargs:
            raise TypeError(
                "Context-independent decode received unsupported arguments: "
                f"{sorted(kwargs)}."
            )
        if (
            self._mid_o is None
            or self._mid_lse is None
            or self._softmax_lse is None
        ):
            raise RuntimeError("Context-independent decode provider was not prepared.")
        payload = view.payload
        if getattr(payload, "backend", None) != "dense":
            raise RuntimeError(
                "Context-independent decode requires dense explicit KV storage."
            )
        batch_size = int(q.shape[0])
        from sparsevllm.kernels.triton.context_independent_flash_decoding import (
            context_independent_flash_decode,
        )

        result = context_independent_flash_decode(
            q,
            payload.k_cache,
            payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            self._mid_o[:batch_size],
            self._mid_lse[:batch_size],
            attn_score=(
                None
                if spec.h2o_layerwise_probability_scores
                else view.meta.attn_score
            ),
            softmax_scale=spec.softmax_scale,
            target_tokens_per_split=self.launch_plan.target_tokens_per_split,
            block_n=self.launch_plan.block_n,
            num_warps=self.launch_plan.stage1_num_warps,
            num_stages=self.launch_plan.stage1_num_stages,
            stage2_num_warps=self.launch_plan.stage2_num_warps,
            stage2_num_stages=self.launch_plan.stage2_num_stages,
            return_softmax_lse=spec.h2o_layerwise_probability_scores,
            output_lse=self._softmax_lse[:, :batch_size],
        )
        if not spec.h2o_layerwise_probability_scores:
            return result
        if not isinstance(result, tuple):
            raise RuntimeError("Context-independent decode did not return softmax LSE.")
        return DecodeAttentionRunResult(output=result[0], softmax_lse=result[1])


class PreparedDecodeAttentionOp:
    """One prepared decode provider shared by all compatible MHA layers."""

    def __init__(
        self,
        spec: DecodeAttentionOpSpec,
        provider: DecodeAttentionProvider,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self._closed = False

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def context_independent_cuda_graph(self) -> bool:
        return bool(self.spec.context_independent_cuda_graph)

    def run(self, q: torch.Tensor, view: Any, **kwargs) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Decode attention operator is closed.")
        if view.meta.attn_score is not None and not self.spec.may_require_attention_scores:
            raise RuntimeError(
                "Decode attention view requested scores after a score-free provider "
                "was bound during model preparation."
            )
        result = self.provider.run(self.spec, q, view, **kwargs)
        if not self.spec.h2o_layerwise_probability_scores:
            if isinstance(result, DecodeAttentionRunResult):
                raise RuntimeError(
                    "Decode provider returned an unrequested softmax LSE."
                )
            return result
        if not isinstance(result, DecodeAttentionRunResult):
            raise RuntimeError(
                "H2O decode requested softmax LSE but the provider returned none."
            )
        score = view.meta.attn_score
        if score is None or score.ndim != 2:
            raise RuntimeError(
                "Layer-wise H2O decode requires a reduced [batch, width] score."
            )
        from sparsevllm.kernels.triton.h2o_decode_score import (
            h2o_probability_from_lse,
        )

        h2o_probability_from_lse(
            q,
            view.payload.k_cache,
            result.softmax_lse,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            score,
            softmax_scale=self.spec.softmax_scale,
        )
        return result.output

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        self._closed = True


def prepare_decode_attention_op(
    spec: DecodeAttentionOpSpec,
    *,
    device_index: int | None = None,
) -> PreparedDecodeAttentionOp:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    provider_kwargs = {}
    if spec.context_independent_cuda_graph:
        provider_kwargs["launch_plan"] = build_graph_stable_decode_launch_plan(
            spec,
            caps,
        )
    resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(
        spec,
        caps,
        **provider_kwargs,
    )
    logger.info(
        "Resolved MHA decode provider={} rejected={}",
        resolved.provider.name,
        dict(resolved.rejected),
    )
    resolved.provider.prepare(spec, device_index=device_index)
    return PreparedDecodeAttentionOp(spec, resolved.provider)


def validate_context_independent_decode_graph_model(model: torch.nn.Module) -> int:
    """Audit every semantic decode path after construction-time binding."""
    from sparsevllm.layers.attention import Attention

    validated = 0
    for module in model.modules():
        if isinstance(module, Attention):
            decode_op = getattr(module, "decode_op", None)
            implementation = (
                decode_op
                if decode_op is not None
                else getattr(module, "attention_backend", None)
            )
            if not bool(
                getattr(implementation, "context_independent_cuda_graph", False)
            ):
                raise RuntimeError(
                    "batch-only decode CUDA Graph requires a context-independent "
                    f"attention provider, got {type(implementation).__name__}."
                )
            validated += 1
        if getattr(module, "is_gated_delta_rule_layer", False):
            op = getattr(module, "gated_delta_rule_op", None)
            if not bool(getattr(op, "context_independent_cuda_graph", False)):
                raise RuntimeError(
                    "batch-only decode CUDA Graph requires a context-independent "
                    "GDN provider."
                )
            validated += 1

    model_body = getattr(model, "model", None)
    mla_attention = getattr(model_body, "mla_attention", None)
    if mla_attention is not None:
        provider = getattr(mla_attention, "provider", None)
        if not bool(
            getattr(provider, "context_independent_cuda_graph", False)
        ):
            raise RuntimeError(
                "batch-only decode CUDA Graph requires a context-independent "
                "MLA provider."
            )
        validated += 1
    if validated == 0:
        raise RuntimeError(
            "batch-only decode CUDA Graph found no validated decode operator."
        )
    return validated


@dataclass(frozen=True)
class DecodeAttentionLaunchSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    activation_dtype: torch.dtype
    page_size: int = 1

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Decode attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Decode query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0:
            raise ValueError("Decode attention dimensions must be positive.")


class DecodeAttentionLaunchProvider:
    name = ""

    def launch_config(
        self,
        *,
        block_seq: int,
        max_context_len: int,
        requires_attention_scores: bool,
    ) -> tuple[int, int, int]:
        raise NotImplementedError


DECODE_ATTENTION_LAUNCH_REGISTRY: OpRegistry[
    DecodeAttentionLaunchSpec, DecodeAttentionLaunchProvider
] = OpRegistry(
    "decode attention launch",
    portfolio=PortfolioPolicy(repo_portable=("default_gqa",)),
    profile_order=("h100_long_gqa_12q_2kv_hd128_profile",),
)


@DECODE_ATTENTION_LAUNCH_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class H100LongGqaDecodeLaunchProvider(DecodeAttentionLaunchProvider):
    name = "h100_long_gqa_12q_2kv_hd128"

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 query/KV tensors, got {spec.activation_dtype}"
            )
        if spec.page_size != 1:
            return SupportResult.unsupported(
                f"requires token-page KV storage (page_size=1), got {spec.page_size}"
            )
        return SupportResult.yes()

    def launch_config(
        self,
        *,
        block_seq: int,
        max_context_len: int,
        requires_attention_scores: bool,
    ) -> tuple[int, int, int]:
        if (
            int(block_seq) == 256
            and int(max_context_len) > 32768
            and not requires_attention_scores
        ):
            return 1024, 128, 4
        return int(block_seq), 16, 2


@DECODE_ATTENTION_LAUNCH_REGISTRY.register_profile
class H100LongGqaDecodeLaunchProfile:
    name = "h100_long_gqa_12q_2kv_hd128_profile"

    @classmethod
    def atomic_provider_names(
        cls,
        spec: DecodeAttentionLaunchSpec,
    ) -> tuple[str, ...]:
        del spec
        return ("h100_long_gqa_12q_2kv_hd128",)

    @classmethod
    def matches(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
    ) -> ProfileMatch:
        expected_shape = (12, 2, 128)
        actual_shape = (
            spec.num_query_heads,
            spec.num_kv_heads,
            spec.head_dim,
        )
        if caps.device_name != "NVIDIA H100 80GB HBM3":
            return ProfileMatch.no(
                "requires profiled NVIDIA H100 80GB HBM3 hardware, "
                f"got {caps.device_name}"
            )
        if actual_shape != expected_shape:
            return ProfileMatch.no(
                f"requires profiled local Q/KV/head shape {expected_shape}, "
                f"got {actual_shape}"
            )
        return ProfileMatch.yes("matched H100 long-GQA launch profile")

    @classmethod
    def bind(cls, spec: DecodeAttentionLaunchSpec, caps: DeviceCaps, **kwargs):
        del spec, caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        return H100LongGqaDecodeLaunchProvider()


@DECODE_ATTENTION_LAUNCH_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class DefaultGqaDecodeLaunchProvider(DecodeAttentionLaunchProvider):
    name = "default_gqa"

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def launch_config(
        self,
        *,
        block_seq: int,
        max_context_len: int,
        requires_attention_scores: bool,
    ) -> tuple[int, int, int]:
        del max_context_len, requires_attention_scores
        return int(block_seq), 16, 2


class PreparedDecodeAttentionLaunchOp:
    def __init__(
        self,
        spec: DecodeAttentionLaunchSpec,
        provider: DecodeAttentionLaunchProvider,
    ) -> None:
        self.spec = spec
        self.provider = provider

    @property
    def name(self) -> str:
        return self.provider.name

    def launch_config(self, **kwargs) -> tuple[int, int, int]:
        return self.provider.launch_config(**kwargs)


def prepare_decode_attention_launch_op(
    spec: DecodeAttentionLaunchSpec,
    *,
    device_index: int | None = None,
) -> PreparedDecodeAttentionLaunchOp:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    provider = OpResolver(DECODE_ATTENTION_LAUNCH_REGISTRY).resolve(spec, caps).provider
    return PreparedDecodeAttentionLaunchOp(spec, provider)
