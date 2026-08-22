from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Decode attention head counts must be positive.")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Decode query heads must be divisible by KV heads.")
        if self.head_dim <= 0 or self.page_size <= 0 or self.max_batch_size <= 0:
            raise ValueError("Decode attention dimensions and capacity must be positive.")
        if self.softmax_scale <= 0:
            raise ValueError("Decode attention softmax_scale must be positive.")

    @property
    def kernel_request(self) -> AttentionKernelRequest:
        return AttentionKernelRequest(
            activation_dtype=self.activation_dtype,
            head_dim=self.head_dim,
            page_size=self.page_size,
            score_output=(
                AttentionScoreKind.RAW_QK_PER_HEAD
                if self.may_require_attention_scores
                else AttentionScoreKind.NONE
            ),
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


DECODE_ATTENTION_REGISTRY: OpRegistry[
    DecodeAttentionOpSpec, DecodeAttentionProvider
] = OpRegistry(
    "paged decode attention",
    portfolio=PortfolioPolicy(
        upstream_standard=("sgl_fa3_paged_decode_sm90",),
        repo_portable=("triton_paged_decode",),
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
        if meta.attn_score is not None:
            raise RuntimeError("SGL FA3 decode does not produce attention scores.")
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
        score_outputs=frozenset(AttentionScoreKind),
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

    def run(self, q: torch.Tensor, view: Any, **kwargs) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Decode attention operator is closed.")
        if view.meta.attn_score is not None and not self.spec.may_require_attention_scores:
            raise RuntimeError(
                "Decode attention view requested scores after a score-free provider "
                "was bound during model preparation."
            )
        return self.provider.run(self.spec, q, view, **kwargs)

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
    resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(spec, caps)
    logger.info(
        "Resolved MHA decode provider={} rejected={}",
        resolved.provider.name,
        dict(resolved.rejected),
    )
    resolved.provider.prepare(spec, device_index=device_index)
    return PreparedDecodeAttentionOp(spec, resolved.provider)


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
