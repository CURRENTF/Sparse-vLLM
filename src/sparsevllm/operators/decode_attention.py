from __future__ import annotations

from dataclasses import dataclass

import torch

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import OpRegistry, OpResolver, SupportResult
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


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
    priority = 0

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
] = OpRegistry("decode attention launch")


@DECODE_ATTENTION_LAUNCH_REGISTRY.register
class H100LongGqaDecodeLaunchProvider(DecodeAttentionLaunchProvider):
    name = "h100_long_gqa_12q_2kv_hd128"
    priority = 100

    @classmethod
    def supports(
        cls,
        spec: DecodeAttentionLaunchSpec,
        caps: DeviceCaps,
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
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 query/KV tensors, got {spec.activation_dtype}"
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
        if spec.page_size != 1:
            return SupportResult.no(
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


@DECODE_ATTENTION_LAUNCH_REGISTRY.register
class DefaultGqaDecodeLaunchProvider(DecodeAttentionLaunchProvider):
    name = "default_gqa"
    priority = 10

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


def validate_context_independent_decode_graph_model(model: torch.nn.Module) -> None:
    """Require every decode-attention component to opt into batch-only graphs."""
    checked: list[str] = []
    rejected: list[str] = []
    for module_name, module in model.named_modules():
        backend = getattr(module, "attention_backend", None)
        if backend is not None:
            name = module_name or type(module).__name__
            checked.append(name)
            if not bool(
                getattr(backend, "cuda_graph_context_independent", False)
            ):
                rejected.append(
                    f"{name}: backend={getattr(backend, 'name', type(backend).__name__)}"
                )
        if type(module).__name__ == "Qwen35LinearAttention":
            name = module_name or type(module).__name__
            checked.append(name)
            if not bool(
                getattr(module, "cuda_graph_context_independent", False)
            ):
                rejected.append(f"{name}: linear GDN decode is not validated")

    model_body = getattr(model, "model", None)
    mla_attention = getattr(model_body, "mla_attention", None)
    if mla_attention is not None:
        provider = getattr(mla_attention, "provider", None)
        checked.append("model.mla_attention")
        if not bool(
            getattr(provider, "cuda_graph_context_independent", False)
        ):
            rejected.append(
                "model.mla_attention: provider="
                f"{getattr(provider, 'name', type(provider).__name__)}"
            )

    if not checked:
        rejected.append("model exposes no decode attention capability")
    if rejected:
        raise RuntimeError(
            "decode_cuda_graph_shape_policy='batch_only' requires validated "
            "context-independent experimental operators; unsupported="
            f"{rejected}. Use 'bucketed' until those providers are selected."
        )
