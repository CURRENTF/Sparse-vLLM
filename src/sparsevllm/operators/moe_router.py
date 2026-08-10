from __future__ import annotations

from dataclasses import dataclass

import torch

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class MoeRouterOpSpec:
    num_experts: int
    top_k: int
    activation_dtype: torch.dtype
    norm_topk_prob: bool
    cuda_graph: bool

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("MoE router num_experts must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"MoE router top_k must be in [1, {self.num_experts}], "
                f"got {self.top_k}."
            )
        if not self.activation_dtype.is_floating_point:
            raise TypeError(
                "MoE router activations must be floating point, "
                f"got {self.activation_dtype}."
            )


class MoeRouterProvider:
    name = ""
    priority = 0

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


MOE_ROUTER_REGISTRY: OpRegistry[MoeRouterOpSpec, MoeRouterProvider] = OpRegistry(
    "MoE router"
)


@MOE_ROUTER_REGISTRY.register
class TritonMoeRouterProvider(MoeRouterProvider):
    name = "triton"
    priority = 10

    @classmethod
    def supports(
        cls,
        spec: MoeRouterOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.no(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.no(
                f"requires BF16 or FP16 logits, got {spec.activation_dtype}"
            )
        if spec.num_experts not in {128, 256} or spec.top_k != 8:
            return SupportResult.no(
                "requires num_experts in {128, 256} and top_k=8"
            )
        return SupportResult.yes()

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sparsevllm.triton_kernel.moe_topk import topk_softmax

        return topk_softmax(
            router_logits,
            top_k=spec.top_k,
            norm_topk_prob=spec.norm_topk_prob,
        )


def resolve_moe_router_provider(
    spec: MoeRouterOpSpec,
    *,
    device_index: int | None = None,
) -> MoeRouterProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(MOE_ROUTER_REGISTRY).resolve(spec, caps).provider
