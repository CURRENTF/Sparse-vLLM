from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import torch

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import OpRegistry, OpResolver, SupportResult
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class MoeOpSpec:
    num_experts: int
    num_local_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    activation_dtype: torch.dtype
    weight_dtype: torch.dtype
    block_shape: tuple[int, int] | None
    ep_size: int
    cuda_graph: bool

    def __post_init__(self) -> None:
        if self.num_experts <= 0 or self.num_local_experts <= 0:
            raise ValueError("MoE expert counts must be positive.")
        if self.ep_size <= 0:
            raise ValueError("MoE ep_size must be positive.")
        if self.num_local_experts * self.ep_size != self.num_experts:
            raise ValueError(
                "MoE local expert topology is inconsistent: "
                f"{self.num_local_experts} * {self.ep_size} != {self.num_experts}."
            )
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("MoE hidden/intermediate sizes must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"MoE top_k must be in [1, {self.num_experts}], got {self.top_k}."
            )
        if self.block_shape is not None and (
            len(self.block_shape) != 2 or any(value <= 0 for value in self.block_shape)
        ):
            raise ValueError(
                f"MoE block_shape must contain two positive values, got {self.block_shape}."
            )


def model_activation_dtype(config) -> torch.dtype:
    value = getattr(config, "dtype", None)
    if value is None:
        value = getattr(config, "torch_dtype", None)
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value or "").lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.bfloat16


class MoeProvider:
    name = ""
    priority = 0
    gate_up_order = "gate_up"

    def packed_projection_offset(
        self,
        projection: str,
        intermediate_size: int,
    ) -> int:
        if projection not in {"gate", "up"}:
            raise ValueError(f"Unknown packed MoE projection {projection!r}.")
        first = "gate" if self.gate_up_order == "gate_up" else "up"
        return 0 if projection == first else int(intermediate_size)

    def run(
        self,
        spec: MoeOpSpec,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale_inv: torch.Tensor | None,
        w2_scale_inv: torch.Tensor | None,
        *,
        local_expert_start: int,
        ep_rank: int,
    ) -> torch.Tensor:
        raise NotImplementedError


MOE_REGISTRY: OpRegistry[MoeOpSpec, MoeProvider] = OpRegistry("routed MoE")


@MOE_REGISTRY.register
class FlashInferCutlassFp8MoeProvider(MoeProvider):
    name = "flashinfer_cutlass_fp8_sm90"
    priority = 100
    gate_up_order = "up_gate"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.weight_dtype != torch.float8_e4m3fn:
            return SupportResult.no(f"requires FP8 E4M3 weights, got {spec.weight_dtype}")
        if spec.block_shape != (128, 128):
            return SupportResult.no(f"requires block_shape=(128, 128), got {spec.block_shape}")
        if spec.hidden_size % 128 or spec.intermediate_size % 128:
            return SupportResult.no("FP8 hidden/intermediate sizes must be 128-aligned")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if find_spec("flashinfer") is None:
            return SupportResult.no("flashinfer is not installed")
        return SupportResult.yes()

    def run(
        self,
        spec,
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_scale_inv,
        w2_scale_inv,
        *,
        local_expert_start,
        ep_rank,
    ):
        del local_expert_start
        if w13_scale_inv is None or w2_scale_inv is None:
            raise RuntimeError("FlashInfer FP8 MoE requires expert scales.")
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.tllm_enums import ActivationType

        output = torch.empty_like(hidden_states)
        cutlass_fused_moe(
            hidden_states,
            topk_ids.to(dtype=torch.int32),
            topk_weights.to(dtype=torch.float32),
            w13_weight,
            w2_weight,
            hidden_states.dtype,
            quant_scales=[w13_scale_inv, w2_scale_inv],
            ep_size=int(spec.ep_size),
            ep_rank=int(ep_rank),
            output=output,
            use_deepseek_fp8_block_scale=True,
            use_fused_finalize=False,
            enable_pdl=False,
            activation_type=ActivationType.Swiglu,
        )
        return output


@MOE_REGISTRY.register
class TritonMoeProvider(MoeProvider):
    name = "triton"
    priority = 10
    gate_up_order = "gate_up"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.no(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.no(
                f"requires BF16 or FP16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype == torch.float8_e4m3fn:
            if spec.block_shape != (128, 128):
                return SupportResult.no(
                    f"FP8 requires block_shape=(128, 128), got {spec.block_shape}"
                )
            if not caps.supports_native_fp8:
                return SupportResult.no("device does not provide native FP8 tensor cores")
            if spec.hidden_size % 128 or spec.intermediate_size % 128:
                return SupportResult.no("FP8 hidden/intermediate sizes must be 128-aligned")
            return SupportResult.yes()
        if spec.weight_dtype != spec.activation_dtype:
            return SupportResult.no(
                "unquantized weights must match the activation dtype"
            )
        return SupportResult.yes()

    def run(
        self,
        spec,
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_scale_inv,
        w2_scale_inv,
        *,
        local_expert_start,
        ep_rank,
    ):
        del ep_rank
        if spec.weight_dtype == torch.float8_e4m3fn:
            if w13_scale_inv is None or w2_scale_inv is None:
                raise RuntimeError("Triton FP8 MoE requires expert scales.")
            from sparsevllm.triton_kernel.moe import fused_moe_fp8

            return fused_moe_fp8(
                hidden_states,
                w13_weight,
                w2_weight,
                w13_scale_inv,
                w2_scale_inv,
                topk_ids,
                topk_weights,
                num_experts=spec.num_experts,
                local_expert_start=local_expert_start,
                gate_up_order=self.gate_up_order,
            )
        from sparsevllm.triton_kernel.moe import fused_moe

        return fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


def resolve_moe_provider(
    spec: MoeOpSpec,
    *,
    device_index: int | None = None,
) -> MoeProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(MOE_REGISTRY).resolve(spec, caps).provider
