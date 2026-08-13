from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import torch

import sparsevllm.platforms as platforms
from sparsevllm.kernels.moe import MoeAlignment
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    runtime_version_at_least,
)
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
    tp_size: int = 1
    routing_method: str = "softmax"
    scale_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.num_experts <= 0 or self.num_local_experts <= 0:
            raise ValueError("MoE expert counts must be positive.")
        if self.ep_size <= 0:
            raise ValueError("MoE ep_size must be positive.")
        if self.tp_size <= 0:
            raise ValueError("MoE tp_size must be positive.")
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
        if self.routing_method not in {"softmax", "biased_sigmoid"}:
            raise ValueError(
                "MoE routing_method must be 'softmax' or 'biased_sigmoid', "
                f"got {self.routing_method!r}."
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

    def _packed_projection_offset(
        self,
        projection: str,
        intermediate_size: int,
    ) -> int:
        if projection not in {"gate", "up"}:
            raise ValueError(f"Unknown packed MoE projection {projection!r}.")
        first = "gate" if self.gate_up_order == "gate_up" else "up"
        return 0 if projection == first else int(intermediate_size)

    def load_expert_projection(
        self,
        spec: MoeOpSpec,
        *,
        local_expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale_inv: torch.Tensor | None,
        w2_scale_inv: torch.Tensor | None,
    ) -> None:
        if not 0 <= local_expert_id < spec.num_local_experts:
            raise ValueError(
                f"Local expert id {local_expert_id} is outside "
                f"[0, {spec.num_local_experts})."
            )
        if projection == "down":
            weight_target = w2_weight[local_expert_id]
            scale_target = (
                None if w2_scale_inv is None else w2_scale_inv[local_expert_id]
            )
        elif projection in {"gate", "up"}:
            weight_offset = self._packed_projection_offset(
                projection,
                spec.intermediate_size,
            )
            weight_target = w13_weight[
                local_expert_id,
                weight_offset : weight_offset + spec.intermediate_size,
            ]
            if w13_scale_inv is None:
                scale_target = None
            else:
                scale_rows = w13_scale_inv.shape[1] // 2
                scale_offset = self._packed_projection_offset(
                    projection,
                    scale_rows,
                )
                scale_target = w13_scale_inv[
                    local_expert_id,
                    scale_offset : scale_offset + scale_rows,
                ]
        else:
            raise ValueError(f"Unknown logical MoE projection {projection!r}.")

        if tuple(loaded_weight.shape) != tuple(weight_target.shape):
            raise ValueError(
                "MoE expert weight shape mismatch: "
                f"expected={tuple(weight_target.shape)}, "
                f"got={tuple(loaded_weight.shape)}."
            )
        if (loaded_scale is None) != (scale_target is None):
            raise ValueError(
                "MoE expert scale presence does not match provider storage."
            )
        if loaded_scale is not None and scale_target is not None:
            if tuple(loaded_scale.shape) != tuple(scale_target.shape):
                raise ValueError(
                    "MoE expert scale shape mismatch: "
                    f"expected={tuple(scale_target.shape)}, "
                    f"got={tuple(loaded_scale.shape)}."
                )
            scale_target.copy_(loaded_scale)
        weight_target.copy_(loaded_weight)

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

_PACKED_SHARED_EXPERT_PROFILES = frozenset(
    {(64, 1, 4, 2048, 1536, 2, 1)}
)


def use_packed_shared_experts(
    *,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    tp_size: int,
    ep_size: int,
    cuda_graph: bool,
) -> bool:
    """Return whether a profiled decode path packs shared experts as routes."""

    profile = (
        int(num_routed_experts),
        int(num_shared_experts),
        int(top_k),
        int(hidden_size),
        int(intermediate_size),
        int(tp_size),
        int(ep_size),
    )
    return bool(cuda_graph) and profile in _PACKED_SHARED_EXPERT_PROFILES


def append_shared_expert_route(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    shared_expert_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sparsevllm.kernels.triton.moe import append_shared_expert_route as run

    return run(
        topk_ids,
        topk_weights,
        shared_expert_id=shared_expert_id,
    )


def _sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> MoeAlignment:
    from sparsevllm.kernels.external.sgl.moe import sgl_moe_align_block_size
    from sparsevllm.kernels.triton.moe import localize_expert_ids

    num_local_experts = int(local_expert_end) - int(local_expert_start)
    has_remote_experts = num_local_experts != int(num_experts)
    if has_remote_experts:
        topk_ids = localize_expert_ids(
            topk_ids,
            local_expert_start=local_expert_start,
            local_expert_end=local_expert_end,
            remote_expert_id=num_local_experts,
        )
    return sgl_moe_align_block_size(
        topk_ids,
        block_size=block_size,
        num_experts=num_local_experts,
    )


@MOE_REGISTRY.register
class SglAlignedTritonGlmMoeProvider(MoeProvider):
    name = "sgl_aligned_triton_glm"
    priority = 30
    gate_up_order = "gate_up"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no("requires CUDA SM90")
        if caps.device_name != "NVIDIA H100 80GB HBM3":
            return SupportResult.no("requires profiled NVIDIA H100 80GB HBM3")
        expected = {
            (64, 64, 2048, 768, 4, 2, 1, "biased_sigmoid"),
            (65, 65, 2048, 768, 5, 2, 1, "biased_sigmoid"),
            (64, 32, 2048, 1536, 4, 1, 2, "biased_sigmoid"),
        }
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
            spec.routing_method,
        )
        if actual not in expected:
            return SupportResult.no(
                f"requires a profiled GLM TP2 MoE shape {expected}, got {actual}"
            )
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no("requires BF16 activations")
        if spec.weight_dtype != torch.bfloat16 or spec.block_shape is not None:
            return SupportResult.no("requires unquantized BF16 expert weights")
        from sparsevllm.kernels.external.sgl.moe import sgl_moe_alignment_support

        supported, reason = sgl_moe_alignment_support()
        return SupportResult.yes() if supported else SupportResult.no(reason)

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
        if w13_scale_inv is not None or w2_scale_inv is not None:
            raise RuntimeError("SGL-aligned BF16 MoE does not accept expert scales.")
        from sparsevllm.kernels.triton.moe import fused_moe

        alignment_impl = None
        num_tokens = int(hidden_states.shape[0])
        if (
            num_tokens <= 64
            and (
                int(spec.ep_size) > 1
                or num_tokens * int(spec.top_k) * 4 > int(spec.num_experts)
            )
        ):
            alignment_impl = _sgl_moe_align_block_size
        return fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
            alignment_impl=alignment_impl,
        )


@MOE_REGISTRY.register
class TritonMinimaxM2FusedMoeProvider(MoeProvider):
    name = "triton_minimax_m2_fused"
    priority = 125
    gate_up_order = "gate_up"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.routing_method != "biased_sigmoid":
            return SupportResult.no("requires biased-sigmoid routing")
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
        if not caps.supports_bfloat16:
            return SupportResult.no("device does not support BF16")
        if not caps.supports_native_fp8:
            return SupportResult.no("device does not provide native FP8 tensor cores")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype != torch.float8_e4m3fn:
            return SupportResult.no(
                f"requires FP8 E4M3 weights, got {spec.weight_dtype}"
            )
        if spec.block_shape != (128, 128):
            return SupportResult.no(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
        if spec.scale_dtype != torch.float32:
            return SupportResult.no(
                f"requires FP32 expert scales, got {spec.scale_dtype}"
            )
        if spec.tp_size not in {1, 2, 4}:
            return SupportResult.no(
                f"requires MoE TP size 1, 2, or 4, got {spec.tp_size}"
            )
        expected_shape = (
            256,
            256 // spec.ep_size,
            3072,
            1536 // spec.tp_size,
            8,
        )
        actual_shape = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
        )
        if actual_shape != expected_shape:
            return SupportResult.no(
                "requires MiniMax M2.7 expert shape "
                f"{expected_shape}, got {actual_shape}"
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
        if w13_scale_inv is None or w2_scale_inv is None:
            raise RuntimeError("MiniMax M2.7 fused MoE requires expert scales.")
        from sparsevllm.kernels.triton.minimax_m2_moe import (
            fused_minimax_m2_moe_fp8,
        )

        return fused_minimax_m2_moe_fp8(
            hidden_states,
            w13_weight,
            w2_weight,
            w13_scale_inv,
            w2_scale_inv,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


@MOE_REGISTRY.register
class FlashInferCutlassFp8MoeProvider(MoeProvider):
    name = "flashinfer_cutlass_fp8_sm90"
    priority = 100
    gate_up_order = "up_gate"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.tp_size != 1:
            return SupportResult.no("does not support tensor-parallel expert shards")
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
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.no(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
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
            enable_pdl=None,
            activation_type=ActivationType.Swiglu,
        )
        return output


@MOE_REGISTRY.register
class TritonHopperFusedMoeProvider(MoeProvider):
    name = "triton_hopper_fused"
    priority = 20
    gate_up_order = "gate_up"
    PROFILED_DEVICE_NAME = "NVIDIA H100 80GB HBM3"
    PROFILED_SHAPES = (
        (128, 64, 2048, 384, 8, 2, 2),
        (256, 256, 2048, 512, 8, 1, 1),
        (256, 256, 2048, 256, 8, 2, 1),
        (256, 128, 2048, 512, 8, 1, 2),
    )

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype != torch.bfloat16 or spec.block_shape is not None:
            return SupportResult.no("requires unquantized BF16 expert weights")
        if caps.device_name != cls.PROFILED_DEVICE_NAME:
            return SupportResult.no(
                f"requires profiled {cls.PROFILED_DEVICE_NAME} hardware, "
                f"got {caps.device_name}"
            )
        if not caps.supports_bfloat16:
            return SupportResult.no("device does not support BF16")
        actual_shape = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if actual_shape not in cls.PROFILED_SHAPES:
            return SupportResult.no(
                "requires a profiled MoE shape in "
                f"{cls.PROFILED_SHAPES}, got {actual_shape}"
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
        if w13_scale_inv is not None or w2_scale_inv is not None:
            raise RuntimeError("Fused Hopper BF16 MoE does not accept expert scales.")
        from sparsevllm.kernels.triton.moe import fused_moe_gate_up_swiglu

        return fused_moe_gate_up_swiglu(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


@MOE_REGISTRY.register
class H20Qwen36FusedMoeProvider(TritonHopperFusedMoeProvider):
    name = "h20_qwen36_fused_bf16"
    priority = 21
    PROFILED_DEVICE_NAME = "NVIDIA H20"
    PROFILED_SHAPES = (
        (256, 256, 2048, 512, 8, 1, 1),
        (256, 256, 2048, 256, 8, 2, 1),
        (256, 128, 2048, 512, 8, 1, 2),
    )


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
            from sparsevllm.kernels.triton.moe import fused_moe_fp8

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
        from sparsevllm.kernels.triton.moe import fused_moe

        return fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


@MOE_REGISTRY.register
class HopperQwen36HybridFp8MoeProvider(FlashInferCutlassFp8MoeProvider):
    """Bind one weight layout and dispatch profiled token buckets by kernel."""

    name = "hopper_qwen36_hybrid_fp8"
    priority = 130
    PROFILED_DEVICE_NAME = "NVIDIA H100 80GB HBM3"
    PROFILED_SHAPES = frozenset(
        {
            (256, 256, 2048, 512, 8, 1, 1),
            (256, 128, 2048, 512, 8, 1, 2),
        }
    )
    TRITON_MAX_TOKENS_BY_EP_SIZE = {1: 8, 2: 4}

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if caps.device_name != cls.PROFILED_DEVICE_NAME:
            return SupportResult.no(
                f"requires profiled {cls.PROFILED_DEVICE_NAME} hardware, "
                f"got {caps.device_name}"
            )
        actual_shape = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if actual_shape not in cls.PROFILED_SHAPES:
            return SupportResult.no(
                "requires profiled Qwen3.6 TP1/EP1 or "
                "global-TP2/MoE-TP1xEP2 shape "
                f"{sorted(cls.PROFILED_SHAPES)}, "
                f"got {actual_shape}"
            )
        flashinfer_support = super().supports(spec, caps)
        if not flashinfer_support.supported:
            return SupportResult.no(
                f"FlashInfer prefill path: {flashinfer_support.reason}"
            )
        triton_support = TritonMoeProvider.supports(spec, caps)
        if not triton_support.supported:
            return SupportResult.no(
                f"Triton decode path: {triton_support.reason}"
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
        triton_max_tokens = self.TRITON_MAX_TOKENS_BY_EP_SIZE[int(spec.ep_size)]
        if int(hidden_states.shape[0]) > triton_max_tokens:
            return super().run(
                spec,
                hidden_states,
                topk_ids,
                topk_weights,
                w13_weight,
                w2_weight,
                w13_scale_inv,
                w2_scale_inv,
                local_expert_start=local_expert_start,
                ep_rank=ep_rank,
            )
        if w13_scale_inv is None or w2_scale_inv is None:
            raise RuntimeError("Qwen3.6 hybrid FP8 MoE requires expert scales.")
        from sparsevllm.kernels.triton.moe import fused_moe_fp8

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


@MOE_REGISTRY.register
class H20Qwen36HybridFp8MoeProvider(HopperQwen36HybridFp8MoeProvider):
    name = "h20_qwen36_hybrid_fp8"
    priority = 131
    PROFILED_DEVICE_NAME = "NVIDIA H20"
    TRITON_MAX_TOKENS_BY_EP_SIZE = {1: 8, 2: 1}


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
