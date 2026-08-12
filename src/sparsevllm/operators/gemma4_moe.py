from __future__ import annotations

import torch
import torch.nn.functional as F

from sparsevllm import platforms
from sparsevllm.operators.moe import MoeOpSpec, MoeProvider
from sparsevllm.operators.registry import OpRegistry, OpResolver, SupportResult
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum

GEMMA4_MOE_REGISTRY: OpRegistry[MoeOpSpec, MoeProvider] = OpRegistry(
    "Gemma 4 routed GEGLU MoE"
)


@GEMMA4_MOE_REGISTRY.register
class TritonGemma4MoeProvider(MoeProvider):
    name = "triton_gemma4_geglu"
    priority = 10
    gate_up_order = "gate_up"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.activation != "gelu_tanh" or spec.routing_method != "softmax":
            return SupportResult.no("requires Gemma 4 GELU-tanh and softmax routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.no("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.no("requires BF16 or FP16 activations")
        if spec.weight_dtype != spec.activation_dtype or spec.block_shape is not None:
            return SupportResult.no(
                "requires unquantized experts matching activation dtype"
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
            raise RuntimeError("Gemma 4 BF16 MoE does not accept expert scales.")
        from sparsevllm.kernels.triton.gemma4_moe import fused_gemma4_moe

        return fused_gemma4_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


@GEMMA4_MOE_REGISTRY.register
class TorchGemma4MoeProvider(MoeProvider):
    name = "torch_gemma4_geglu"
    priority = 0

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        del caps
        if spec.activation != "gelu_tanh" or spec.routing_method != "softmax":
            return SupportResult.no("requires Gemma 4 GELU-tanh and softmax routing")
        if spec.weight_dtype != spec.activation_dtype or spec.block_shape is not None:
            return SupportResult.no("requires unquantized Gemma 4 GELU-tanh experts")
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
            raise RuntimeError("Gemma 4 Torch MoE does not accept expert scales.")
        output = torch.zeros_like(hidden_states)
        for local_id in range(spec.num_local_experts):
            global_id = int(local_expert_start) + local_id
            token_ids, routes = torch.where(topk_ids == global_id)
            if token_ids.numel() == 0:
                continue
            gate, up = F.linear(hidden_states[token_ids], w13_weight[local_id]).chunk(
                2, -1
            )
            routed = F.linear(
                F.gelu(gate, approximate="tanh") * up, w2_weight[local_id]
            )
            output.index_add_(
                0, token_ids, routed * topk_weights[token_ids, routes, None]
            )
        return output


def resolve_gemma4_moe_provider(
    spec: MoeOpSpec,
    *,
    device_index: int | None = None,
) -> MoeProvider:
    if spec.activation != "gelu_tanh":
        raise ValueError(
            "Gemma 4 MoE resolver requires activation='gelu_tanh', "
            f"got {spec.activation!r}."
        )
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GEMMA4_MOE_REGISTRY).resolve(spec, caps).provider


__all__ = [
    "GEMMA4_MOE_REGISTRY",
    "TorchGemma4MoeProvider",
    "TritonGemma4MoeProvider",
    "resolve_gemma4_moe_provider",
]
