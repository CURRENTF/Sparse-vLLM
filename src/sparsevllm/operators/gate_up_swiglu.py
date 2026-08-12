from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import OpRegistry, OpResolver, SupportResult
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class GateUpSwiGLUOpSpec:
    hidden_size: int
    intermediate_size: int
    tp_size: int
    activation_dtype: torch.dtype
    weight_dtype: torch.dtype
    cuda_graph: bool

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.intermediate_size, self.tp_size) <= 0:
            raise ValueError("Gate/up SwiGLU dimensions and TP size must be positive.")
        if self.intermediate_size % self.tp_size:
            raise ValueError(
                "Gate/up SwiGLU intermediate size must be divisible by TP size."
            )
        if not (
            self.activation_dtype.is_floating_point
            and self.weight_dtype.is_floating_point
        ):
            raise TypeError(
                "Gate/up SwiGLU activations and weights must be floating point."
            )


class GateUpSwiGLUProvider:
    name = ""
    priority = 0

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        raise NotImplementedError


GATE_UP_SWIGLU_REGISTRY: OpRegistry[
    GateUpSwiGLUOpSpec, GateUpSwiGLUProvider
] = OpRegistry("gate/up SwiGLU")


@GATE_UP_SWIGLU_REGISTRY.register
class NativeGateUpSwiGLUProvider(GateUpSwiGLUProvider):
    name = "native"
    priority = 0

    @classmethod
    def supports(
        cls,
        spec: GateUpSwiGLUOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        del spec
        gate, up = projection(inputs).chunk(2, dim=-1)
        return F.silu(gate, inplace=True).mul_(up)


@GATE_UP_SWIGLU_REGISTRY.register
class H20GateUpSwiGLUProvider(NativeGateUpSwiGLUProvider):
    name = "h20_triton_decode"
    priority = 20

    @classmethod
    def supports(
        cls,
        spec: GateUpSwiGLUOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no(
                f"requires CUDA SM90, got {caps.platform.name} "
                f"{caps.compute_capability}"
            )
        if caps.device_name != "NVIDIA H20":
            return SupportResult.no(
                f"requires profiled NVIDIA H20 hardware, got {caps.device_name}"
            )
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if not caps.supports_bfloat16:
            return SupportResult.no("device does not support BF16")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16 or spec.weight_dtype != torch.bfloat16:
            return SupportResult.no(
                "requires BF16 activations and weights, got "
                f"{spec.activation_dtype} and {spec.weight_dtype}"
            )
        if (spec.hidden_size, spec.intermediate_size, spec.tp_size) not in {
            (2048, 512, 1),
            (2048, 512, 2),
        }:
            return SupportResult.no(
                "requires profiled (hidden, intermediate, TP) shape in "
                "{(2048, 512, 1), (2048, 512, 2)}"
            )
        return SupportResult.yes()

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        if inputs.shape[0] != 1:
            return super().run(spec, inputs, projection)
        from sparsevllm.triton_kernel.gate_up_swiglu import h20_gate_up_swiglu

        return h20_gate_up_swiglu(inputs, projection.weight)


def resolve_gate_up_swiglu_provider(
    spec: GateUpSwiGLUOpSpec,
    *,
    device_index: int | None = None,
) -> GateUpSwiGLUProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GATE_UP_SWIGLU_REGISTRY).resolve(spec, caps).provider
