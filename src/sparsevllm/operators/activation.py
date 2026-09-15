from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class SiluAndMulSpec:
    activation_dtype: torch.dtype
    input_ndim: int = 2
    contiguous: bool = True
    swiglu_limit: float | None = None

    def __post_init__(self) -> None:
        if int(self.input_ndim) <= 0:
            raise ValueError("SiluAndMul input_ndim must be positive.")
        if self.swiglu_limit is not None and (not math.isfinite(self.swiglu_limit) or self.swiglu_limit <= 0):
            raise ValueError("SwiGLU limit must be finite and positive.")


def _validate_input(x: torch.Tensor) -> None:
    if int(x.shape[-1]) % 2:
        raise ValueError(
            "SiluAndMul requires an even final dimension, got "
            f"{int(x.shape[-1])}."
        )


def _validate_bound_input(x: torch.Tensor, spec: SiluAndMulSpec) -> None:
    _validate_input(x)
    if x.dtype != spec.activation_dtype:
        raise TypeError(
            "Bound SiluAndMul provider requires "
            f"dtype={spec.activation_dtype}, got {x.dtype}."
        )
    if x.ndim != int(spec.input_ndim):
        raise ValueError(
            "Bound SiluAndMul provider requires "
            f"ndim={spec.input_ndim}, got {x.ndim}."
        )
    if spec.contiguous and not x.is_contiguous():
        raise ValueError("Bound SiluAndMul provider requires contiguous input.")


class SiluAndMulProvider:
    name = ""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


SILU_AND_MUL_REGISTRY: OpRegistry[SiluAndMulSpec, SiluAndMulProvider] = OpRegistry(
    "SiLU-and-multiply",
    portfolio=PortfolioPolicy(upstream_standard=("vllm_clipped",), repo_nonstandard=("triton_clipped",), repo_portable=("triton", "torch")),
)


@SILU_AND_MUL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonSiluAndMulProvider(SiluAndMulProvider):
    name = "triton"

    def __init__(self, *, op_spec: SiluAndMulSpec) -> None:
        self.spec = op_spec

    @classmethod
    def supports(cls, spec: SiluAndMulSpec, caps: DeviceCaps) -> SupportResult:
        if spec.swiglu_limit is not None:
            return SupportResult.unsupported("requires unclipped SwiGLU")
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.activation_dtype not in (torch.float16, torch.bfloat16):
            return SupportResult.unsupported(
                "requires FP16 or BF16 activations, "
                f"got {spec.activation_dtype}"
            )
        if int(spec.input_ndim) != 2 or not spec.contiguous:
            return SupportResult.unsupported("requires contiguous rank-2 inputs")
        return SupportResult.yes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _validate_bound_input(x, self.spec)
        if not x.is_cuda:
            raise ValueError("Triton SiluAndMul provider requires a CUDA input.")
        from sparsevllm.kernels.triton.silu_and_mul import silu_and_mul_fwd

        return silu_and_mul_fwd(x)


@SILU_AND_MUL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchSiluAndMulProvider(SiluAndMulProvider):
    name = "torch"

    def __init__(self, *, op_spec: SiluAndMulSpec | None = None) -> None:
        self.spec = op_spec

    @classmethod
    def supports(cls, spec: SiluAndMulSpec, caps: DeviceCaps) -> SupportResult:
        if spec.swiglu_limit is not None:
            return SupportResult.unsupported("requires unclipped SwiGLU")
        return SupportResult.yes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec is None:
            _validate_input(x)
        else:
            _validate_bound_input(x, self.spec)
        gate, up = x.chunk(2, -1)
        F.silu(gate, inplace=True)
        gate.mul_(up)
        return gate


@SILU_AND_MUL_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonClippedSiluAndMulProvider(SiluAndMulProvider):
    name = "triton_clipped"

    def __init__(self, *, op_spec: SiluAndMulSpec):
        self.spec = op_spec

    @classmethod
    def supports(cls, spec, caps):
        if spec.swiglu_limit is None:
            return SupportResult.unsupported("requires clipped SwiGLU")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA and Triton")
        if spec.activation_dtype != torch.bfloat16 or spec.input_ndim != 2 or not spec.contiguous:
            return SupportResult.unsupported("requires contiguous rank-2 BF16 inputs")
        return SupportResult.yes()

    def __call__(self, x):
        _validate_bound_input(x, self.spec)
        if not x.is_cuda:
            raise ValueError("Clipped SwiGLU requires a CUDA input.")
        from sparsevllm.kernels.triton.silu_and_mul import clipped_swiglu
        output = x.new_empty((x.shape[0], x.shape[1] // 2))
        clipped_swiglu(x, output, limit=self.spec.swiglu_limit)
        return output


@SILU_AND_MUL_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class VllmClippedSiluAndMulProvider(SiluAndMulProvider):
    name = "vllm_clipped"

    def __init__(self, *, op_spec: SiluAndMulSpec):
        from sparsevllm.kernels.external.vllm_moe import clipped_swiglu_op
        self.spec = op_spec
        self._op = clipped_swiglu_op()

    @classmethod
    def supports(cls, spec, caps):
        if spec.swiglu_limit is None:
            return SupportResult.unsupported("requires clipped SwiGLU")
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported("requires CUDA")
        if spec.activation_dtype != torch.bfloat16 or spec.input_ndim != 2 or not spec.contiguous:
            return SupportResult.unsupported("requires contiguous rank-2 BF16 inputs")
        from sparsevllm.kernels.external.vllm_moe import clipped_swiglu_op
        if clipped_swiglu_op() is None:
            return SupportResult.unsupported("optional vLLM stable-ABI library is unavailable")
        return SupportResult.yes()

    def binding_metadata(self):
        return {"kernel_path": "vllm_stable_abi._C.silu_and_mul_with_clamp",
                "interface": "raw stable-ABI kernel; no vLLM engine dispatcher"}

    def __call__(self, x):
        _validate_bound_input(x, self.spec)
        if not x.is_cuda:
            raise ValueError("Clipped SwiGLU requires a CUDA input.")
        output = x.new_empty((x.shape[0], x.shape[1] // 2))
        if output.numel():
            self._op(output, x, self.spec.swiglu_limit, 1.0, 0.0)
        return output


def resolve_silu_and_mul_provider(
    *,
    activation_dtype: torch.dtype,
    input_ndim: int = 2,
    contiguous: bool = True,
    swiglu_limit: float | None = None,
    device_index: int | None = None,
) -> SiluAndMulProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    spec = SiluAndMulSpec(
        activation_dtype=activation_dtype,
        input_ndim=int(input_ndim),
        contiguous=bool(contiguous),
        swiglu_limit=swiglu_limit,
    )
    return OpResolver(SILU_AND_MUL_REGISTRY).resolve(
        spec,
        caps,
        op_spec=spec,
    ).provider


__all__ = [
    "SILU_AND_MUL_REGISTRY",
    "SiluAndMulProvider",
    "SiluAndMulSpec",
    "TorchSiluAndMulProvider",
    "TritonSiluAndMulProvider",
    "resolve_silu_and_mul_provider",
]
