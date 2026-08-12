from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import torch

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    runtime_version_at_least,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class Fp8LinearSpec:
    block_shape: tuple[int, int]
    input_features: int
    output_features: int
    activation_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.input_features <= 0 or self.output_features <= 0:
            raise ValueError("FP8 Linear feature sizes must be positive.")


class Fp8LinearProvider:
    name = ""
    priority = 0

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


FP8_LINEAR_REGISTRY: OpRegistry[Fp8LinearSpec, Fp8LinearProvider] = OpRegistry(
    "block-scaled FP8 Linear"
)


@FP8_LINEAR_REGISTRY.register
class FlashInferSm90Fp8LinearProvider(Fp8LinearProvider):
    name = "flashinfer_sm90"
    priority = 100

    @classmethod
    def supports(cls, spec: Fp8LinearSpec, caps: DeviceCaps) -> SupportResult:
        if spec.block_shape != (128, 128):
            return SupportResult.no(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
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
        if spec.input_features % 128:
            return SupportResult.no(
                f"requires input_features divisible by 128, got {spec.input_features}"
            )
        if spec.output_features % 64:
            return SupportResult.no(
                f"requires output_features divisible by 64, got {spec.output_features}"
            )
        if find_spec("flashinfer") is None:
            return SupportResult.no("flashinfer is not installed")
        return SupportResult.yes()

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        if x.dtype != torch.bfloat16:
            raise TypeError(
                f"FlashInfer SM90 FP8 Linear requires BF16 activations, got {x.dtype}."
            )
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        original_shape = x.shape[:-1]
        output = fp8_blockscale_gemm_sm90(
            x.reshape(-1, x.shape[-1]).contiguous(),
            weight,
            weight_scale=weight_scale_inv,
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*original_shape, weight.shape[0])


@FP8_LINEAR_REGISTRY.register
class TritonFp8LinearProvider(Fp8LinearProvider):
    name = "triton"
    priority = 10

    @classmethod
    def supports(cls, spec: Fp8LinearSpec, caps: DeviceCaps) -> SupportResult:
        if spec.block_shape != (128, 128):
            return SupportResult.no(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.no(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if not caps.supports_native_fp8:
            return SupportResult.no("device does not provide native FP8 tensor cores")
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.no(
                f"requires BF16 or FP16 activations, got {spec.activation_dtype}"
            )
        return SupportResult.yes()

    def __call__(self, x, weight, weight_scale_inv, bias=None):
        from sparsevllm.triton_kernel.fp8_blockwise import fp8_blockwise_matmul

        original_shape = x.shape[:-1]
        output = fp8_blockwise_matmul(
            x.reshape(-1, x.shape[-1]).contiguous(),
            weight,
            weight_scale_inv,
            output_dtype=x.dtype,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*original_shape, weight.shape[0])


def resolve_fp8_linear_provider(
    block_shape: tuple[int, int],
    *,
    input_features: int,
    output_features: int,
    activation_dtype: torch.dtype = torch.bfloat16,
    device_index: int | None = None,
) -> Fp8LinearProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(FP8_LINEAR_REGISTRY).resolve(
        Fp8LinearSpec(
            tuple(int(value) for value in block_shape),
            input_features=int(input_features),
            output_features=int(output_features),
            activation_dtype=activation_dtype,
        ),
        caps,
    ).provider
