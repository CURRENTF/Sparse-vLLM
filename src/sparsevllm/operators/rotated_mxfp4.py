"""Normalized Hadamard rotation followed by native MXFP4 QAT rounding."""

from dataclasses import dataclass

import torch

from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class RotatedMxfp4Spec:
    head_dim: int

    def __post_init__(self):
        if self.head_dim < 32 or self.head_dim > 32768 or self.head_dim & (self.head_dim - 1):
            raise ValueError("Rotated MXFP4 requires a power-of-two dimension between 32 and 32768.")


ROTATED_MXFP4_REGISTRY = OpRegistry(
    "Hadamard rotated MXFP4 activation simulation",
    portfolio=PortfolioPolicy(repo_nonstandard=("fht_triton_qat",)),
)


@ROTATED_MXFP4_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class RotatedMxfp4Provider:
    name = "fht_triton_qat"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA, BF16 and Triton")
        from sparsevllm.kernels.external.hadamard import hadamard_op
        hadamard_op()
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        from sparsevllm.kernels.external.hadamard import hadamard_op
        self.spec, self.device = spec, device
        self._rotate = hadamard_op()

    def run(self, x):
        if x.ndim < 2 or x.shape[-1] != self.spec.head_dim or x.dtype != torch.bfloat16:
            raise ValueError("Rotated MXFP4 requires BF16 rows of the prepared head dimension.")
        if x.device != self.device or not x.is_contiguous():
            raise ValueError("Rotated MXFP4 requires contiguous inputs on the prepared device.")
        if not x.numel():
            return torch.empty_like(x)
        out = self._rotate(x, scale=self.spec.head_dim ** -.5)
        from sparsevllm.kernels.triton.mxfp4_qat import simulate_mxfp4
        simulate_mxfp4(out, out)
        return out


def prepare_rotated_mxfp4(spec, *, device_index):
    return OpResolver(ROTATED_MXFP4_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
