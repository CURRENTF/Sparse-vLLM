"""Prepared rotary semantics for native shared-KV and index representations."""

from dataclasses import dataclass
import math

import torch

from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class NativeRotarySpec:
    head_dim: int
    rotary_dim: int
    norm: str = "none"
    eps: float = 1e-6
    inverse: bool = False
    quantize_nope: bool = False

    def __post_init__(self):
        if self.head_dim <= 0 or self.head_dim % 2 or not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
            raise ValueError("Native rotary dimensions must be positive even dimensions within the head.")
        if self.norm not in ("none", "weighted_fp32", "query_bf16"):
            raise ValueError(f"Unsupported native normalization {self.norm!r}.")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Native normalization epsilon must be finite and positive.")
        if self.quantize_nope and (self.head_dim - self.rotary_dim) % 64:
            raise ValueError("Native KV QAT requires NoPE dimensions aligned to groups of 64.")


NATIVE_ROTARY_REGISTRY = OpRegistry(
    "native normalization and adjacent-pair rotary transform",
    portfolio=PortfolioPolicy(repo_nonstandard=("triton_native",)),
)


@NATIVE_ROTARY_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonNativeRotaryProvider:
    name = "triton_native"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA, Triton and BF16")
        if spec.quantize_nope and not caps.supports_native_fp8:
            return SupportResult.unsupported("KV QAT requires native E4M3 conversion")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        self.spec, self.device = spec, device

    def run(self, x, positions, inv_freq, *, weight=None, out=None):
        s = self.spec
        if x.ndim != 3 or x.shape[-1] != s.head_dim or x.shape[1] <= 0 or x.dtype != torch.bfloat16:
            raise ValueError("Native rotary input must be BF16 [tokens, heads, prepared head_dim].")
        if positions.shape != (x.shape[0],) or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("Native rotary positions must be integer token rows.")
        if inv_freq.shape != (s.rotary_dim // 2,) or inv_freq.dtype != torch.float32:
            raise ValueError("Native rotary frequencies must be FP32 with half the rotary dimension.")
        if s.norm == "weighted_fp32":
            if weight is None or weight.shape != (s.head_dim,) or weight.dtype != torch.float32:
                raise ValueError("Weighted native RMSNorm requires FP32 per-channel weights.")
        elif weight is not None:
            raise ValueError("Unweighted native rotary transforms cannot take normalization weights.")
        if out is None:
            out = torch.empty_like(x)
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("Native rotary output must match the input shape and dtype.")
        tensors = (x, positions, inv_freq, out) + (() if weight is None else (weight,))
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Native rotary tensors must be contiguous on the prepared device.")
        from sparsevllm.kernels.triton.native_rope import native_rotary_transform
        native_rotary_transform(x, positions, inv_freq, weight, out, norm=s.norm,
                                eps=s.eps, inverse=s.inverse, quantize_nope=s.quantize_nope)
        return out


def prepare_native_rotary(spec, *, device_index):
    return OpResolver(NATIVE_ROTARY_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
