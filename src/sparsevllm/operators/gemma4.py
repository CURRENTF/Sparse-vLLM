from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import torch
import torch.nn.functional as F

import sparsevllm.platforms as platforms
from sparsevllm.layers.rotary_embedding import apply_rotary_emb
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    runtime_version_at_least,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class Gemma4OpSpec:
    activation_dtype: torch.dtype
    head_dims: tuple[int, ...]
    cuda_graph: bool

    def __post_init__(self) -> None:
        if not self.head_dims or any(int(value) <= 0 for value in self.head_dims):
            raise ValueError("Gemma 4 head dimensions must be positive.")


class Gemma4OperatorProvider:
    name = ""
    priority = 0

    def attention_backend(self, *, sliding_window: int | None):
        raise NotImplementedError

    def rmsnorm(
        self, x: torch.Tensor, weight: torch.Tensor | None, eps: float
    ) -> torch.Tensor:
        raise NotImplementedError

    def qkv_norm_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor | None,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def router_input(
        self,
        hidden_states: torch.Tensor,
        scale: torch.Tensor,
        root_size: float,
        eps: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def gelu_tanh_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gelu_mul(self, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rmsnorm_residual(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


GEMMA4_REGISTRY: OpRegistry[Gemma4OpSpec, Gemma4OperatorProvider] = OpRegistry(
    "Gemma 4 model operations"
)


@GEMMA4_REGISTRY.register
class TritonGemma4OperatorProvider(Gemma4OperatorProvider):
    name = "triton"
    priority = 10

    @classmethod
    def supports(cls, spec: Gemma4OpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.no("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.no("requires BF16 or FP16 activations")
        if any(head_dim not in {256, 512} for head_dim in spec.head_dims):
            return SupportResult.no("requires attention head dimensions 256 or 512")
        return SupportResult.yes()

    def attention_backend(self, *, sliding_window: int | None):
        from sparsevllm.operators.gemma4_attention import Gemma4AttentionBackend

        return Gemma4AttentionBackend(sliding_window=sliding_window)

    def rmsnorm(self, x, weight, eps):
        from sparsevllm.kernels.triton.gemma4_rmsnorm import gemma4_rmsnorm

        return gemma4_rmsnorm(x, weight, eps)

    def qkv_norm_rope(
        self, q, k, v, q_weight, k_weight, rope_cache, positions, eps
    ):
        from sparsevllm.kernels.triton.gemma4_qkv_norm_rope import (
            gemma4_qkv_norm_rope,
        )

        gemma4_qkv_norm_rope(
            q, k, v, q_weight, k_weight, rope_cache, positions, eps
        )
        return q, k, v

    def router_input(self, hidden_states, scale, root_size, eps):
        from sparsevllm.kernels.triton.gemma4_router import gemma4_router_input

        return gemma4_router_input(hidden_states, scale, root_size, eps)

    def gelu_tanh_and_mul(self, x):
        from sparsevllm.kernels.triton.gemma4_gelu_and_mul import (
            gelu_tanh_and_mul_fwd,
        )

        return gelu_tanh_and_mul_fwd(x)

    def gelu_mul(self, gate, value):
        from sparsevllm.kernels.triton.gemma4_fused_ops import gemma4_gelu_mul

        return gemma4_gelu_mul(gate, value)

    def rmsnorm_residual(self, x, weight, residual, eps, scalar=None):
        from sparsevllm.kernels.triton.gemma4_fused_ops import (
            gemma4_rmsnorm_residual,
        )

        return gemma4_rmsnorm_residual(x, weight, residual, eps, scalar)


@GEMMA4_REGISTRY.register
class H20Gemma4OperatorProvider(TritonGemma4OperatorProvider):
    """Profiled H20 provider; generic Gemma kernels remain unchanged."""

    name = "gemma4_h20"
    priority = 100

    @classmethod
    def supports(cls, spec: Gemma4OpSpec, caps: DeviceCaps) -> SupportResult:
        triton = super().supports(spec, caps)
        if not triton.supported:
            return triton
        if caps.compute_capability != (9, 0) or caps.device_name != "NVIDIA H20":
            return SupportResult.no(
                "requires profiled NVIDIA H20 SM90 hardware, "
                f"got {caps.device_name} {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.no(
                f"requires CUDA runtime >= 12.8, got {caps.runtime_version or 'unknown'}"
            )
        if find_spec("flashinfer") is None:
            return SupportResult.no("flashinfer is not installed")
        try:
            installed = version("flashinfer-python")
        except PackageNotFoundError:
            return SupportResult.no("flashinfer-python package metadata is unavailable")
        try:
            numeric = tuple(int(part) for part in installed.split(".")[:3])
        except ValueError:
            return SupportResult.no(
                f"cannot parse flashinfer-python version {installed!r}"
            )
        if numeric < (0, 6, 15):
            return SupportResult.no(
                f"requires flashinfer-python >= 0.6.15, got {installed}"
            )
        return SupportResult.yes()

    def __init__(self) -> None:
        from sparsevllm.operators.gemma4_attention import Gemma4FlashInferPrefill

        self._prefill = Gemma4FlashInferPrefill()

    def attention_backend(self, *, sliding_window: int | None):
        from sparsevllm.operators.gemma4_attention import Gemma4AttentionBackend

        return Gemma4AttentionBackend(
            sliding_window=sliding_window,
            flashinfer_prefill=self._prefill,
            use_window_decode=True,
            global_decode_heads_per_program=4,
        )


class TorchGemma4OperatorProvider(Gemma4OperatorProvider):
    """Explicit correctness oracle; never selected for production inference."""

    name = "torch_oracle"

    def attention_backend(self, *, sliding_window: int | None):
        from sparsevllm.operators.gemma4_attention import Gemma4AttentionBackend

        return Gemma4AttentionBackend(sliding_window=sliding_window)

    def rmsnorm(self, x, weight, eps):
        output = x.float()
        output *= torch.rsqrt(output.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            output *= weight.float()
        return output.to(x.dtype)

    def qkv_norm_rope(
        self, q, k, v, q_weight, k_weight, rope_cache, positions, eps
    ):
        q = self.rmsnorm(q, q_weight, eps)
        cos, sin = rope_cache[positions].chunk(2, -1)
        q = apply_rotary_emb(q, cos, sin)
        if k is not None:
            k = apply_rotary_emb(self.rmsnorm(k, k_weight, eps), cos, sin)
            v = self.rmsnorm(v, None, eps)
        return q, k, v

    def router_input(self, hidden_states, scale, root_size, eps):
        return self.rmsnorm(hidden_states, None, eps) * scale * root_size

    def gelu_tanh_and_mul(self, x):
        gate, up = x.chunk(2, -1)
        return F.gelu(gate, approximate="tanh") * up

    def gelu_mul(self, gate, value):
        return F.gelu(gate, approximate="tanh") * value

    def rmsnorm_residual(self, x, weight, residual, eps, scalar=None):
        output = self.rmsnorm(x, weight, eps) + residual
        return output if scalar is None else output * scalar


def resolve_gemma4_provider(
    spec: Gemma4OpSpec, *, device_index: int | None = None
) -> Gemma4OperatorProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GEMMA4_REGISTRY).resolve(spec, caps).provider


__all__ = [
    "GEMMA4_REGISTRY",
    "H20Gemma4OperatorProvider",
    "Gemma4OperatorProvider",
    "Gemma4OpSpec",
    "TorchGemma4OperatorProvider",
    "TritonGemma4OperatorProvider",
    "resolve_gemma4_provider",
]
