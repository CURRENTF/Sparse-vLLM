"""FP32-projected, Sinkhorn-normalized residual stream mixing."""

from dataclasses import dataclass
import math

import torch

from sparsevllm.operators.float32_linear import Float32LinearSpec, prepare_float32_linear
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class HyperConnectionSpec:
    hidden_size: int
    num_streams: int
    max_num_tokens: int
    norm_eps: float = 1e-6
    mixing_eps: float = 1e-6
    sinkhorn_iterations: int = 20

    def __post_init__(self):
        if min(self.hidden_size, self.num_streams, self.max_num_tokens, self.sinkhorn_iterations) <= 0:
            raise ValueError("Hyper-connection dimensions and iteration count must be positive.")
        if any(not math.isfinite(eps) or eps <= 0 for eps in (self.norm_eps, self.mixing_eps)):
            raise ValueError("Hyper-connection epsilons must be finite and positive.")


HYPER_CONNECTION_REGISTRY = OpRegistry(
    "manifold-constrained hyper connection",
    portfolio=PortfolioPolicy(upstream_standard=("torch_flashinfer",)),
)


@HYPER_CONNECTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferHyperConnectionProvider:
    name = "torch_flashinfer"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA and BF16")
        if spec.num_streams != 4 or spec.hidden_size % 8:
            return SupportResult.unsupported("FlashInfer mHC requires four streams and hidden size divisible by eight")
        from sparsevllm.kernels.external.flashinfer.mhc import mhc_ops
        mhc_ops()
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        from sparsevllm.kernels.external.flashinfer.mhc import mhc_ops
        self._pre, self._post = mhc_ops()
        self.spec, self.device = spec, device
        self._mix_dim = (2 + spec.num_streams) * spec.num_streams
        self._projection = prepare_float32_linear(
            Float32LinearSpec(spec.num_streams * spec.hidden_size, self._mix_dim,
                              max(2, spec.max_num_tokens)),
            device_index=device.index,
        )

    def _validate(self, tensor, shape, dtype):
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(f"Hyper-connection input requires contiguous {dtype} {shape} on {self.device}.")

    def _residual_rows(self, residual):
        if residual.ndim != 3:
            raise ValueError("Hyper-connection residual requires [tokens, streams, hidden_size].")
        rows = len(residual)
        if rows > self.spec.max_num_tokens:
            raise ValueError("Hyper-connection input exceeds prepared token capacity.")
        self._validate(residual, (rows, self.spec.num_streams, self.spec.hidden_size), torch.bfloat16)
        return rows

    def pre(self, residual, weight, scale, base):
        s = self.spec
        rows = self._residual_rows(residual)
        width = s.num_streams * s.hidden_size
        self._validate(weight, (self._mix_dim, width), torch.float32)
        self._validate(scale, (3,), torch.float32)
        self._validate(base, (self._mix_dim,), torch.float32)
        # Fix the projection shape across decode batches and prefill chunks:
        # FP32 reduction differences can cross BF16 residual midpoints.
        dot_mix = self._projection.run(residual.view(rows, width), weight)
        # Public FlashInfer owns result allocations; capture retains them in the
        # graph pool. No per-layer persistent copy of projection scratch is kept.
        post, comb, layer_input = self._pre(
            dot_mix, residual, scale, base, rms_eps=s.norm_eps,
            mhc_pre_eps=s.mixing_eps, mhc_sinkhorn_eps=s.mixing_eps,
            mhc_post_mult_value=2., sinkhorn_repeat=s.sinkhorn_iterations,
        )
        return layer_input, post.squeeze(-1), comb

    def post(self, x, residual, post, comb):
        s = self.spec
        rows = self._residual_rows(residual)
        self._validate(x, (rows, s.hidden_size), torch.bfloat16)
        self._validate(post, (rows, s.num_streams), torch.float32)
        self._validate(comb, (rows, s.num_streams, s.num_streams), torch.float32)
        return self._post(x, residual, post, comb)


def prepare_hyper_connection(spec, *, device_index):
    return OpResolver(HYPER_CONNECTION_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider


@dataclass(frozen=True)
class HyperConnectionHeadSpec:
    hidden_size: int
    num_streams: int
    max_num_tokens: int
    norm_eps: float = 1e-6
    mixing_eps: float = 1e-6

    def __post_init__(self):
        if min(self.hidden_size, self.num_streams, self.max_num_tokens) <= 0:
            raise ValueError("Hyper-connection head dimensions must be positive.")
        if any(not math.isfinite(eps) or eps <= 0 for eps in (self.norm_eps, self.mixing_eps)):
            raise ValueError("Hyper-connection head epsilons must be finite and positive.")


HYPER_CONNECTION_HEAD_REGISTRY = OpRegistry(
    "hyper connection head", portfolio=PortfolioPolicy(upstream_standard=("torch",)),
)


@HYPER_CONNECTION_HEAD_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class TorchHyperConnectionHeadProvider:
    name = "torch"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA and BF16")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self._projection = prepare_float32_linear(
            Float32LinearSpec(spec.num_streams * spec.hidden_size, spec.num_streams,
                              max(2, spec.max_num_tokens)),
            device_index=device.index,
        )
        self._sizes = (spec.max_num_tokens * spec.num_streams * spec.hidden_size,
                       spec.max_num_tokens,
                       spec.max_num_tokens * spec.hidden_size)
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            4 * sum(self._sizes), label="hyper_connection_head", lane="hyper_connection_projection",
        )

    def run(self, residual, weight, scale, base):
        s = self.spec
        rows = len(residual)
        if rows > s.max_num_tokens:
            raise ValueError("Hyper-connection head exceeds prepared token capacity.")
        for tensor, shape, dtype in (
            (residual, (rows, s.num_streams, s.hidden_size), torch.bfloat16),
            (weight, (s.num_streams, s.num_streams * s.hidden_size), torch.float32),
            (scale, (1,), torch.float32), (base, (s.num_streams,), torch.float32),
        ):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError(f"Hyper-connection head requires contiguous {dtype} {shape} on {self.device}.")
        output = torch.empty((rows, s.hidden_size), device=self.device, dtype=torch.bfloat16)
        if rows == 0:
            return output
        x, norm, reduced = self._lease.buffer.view(torch.float32).split(self._sizes)
        x = x.view(s.max_num_tokens, -1)
        mixes = self._projection.run(residual.flatten(1), weight)
        norm = norm[:, None]
        reduced = reduced[:rows * s.hidden_size].view(rows, s.hidden_size)
        x[:rows].copy_(residual.flatten(1))
        x[rows:].zero_()
        # Normalize the FP32 projection. Normalizing BF16 input before GEMM
        # changes the native checkpoint's rounding contract.
        x.square_()
        torch.mean(x, dim=1, keepdim=True, out=norm)
        norm.add_(s.norm_eps).rsqrt_()
        mixes.mul_(norm[:rows]).mul_(scale).add_(base).sigmoid_().add_(s.mixing_eps)
        # Fixed RMS rows avoid reduction-shape rounding; explicit multiply/sum
        # preserves the reference's stream contraction without batched GEMM.
        weighted = x[:rows].view(rows, s.num_streams, s.hidden_size)
        weighted.copy_(residual).mul_(mixes[:, :, None])
        torch.sum(weighted, dim=1, out=reduced)
        output.copy_(reduced)
        return output


def prepare_hyper_connection_head(spec, *, device_index):
    return OpResolver(HYPER_CONNECTION_HEAD_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
