"""Prepared gated pooling over cache-owned carry state."""

from dataclasses import dataclass

import torch

from sparsevllm.engine.cache_manager.native_attention import CompressionBatchView
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class CompressionOpSpec:
    ratio: int
    head_dim: int
    max_num_tokens: int
    max_num_requests: int

    def __post_init__(self):
        if self.ratio not in (4, 128) or min(self.head_dim, self.max_num_tokens, self.max_num_requests) <= 0:
            raise ValueError("Native compression requires ratio 4/128 and positive prepared dimensions.")


COMPRESSION_REGISTRY = OpRegistry(
    "request-local native gated compression",
    portfolio=PortfolioPolicy(repo_nonstandard=("triton_gated_pool",)),
)


@COMPRESSION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class GatedCompressionProvider:
    name = "triton_gated_pool"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA, Triton and BF16")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        self.spec = spec
        self.max_events = spec.max_num_requests + spec.max_num_tokens // spec.ratio
        self._value_bytes = (self.max_events * spec.head_dim * 2 + 255) // 256 * 256
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            self._value_bytes + self.max_events * 4, label="gated_compression", lane="gated_compression",
        )

    def run(self, kv, gate, ape, view: CompressionBatchView):
        spec = self.spec
        if kv.shape[0] > spec.max_num_tokens or view.request_rows.numel() > spec.max_num_requests:
            raise ValueError("Compression batch exceeds prepared workspace capacity.")
        events = view.request_rows.numel() if view.decode else view.boundary_requests.numel()
        if events > self.max_events:
            raise ValueError("Compression boundaries exceed prepared workspace capacity.")
        storage = self._lease.buffer
        values = storage[:self._value_bytes].view(torch.bfloat16)[:events * spec.head_dim].view(events, spec.head_dim)
        positions = storage[self._value_bytes:].view(torch.int32)[:events]
        from sparsevllm.kernels.triton.deepseek_v4.compression import compress_projected
        compress_projected(kv, gate, ape, view.state_kv, view.state_gate,
                           view.request_rows, view.cu_seqlens, view.start_positions,
                           view.boundary_requests, view.boundary_ends, values, positions,
                           ratio=spec.ratio, decode=view.decode, snapshots=view.snapshots)
        return values, positions


def prepare_compression(spec, *, device_index):
    return OpResolver(COMPRESSION_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
