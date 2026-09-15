"""Sparse attention over shared K/V vectors with a zero-valued sink logit."""

from dataclasses import dataclass
import math

import torch

from sparsevllm.engine.cache_manager.native_attention import IndexedSharedKVView
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class IndexedSharedKVAttentionSpec:
    num_heads: int
    head_dim: int
    selection_capacity: int
    softmax_scale: float
    max_query_tokens: int
    activation_dtype: torch.dtype = torch.bfloat16
    cuda_graph: bool = True

    def __post_init__(self):
        if min(self.num_heads, self.head_dim, self.selection_capacity, self.max_query_tokens) <= 0:
            raise ValueError("Indexed attention dimensions and selection capacity must be positive.")
        if not math.isfinite(self.softmax_scale) or self.softmax_scale <= 0:
            raise ValueError("Indexed attention softmax scale must be finite and positive.")


class IndexedSharedKVAttentionProvider:
    name = ""

    def run(self, query: torch.Tensor, view: IndexedSharedKVView, sink: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


INDEXED_SHARED_KV_REGISTRY = OpRegistry(
    "indexed shared-KV attention",
    portfolio=PortfolioPolicy(upstream_standard=("sgl_flashmla",)),
)


@INDEXED_SHARED_KV_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SglIndexedSharedKVAttentionProvider(IndexedSharedKVAttentionProvider):
    name = "sgl_flashmla"
    supports_decode_graph = True

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability not in ((9, 0), (10, 0)):
            return SupportResult.unsupported("SGL sparse FlashMLA requires SM90 or SM100")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support graph capture")
        if spec.head_dim != 512 or spec.num_heads not in (64, 128):
            return SupportResult.unsupported("SGL shared-KV kernel requires dimension 512 and 64 or 128 local heads")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported("SGL sparse FlashMLA requires BF16 query and cache")
        from sparsevllm.kernels.external.sgl.sparse_mla import sparse_mla_op
        sparse_mla_op()
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, caps.device_index)

    def __init__(self, spec, device_index):
        from sparsevllm.kernels.external.sgl.sparse_mla import sparse_mla_op
        self.spec = spec
        self._op = sparse_mla_op()
        self._device = current_platform.get_device(device_index)
        aligned_capacity = (spec.selection_capacity + 127) // 128 * 128
        self._padded_indices = None
        if aligned_capacity != spec.selection_capacity:
            self._padded_indices = torch.full(
                (spec.max_query_tokens, 1, aligned_capacity), -1,
                dtype=torch.int32, device=self._device,
            )

    def binding_metadata(self):
        return {"implementation_kind": "atomic_provider", "implementation_source": "sglang-kernel",
                "kernel_path": "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
                "cache_contract": "indexed_bf16_shared_kv", "sink": "zero_value_logit"}

    def run(self, query, view, sink):
        spec = self.spec
        if query.device != self._device:
            raise ValueError("Indexed attention query must use the prepared device.")
        if query.ndim != 3 or query.shape[1:] != (spec.num_heads, spec.head_dim):
            raise ValueError("Indexed attention query shape differs from the prepared contract.")
        if query.shape[0] > spec.max_query_tokens:
            raise ValueError("Indexed attention query count exceeds the prepared workspace capacity.")
        if view.kv.ndim != 3 or view.kv.shape[1:] != (1, spec.head_dim):
            raise ValueError("Indexed attention cache requires [slots, 1, head_dim].")
        if view.indices.shape != (query.shape[0], 1, spec.selection_capacity):
            raise ValueError("Indexed attention selection capacity differs from the prepared contract.")
        if sink.shape != (spec.num_heads,):
            raise ValueError("Indexed attention requires one sink logit per head.")
        if query.dtype != spec.activation_dtype or view.kv.dtype != spec.activation_dtype:
            raise TypeError("Indexed attention query and cache must use the prepared activation dtype.")
        if view.indices.dtype != torch.int32 or sink.dtype != torch.float32:
            raise TypeError("Indexed attention requires int32 indices and FP32 sink logits.")
        if any(t.device != query.device or not t.is_contiguous() for t in (query, view.kv, view.indices, sink)):
            raise ValueError("Indexed attention inputs must be contiguous and share a device.")
        if query.shape[0] == 0:
            return torch.empty_like(query)
        indices = view.indices
        if self._padded_indices is not None:
            indices = self._padded_indices[:query.shape[0]]
            indices[:, :, :spec.selection_capacity].copy_(view.indices)
        # The public upstream API owns the result and its reduction workspace.
        # During capture these allocations are retained by the graph memory pool.
        return self._op(query, view.kv, indices, spec.softmax_scale,
                        d_v=spec.head_dim, attn_sink=sink)[0]


def prepare_indexed_shared_kv_attention(spec, *, device_index: int):
    caps = current_platform.get_device_caps(device_index)
    return OpResolver(INDEXED_SHARED_KV_REGISTRY).resolve(spec, caps).provider
