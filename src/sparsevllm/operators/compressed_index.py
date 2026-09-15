"""Prepared native index scoring and top-k over cache-owned physical slots."""

from dataclasses import dataclass

import torch

from sparsevllm.engine.cache_manager.native_attention import CompressedIndexView
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class CompressedIndexSpec:
    num_heads: int
    head_dim: int
    max_index_tokens: int
    query_chunk_size: int
    top_k: int

    def __post_init__(self):
        if min(self.num_heads, self.head_dim, self.max_index_tokens, self.query_chunk_size, self.top_k) <= 0:
            raise ValueError("Compressed index dimensions and capacities must be positive.")


COMPRESSED_INDEX_REGISTRY = OpRegistry(
    "native compressed-key index scoring and selection",
    portfolio=PortfolioPolicy(repo_nonstandard=("triton_flashinfer",)),
)


@COMPRESSED_INDEX_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class CompressedIndexProvider:
    name = "triton_flashinfer"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_bfloat16 or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA, BF16 and Triton")
        if spec.head_dim != 128 or spec.num_heads > 64:
            return SupportResult.unsupported("native index scores require dimension 128 and at most 64 local heads")
        if spec.top_k != 512:
            return SupportResult.unsupported("this prepared native index contract requires top-512")
        from sparsevllm.kernels.external.flashinfer.index_topk import index_topk_op
        index_topk_op()
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index), caps.multi_processor_count)

    def __init__(self, spec, device, num_sms):
        from sparsevllm.kernels.external.flashinfer.index_topk import index_topk_op
        self.spec, self.device, self.num_sms = spec, device, num_sms
        self._select = index_topk_op()
        self._score_bytes = (spec.query_chunk_size * spec.max_index_tokens * 2 + 255) // 256 * 256
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            self._score_bytes + spec.query_chunk_size * 4,
            label="compressed_index_scores", lane="compressed_index_scores",
        )

    def graph_keepalive_tensors(self):
        return (self._lease.buffer,)

    def _validate_tensor(self, tensor, shape, dtype):
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(f"Compressed index requires contiguous {dtype} {shape} on {self.device}.")

    def _validate_view(self, view, rows):
        s = self.spec
        if rows > s.query_chunk_size:
            raise ValueError("Compressed index query chunk exceeds prepared capacity.")
        if view.keys.ndim != 2 or view.slots.ndim != 2:
            raise ValueError("Compressed index keys and slot tables require matrices.")
        self._validate_tensor(view.keys, (len(view.keys), s.head_dim), torch.bfloat16)
        self._validate_tensor(view.slots, (len(view.slots), s.max_index_tokens), torch.int32)
        self._validate_tensor(view.query_rows, (rows,), torch.int32)
        self._validate_tensor(view.visible_lengths, (rows,), torch.int32)

    def score(self, query, weights, view: CompressedIndexView):
        s = self.spec
        rows = len(query)
        self._validate_view(view, rows)
        self._validate_tensor(query, (rows, s.num_heads, s.head_dim), torch.bfloat16)
        self._validate_tensor(weights, (rows, s.num_heads), torch.bfloat16)
        scores = self._lease.buffer[:self._score_bytes].view(torch.bfloat16)
        scores = scores[:rows * s.max_index_tokens].view(rows, s.max_index_tokens)
        from sparsevllm.kernels.triton.deepseek_v4.index_score import compressed_index_scores
        compressed_index_scores(query, weights, view, scores, num_sms=self.num_sms)
        # Only [0, visible_length) is initialized. Selection consumes these
        # lengths; attention-TP reduction must likewise use a masked buffer.
        return scores

    def select(self, scores, view: CompressedIndexView, *, out, raw_indices=None):
        s = self.spec
        rows = len(scores)
        self._validate_view(view, rows)
        self._validate_tensor(scores, (rows, s.max_index_tokens), torch.bfloat16)
        self._validate_tensor(out, (rows, s.top_k), torch.int32)
        if raw_indices is not None:
            self._validate_tensor(raw_indices, out.shape, torch.int32)
        if rows:
            safe_rows = self._lease.buffer[self._score_bytes:].view(torch.int32)[:rows]
            torch.clamp(view.query_rows, min=0, out=safe_rows)
            self._select(scores, view.slots, view.visible_lengths, s.top_k,
                         row_to_batch=safe_rows, deterministic=True, dsa_graph_safe=True,
                         out=out, out_raw_indices=raw_indices)
        return out


def prepare_compressed_index(spec, *, device_index):
    return OpResolver(COMPRESSED_INDEX_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
