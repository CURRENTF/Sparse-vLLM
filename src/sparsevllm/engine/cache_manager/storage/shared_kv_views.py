"""Cache-side construction of physical native attention and ring-write views."""

from dataclasses import dataclass

import torch

from ..native_attention import IndexedSharedKVView, SharedKVWindowBatch


@dataclass(frozen=True)
class SharedKVRegions:
    num_request_rows: int
    window_size: int
    compressed_capacity: int
    prefill_capacity: int

    def __post_init__(self):
        if min(self.num_request_rows, self.window_size, self.prefill_capacity) <= 0 or self.compressed_capacity < 0:
            raise ValueError("Shared KV requires positive request/window/prefill capacities and nonnegative compression capacity.")

    @property
    def compressed_offset(self):
        return self.num_request_rows * self.window_size

    @property
    def temporary_offset(self):
        return self.compressed_offset + self.compressed_capacity

    @property
    def num_slots(self):
        return self.temporary_offset + self.prefill_capacity

    def _validate_batch(self, batch):
        n, device = batch.positions.numel(), batch.positions.device
        prefill = (batch.chunk_starts, batch.chunk_ends, batch.packed_starts)
        if any(t is None for t in prefill) and any(t is not None for t in prefill):
            raise ValueError("Prefill window metadata requires starts, ends and packed offsets together.")
        if batch.chunk_starts is not None and n > self.prefill_capacity:
            raise ValueError("Window prefill batch exceeds the prepared temporary region.")
        tensors = (batch.request_rows, batch.positions) + tuple(t for t in prefill if t is not None)
        if any(t.shape != (n,) or t.dtype != torch.int32 or t.device != device or not t.is_contiguous() for t in tensors):
            raise ValueError("Window batch metadata requires contiguous int32 token vectors on one device.")
        return n, device

    def attention_view(self, kv, batch: SharedKVWindowBatch, indices, *, compressed=None,
                       compressed_lengths=None, compressed_per_request=False):
        if kv.ndim != 3 or kv.shape[:2] != (self.num_slots, 1) or kv.device != batch.positions.device:
            raise ValueError("Shared KV payload does not match the prepared physical regions.")
        self.attention_indices(batch, indices, compressed=compressed, compressed_lengths=compressed_lengths,
                               compressed_per_request=compressed_per_request)
        return IndexedSharedKVView(kv, indices)

    def attention_indices(self, batch, indices, *, compressed=None, compressed_lengths=None,
                          compressed_per_request=False):
        n, device = self._validate_batch(batch)
        capacity = 0
        if compressed is not None:
            expected_rows = self.num_request_rows if compressed_per_request else n
            if (compressed.ndim != 2 or len(compressed) != expected_rows or compressed.dtype != torch.int32
                    or compressed.device != device or not compressed.is_contiguous()):
                raise ValueError("Compressed physical selections require contiguous int32 rows on the cache device.")
            capacity = compressed.shape[1]
        if compressed_per_request:
            if compressed is None or compressed_lengths is None:
                raise ValueError("Per-request compressed tables require per-query causal lengths.")
            if (compressed_lengths.shape != (n,) or compressed_lengths.dtype != torch.int32
                    or compressed_lengths.device != device or not compressed_lengths.is_contiguous()):
                raise ValueError("Compressed visibility requires contiguous int32 token lengths.")
        elif compressed_lengths is not None:
            raise ValueError("Selected compressed slots encode their own padding and do not take lengths.")
        if indices.shape != (n, 1, self.window_size + capacity) or indices.dtype != torch.int32 or indices.device != device or not indices.is_contiguous():
            raise ValueError("Attention index output must match packed queries and window/selection capacity.")
        from sparsevllm.kernels.triton.deepseek_v4.window_indices import shared_kv_indices
        shared_kv_indices(batch, compressed, compressed_lengths, indices, window_size=self.window_size,
                          window_offset=0, temporary_offset=self.temporary_offset,
                          compressed_offset=self.compressed_offset, compressed_per_request=compressed_per_request)
        return indices

    def commit_slots(self, batch: SharedKVWindowBatch, out):
        n, device = self._validate_batch(batch)
        if out.shape != (n,) or out.dtype != torch.int32 or out.device != device or not out.is_contiguous():
            raise ValueError("Window commit output requires contiguous int32 token slots on the cache device.")
        from sparsevllm.kernels.triton.deepseek_v4.window_indices import window_commit_slots
        window_commit_slots(batch, out, window_size=self.window_size, window_offset=0)
        return out
