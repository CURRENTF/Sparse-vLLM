"""Physical DSv4 FP8 pages; logical slots and prefix ownership stay above storage."""

import torch

from ..base import SharedKVWrite
from ..native_attention import PackedSharedKVPayload
from .base import CacheLayout
from .components import CacheComponentSpec


PAGE_SIZE = 64
HEAD_DIM = 512
DATA_BYTES = 576
SCALE_BYTES = 8
TOKEN_BYTES = DATA_BYTES + SCALE_BYTES
PAGE_BYTES = (PAGE_SIZE * TOKEN_BYTES + DATA_BYTES - 1) // DATA_BYTES * DATA_BYTES


def packed_shared_kv_bytes(num_slots):
    if num_slots < 0:
        raise ValueError("Packed shared KV capacity must be nonnegative.")
    return (num_slots + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_BYTES


class PackedSharedKVStorage:
    layout = CacheLayout.SHARED_KV

    def __init__(self):
        self.cache = None
        self._ops = None

    def allocate(self, *, num_layers, num_slots, device):
        if num_layers <= 0 or num_slots <= 0:
            raise ValueError("Packed shared KV requires positive layer and slot capacities.")
        self._num_slots = num_slots
        self._ops = None
        pages = packed_shared_kv_bytes(num_slots) // PAGE_BYTES
        self.cache = torch.empty((num_layers, pages, PAGE_BYTES), device=device, dtype=torch.uint8)
        self._identity_rope = torch.cat([
            torch.ones(1, 32, device=device, dtype=torch.float32),
            torch.zeros(1, 32, device=device, dtype=torch.float32),
        ], -1)
        if self.cache.is_cuda:
            from sparsevllm.kernels.external.vllm_cache import shared_kv_cache_ops
            self._ops = shared_kv_cache_ops()
            if self._ops is None:
                raise RuntimeError("Packed shared KV requires the optional vLLM cache kernels.")

    def _pages(self, layer_idx):
        if self.cache is None:
            raise RuntimeError("Packed shared KV storage is not allocated.")
        if not 0 <= layer_idx < len(self.cache):
            raise IndexError("Packed shared KV layer index is outside allocated storage.")
        return self.cache[layer_idx]

    def layer_payload(self, layer_idx):
        pages = self._pages(layer_idx)
        view = torch.as_strided(pages, (len(pages), PAGE_SIZE, 1, TOKEN_BYTES),
                                (PAGE_BYTES, TOKEN_BYTES, TOKEN_BYTES, 1))
        return PackedSharedKVPayload(view, self._num_slots)

    def slot_capacity(self):
        self._pages(0)
        return self._num_slots

    def accounting_tensors(self):
        self._pages(0)
        return self.cache, self._identity_rope

    def component_specs(self, layer_idx):
        self._pages(layer_idx)
        return (CacheComponentSpec("shared_kv_fp8", (PAGE_BYTES,), torch.uint8, index_unit="page"),)

    def component_tensors(self, layer_idx):
        return (self._pages(layer_idx),)

    def validate_slot_mapping(self, slots):
        cache = self._pages(0)
        if slots.ndim != 1 or slots.dtype != torch.int32 or slots.device != cache.device:
            raise ValueError("Packed slot mappings must be int32 vectors on the cache device.")
        valid = ((slots >= -1) & (slots < self._num_slots)).all()
        if valid.is_cuda:
            torch._assert_async(valid, "Packed shared KV slot mapping is out of bounds")
        elif not valid.item():
            raise ValueError("Packed shared KV slot mapping is out of bounds.")

    def validate_slot_mappings(self, mappings):
        for mapping in mappings:
            self.validate_slot_mapping(mapping)

    def store(self, layer_idx, slot_mapping, payload):
        cache = self._pages(layer_idx)
        if not isinstance(payload, SharedKVWrite):
            raise TypeError("Packed shared KV requires a SharedKVWrite payload.")
        values, positions = payload.values, payload.positions
        if values.ndim != 3 or values.shape[1:] != (1, HEAD_DIM) or values.dtype != torch.bfloat16:
            raise ValueError("Packed shared KV values must be BF16 [tokens, 1, 512].")
        if slot_mapping.shape != (len(values),) or slot_mapping.dtype != torch.int32:
            raise ValueError("Packed slot mappings must be int32 token rows.")
        if positions is not None and (positions.shape != slot_mapping.shape or positions.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Packed shared KV validity positions must be integer token rows.")
        tensors = (values, slot_mapping) + (() if positions is None else (positions,))
        if any(t.device != cache.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Packed shared KV writes require contiguous tensors on the cache device.")
        if self._ops is None:
            raise RuntimeError("Numerical packed shared KV operations require CUDA kernels.")
        if not len(values):
            return
        slots = slot_mapping.to(torch.int64)
        if positions is not None:
            slots = torch.where(positions >= 0, slots, -1)
        # Values already include the model's normalization and rotary transform.
        # Identity rotary permits reuse of the upstream fused cache writer.
        query = values.new_zeros((len(values), 1, HEAD_DIM))
        identity_positions = torch.zeros(len(values), device=cache.device, dtype=torch.int64)
        self._ops[0](query, values[:, 0], cache, slots, identity_positions,
                     self._identity_rope, 8, 1e-6, PAGE_SIZE)

    def copy_slots(self, layer_idx, source_slots, destination_slots):
        pages = self._pages(layer_idx)
        if source_slots.ndim != 1 or source_slots.shape != destination_slots.shape:
            raise ValueError("Packed slot copies require equal source/destination vectors.")
        if any(t.dtype not in (torch.int32, torch.int64) or t.device != pages.device
               for t in (source_slots, destination_slots)):
            raise ValueError("Packed slot copies require integer indices on the cache device.")
        data = pages[:, :PAGE_SIZE * DATA_BYTES].view(-1, PAGE_SIZE, DATA_BYTES)
        scales = pages[:, PAGE_SIZE * DATA_BYTES:PAGE_SIZE * TOKEN_BYTES].view(-1, PAGE_SIZE, SCALE_BYTES)
        source = source_slots.long()
        destination = destination_slots.long()
        valid = ((source >= 0) & (source < self._num_slots)
                 & (destination >= 0) & (destination < self._num_slots)).all()
        if valid.is_cuda:
            torch._assert_async(valid, "Packed slot copy is outside logical capacity")
        elif not valid.item():
            raise ValueError("Packed slot copy is outside logical capacity.")
        # Advanced indexing snapshots both byte regions before either write.
        values = data[source // PAGE_SIZE, source % PAGE_SIZE]
        factors = scales[source // PAGE_SIZE, source % PAGE_SIZE]
        data[destination // PAGE_SIZE, destination % PAGE_SIZE] = values
        scales[destination // PAGE_SIZE, destination % PAGE_SIZE] = factors

    def gather(self, layer_idx, out, block_table, seq_lens, gather_lens=None, offset=0):
        cache = self._pages(layer_idx)
        if out.ndim != 3 or out.shape[-1] != HEAD_DIM or out.dtype != torch.bfloat16:
            raise ValueError("Packed gather output must be BF16 [requests, tokens, 512].")
        if (seq_lens.shape != (len(out),) or block_table.ndim != 2
                or len(block_table) != len(out) or not 0 <= offset <= out.shape[1]):
            raise ValueError("Packed gather metadata must match output requests and offset.")
        if gather_lens is not None and gather_lens.shape != seq_lens.shape:
            raise ValueError("Packed gather lengths must match sequence lengths.")
        metadata = (block_table, seq_lens) + (() if gather_lens is None else (gather_lens,))
        if any(t.dtype != torch.int32 for t in metadata):
            raise ValueError("Packed gather metadata must use int32.")
        if any(t.device != cache.device or not t.is_contiguous() for t in (out, *metadata)):
            raise ValueError("Packed gather tensors must be contiguous on the cache device.")
        if self._ops is None:
            raise RuntimeError("Numerical packed shared KV operations require CUDA kernels.")
        self._ops[1](out, self.layer_payload(layer_idx).cache[:, :, 0], seq_lens,
                     gather_lens, block_table, PAGE_SIZE, offset)
