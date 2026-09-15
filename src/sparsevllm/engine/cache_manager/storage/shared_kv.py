"""Native KV vectors and raw compression carry, owned by cache managers."""

import torch

from ..base import SharedKVPayload, SharedKVWrite
from ..methods.deepseek_v4 import CompressionStateShape
from .base import CacheLayout
from .components import CacheComponentSpec


class SharedKVStorage:
    layout = CacheLayout.SHARED_KV

    def __init__(self, *, head_dim, dtype=torch.bfloat16):
        if head_dim <= 0 or dtype != torch.bfloat16:
            raise ValueError("Native shared KV storage requires a positive dimension and BF16.")
        self.head_dim, self.dtype = head_dim, dtype
        self.cache = None

    def allocate(self, *, num_layers, num_slots, device):
        if num_layers <= 0 or num_slots <= 0:
            raise ValueError("Shared KV storage requires positive layer and slot capacities.")
        self.cache = torch.empty((num_layers, num_slots, 1, self.head_dim), device=device, dtype=self.dtype)

    def layer_payload(self, layer_idx):
        if self.cache is None:
            raise RuntimeError("Shared KV storage is not allocated.")
        if not 0 <= layer_idx < self.cache.shape[0]:
            raise IndexError("Shared KV layer index is outside allocated storage.")
        return SharedKVPayload(self.cache[layer_idx])

    def slot_capacity(self):
        return self.layer_payload(0).cache.shape[0]

    def bytes_per_slot_per_layer(self):
        return self.head_dim * self.dtype.itemsize

    def accounting_tensors(self):
        self.layer_payload(0)
        return (self.cache,)

    def component_specs(self, layer_idx):
        return (CacheComponentSpec("shared_kv", (1, self.head_dim), self.dtype),)

    def component_tensors(self, layer_idx):
        return (self.layer_payload(layer_idx).cache,)

    def validate_slot_mapping(self, slot_mapping):
        cache = self.layer_payload(0).cache
        if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int32 or slot_mapping.device != cache.device:
            raise ValueError("Shared KV slot mappings must be int32 vectors on the cache device.")
        valid = ((slot_mapping >= -1) & (slot_mapping < self.slot_capacity())).all()
        if valid.is_cuda:
            torch._assert_async(valid, "Shared KV slot mapping is out of bounds")
        elif not valid.item():
            raise ValueError("Shared KV slot mapping is out of bounds.")

    def validate_slot_mappings(self, mappings):
        for mapping in mappings:
            self.validate_slot_mapping(mapping)

    def store(self, layer_idx, slot_mapping, payload):
        if not isinstance(payload, SharedKVWrite):
            raise TypeError("Shared KV storage requires a SharedKVWrite payload.")
        values, positions = payload.values, payload.positions
        cache = self.layer_payload(layer_idx).cache
        if values.ndim != 3 or values.shape[1:] != (1, self.head_dim) or values.dtype != self.dtype:
            raise ValueError("Shared KV values must be BF16 [tokens, 1, head_dim].")
        if slot_mapping.shape != (len(values),) or slot_mapping.dtype != torch.int32:
            raise ValueError("Shared KV slot mappings must be int32 token rows.")
        if positions is not None and (positions.shape != slot_mapping.shape or positions.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Shared KV valid positions must be integer token rows.")
        tensors = (values, slot_mapping) + (() if positions is None else (positions,))
        if any(t.device != cache.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Shared KV write tensors must be contiguous on the cache device.")
        from sparsevllm.kernels.triton.store_shared_kv import store_shared_kv
        store_shared_kv(values, slot_mapping, cache, positions)

    def copy_slots(self, layer_idx, source_slots, destination_slots):
        cache = self.layer_payload(layer_idx).cache
        if source_slots.shape != destination_slots.shape:
            raise ValueError("Shared KV slot copies require equal source/destination shapes.")
        # Snapshot before writing so overlapping copies preserve every source.
        values = cache.index_select(0, source_slots.long())
        cache.index_copy_(0, destination_slots.long(), values)


class CompressionCarryStorage:
    """Raw FP32 rings; cache lifecycle code copies/clears rows outside forward."""

    def __init__(self, *, num_rows, ratio, head_dim, device):
        if num_rows <= 0:
            raise ValueError("Compression carry requires a positive request capacity.")
        self.shape = CompressionStateShape(ratio, head_dim)
        self.kv = torch.zeros((num_rows, *self.shape.row_shape), device=device, dtype=torch.float32)
        self.gate = torch.zeros_like(self.kv)

    def accounting_tensors(self):
        return self.kv, self.gate

    def clear_rows(self, rows):
        rows = rows.long()
        for tensor in self.accounting_tensors():
            tensor.index_fill_(0, rows, 0.)

    def copy_rows(self, source_rows, destination_rows):
        if source_rows.shape != destination_rows.shape:
            raise ValueError("Compression carry copies require equal source/destination shapes.")
        source_rows, destination_rows = source_rows.long(), destination_rows.long()
        for tensor in self.accounting_tensors():
            tensor.index_copy_(0, destination_rows, tensor.index_select(0, source_rows))
