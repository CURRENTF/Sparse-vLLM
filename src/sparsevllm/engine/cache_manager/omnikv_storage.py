"""Split residency with a common logical slot space for full and sparse layers."""

from __future__ import annotations

import torch

from sparsevllm.kernels.triton.indexed_host_copy import store_rows

from .base import ExplicitKVPayload, ExplicitKVWrite, MlaLatentPayload, MlaLatentWrite
from .storage.base import CacheLayout


def payload_tensors(payload):
    if isinstance(payload, (ExplicitKVPayload, ExplicitKVWrite)):
        return (
            (payload.k_cache, payload.v_cache)
            if isinstance(payload, ExplicitKVPayload)
            else (payload.key, payload.value)
        )
    if isinstance(payload, (MlaLatentPayload, MlaLatentWrite)):
        return (
            (payload.latent_cache, payload.rope_cache)
            if isinstance(payload, MlaLatentPayload)
            else (payload.latent, payload.rope)
        )
    raise TypeError(f"Unsupported offload payload: {type(payload).__name__}")


class OmniKVStorage:
    def __init__(
        self, original, *, num_layers, num_slots, full_layers, device, prefix_slots=0
    ):
        self.layout = original.layout
        self.dtype = original.dtype
        self.full_layers = frozenset(full_layers)
        self.num_slots = num_slots
        if self.layout is CacheLayout.EXPLICIT_KV:
            shapes = [(original.num_kv_heads, original.head_dim)] * 2
        else:
            shapes = [(1, original.kv_lora_rank), (1, original.rope_dim)]
        self.shapes = shapes
        self.host_slot_map = torch.arange(num_slots, dtype=torch.int32, device=device)
        self.layers = []
        self.pointers = []
        for layer in range(num_layers):
            on_host = layer not in self.full_layers
            parts = tuple(
                torch.empty(
                    num_slots + (prefix_slots if on_host else 0),
                    *shape,
                    dtype=self.dtype,
                    device="cpu" if on_host else device,
                    pin_memory=on_host,
                )
                for shape in shapes
            )
            self.layers.append(parts)
            self.pointers.append(
                torch.tensor(
                    [x.data_ptr() for x in parts], dtype=torch.uint64, device=device
                )
            )

    def make_payload(self, parts):
        return (
            ExplicitKVPayload(*parts)
            if self.layout is CacheLayout.EXPLICIT_KV
            else MlaLatentPayload(*parts)
        )

    def layer_payload(self, layer_idx):
        return self.make_payload(self.layers[layer_idx])

    def bytes_per_slot_per_layer(self):
        return (
            sum(h * d for h, d in self.shapes)
            * torch.tensor([], dtype=self.dtype).element_size()
        )

    def slot_capacity(self):
        return self.num_slots

    def validate_slot_mapping(self, slots):
        if slots.ndim != 1 or slots.dtype != torch.int32 or slots.device.type != "cuda":
            raise ValueError("OmniKV writes require a CUDA int32 slot vector.")

    def validate_slot_mappings(self, mappings):
        for slots in mappings:
            self.validate_slot_mapping(slots)

    def store(self, layer_idx, slots, payload):
        if layer_idx in self.full_layers:
            first, second = payload_tensors(payload)
            a_cache, b_cache = self.layers[layer_idx]
            if self.layout is CacheLayout.EXPLICIT_KV:
                from sparsevllm.kernels.triton.store_kvcache import store_kvcache

                store_kvcache(first, second, a_cache, b_cache, slots)
            else:
                from sparsevllm.kernels.triton.mla.copy_latent import (
                    copy_latent_to_cache,
                )

                copy_latent_to_cache(
                    first, second, slots, a_cache, b_cache, validate_slots=False
                )
            return
        for component, source in enumerate(payload_tensors(payload)):
            store_rows(
                source,
                self.pointers[layer_idx],
                slots,
                component,
                self.host_slot_map,
            )

    def accounting_tensors(self):
        return (
            tuple(x for parts in self.layers for x in parts)
            + tuple(self.pointers)
            + (self.host_slot_map,)
        )
