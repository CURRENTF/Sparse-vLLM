from __future__ import annotations

import torch

from sparsevllm.triton_kernel.store_kvcache import store_kvcache

from ..base import AttentionCacheWrite, ExplicitKVPayload, ExplicitKVWrite
from .base import CacheLayout


class ExplicitKVStorage:
    """The ordinary two-tensor K/V cache layout."""

    layout = CacheLayout.EXPLICIT_KV

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        if self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError(
                "Explicit KV dimensions must be positive, got "
                f"num_kv_heads={self.num_kv_heads} head_dim={self.head_dim}."
            )
        self.kv_cache: torch.Tensor | None = None

    def allocate(
        self,
        *,
        num_layers: int,
        num_slots: int,
        device: torch.device,
    ) -> None:
        num_layers = int(num_layers)
        num_slots = int(num_slots)
        if num_layers <= 0 or num_slots <= 0:
            raise ValueError(
                "Explicit KV allocation requires positive layers and slots, got "
                f"num_layers={num_layers} num_slots={num_slots}."
            )
        self.kv_cache = torch.empty(
            2,
            num_layers,
            num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=device,
        )

    def _require_cache(self) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("Explicit KV storage has not been allocated.")
        return self.kv_cache

    @property
    def cache(self) -> torch.Tensor:
        return self._require_cache()

    def layer_payload(self, layer_idx: int) -> ExplicitKVPayload:
        cache = self._require_cache()
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < int(cache.shape[1]):
            raise IndexError(
                f"Explicit KV layer index {layer_idx} is outside [0, {int(cache.shape[1])})."
            )
        return ExplicitKVPayload(
            k_cache=cache[0, layer_idx],
            v_cache=cache[1, layer_idx],
        )

    def validate_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        cache = self._require_cache()
        if slot_mapping.ndim != 1:
            raise ValueError(
                f"Explicit KV slot_mapping must be 1D, got {tuple(slot_mapping.shape)}."
            )
        if slot_mapping.dtype != torch.int32:
            raise TypeError(
                f"Explicit KV slot_mapping must use torch.int32, got {slot_mapping.dtype}."
            )
        if slot_mapping.device != cache.device:
            raise ValueError(
                "Explicit KV slot_mapping must share the cache device, got "
                f"slots={slot_mapping.device} cache={cache.device}."
            )

    def store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
    ) -> None:
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError(
                "ExplicitKVStorage.store requires ExplicitKVWrite, got "
                f"{type(payload).__name__}."
            )
        destination = self.layer_payload(layer_idx)
        if payload.key.shape != payload.value.shape:
            raise ValueError(
                "Explicit KV store tensors must have equal shapes, got "
                f"k={tuple(payload.key.shape)} v={tuple(payload.value.shape)}."
            )
        expected_tail = (self.num_kv_heads, self.head_dim)
        if payload.key.ndim != 3 or tuple(payload.key.shape[1:]) != expected_tail:
            raise ValueError(
                "Explicit KV store tensors must have shape [tokens, "
                f"{self.num_kv_heads}, {self.head_dim}], got "
                f"{tuple(payload.key.shape)}."
            )
        if slot_mapping.shape != (int(payload.key.shape[0]),):
            raise ValueError(
                "Explicit KV slot_mapping must match the token dimension, got "
                f"slots={tuple(slot_mapping.shape)} tokens={int(payload.key.shape[0])}."
            )
        if payload.key.dtype != self.dtype or payload.value.dtype != self.dtype:
            raise TypeError(
                f"Explicit KV store requires dtype={self.dtype}, got "
                f"k={payload.key.dtype} v={payload.value.dtype}."
            )
        destination_device = destination.k_cache.device
        if (
            payload.key.device != destination_device
            or payload.value.device != destination_device
            or slot_mapping.device != destination_device
        ):
            raise ValueError(
                "Explicit KV store tensors must share the destination device, got "
                f"destination={destination_device} key={payload.key.device} "
                f"value={payload.value.device} slots={slot_mapping.device}."
            )
        self.validate_slot_mapping(slot_mapping)
        store_kvcache(
            payload.key,
            payload.value,
            destination.k_cache,
            destination.v_cache,
            slot_mapping,
        )

    def bytes_per_slot_per_layer(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return int(2 * self.num_kv_heads * self.head_dim * element_size)

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self._require_cache(),)
