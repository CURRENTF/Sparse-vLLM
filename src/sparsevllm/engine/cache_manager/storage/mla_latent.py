from __future__ import annotations

import torch

from sparsevllm.triton_kernel.mla.copy_latent import (
    copy_latent_to_cache,
    validate_copy_slot_mapping,
)
from sparsevllm.triton_kernel.mla.decode_stage1 import MLA_LATENT_DIM, MLA_ROPE_DIM

from ..base import AttentionCacheWrite, MlaLatentPayload, MlaLatentWrite
from .base import CacheLayout


class MlaLatentStorage:
    """Persistent latent and RoPE caches for GLM-style MLA."""

    layout = CacheLayout.MLA_LATENT

    def __init__(
        self,
        *,
        kv_lora_rank: int,
        rope_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.kv_lora_rank = int(kv_lora_rank)
        self.rope_dim = int(rope_dim)
        self.dtype = dtype
        if self.kv_lora_rank != MLA_LATENT_DIM or self.rope_dim != MLA_ROPE_DIM:
            raise ValueError(
                "The vendored MLA storage kernel requires "
                f"kv_lora_rank={MLA_LATENT_DIM} and rope_dim={MLA_ROPE_DIM}, got "
                f"kv_lora_rank={self.kv_lora_rank} rope_dim={self.rope_dim}."
            )
        if self.dtype != torch.bfloat16:
            raise TypeError(
                f"MLA latent storage v1 requires torch.bfloat16, got {self.dtype}."
            )
        self.latent_cache: torch.Tensor | None = None
        self.rope_cache: torch.Tensor | None = None
        self._validated_slot_mapping_key: tuple[str, int, int] | None = None
        self._validated_store_calls_remaining = 0

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
                "MLA allocation requires positive layers and slots, got "
                f"num_layers={num_layers} num_slots={num_slots}."
            )
        self.latent_cache = torch.empty(
            num_layers,
            num_slots,
            1,
            self.kv_lora_rank,
            dtype=self.dtype,
            device=device,
        )
        self.rope_cache = torch.empty(
            num_layers,
            num_slots,
            1,
            self.rope_dim,
            dtype=self.dtype,
            device=device,
        )
        self._validated_slot_mapping_key = None
        self._validated_store_calls_remaining = 0

    def _require_caches(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent_cache is None or self.rope_cache is None:
            raise RuntimeError("MLA latent storage has not been allocated.")
        return self.latent_cache, self.rope_cache

    def layer_payload(self, layer_idx: int) -> MlaLatentPayload:
        latent_cache, rope_cache = self._require_caches()
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < int(latent_cache.shape[0]):
            raise IndexError(
                f"MLA layer index {layer_idx} is outside [0, {int(latent_cache.shape[0])})."
            )
        return MlaLatentPayload(
            latent_cache=latent_cache[layer_idx],
            rope_cache=rope_cache[layer_idx],
        )

    @staticmethod
    def _slot_mapping_key(slot_mapping: torch.Tensor) -> tuple[str, int, int]:
        return (
            str(slot_mapping.device),
            int(slot_mapping.data_ptr()),
            int(slot_mapping.numel()),
        )

    def validate_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        latent_cache, _ = self._require_caches()
        if slot_mapping.ndim != 1:
            raise ValueError(
                f"MLA slot_mapping must be 1D, got {tuple(slot_mapping.shape)}."
            )
        if slot_mapping.dtype != torch.int32:
            raise TypeError(
                f"MLA slot_mapping must use torch.int32, got {slot_mapping.dtype}."
            )
        if slot_mapping.device != latent_cache.device:
            raise ValueError(
                "MLA slot_mapping must share the cache device, got "
                f"slots={slot_mapping.device} cache={latent_cache.device}."
            )
        validate_copy_slot_mapping(
            slot_mapping,
            cache_slot_count=int(latent_cache.shape[1]),
        )
        self._validated_slot_mapping_key = self._slot_mapping_key(slot_mapping)
        self._validated_store_calls_remaining = int(latent_cache.shape[0])

    def store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
    ) -> None:
        if not isinstance(payload, MlaLatentWrite):
            raise TypeError(
                "MlaLatentStorage.store requires MlaLatentWrite, got "
                f"{type(payload).__name__}."
            )
        destination = self.layer_payload(layer_idx)
        slot_mapping_key = self._slot_mapping_key(slot_mapping)
        use_prevalidated_mapping = (
            slot_mapping_key == self._validated_slot_mapping_key
            and self._validated_store_calls_remaining > 0
        )
        copy_latent_to_cache(
            payload.latent,
            payload.rope,
            slot_mapping,
            destination.latent_cache,
            destination.rope_cache,
            validate_slots=not use_prevalidated_mapping,
        )
        if use_prevalidated_mapping:
            self._validated_store_calls_remaining -= 1
            if self._validated_store_calls_remaining == 0:
                self._validated_slot_mapping_key = None

    def bytes_per_slot_per_layer(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return int((self.kv_lora_rank + self.rope_dim) * element_size)

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]:
        return self._require_caches()
