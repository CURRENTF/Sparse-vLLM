from __future__ import annotations

from typing import Any

from .base import AttentionCacheStorage, CacheLayout
from .explicit_kv import ExplicitKVStorage
from .mla_latent import MlaLatentStorage


def create_attention_cache_storage(
    config: Any,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> AttentionCacheStorage:
    configured_layout = getattr(config, "attention_cache_layout", None)
    if configured_layout is None:
        model_type = str(getattr(config.hf_config, "model_type", "") or "")
        configured_layout = (
            CacheLayout.MLA_LATENT.value
            if model_type == "glm4_moe_lite"
            else CacheLayout.EXPLICIT_KV.value
        )
    layout = (
        configured_layout
        if isinstance(configured_layout, CacheLayout)
        else CacheLayout(str(configured_layout))
    )
    dtype = config.hf_config.torch_dtype
    if layout is CacheLayout.EXPLICIT_KV:
        return ExplicitKVStorage(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
    if layout is CacheLayout.MLA_LATENT:
        return MlaLatentStorage(
            kv_lora_rank=int(config.hf_config.kv_lora_rank),
            rope_dim=int(config.hf_config.qk_rope_head_dim),
            dtype=dtype,
        )
    raise AssertionError(f"Unhandled attention cache layout: {layout!r}")


__all__ = [
    "AttentionCacheStorage",
    "CacheLayout",
    "ExplicitKVStorage",
    "MlaLatentStorage",
    "create_attention_cache_storage",
]
