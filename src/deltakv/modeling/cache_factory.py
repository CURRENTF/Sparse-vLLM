from __future__ import annotations

from typing import Any

from deltakv.modeling.cache_pipeline import (
    DELTA_COMPRESSED_LATENT_WO_FULL,
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_WO_FULL,
    DELTA_ORIGIN_W_FULL,
    DeltaCompressedLatentWoFullCache,
    DeltaCompressedLatentWFullCache,
    DeltaOriginWoFullCache,
    DeltaOriginWFullCache,
)


_VALID_CACHE_IMPLS = {
    DELTA_COMPRESSED_LATENT_WO_FULL,
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_WO_FULL,
    DELTA_ORIGIN_W_FULL,
}


def set_deltakv_cache_impl(config: Any, cache_impl: str) -> None:
    cache_impl = str(cache_impl).strip()
    if cache_impl not in _VALID_CACHE_IMPLS:
        raise ValueError(
            f"Unknown deltakv_cache_impl={cache_impl!r}. "
            f"Expected one of {sorted(_VALID_CACHE_IMPLS)}."
        )
    setattr(config, "deltakv_cache_impl", cache_impl)


def get_deltakv_cache_impl(config: Any) -> str:
    cache_impl = getattr(config, "deltakv_cache_impl", DELTA_COMPRESSED_LATENT_WO_FULL)
    cache_impl = DELTA_COMPRESSED_LATENT_WO_FULL if cache_impl is None else str(cache_impl).strip()
    if cache_impl == "":
        cache_impl = DELTA_COMPRESSED_LATENT_WO_FULL
    if cache_impl not in _VALID_CACHE_IMPLS:
        raise ValueError(
            f"Unknown deltakv_cache_impl={cache_impl!r}. "
            f"Expected one of {sorted(_VALID_CACHE_IMPLS)}."
        )
    return cache_impl


def _expected_cache_types(config: Any) -> tuple[type, ...]:
    cache_impl = get_deltakv_cache_impl(config)
    if cache_impl == DELTA_COMPRESSED_LATENT_WO_FULL:
        return (DeltaCompressedLatentWoFullCache,)
    if cache_impl == DELTA_COMPRESSED_LATENT_W_FULL:
        return (DeltaCompressedLatentWFullCache,)
    if cache_impl == DELTA_ORIGIN_WO_FULL:
        return (DeltaOriginWoFullCache,)
    if cache_impl == DELTA_ORIGIN_W_FULL:
        return (DeltaOriginWFullCache,)
    raise AssertionError(f"Unhandled deltakv_cache_impl={cache_impl!r}")


def is_deltakv_cache_instance(past_key_values: Any, config: Any) -> bool:
    return isinstance(past_key_values, _expected_cache_types(config))


def create_deltakv_cache(config: Any):
    if not getattr(config, "use_cluster", False):
        raise ValueError("HF DeltaKV modeling is cluster-only; set use_cluster=True or use Sparse-vLLM.")
    cache_impl = get_deltakv_cache_impl(config)
    if cache_impl == DELTA_COMPRESSED_LATENT_WO_FULL:
        return DeltaCompressedLatentWoFullCache(config=config)
    if cache_impl == DELTA_COMPRESSED_LATENT_W_FULL:
        return DeltaCompressedLatentWFullCache(config=config)
    if cache_impl == DELTA_ORIGIN_WO_FULL:
        return DeltaOriginWoFullCache(config=config)
    if cache_impl == DELTA_ORIGIN_W_FULL:
        return DeltaOriginWFullCache(config=config)
    raise AssertionError(f"Unhandled deltakv_cache_impl={cache_impl!r}")
