from __future__ import annotations

from typing import Any

from deltakv.modeling.all_origin_residual_quant_cache import AllOriginResidualQuantClusterCompressedKVCache
from deltakv.modeling.kv_cache import ClusterCompressedKVCache, CompressedKVCache
from deltakv.modeling.origin_residual_quant_cache import (
    OriginResidualQuantClusterCompressedKVCache,
    OriginResidualQuantCompressedKVCache,
)


STANDARD_CACHE = "standard"
ORIGIN_RESIDUAL_QUANT_CACHE = "origin_residual_quant"
ALL_ORIGIN_RESIDUAL_QUANT_CACHE = "all_origin_residual_quant"

_VALID_CACHE_IMPLS = {
    STANDARD_CACHE,
    ORIGIN_RESIDUAL_QUANT_CACHE,
    ALL_ORIGIN_RESIDUAL_QUANT_CACHE,
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
    cache_impl = getattr(config, "deltakv_cache_impl", STANDARD_CACHE)
    cache_impl = STANDARD_CACHE if cache_impl is None else str(cache_impl).strip()
    if cache_impl == "":
        cache_impl = STANDARD_CACHE
    if cache_impl not in _VALID_CACHE_IMPLS:
        raise ValueError(
            f"Unknown deltakv_cache_impl={cache_impl!r}. "
            f"Expected one of {sorted(_VALID_CACHE_IMPLS)}."
        )
    return cache_impl


def _expected_cache_types(config: Any) -> tuple[type, ...]:
    cache_impl = get_deltakv_cache_impl(config)
    if cache_impl == STANDARD_CACHE:
        return (CompressedKVCache, ClusterCompressedKVCache)
    if cache_impl == ORIGIN_RESIDUAL_QUANT_CACHE:
        return (OriginResidualQuantCompressedKVCache, OriginResidualQuantClusterCompressedKVCache)
    if cache_impl == ALL_ORIGIN_RESIDUAL_QUANT_CACHE:
        return (AllOriginResidualQuantClusterCompressedKVCache,)
    raise AssertionError(f"Unhandled deltakv_cache_impl={cache_impl!r}")


def is_deltakv_cache_instance(past_key_values: Any, config: Any) -> bool:
    return isinstance(past_key_values, _expected_cache_types(config))


def create_deltakv_cache(config: Any):
    cache_impl = get_deltakv_cache_impl(config)
    if cache_impl == STANDARD_CACHE:
        return ClusterCompressedKVCache(config=config) if config.use_cluster else CompressedKVCache(config=config)
    if cache_impl == ORIGIN_RESIDUAL_QUANT_CACHE:
        return (
            OriginResidualQuantClusterCompressedKVCache(config=config)
            if config.use_cluster
            else OriginResidualQuantCompressedKVCache(config=config)
        )
    if cache_impl == ALL_ORIGIN_RESIDUAL_QUANT_CACHE:
        if not config.use_cluster:
            raise ValueError("all_origin_residual_quant cache requires use_cluster=True.")
        return AllOriginResidualQuantClusterCompressedKVCache(config=config)
    raise AssertionError(f"Unhandled deltakv_cache_impl={cache_impl!r}")
