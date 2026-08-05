"""Decode CUDA Graph normalization and validation."""

from collections.abc import Callable
from typing import Any

from sparsevllm.configs.common import _coerce_bool_config
from sparsevllm.method_registry import (
    DECODE_CUDA_GRAPH_SUPPORTED_METHODS,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
)
from sparsevllm.utils.log import log_once

def _default_decode_cuda_graph_capture_sizes(max_decoding_seqs: int) -> list[int]:
    max_decoding_seqs = int(max_decoding_seqs)
    if max_decoding_seqs <= 0:
        raise ValueError(f"max_decoding_seqs must be > 0, got {max_decoding_seqs}.")

    sizes: list[int] = []
    size = 1
    while size < max_decoding_seqs:
        sizes.append(size)
        size *= 2
    if not sizes or sizes[-1] != max_decoding_seqs:
        sizes.append(max_decoding_seqs)
    return sizes


def _resolve_positive_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    *,
    name: str,
    default_factory: Callable[[], list[int]],
) -> list[int]:
    if value is None:
        sizes = default_factory()
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            sizes = default_factory()
        else:
            try:
                sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be 'auto' or a comma-separated "
                    f"integer list, got {value!r}."
                ) from exc
    elif isinstance(value, int):
        sizes = [int(value)]
    elif isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value]
    else:
        raise ValueError(
            f"{name} must be 'auto', an int, a list/tuple of ints, "
            f"or None, got {type(value).__name__}."
        )

    sizes = sorted(set(sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"{name} must contain positive integers, got {sizes}.")
    return sizes


def _resolve_decode_cuda_graph_capture_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_decoding_seqs: int,
) -> list[int]:
    sizes = _resolve_positive_sizes(
        value,
        name="decode_cuda_graph_capture_sizes",
        default_factory=lambda: _default_decode_cuda_graph_capture_sizes(
            max_decoding_seqs
        ),
    )
    if sizes[-1] < int(max_decoding_seqs):
        raise ValueError(
            "decode_cuda_graph_capture_sizes must cover max_decoding_seqs: "
            f"max capture size {sizes[-1]} < max_decoding_seqs {int(max_decoding_seqs)}."
        )
    return sizes


def _default_decode_cuda_graph_context_sizes(max_model_len: int) -> list[int]:
    """Default decode graph context buckets: 1k, 2k, 4k, ... up to max_model_len."""
    max_model_len = int(max_model_len)
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be > 0, got {max_model_len}.")

    size = min(1024, max_model_len)
    sizes: list[int] = []
    while size < max_model_len:
        sizes.append(size)
        size *= 2
    sizes.append(size)
    return sorted(set(sizes))


def _resolve_decode_cuda_graph_context_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_model_len: int,
) -> list[int]:
    return _resolve_positive_sizes(
        value,
        name="decode_cuda_graph_context_sizes",
        default_factory=lambda: _default_decode_cuda_graph_context_sizes(max_model_len),
    )


def _normalize_decode_cuda_graph_context_policy(value: str | None) -> str:
    policy = str(value or "current").strip().lower()
    if policy in {"cur", "now"}:
        return "current"
    if policy in {"request", "final"}:
        return "requested"
    if policy not in {"current", "requested"}:
        raise ValueError(
            "decode_cuda_graph_context_policy must be 'current' or 'requested', "
            f"got {policy!r}."
        )
    return policy


def normalize_decode_cuda_graph(config, *, legacy_deltakv_graph_method: bool) -> None:
    if legacy_deltakv_graph_method:
        config.decode_cuda_graph = True
        config.decode_graph = True
    if config.decode_cuda_graph_max_cached_graphs is not None:
        config.decode_cuda_graph_max_cached_graphs = int(config.decode_cuda_graph_max_cached_graphs)
        if config.decode_cuda_graph_max_cached_graphs <= 0:
            raise ValueError(
                "decode_cuda_graph_max_cached_graphs must be a positive integer or None, "
                f"got {config.decode_cuda_graph_max_cached_graphs}."
            )
    if config.decode_cuda_graph_capture_sampling and not config.decode_cuda_graph:
        raise ValueError("decode_cuda_graph_capture_sampling requires decode_cuda_graph=True.")
    config.decode_cuda_graph_context_policy = _normalize_decode_cuda_graph_context_policy(
        config.decode_cuda_graph_context_policy
    )
    if config.decode_cuda_graph:
        if config.enable_prefix_caching:
            if config.decode_cuda_graph_capture_sampling:
                raise ValueError(
                    "prefix caching with decode_cuda_graph does not support "
                    "decode_cuda_graph_capture_sampling=True yet."
                )
        if config.tensor_parallel_size > 1:
            if config.decode_cuda_graph_capture_sampling:
                raise ValueError(
                    "decode_cuda_graph_capture_sampling is disabled when tensor_parallel_size > 1 "
                    "because TP workers do not materialize rank-0 gathered logits."
                )
            if not is_tp_decode_cuda_graph_supported(config.vllm_sparse_method):
                supported = ", ".join(
                    repr(method)
                    for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS)
                    if method and is_tp_decode_cuda_graph_supported(method)
                )
                raise ValueError(
                    "decode_cuda_graph with tensor_parallel_size > 1 supports these methods only: "
                    f"'', {supported}. DeltaKV is not supported."
                )
            log_once(
                "decode_cuda_graph with tensor_parallel_size > 1 uses TP-local sparse selection: "
                "each rank selects sparse tokens from its local heads/KV heads without cross-rank "
                "sparse-index aggregation, so sparse behavior is not guaranteed equivalent to TP=1 "
                "or global-head sparse selection.",
                level="WARNING",
            )
        elif not is_decode_cuda_graph_supported(config.vllm_sparse_method):
            supported = ", ".join(
                repr(method) for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS) if method
            )
            raise ValueError(f"decode_cuda_graph supports these methods only: '', {supported}.")
        config.decode_cuda_graph_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
            config.decode_cuda_graph_capture_sizes,
            config.max_decoding_seqs,
        )
        config.decode_cuda_graph_context_sizes = _resolve_decode_cuda_graph_context_sizes(
            config.decode_cuda_graph_context_sizes,
            config.max_model_len,
        )
    config.decode_graph = bool(config.decode_cuda_graph)
    config.decode_graph_capture_sampling = bool(config.decode_cuda_graph_capture_sampling)
    config.decode_graph_capture_sizes = config.decode_cuda_graph_capture_sizes
