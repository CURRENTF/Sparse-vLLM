"""Decode CUDA Graph normalization and validation."""

from collections.abc import Callable
from typing import Any

from sparsevllm.configs.common import _coerce_bool_config
from sparsevllm.method_registry import (
    DECODE_CUDA_GRAPH_SUPPORTED_METHODS,
    decode_sparse_long_text_threshold,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
)
from sparsevllm.utils.log import log_once


def _default_decode_cuda_graph_capture_sizes(max_decoding_seqs: int) -> list[int]:
    """Return at most 32 batch buckets, dense where padding hurts most."""
    max_decoding_seqs = int(max_decoding_seqs)
    if max_decoding_seqs <= 0:
        raise ValueError(f"max_decoding_seqs must be > 0, got {max_decoding_seqs}.")

    dense_limit = min(8, max_decoding_seqs)
    sizes = list(range(1, dense_limit + 1))
    if max_decoding_seqs <= dense_limit:
        return sizes

    # Keep small decode batches exact, then use aligned, bounded-width buckets.
    # The adaptive stride caps the auto plan at 32 batch families even for a
    # very large scheduler limit; explicit capture sizes remain unrestricted.
    remaining_bucket_budget = 32 - dense_limit
    span = max_decoding_seqs - dense_limit
    stride = max(4, (span + remaining_bucket_budget - 1) // remaining_bucket_budget)
    stride = ((stride + 3) // 4) * 4
    sizes.extend(range(dense_limit + stride, max_decoding_seqs, stride))
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
        name="decode_graph_capture_sizes",
        default_factory=lambda: _default_decode_cuda_graph_capture_sizes(
            max_decoding_seqs
        ),
    )
    if sizes[-1] < int(max_decoding_seqs):
        raise ValueError(
            "decode_graph_capture_sizes must cover max_decoding_seqs: "
            f"max capture size {sizes[-1]} < max_decoding_seqs {int(max_decoding_seqs)}."
        )
    return sizes


def _select_decode_cuda_graph_batch_size(
    real_batch_size: int,
    capture_sizes: list[int] | tuple[int, ...],
) -> int:
    real_batch_size = int(real_batch_size)
    if real_batch_size <= 0:
        raise ValueError(
            f"decode batch size must be > 0, got {real_batch_size}."
        )
    sizes = sorted(set(int(size) for size in capture_sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(
            "decode_graph_capture_sizes must contain positive integers, "
            f"got {sizes}."
        )
    for size in sizes:
        if size >= real_batch_size:
            return size
    raise ValueError(
        "decode_cuda_graph capture sizes do not cover current decode batch: "
        f"batch_size={real_batch_size}, capture_sizes={sizes}."
    )


def _resolve_decode_static_batch_capacity(
    capture_sizes: list[int] | tuple[int, ...],
    *,
    max_num_seqs_in_batch: int,
    max_decoding_seqs: int,
) -> int:
    """Return the largest padded decode batch reachable by the scheduler."""

    max_real_batch_size = min(
        int(max_num_seqs_in_batch),
        int(max_decoding_seqs),
    )
    return _select_decode_cuda_graph_batch_size(
        max_real_batch_size,
        capture_sizes,
    )


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
    sizes.append(max_model_len)
    return sorted(set(sizes))


def _resolve_decode_cuda_graph_context_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_model_len: int,
) -> list[int]:
    return _resolve_positive_sizes(
        value,
        name="decode_graph_context_sizes",
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
            "decode_graph_context_policy must be 'current' or 'requested', "
            f"got {policy!r}."
        )
    return policy


def build_decode_cuda_graph_startup_plan(
    capture_sizes: list[int] | tuple[int, ...],
    context_sizes: list[int] | tuple[int, ...],
    limit: int,
    *,
    mandatory: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Select dense batch coverage and coarse context coverage within ``limit``."""
    batches = sorted(set(int(size) for size in capture_sizes))
    contexts = sorted(set(int(size) for size in context_sizes))
    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"decode_graph_startup_capture_limit must be positive, got {limit}.")
    if not batches or not contexts:
        return []
    if any(batch <= 0 for batch in batches) or any(context <= 0 for context in contexts):
        raise ValueError(
            "decode CUDA Graph startup buckets must be positive: "
            f"batch_sizes={batches}, context_sizes={contexts}."
        )
    if limit < len(batches):
        raise ValueError(
            "decode CUDA Graph startup capture limit must cover every batch bucket: "
            f"limit={limit}, batch_buckets={len(batches)}."
        )

    full_plan = [(batch, context) for batch in batches for context in contexts]
    if len(full_plan) <= limit:
        return full_plan

    # Every batch family gets its largest context first. Remaining quota is
    # biased toward smaller batches and spread over the existing power-of-two
    # context buckets. A missing exact context can still reuse that batch's
    # next larger captured graph, at the context-padding cost measured by the
    # benchmark rather than triggering a runtime capture.
    selected: list[tuple[int, int]] = []
    quotas = [limit // len(batches)] * len(batches)
    for idx in range(limit % len(batches)):
        quotas[idx] += 1
    for batch, quota in zip(batches, quotas):
        if quota <= 0:
            continue
        if quota >= len(contexts):
            chosen = contexts
        elif quota == 1:
            chosen = [contexts[-1]]
        else:
            indices = {
                round(idx * (len(contexts) - 1) / (quota - 1))
                for idx in range(quota)
            }
            chosen = [contexts[idx] for idx in sorted(indices)]
        selected.extend((batch, context) for context in chosen)

    if mandatory is not None:
        mandatory = (int(mandatory[0]), int(mandatory[1]))
    if mandatory is not None and mandatory in full_plan and mandatory not in selected:
        mandatory_batch = int(mandatory[0])
        replace_idx = next(
            (
                idx for idx, pair in enumerate(selected)
                if pair[0] == mandatory_batch
                and pair[1] != contexts[-1]
                and sum(selected_pair[0] == mandatory_batch for selected_pair in selected) > 1
            ),
            -1,
        )
        if replace_idx >= 0:
            selected[replace_idx] = mandatory

    plan = sorted(set(selected))
    if len(plan) != min(limit, len(full_plan)):
        raise RuntimeError(
            "decode CUDA Graph startup planner produced an incomplete plan: "
            f"expected={min(limit, len(full_plan))}, actual={len(plan)}."
        )
    missing_max_batches = [
        batch for batch in batches if (batch, contexts[-1]) not in plan
    ]
    if missing_max_batches:
        raise RuntimeError(
            "decode CUDA Graph startup plan must retain the largest context for "
            f"every batch bucket, missing={missing_max_batches}."
        )
    return plan


def build_decode_cuda_graph_startup_family_plan(config) -> list[tuple[int, int, bool]]:
    """Build graph keys largest-first so captures reuse the shared graph pool."""
    batches = sorted(set(int(size) for size in config.decode_graph_capture_sizes))
    contexts = sorted(set(int(size) for size in config.decode_graph_context_sizes))
    limit = min(
        int(config.decode_graph_startup_capture_limit),
        int(config.decode_graph_max_cached_graphs),
    )
    method = str(config.sparse_method or "")
    if not method:
        return sorted(
            (
                (batch, context, False)
                for batch, context in build_decode_cuda_graph_startup_plan(
                    batches,
                    contexts,
                    limit,
                )
            ),
            reverse=True,
        )

    threshold = decode_sparse_long_text_threshold(
        method,
        num_sink_tokens=config.sink_keep_tokens,
        decode_keep_tokens=config.decode_keep_tokens,
        num_recent_tokens=config.recent_keep_tokens,
    )
    family_contexts: list[tuple[bool, list[int]]] = []
    if threshold >= 2:
        family_contexts.append((False, contexts))
    if threshold + 2 <= int(config.max_model_len):
        long_contexts = [context for context in contexts if context > threshold]
        if long_contexts:
            family_contexts.append((True, long_contexts))
    if not family_contexts:
        raise ValueError(
            "No reachable sparse decode CUDA Graph family for startup capture: "
            f"method={method!r}, threshold={threshold}, max_model_len={config.max_model_len}."
        )

    lanes = [
        (batch, is_long_text, lane_contexts)
        for batch in batches
        for is_long_text, lane_contexts in family_contexts
    ]
    if limit < len(lanes):
        raise ValueError(
            "decode CUDA Graph sparse startup capture limit must cover every "
            "batch/family lane: "
            f"limit={limit}, required={len(lanes)}, batch_buckets={len(batches)}, "
            f"families={len(family_contexts)}."
        )

    full_plan = [
        (batch, context, is_long_text)
        for batch, is_long_text, lane_contexts in lanes
        for context in lane_contexts
    ]
    if len(full_plan) <= limit:
        return sorted(full_plan, reverse=True)

    target_size = min(limit, len(full_plan))
    quotas = [1] * len(lanes)
    remaining = target_size - len(lanes)
    while remaining > 0:
        progressed = False
        for lane_idx, (_, _, lane_contexts) in enumerate(lanes):
            if quotas[lane_idx] >= len(lane_contexts):
                continue
            quotas[lane_idx] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError(
                "decode CUDA Graph sparse startup planner could not allocate "
                f"remaining budget={remaining}."
            )
    selected: list[tuple[int, int, bool]] = []
    for (batch, is_long_text, lane_contexts), quota in zip(lanes, quotas):
        if quota >= len(lane_contexts):
            chosen = lane_contexts
        elif quota == 1:
            chosen = [lane_contexts[-1]]
        else:
            indices = {
                round(idx * (len(lane_contexts) - 1) / (quota - 1))
                for idx in range(quota)
            }
            chosen = [lane_contexts[idx] for idx in sorted(indices)]
        selected.extend(
            (batch, context, is_long_text) for context in chosen
        )
    plan = sorted(set(selected), reverse=True)
    if len(plan) != target_size:
        raise RuntimeError(
            "decode CUDA Graph sparse startup planner produced an incomplete plan: "
            f"expected={target_size}, actual={len(plan)}."
        )
    return plan


def normalize_decode_cuda_graph(config) -> None:
    if config.decode_graph_max_cached_graphs is not None:
        config.decode_graph_max_cached_graphs = int(config.decode_graph_max_cached_graphs)
        if config.decode_graph_max_cached_graphs <= 0:
            raise ValueError(
                "decode_graph_max_cached_graphs must be a positive integer or None, "
                f"got {config.decode_graph_max_cached_graphs}."
            )
    startup_capture_setting = config.decode_graph_startup_capture
    startup_capture_auto = startup_capture_setting is None
    if startup_capture_auto:
        config.decode_graph_startup_capture = bool(config.decode_graph)
    else:
        config.decode_graph_startup_capture = _coerce_bool_config(
            "decode_graph_startup_capture",
            startup_capture_setting,
        )
    if config.decode_graph_startup_capture_limit is None:
        config.decode_graph_startup_capture_limit = (
            48 if config.sparse_method else 32
        )
    config.decode_graph_startup_capture_limit = int(
        config.decode_graph_startup_capture_limit
    )
    if config.decode_graph_startup_capture_limit <= 0:
        raise ValueError(
            "decode_graph_startup_capture_limit must be a positive integer, "
            f"got {config.decode_graph_startup_capture_limit}."
        )
    if config.decode_graph_startup_capture:
        if not config.decode_graph:
            raise ValueError("decode_graph_startup_capture requires decode_graph=True.")
        if config.decode_graph_max_cached_graphs is None:
            config.decode_graph_max_cached_graphs = (
                config.decode_graph_startup_capture_limit
            )
    if config.decode_graph_capture_sampling and not config.decode_graph:
        raise ValueError("decode_graph_capture_sampling requires decode_graph=True.")
    config.decode_graph_context_policy = _normalize_decode_cuda_graph_context_policy(
        config.decode_graph_context_policy
    )
    context_sizes = config.decode_graph_context_sizes
    config.decode_graph_context_sizes_auto = context_sizes is None or (
        isinstance(context_sizes, str) and context_sizes.strip().lower() in {"", "auto"}
    )
    if config.decode_graph:
        if config.enable_prefix_caching:
            if config.decode_graph_capture_sampling:
                raise ValueError(
                    "prefix caching with decode_graph does not support "
                    "decode_graph_capture_sampling=True yet."
                )
        if config.tensor_parallel_size > 1:
            if config.decode_graph_capture_sampling:
                raise ValueError(
                    "decode_graph_capture_sampling is disabled when tensor_parallel_size > 1 "
                    "because TP workers do not materialize rank-0 gathered logits."
                )
            if not is_tp_decode_cuda_graph_supported(config.sparse_method):
                supported = ", ".join(
                    repr(method)
                    for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS)
                    if method and is_tp_decode_cuda_graph_supported(method)
                )
                raise ValueError(
                    "decode_graph with tensor_parallel_size > 1 supports these methods only: "
                    f"'', {supported}. DeltaKV is not supported."
                )
            log_once(
                "decode_graph with tensor_parallel_size > 1 uses TP-local sparse selection: "
                "each rank selects sparse tokens from its local heads/KV heads without cross-rank "
                "sparse-index aggregation, so sparse behavior is not guaranteed equivalent to TP=1 "
                "or global-head sparse selection.",
                level="WARNING",
            )
        elif not is_decode_cuda_graph_supported(config.sparse_method):
            supported = ", ".join(
                repr(method) for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS) if method
            )
            raise ValueError(f"decode_graph supports these methods only: '', {supported}.")
        config.decode_graph_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
            config.decode_graph_capture_sizes,
            config.max_decoding_seqs,
        )
        config.decode_graph_context_sizes = _resolve_decode_cuda_graph_context_sizes(
            config.decode_graph_context_sizes,
            config.max_model_len,
        )
        if config.decode_graph_startup_capture:
            startup_plan = build_decode_cuda_graph_startup_family_plan(config)
            log_once(
                "Decode CUDA Graph startup precapture enabled "
                f"({'default' if startup_capture_auto else 'explicit'}): "
                f"budget={config.decode_graph_startup_capture_limit}, "
                f"cache_limit={config.decode_graph_max_cached_graphs}, "
                f"planned_graphs={len(startup_plan)}, "
                f"batch_buckets={config.decode_graph_capture_sizes}, "
                f"context_buckets={config.decode_graph_context_sizes}."
            )
