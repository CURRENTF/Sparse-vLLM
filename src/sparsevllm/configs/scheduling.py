"""Batch-capacity and prefill-scheduling normalization."""

from sparsevllm.configs.common import (
    _coerce_optional_positive_int,
    _resolve_long_prefill_offload_threshold,
)
from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.method_registry import (
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    resolve_prefill_schedule_policy,
)
from sparsevllm.utils.log import log_once

def normalize_scheduling(config) -> None:
    config.max_num_seqs_in_batch = int(config.max_num_seqs_in_batch)
    if config.max_num_seqs_in_batch <= 0:
        raise ValueError(
            "max_num_seqs_in_batch must be > 0, "
            f"got {config.max_num_seqs_in_batch}."
        )
    config.max_decoding_seqs = int(config.max_decoding_seqs)
    if config.max_decoding_seqs <= 0:
        raise ValueError(
            f"max_decoding_seqs must be > 0, got {config.max_decoding_seqs}."
        )
    configured_max_num_seqs_in_gpu = _coerce_optional_positive_int(
        "max_num_seqs_in_gpu",
        config.max_num_seqs_in_gpu,
    )
    if configured_max_num_seqs_in_gpu is None:
        configured_max_num_seqs_in_gpu = max(
            config.max_num_seqs_in_batch * REDUNDANCY_BATCH_SIZE_FACTOR,
            config.max_decoding_seqs,
        )
    if configured_max_num_seqs_in_gpu < config.max_num_seqs_in_batch:
        raise ValueError(
            "max_num_seqs_in_gpu must be >= max_num_seqs_in_batch: "
            f"{configured_max_num_seqs_in_gpu} < {config.max_num_seqs_in_batch}."
        )
    if configured_max_num_seqs_in_gpu < config.max_decoding_seqs:
        raise ValueError(
            "max_num_seqs_in_gpu must be >= max_decoding_seqs: "
            f"{configured_max_num_seqs_in_gpu} < {config.max_decoding_seqs}."
        )
    config.max_num_seqs_in_gpu = int(configured_max_num_seqs_in_gpu)

    config.prefill_schedule_policy = resolve_prefill_schedule_policy(
        config.vllm_sparse_method,
        config.prefill_schedule_policy,
    )
    config.max_num_batched_tokens = int(config.max_num_batched_tokens)
    if config.max_num_batched_tokens <= 0:
        raise ValueError(
            "max_num_batched_tokens must be > 0, "
            f"got {config.max_num_batched_tokens}."
        )
    configured_chunk_prefill_size = (
        None if config.chunk_prefill_size is None else int(config.chunk_prefill_size)
    )
    if config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
        config.long_prefill_offload_threshold = _resolve_long_prefill_offload_threshold(
            config.long_prefill_offload_threshold
        )
        if (
            configured_chunk_prefill_size is not None
            and configured_chunk_prefill_size != config.long_prefill_offload_threshold
        ):
            log_once(
                "long_bs1full_short_batch derives chunk_prefill_size from "
                "long_prefill_offload_threshold; ignoring "
                f"chunk_prefill_size={configured_chunk_prefill_size} and using "
                f"{config.long_prefill_offload_threshold}.",
                level="WARNING",
            )
        config.chunk_prefill_size = config.long_prefill_offload_threshold
    else:
        resolved_offload_threshold = _coerce_optional_positive_int(
            "long_prefill_offload_threshold",
            config.long_prefill_offload_threshold,
        )
        if resolved_offload_threshold is None:
            raise ValueError("long_prefill_offload_threshold must be a positive integer.")
        config.long_prefill_offload_threshold = int(resolved_offload_threshold)
        config.chunk_prefill_size = (
            8192
            if configured_chunk_prefill_size is None
            else configured_chunk_prefill_size
        )
    if config.chunk_prefill_size <= 0:
        raise ValueError(
            f"chunk_prefill_size must be > 0, got {config.chunk_prefill_size}."
        )
    if (
        config.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
        and config.max_num_batched_tokens < config.chunk_prefill_size
    ):
        log_once(
            "long_bs1full_short_batch requires one short-boundary prefill to fit; "
            f"raising max_num_batched_tokens from {config.max_num_batched_tokens} "
            f"to {config.chunk_prefill_size}.",
            level="WARNING",
        )
        config.max_num_batched_tokens = config.chunk_prefill_size

    if int(config.mlp_chunk_size) <= 0:
        raise ValueError(f"mlp_chunk_size must be > 0, got {config.mlp_chunk_size}.")
    config.mlp_chunk_size = int(config.mlp_chunk_size)
