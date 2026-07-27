"""Sparse-method normalization and layout-dependent validation."""

from sparsevllm.configs.delta import SUPPORTED_SKIPKV_MODEL_NAMES
from sparsevllm.method_registry import SUPPORTED_SPARSE_METHODS, normalize_sparse_method
from sparsevllm.utils.log import logger, log_once

def normalize_sparse_method_name(config) -> bool:
    raw_sparse_method = config.vllm_sparse_method
    raw_sparse_method_normalized = "" if raw_sparse_method is None else str(raw_sparse_method).strip().lower()
    legacy_deltakv_graph_method = raw_sparse_method_normalized in {
        "deltakv-less-memory-cudagraph",
        "deltakv_less_memory_cudagraph",
    }

    config.vllm_sparse_method = normalize_sparse_method(config.vllm_sparse_method)
    if config.vllm_sparse_method not in SUPPORTED_SPARSE_METHODS:
        supported = ", ".join(repr(method) for method in sorted(SUPPORTED_SPARSE_METHODS) if method)
        raise ValueError(
            f"Unsupported vllm_sparse_method={config.vllm_sparse_method!r}. "
            f"Supported methods: '', {supported}."
        )
    return legacy_deltakv_graph_method

def normalize_sparse_methods(config) -> None:
    if isinstance(config.full_attn_layers, str):
        layers = config.full_attn_layers.strip()
        config.full_attn_layers = [] if not layers else [int(x) for x in layers.split(",")]

    if config.quest_chunk_size <= 0:
        raise ValueError("quest_chunk_size 必须 > 0")
    config.quest_token_budget = 0
    if config.vllm_sparse_method == "quest":
        for name in ("num_sink_tokens", "decode_keep_tokens", "num_recent_tokens"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"QuEST {name} must be a non-negative integer, got {value!r}."
                )
        config.quest_token_budget = (
            config.num_sink_tokens
            + config.decode_keep_tokens
            + config.num_recent_tokens
        )
        if config.quest_token_budget <= 0:
            raise ValueError(
                "QuEST derived token budget must be > 0: "
                "num_sink_tokens + decode_keep_tokens + num_recent_tokens "
                f"= {config.quest_token_budget}."
            )
    if config.quest_skip_layers < 0:
        raise ValueError("quest_skip_layers 不能 < 0")
    config.h2o_decode_budget = int(config.h2o_decode_budget or 0)
    if config.h2o_decode_budget <= 0:
        raise ValueError(
            f"h2o_decode_budget must be > 0, got {config.h2o_decode_budget}."
        )
    config.h2o_prefill_budget = int(config.h2o_prefill_budget or 0)
    if config.h2o_prefill_budget < config.h2o_decode_budget:
        raise ValueError(
            "h2o_prefill_budget must be >= h2o_decode_budget, "
            f"got prefill={config.h2o_prefill_budget} decode={config.h2o_decode_budget}."
        )
    config.h2o_recent_ratio = float(config.h2o_recent_ratio)
    if not 0.0 < config.h2o_recent_ratio < 1.0:
        raise ValueError(
            f"h2o_recent_ratio must be in (0, 1), got {config.h2o_recent_ratio}."
        )
    config.h2o_prefill_score_window = int(config.h2o_prefill_score_window or 0)
    if not 1 <= config.h2o_prefill_score_window <= 128:
        raise ValueError(
            "h2o_prefill_score_window must be in [1, 128] because the prefill "
            f"score kernel supports at most 128 query tokens, got {config.h2o_prefill_score_window}."
        )
    config.rkv_compression_interval = int(config.rkv_compression_interval or 0)
    if config.rkv_compression_interval <= 0:
        raise ValueError(
            f"rkv_compression_interval must be > 0, got {config.rkv_compression_interval}."
        )
    config.rkv_observation_tokens = int(config.rkv_observation_tokens or 0)
    if config.rkv_observation_tokens <= 0:
        raise ValueError(
            f"rkv_observation_tokens must be > 0, got {config.rkv_observation_tokens}."
        )
    if config.rkv_observation_tokens > 128:
        raise ValueError(
            "rkv_observation_tokens must be <= 128 because the prefill score kernel "
            f"supports at most 128 query tokens, got {config.rkv_observation_tokens}."
        )
    if config.rkv_observation_tokens > config.rkv_compression_interval:
        raise ValueError(
            "rkv_observation_tokens must be <= rkv_compression_interval so the query cache "
            "can be refreshed between decode evictions, "
            f"got observation={config.rkv_observation_tokens} interval={config.rkv_compression_interval}."
        )
    config.rkv_alpha = float(config.rkv_alpha)
    if not 0.0 <= config.rkv_alpha <= 1.0:
        raise ValueError(f"rkv_alpha must be in [0, 1], got {config.rkv_alpha}.")
    config.rkv_similarity_threshold = float(config.rkv_similarity_threshold)
    if not 0.0 <= config.rkv_similarity_threshold <= 1.0:
        raise ValueError(
            "rkv_similarity_threshold must be in [0, 1], "
            f"got {config.rkv_similarity_threshold}."
        )
    config.rkv_recent_similar_keep = int(config.rkv_recent_similar_keep)
    if config.rkv_recent_similar_keep < 0:
        raise ValueError(
            f"rkv_recent_similar_keep must be >= 0, got {config.rkv_recent_similar_keep}."
        )
    config.rkv_max_redundancy_tokens = int(config.rkv_max_redundancy_tokens or 0)
    if config.rkv_max_redundancy_tokens <= 0:
        raise ValueError(
            f"rkv_max_redundancy_tokens must be > 0, got {config.rkv_max_redundancy_tokens}."
        )
    config.rkv_redundancy_window = int(config.rkv_redundancy_window or 0)
    if config.rkv_redundancy_window < 0:
        raise ValueError(
            f"rkv_redundancy_window must be >= 0, got {config.rkv_redundancy_window}."
        )
    if 0 < config.rkv_redundancy_window > config.rkv_max_redundancy_tokens:
        raise ValueError(
            "rkv_redundancy_window must be <= rkv_max_redundancy_tokens, "
            f"got window={config.rkv_redundancy_window} max={config.rkv_max_redundancy_tokens}."
        )
    if config.vllm_sparse_method == "rkv":
        log_once(
            "R-KV support is an approximation of the official implementation: "
            "Sparse-VLLM uses one shared physical token index set across KV heads, "
            "so official per-KV-head token selection is not fully reproduced. "
            f"rkv_redundancy_window={config.rkv_redundancy_window}; values > 0 score "
            "redundancy only over the trailing candidate tokens.",
            level="WARNING",
        )
    config.skipkv_compression_interval = int(config.skipkv_compression_interval or 0)
    if config.skipkv_compression_interval <= 0:
        raise ValueError(
            "skipkv_compression_interval must be > 0, "
            f"got {config.skipkv_compression_interval}."
        )
    config.skipkv_alpha = float(config.skipkv_alpha)
    if config.skipkv_alpha < 0.0:
        raise ValueError(f"skipkv_alpha must be >= 0, got {config.skipkv_alpha}.")
    config.skipkv_similarity_threshold = float(config.skipkv_similarity_threshold)
    if not 0.0 <= config.skipkv_similarity_threshold <= 1.0:
        raise ValueError(
            "skipkv_similarity_threshold must be in [0, 1], "
            f"got {config.skipkv_similarity_threshold}."
        )
    config.skipkv_segment_size = int(config.skipkv_segment_size or 0)
    if config.skipkv_segment_size <= 0:
        raise ValueError(f"skipkv_segment_size must be > 0, got {config.skipkv_segment_size}.")
    config.skipkv_max_redundancy_tokens = int(config.skipkv_max_redundancy_tokens or 0)
    if config.skipkv_max_redundancy_tokens <= 0:
        raise ValueError(
            "skipkv_max_redundancy_tokens must be > 0, "
            f"got {config.skipkv_max_redundancy_tokens}."
        )
    config.skipkv_redundancy_window = int(config.skipkv_redundancy_window or 0)
    if config.skipkv_redundancy_window <= 0:
        raise ValueError(
            "skipkv_redundancy_window must be > 0, "
            f"got {config.skipkv_redundancy_window}."
        )
    if config.skipkv_redundancy_window > config.skipkv_max_redundancy_tokens:
        raise ValueError(
            "skipkv_redundancy_window must be <= skipkv_max_redundancy_tokens, "
            f"got window={config.skipkv_redundancy_window} max={config.skipkv_max_redundancy_tokens}."
        )
    config.skipkv_enable_sentence_scoring = bool(config.skipkv_enable_sentence_scoring)
    config.skipkv_sentence_score_weight = float(config.skipkv_sentence_score_weight)
    if config.skipkv_sentence_score_weight < 0.0:
        raise ValueError(
            "skipkv_sentence_score_weight must be >= 0, "
            f"got {config.skipkv_sentence_score_weight}."
        )
    config.skipkv_sentence_min_tokens = int(config.skipkv_sentence_min_tokens or 0)
    if config.skipkv_sentence_min_tokens <= 0:
        raise ValueError(
            "skipkv_sentence_min_tokens must be > 0, "
            f"got {config.skipkv_sentence_min_tokens}."
        )
    config.skipkv_sentence_max_tokens = int(config.skipkv_sentence_max_tokens or 0)
    if config.skipkv_sentence_max_tokens < config.skipkv_sentence_min_tokens:
        raise ValueError(
            "skipkv_sentence_max_tokens must be >= skipkv_sentence_min_tokens, "
            f"got max={config.skipkv_sentence_max_tokens} min={config.skipkv_sentence_min_tokens}."
        )
    config.skipkv_sentence_embedding_layer = int(config.skipkv_sentence_embedding_layer)
    config.skipkv_max_tracked_sentences = int(config.skipkv_max_tracked_sentences or 0)
    if config.skipkv_max_tracked_sentences <= 0:
        raise ValueError(
            "skipkv_max_tracked_sentences must be > 0, "
            f"got {config.skipkv_max_tracked_sentences}."
        )
    config.skipkv_enable_activation_steering = bool(config.skipkv_enable_activation_steering)
    config.skipkv_steering_layer = int(config.skipkv_steering_layer)
    config.skipkv_steering_alpha = float(config.skipkv_steering_alpha)
    config.skipkv_steering_alpha_increment = float(config.skipkv_steering_alpha_increment)
    config.skipkv_steering_alpha_max = float(config.skipkv_steering_alpha_max)
    if config.skipkv_enable_activation_steering and not config.skipkv_steering_vector_path:
        raise ValueError(
            "skipkv_enable_activation_steering=True requires skipkv_steering_vector_path. "
            "Official SkipKV support is limited to the released steering vectors for "
            f"{', '.join(sorted(SUPPORTED_SKIPKV_MODEL_NAMES))}."
        )

def finalize_sparse_layout(config) -> None:
    configured_full_layers = {int(layer) for layer in config.full_attn_layers}
    kv_layers = tuple(int(layer) for layer in config.runtime_layout.kv_idx_to_layer_idx)
    kv_positions = {layer: index for index, layer in enumerate(kv_layers)}
    unknown_full_layers = sorted(configured_full_layers - set(kv_layers))
    if unknown_full_layers and config.vllm_sparse_method in {"omnikv", "deltakv"}:
        raise ValueError(
            "full_attn_layers must contain KV/full-attention layer indices for "
            f"{config.vllm_sparse_method}; non-KV layers={unknown_full_layers}."
        )
    config.obs_layer_ids = []
    for layer in config.full_attn_layers:
        layer = int(layer)
        kv_position = kv_positions.get(layer)
        if kv_position is None or kv_position + 1 >= len(kv_layers):
            continue
        if kv_layers[kv_position + 1] not in configured_full_layers:
            config.obs_layer_ids.append(layer)

    # PyramidKV 配置验证与智能生成
    if 'pyramidkv' == config.vllm_sparse_method:
        num_layers = int(config.runtime_layout.num_layers)
        num_kv_layers = int(config.runtime_layout.num_kv_layers)
        if config.pyramid_layer_ratios is None:
            start_l = int(config.pyramidkv_start_layer)
            least_l = (
                int(config.pyramidkv_least_layer)
                if config.pyramidkv_least_layer is not None
                else num_kv_layers - 1
            )
            start_r = float(config.pyramidkv_start_ratio)
            least_r = float(config.pyramidkv_least_ratio)
            if not 0 <= start_l < num_kv_layers:
                raise ValueError(
                    f"pyramidkv_start_layer must be a KV layer position in [0, {num_kv_layers}), "
                    f"got {start_l}."
                )
            if not start_l <= least_l < num_kv_layers:
                raise ValueError(
                    "pyramidkv_least_layer must be a KV layer position between "
                    f"start_layer={start_l} and {num_kv_layers - 1}, got {least_l}."
                )

            ratios = [1.0] * num_kv_layers
            for i in range(start_l, num_kv_layers):
                if i <= least_l:
                    if least_l > start_l:
                        ratio = start_r - (start_r - least_r) * (i - start_l) / (least_l - start_l)
                    else:
                        ratio = least_r
                    ratios[i] = ratio
                else:
                    ratios[i] = least_r
            config.pyramid_layer_ratios = ratios
            logger.info(f"PyramidKV 自动生成 KV layer_ratios = {[f'{r:.3f}' for r in ratios]}")
        else:
            ratios = [float(ratio) for ratio in config.pyramid_layer_ratios]
            if len(ratios) == num_layers and num_layers != num_kv_layers:
                ratios = [ratios[layer_idx] for layer_idx in config.runtime_layout.kv_idx_to_layer_idx]
            config.pyramid_layer_ratios = ratios

    if config.pyramid_layer_ratios is not None:
        # PyramidKV 模式自动启用 SnapKV 逻辑
        if 'pyramidkv' != config.vllm_sparse_method:
            raise ValueError('vllm_sparse_method 应为 pyramidkv')

        num_kv_layers = int(config.runtime_layout.num_kv_layers)
        if len(config.pyramid_layer_ratios) != num_kv_layers:
            raise ValueError(
                f"pyramid_layer_ratios length ({len(config.pyramid_layer_ratios)}) must equal "
                f"the number of KV/full-attention layers ({num_kv_layers})."
            )

        if any(r <= 0 or r > 1.0 for r in config.pyramid_layer_ratios):
            raise ValueError("pyramid_layer_ratios 的所有值必须在 (0, 1.0] 范围内")
