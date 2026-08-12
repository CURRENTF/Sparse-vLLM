"""Model metadata loading and validation orchestration."""

import json
import os
from types import SimpleNamespace
from typing import Any

from transformers import AutoConfig

from sparsevllm.method_registry import (
    validate_model_runtime_compatibility,
    validate_sparse_method_assets,
)
from sparsevllm.models.checkpoint import validate_checkpoint
from sparsevllm.models.layout import RuntimeLayout
from sparsevllm.models.spec import ModelSpec, resolve_model_spec
from sparsevllm.quantization import QuantizationConfig
from sparsevllm.utils.config import config_get
from sparsevllm.utils.log import logger, log_once


def _config_to_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def _load_model_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as error:
        load_error = error
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError(
            "AutoConfig.from_pretrained failed and no config.json exists for an "
            f"explicit raw-config fallback. model={model_path} "
            f"error={type(load_error).__name__}: {load_error}"
        ) from load_error
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    model_type = str(config_get(raw_config, "model_type", "") or "")
    model_spec = resolve_model_spec(model_type)
    if not model_spec.allow_raw_config:
        raise RuntimeError(
            "AutoConfig.from_pretrained failed. Refusing to silently fall back to raw "
            f"config.json for {model_spec.name}. model={model_path} "
            f"error={type(load_error).__name__}: {load_error}"
        ) from load_error
    log_once(
        f"AutoConfig.from_pretrained failed for {model_spec.name}; loading raw "
        "config.json through its explicit model specification.",
        level="WARNING",
    )
    return _config_to_namespace(raw_config)


def _extract_text_config(config: Any) -> Any:
    text_config = config_get(config, "text_config", None)
    if text_config is None:
        return config
    if isinstance(text_config, dict):
        return _config_to_namespace(text_config)
    return text_config


def _finalize_model_config(config, model_spec: ModelSpec) -> None:
    config.runtime_layout = RuntimeLayout.from_config(
        config.hf_config,
        require_mixed=model_spec.mixed_attention,
    )
    if config.max_model_len > config.hf_config.max_position_embeddings:
        logger.warning('max_model_len > model.max_position_embeddings 输出可能不正常')
        config.hf_config.max_position_embeddings = config.max_model_len

    if config.max_num_seqs_in_batch > 32:
        logger.warning('max_num_seqs_in_batch 过大或许会占用太多显存')


def load_and_validate_model(config) -> None:
    validate_sparse_method_assets(config.vllm_sparse_method, config.model)
    if isinstance(config.deltakv_path, str):
        deltakv_path = config.deltakv_path.strip()
        config.deltakv_path = (
            None
            if deltakv_path.lower() in {"", "none", "null"}
            else deltakv_path
        )
    if config.tiny_random and config.vllm_sparse_method == "deltakv":
        raise NotImplementedError(
            "Tiny random mode does not support DeltaKV compressor weights yet."
        )
    config.outer_hf_config = _load_model_config(config.model)
    model_type = str(config_get(config.outer_hf_config, "model_type", "") or "")
    model_spec = resolve_model_spec(model_type)
    config.hf_config = _extract_text_config(config.outer_hf_config)
    config.model_spec = model_spec
    config.parallel_topology = model_spec.topology(
        config.tensor_parallel_size,
        config.expert_parallel_size,
        config.data_parallel_size,
    )
    if config.tiny_random:
        from sparsevllm.debug.tiny_random import apply_tiny_random_overrides

        if not model_spec.supports_tiny_random:
            raise NotImplementedError(
                f"Tiny random mode does not support {model_spec.name} yet."
            )
        config.tiny_random_overrides = apply_tiny_random_overrides(
            config.hf_config,
            config.tiny_random_config,
        )
        log_once(
            "TINY RANDOM MODE is enabled: checkpoint weights will not be read and "
            f"model-quality results are invalid. config={config.tiny_random_config} "
            f"seed={config.tiny_random_seed} overrides={config.tiny_random_overrides}",
            level="WARNING",
        )
    model_spec.validate_sharding(config.hf_config, config.parallel_topology)

    raw_quantization_config = config_get(
        config.hf_config,
        "quantization_config",
        config_get(config.outer_hf_config, "quantization_config", None),
    )
    config.quantization_config = QuantizationConfig.from_hf_config(
        raw_quantization_config,
        required_fp8=model_spec.requires_fp8,
        model_name=model_spec.name,
    )
    if config.tiny_random and config.quantization_config.enabled:
        raise NotImplementedError(
            "Tiny random mode does not support quantized model weights."
        )
    setattr(config.hf_config, "quantization_config", config.quantization_config)
    validate_checkpoint(
        model_type,
        outer_config=config.outer_hf_config,
        config=config.hf_config,
        raw_quantization_config=raw_quantization_config,
        quantization=config.quantization_config,
        topology=config.parallel_topology,
    )

    validate_model_runtime_compatibility(
        model_type=model_type,
        sparse_method=config.vllm_sparse_method,
        topology=config.parallel_topology,
        enforce_eager=config.enforce_eager,
        decode_cuda_graph=config.decode_cuda_graph,
        enable_prefix_caching=config.enable_prefix_caching,
    )
    _finalize_model_config(config, model_spec)
