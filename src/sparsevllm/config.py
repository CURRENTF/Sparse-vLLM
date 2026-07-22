import importlib.util
import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Union

import torch
from transformers import AutoConfig

from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.method_registry import (
    DECODE_CUDA_GRAPH_SUPPORTED_METHODS,
    PREFILL_POLICY_AUTO,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    PREFIX_CACHE_SUPPORTED_METHODS,
    SUPPORTED_SPARSE_METHODS,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
    normalize_sparse_method,
    resolve_prefill_schedule_policy,
    validate_model_runtime_compatibility,
)
from sparsevllm.engine.prefix_cache import resolve_prefix_cache_block_size
from sparsevllm.utils.log import logger, log_once

try:
    from transformers import Qwen3Config
except ImportError:
    Qwen3Config = AutoConfig


def _coerce_bool_config(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or explicit true/false string, got {value!r}.")


def _coerce_optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw.isdecimal():
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        parsed = int(raw)
    else:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {parsed}.")
    return parsed


def _resolve_long_prefill_offload_threshold(configured: Any) -> int:
    raw = os.getenv("SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS")
    legacy_raw = os.getenv("SPARSEVLLM_DEFERRED_PREFILL_MIN_TOKENS")
    if raw is not None and legacy_raw is not None and raw != legacy_raw:
        raise ValueError(
            "SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS and "
            "SPARSEVLLM_DEFERRED_PREFILL_MIN_TOKENS are both set with different values."
        )
    value = raw if raw is not None else legacy_raw
    resolved = _coerce_optional_positive_int(
        "long_prefill_offload_threshold",
        configured if value is None else value,
    )
    if resolved is None:
        raise ValueError("long_prefill_offload_threshold must be a positive integer.")
    return int(resolved)


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_deltakv_sparse_decode_backend(value: Any) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in {"auto", "custom", "fa2"}:
        raise ValueError(
            "deltakv_sparse_decode_backend must be one of 'auto', 'custom', or 'fa2', "
            f"got {value!r}."
        )
    if backend == "auto":
        resolved = "fa2" if _flash_attn_available() else "custom"
        reason = "flash_attn available" if resolved == "fa2" else "flash_attn not available"
        log_once(
            f"DeltaKV sparse decode backend auto-selected {resolved!r} ({reason}).",
            level="INFO",
        )
        return resolved
    if backend == "fa2" and not _flash_attn_available():
        raise ValueError(
            "deltakv_sparse_decode_backend='fa2' requires the flash_attn package; "
            "use 'custom' or leave it as 'auto' when flash_attn is not installed."
        )
    return backend


SUPPORTED_SKIPKV_MODEL_NAMES = frozenset(
    {
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
    }
)


def _model_path_basename(model_path: str) -> str:
    return str(model_path).rstrip("/").split("/")[-1]


def _default_decode_cuda_graph_capture_sizes(max_decoding_seqs: int) -> list[int]:
    max_decoding_seqs = int(max_decoding_seqs)
    if max_decoding_seqs <= 0:
        raise ValueError(f"max_decoding_seqs must be > 0, got {max_decoding_seqs}.")

    sizes: list[int] = []
    size = 1
    while size < max_decoding_seqs:
        sizes.append(size)
        size *= 2
    sizes.append(size)
    return sizes


def _resolve_decode_cuda_graph_capture_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_decoding_seqs: int,
) -> list[int]:
    if value is None:
        sizes = _default_decode_cuda_graph_capture_sizes(max_decoding_seqs)
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            sizes = _default_decode_cuda_graph_capture_sizes(max_decoding_seqs)
        else:
            try:
                sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    "decode_cuda_graph_capture_sizes must be 'auto' or a comma-separated "
                    f"integer list, got {value!r}."
                ) from exc
    elif isinstance(value, int):
        sizes = [int(value)]
    elif isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value]
    else:
        raise ValueError(
            "decode_cuda_graph_capture_sizes must be 'auto', an int, a list/tuple of ints, "
            f"or None, got {type(value).__name__}."
        )

    sizes = sorted(set(sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"decode_cuda_graph_capture_sizes must contain positive integers, got {sizes}.")
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
    if value is None:
        sizes = _default_decode_cuda_graph_context_sizes(max_model_len)
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            sizes = _default_decode_cuda_graph_context_sizes(max_model_len)
        else:
            try:
                sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    "decode_cuda_graph_context_sizes must be 'auto' or a comma-separated "
                    f"integer list, got {value!r}."
                ) from exc
    elif isinstance(value, int):
        sizes = [int(value)]
    elif isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value]
    else:
        raise ValueError(
            "decode_cuda_graph_context_sizes must be 'auto', an int, a list/tuple of ints, "
            f"or None, got {type(value).__name__}."
        )

    sizes = sorted(set(sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"decode_cuda_graph_context_sizes must contain positive integers, got {sizes}.")
    return sizes


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


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_to_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def _load_raw_qwen35_config(model_path: str, error: Exception) -> SimpleNamespace:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError(
            "AutoConfig.from_pretrained failed and no config.json exists for explicit "
            f"qwen3_5 fallback. model={model_path} error={type(error).__name__}: {error}"
        ) from error
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    if not _is_qwen35_outer_config(raw_config):
        raise RuntimeError(
            "AutoConfig.from_pretrained failed. Refusing to silently fall back to raw "
            f"`config.json` for non-qwen3_5 model. model={model_path} "
            f"error={type(error).__name__}: {error}"
        ) from error
    log_once(
        "AutoConfig.from_pretrained failed for qwen3_5/qwen3_6; loading raw config.json "
        "through Sparse-vLLM's explicit mixed-runtime parser.",
        level="WARNING",
    )
    return _config_to_namespace(raw_config)


def _coerce_int_list(name: str, value: Any, *, allow_none: bool = False) -> list[int] | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required.")
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"{name} must be a list/tuple of ints or a comma-separated string, got {value!r}.")


def _attention_type_is_full(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"full", "full_attention", "attention", "self_attention", "sliding_attention"}


def _attention_type_is_linear(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"linear", "linear_attention", "recurrent", "recurrent_attention", "gated_delta", "gated_delta_net"}


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = False
    quant_method: str = ""
    weight_dtype: str = ""
    activation_scheme: str = ""
    weight_block_size: tuple[int, int] | None = None
    backend: str = "auto"
    model_name: str = "qwen3_5"

    @classmethod
    def disabled(cls) -> "QuantizationConfig":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        payload: dict[str, Any] = {
            "quant_method": self.quant_method,
            "fmt": self.weight_dtype,
            "activation_scheme": self.activation_scheme,
            "backend": self.backend,
        }
        if self.weight_block_size is not None:
            payload["weight_block_size"] = list(self.weight_block_size)
        return payload

    @classmethod
    def from_hf_config(
        cls,
        value: Any,
        *,
        required_fp8: bool = False,
        model_name: str = "qwen3_5",
    ) -> "QuantizationConfig":
        if value is None:
            if required_fp8:
                raise ValueError(
                    f"{model_name} requires FP8 quantization_config; "
                    "BF16/FP16 fallback is not supported."
                )
            return cls.disabled()

        quant_method = str(
            _config_get(value, "quant_method", _config_get(value, "method", ""))
            or ""
        ).strip().lower()
        if quant_method not in {"fp8", "fbgemm_fp8"}:
            if required_fp8:
                raise ValueError(
                    f"{model_name} requires quantization_config.quant_method='fp8', "
                    f"got {quant_method!r}."
                )
            return cls.disabled()

        weight_dtype = str(
            _config_get(
                value,
                "weight_dtype",
                _config_get(value, "fmt", _config_get(value, "format", "e4m3")),
            )
            or ""
        ).strip().lower()
        if "e4m3" not in weight_dtype:
            raise ValueError(
                f"Sparse-vLLM {model_name} FP8 supports e4m3 weights only, "
                f"got weight_dtype={weight_dtype!r}."
            )

        activation_scheme = str(
            _config_get(value, "activation_scheme", _config_get(value, "activation", "dynamic"))
            or ""
        ).strip().lower()
        if activation_scheme != "dynamic":
            raise ValueError(
                f"Sparse-vLLM {model_name} FP8 supports dynamic activation only, "
                f"got activation_scheme={activation_scheme!r}."
            )

        block_size = _config_get(
            value,
            "weight_block_size",
            _config_get(value, "weight_block_shape", _config_get(value, "block_size", (128, 128))),
        )
        if isinstance(block_size, int):
            block_tuple = (int(block_size), int(block_size))
        elif isinstance(block_size, (list, tuple)) and len(block_size) == 2:
            block_tuple = (int(block_size[0]), int(block_size[1]))
        else:
            raise ValueError(f"weight_block_size must be a pair, got {block_size!r}.")
        if block_tuple != (128, 128):
            raise ValueError(
                f"Sparse-vLLM {model_name} FP8 supports "
                "weight_block_size=(128, 128) only, "
                f"got {block_tuple}."
            )

        backend = str(_config_get(value, "backend", "auto") or "auto").strip().lower()
        return cls(
            enabled=True,
            quant_method="fp8",
            weight_dtype="e4m3",
            activation_scheme="dynamic",
            weight_block_size=block_tuple,
            backend=backend,
            model_name=model_name,
        )


_MINIMAX_M2_FIXED_FIELDS = {
    "vocab_size": 200064,
    "hidden_size": 3072,
    "intermediate_size": 1536,
    "num_hidden_layers": 62,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rotary_dim": 64,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "max_position_embeddings": 204800,
    "shared_intermediate_size": 0,
    "mtp_transformer_layers": 1,
    "num_mtp_modules": 3,
}


def _validate_minimax_m2_checkpoint_config(
    hf_config: Any,
    raw_quantization_config: Any,
) -> None:
    architectures = tuple(_config_get(hf_config, "architectures", ()) or ())
    if architectures != ("MiniMaxM2ForCausalLM",):
        raise ValueError(
            "MiniMax M2.7 requires architectures=['MiniMaxM2ForCausalLM'], "
            f"got {list(architectures)}."
        )
    for field_name, expected in _MINIMAX_M2_FIXED_FIELDS.items():
        actual = _config_get(hf_config, field_name, None)
        if actual != expected:
            raise ValueError(
                f"MiniMax M2.7 requires {field_name}={expected!r}, got {actual!r}."
            )

    expected_values = {
        "hidden_act": "silu",
        "qk_norm_type": "per_layer",
        "scoring_func": "sigmoid",
        "use_qk_norm": True,
        "use_routing_bias": True,
        "use_mtp": True,
        "tie_word_embeddings": False,
    }
    for field_name, expected in expected_values.items():
        actual = _config_get(hf_config, field_name, None)
        if actual != expected:
            raise ValueError(
                f"MiniMax M2.7 requires {field_name}={expected!r}, got {actual!r}."
            )

    configured_dtype = _config_get(hf_config, "torch_dtype", None)
    if configured_dtype is None:
        configured_dtype = _config_get(hf_config, "dtype", None)
    if configured_dtype not in {torch.bfloat16, "bfloat16"}:
        raise ValueError(
            "MiniMax M2.7 requires BF16 non-quantized parameters, "
            f"got dtype={configured_dtype!r}."
        )

    excluded_modules = {
        str(name)
        for name in (
            _config_get(raw_quantization_config, "modules_to_not_convert", ()) or ()
        )
    }
    required_exclusions = {"gate", "e_score_correction_bias", "lm_head"}
    missing_exclusions = sorted(required_exclusions - excluded_modules)
    if missing_exclusions:
        raise ValueError(
            "MiniMax M2.7 quantization_config must exclude gate, "
            "e_score_correction_bias, and lm_head; missing "
            f"{missing_exclusions}."
        )


@dataclass(frozen=True)
class RuntimeLayout:
    num_layers: int
    num_kv_layers: int
    full_attention_layer_indices: tuple[int, ...]
    linear_attention_layer_indices: tuple[int, ...]
    layer_idx_to_kv_idx: tuple[int | None, ...]
    kv_idx_to_layer_idx: tuple[int, ...]

    @classmethod
    def dense(cls, num_layers: int) -> "RuntimeLayout":
        num_layers = int(num_layers)
        layers = tuple(range(num_layers))
        return cls(
            num_layers=num_layers,
            num_kv_layers=num_layers,
            full_attention_layer_indices=layers,
            linear_attention_layer_indices=(),
            layer_idx_to_kv_idx=tuple(range(num_layers)),
            kv_idx_to_layer_idx=layers,
        )

    @classmethod
    def from_config(cls, hf_config: Any, *, require_mixed: bool = False) -> "RuntimeLayout":
        num_layers = int(_config_get(hf_config, "num_hidden_layers"))
        layer_types = _config_get(hf_config, "layer_types", None)
        full_layers = _coerce_int_list(
            "full_attention_layer_indices",
            _config_get(
                hf_config,
                "full_attention_layer_indices",
                _config_get(hf_config, "attention_layer_indices", None),
            ),
            allow_none=True,
        )
        linear_layers = _coerce_int_list(
            "linear_attention_layer_indices",
            _config_get(hf_config, "linear_attention_layer_indices", None),
            allow_none=True,
        )

        if layer_types is not None:
            if len(layer_types) != num_layers:
                raise ValueError(
                    f"runtime layer_types length must equal num_hidden_layers: "
                    f"{len(layer_types)} != {num_layers}."
                )
            inferred_full: list[int] = []
            inferred_linear: list[int] = []
            for idx, layer_type in enumerate(layer_types):
                if _attention_type_is_full(layer_type):
                    inferred_full.append(idx)
                elif _attention_type_is_linear(layer_type):
                    inferred_linear.append(idx)
                else:
                    raise ValueError(f"Unsupported qwen3_5 layer_types[{idx}]={layer_type!r}.")
            full_layers = inferred_full if full_layers is None else full_layers
            linear_layers = inferred_linear if linear_layers is None else linear_layers

        if full_layers is None and linear_layers is None:
            if require_mixed:
                raise ValueError(
                    "qwen3_5 requires a mixed attention layer map: provide layer_types or "
                    "full_attention_layer_indices/linear_attention_layer_indices."
                )
            return cls.dense(num_layers)
        if full_layers is None:
            linear_set = set(linear_layers or [])
            full_layers = [idx for idx in range(num_layers) if idx not in linear_set]
        if linear_layers is None:
            full_set = set(full_layers or [])
            linear_layers = [idx for idx in range(num_layers) if idx not in full_set]

        full_tuple = tuple(sorted(int(idx) for idx in full_layers))
        linear_tuple = tuple(sorted(int(idx) for idx in linear_layers))
        full_set = set(full_tuple)
        linear_set = set(linear_tuple)
        expected = set(range(num_layers))
        if full_set & linear_set:
            overlap = sorted(full_set & linear_set)
            raise ValueError(f"RuntimeLayout full and linear layer sets overlap: {overlap}.")
        if full_set | linear_set != expected:
            missing = sorted(expected - (full_set | linear_set))
            extra = sorted((full_set | linear_set) - expected)
            raise ValueError(f"RuntimeLayout layer map is incomplete: missing={missing}, extra={extra}.")

        raw_layer_to_kv = _config_get(hf_config, "layer_idx_to_kv_idx", None)
        if raw_layer_to_kv is None:
            layer_to_kv: list[int | None] = [None] * num_layers
            for kv_idx, layer_idx in enumerate(full_tuple):
                layer_to_kv[layer_idx] = kv_idx
        else:
            if len(raw_layer_to_kv) != num_layers:
                raise ValueError(
                    "layer_idx_to_kv_idx length must equal num_hidden_layers: "
                    f"{len(raw_layer_to_kv)} != {num_layers}."
                )
            layer_to_kv = []
            for idx, value in enumerate(raw_layer_to_kv):
                if value is None or int(value) < 0:
                    layer_to_kv.append(None)
                else:
                    layer_to_kv.append(int(value))
            for layer_idx in linear_tuple:
                if layer_to_kv[layer_idx] is not None:
                    raise ValueError(
                        f"layer_idx_to_kv_idx[{layer_idx}] must be None/-1 for linear_attention layers."
                    )

        kv_pairs = [(kv_idx, layer_idx) for layer_idx, kv_idx in enumerate(layer_to_kv) if kv_idx is not None]
        if len(kv_pairs) != len(full_tuple):
            raise ValueError(
                "RuntimeLayout must assign exactly one KV index to each full_attention layer: "
                f"full_layers={len(full_tuple)} assigned={len(kv_pairs)}."
            )
        kv_pairs.sort()
        kv_indices = [kv_idx for kv_idx, _ in kv_pairs]
        if kv_indices != list(range(len(kv_pairs))):
            raise ValueError(f"KV layer indices must be contiguous from 0, got {kv_indices}.")
        kv_tuple = tuple(layer_idx for _, layer_idx in kv_pairs)

        configured_num_kv_layers = _config_get(hf_config, "num_kv_layers", None)
        if configured_num_kv_layers is not None and int(configured_num_kv_layers) != len(kv_tuple):
            raise ValueError(
                f"num_kv_layers={configured_num_kv_layers} does not match full_attention layers={len(kv_tuple)}."
            )
        return cls(
            num_layers=num_layers,
            num_kv_layers=len(kv_tuple),
            full_attention_layer_indices=full_tuple,
            linear_attention_layer_indices=linear_tuple,
            layer_idx_to_kv_idx=tuple(layer_to_kv),
            kv_idx_to_layer_idx=kv_tuple,
        )

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_idx_to_kv_idx[int(layer_idx)] is not None

    def is_linear_attention(self, layer_idx: int) -> bool:
        return self.layer_idx_to_kv_idx[int(layer_idx)] is None

    def kv_layer_index(self, layer_idx: int) -> int:
        layer_idx = int(layer_idx)
        kv_idx = self.layer_idx_to_kv_idx[layer_idx]
        if kv_idx is None:
            raise RuntimeError(f"layer_idx={layer_idx} is linear_attention and has no KV cache")
        return int(kv_idx)


def _is_qwen35_outer_config(config: Any) -> bool:
    return str(_config_get(config, "model_type", "") or "").strip().lower() in {"qwen3_5", "qwen3_6"}


def _extract_text_config(config: Any) -> Any:
    text_config = _config_get(config, "text_config", None)
    if text_config is None:
        return config
    if isinstance(text_config, dict):
        return _config_to_namespace(text_config)
    return text_config


def _qwen35_deltakv_message() -> str:
    return (
        "DeltaKV for qwen3_5 requires a qwen3_5-compatible deltakv_path. "
        "Use vllm_sparse_method='' to run quantized vanilla inference."
    )


def _is_qwen35_deltakv_checkpoint(path: str | None) -> bool:
    if path is None or not os.path.exists(path):
        return False
    config_path = os.path.join(path, "config.json") if os.path.isdir(path) else None
    if config_path is None or not os.path.isfile(config_path):
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        checkpoint_config = json.load(f)
    candidates = [
        checkpoint_config.get("model_type"),
        checkpoint_config.get("base_model_type"),
        checkpoint_config.get("target_model_type"),
        checkpoint_config.get("runtime_model_type"),
    ]
    return any(str(value).strip().lower() in {"qwen3_5", "qwen3_6"} for value in candidates if value)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs_in_batch: int = 32  # 不能设置太大
    max_model_len: int = 128_000
    max_decoding_seqs: int = 64
    max_num_seqs_in_gpu: int | None = None

    chunk_prefill_size: int | None = None
    long_prefill_offload_threshold: int = 96 * 1024
    mlp_chunk_size: int = 16384
    prefill_schedule_policy: str = PREFILL_POLICY_AUTO
    gpu_memory_utilization: float = 0.8
    device_memory_utilization: float | None = None
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    # Total host-side I/O worker budget shared by all distributed ranks.
    weight_loading_workers: int = 8
    moe_backend: str = "triton"
    enforce_eager: bool = True
    hf_config: Union[Qwen3Config, AutoConfig] | None = None
    outer_hf_config: Any | None = None
    runtime_layout: RuntimeLayout | None = None
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig.disabled)
    eos: int = -1
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    num_kvcache_slots: int | list = -1

    # Sparse Attention Config
    vllm_sparse_method: str = ""  # "", "streamingllm", "snapkv", "pyramidkv", "omnikv", "quest", "rkv", "skipkv", "deltakv"; legacy deltakv-less-memory aliases normalize to deltakv.

    # Prefix Cache Config
    enable_prefix_caching: bool = False
    prefix_cache_block_size: int | None = None
    prefix_cache_max_blocks: int | None = None
    recurrent_state_max_bytes: int | None = None
    prefix_cache_max_recurrent_bytes: int | None = None
    prefix_cache_salt: str = ""

    # General Sparse Config
    num_sink_tokens: int = 64
    num_recent_tokens: int = 512
    decode_keep_tokens: int = 4096

    # OmniKV Config
    obs_layer_ids: list[int] = field(default=None, init=False)
    full_attn_layers: str | list[int] = "0" # useful for omnikv

    # Decode CUDA Graph Config
    decode_cuda_graph: bool = False
    decode_cuda_graph_capture_sampling: bool = False
    decode_cuda_graph_capture_sizes: str | int | list[int] | tuple[int, ...] | None = "auto"
    # Static decode/CUDA graph context buckets.  The default produces
    # 1024, 2048, 4096, ... up to max_model_len so a short request never
    # inherits a previously captured 128k graph.
    decode_cuda_graph_context_sizes: str | int | list[int] | tuple[int, ...] | None = "auto"
    # "current" buckets by the real current decode length; "requested" keeps
    # the old final-length capacity behavior for compatibility/debugging.
    decode_cuda_graph_context_policy: str = "current"
    # Optional LRU cap for captured graphs.  None keeps all bucketed graphs.
    decode_cuda_graph_max_cached_graphs: int | None = None
    sparse_attn_score_dtype: str = "float32"
    decode_graph: bool | None = None
    decode_graph_capture_sampling: bool | None = None
    decode_graph_capture_sizes: str | int | list[int] | tuple[int, ...] | None = None

    # QuEST Config
    quest_chunk_size: int = 16
    quest_token_budget: int = 1024
    quest_skip_layers: int = 2

    # SnapKV Config
    snapkv_window_size: int = 32
    snapkv_num_full_layers: int = 0  # 前多少层不进行驱逐

    # R-KV Config
    rkv_compression_interval: int = 128
    rkv_observation_tokens: int = 8
    rkv_alpha: float = 0.1
    rkv_similarity_threshold: float = 0.8
    rkv_recent_similar_keep: int = 1
    rkv_max_redundancy_tokens: int = 4096
    # 0 means score the full candidate set, matching R-KV's budget-candidate selection.
    # Positive values are an explicit speed/quality approximation.
    rkv_redundancy_window: int = 0

    # SkipKV Config.
    skipkv_compression_interval: int = 128
    skipkv_alpha: float = 0.1
    skipkv_similarity_threshold: float = 0.95
    skipkv_segment_size: int = 32
    skipkv_max_redundancy_tokens: int = 4096
    skipkv_redundancy_window: int = 64
    skipkv_enable_sentence_scoring: bool = True
    skipkv_sentence_score_weight: float = 1.0
    skipkv_sentence_min_tokens: int = 4
    skipkv_sentence_max_tokens: int = 256
    skipkv_sentence_embedding_layer: int = -1
    skipkv_max_tracked_sentences: int = 256
    skipkv_enable_activation_steering: bool = False
    skipkv_steering_vector_path: str | None = None
    skipkv_steering_layer: int = -1
    skipkv_steering_alpha: float = 0.0
    skipkv_steering_alpha_increment: float = 0.0
    skipkv_steering_alpha_max: float = 0.0
    
    # PyramidKV Config
    pyramid_layer_ratios: list[float] | None = None  # 每层的 KV budget 比例
    pyramidkv_start_layer: int = 0
    pyramidkv_start_ratio: float = 0.6
    pyramidkv_least_layer: int | None = None
    pyramidkv_least_ratio: float = 0.01

    # DeltaKV Config
    deltakv_path: str | None = None
    deltakv_k_neighbors: int = 4
    cluster_ratio: float = 0.1
    cluster_metric: str = 'l2'  # 'l2', 'dot', 'cosine', 'fastdot' (approx; fastest)
    cluster_on_kv: bool = True
    use_compression: bool = True
    kv_compressed_size: int = 128
    kv_quant_bits: int = 4
    kv_quant_group_size: int = 0
    enable_sparse_ref_fp8: bool = False
    # Optional residual quantization for layers listed in full_attn_layers.
    # 0 keeps the previous BF16/FP16 full-layer KV behavior. 2/4 store old
    # full-layer tokens as DeltaKV-style residuals and reconstruct a dense view
    # before full attention.
    full_layer_kv_quant_bits: int = 0
    full_layer_cluster_ratio: float = 0.0  # <=0 reuses cluster_ratio
    full_layer_kivi_group_size: int = 32
    full_layer_kivi_residual_length: int = 32
    full_layer_kivi_decode_block_seq: int = 256
    full_layer_kivi_decode_block_n: int = 16
    full_layer_kivi_decode_num_warps: int = 2
    full_layer_kivi_decode_num_stages: int = 3
    enable_full_layer_kivi_quant: bool = True
    enable_full_layer_kivi_fused_decode: bool = False
    enable_full_layer_kivi_grouped_decode: bool = False
    # Legacy config compatibility only. Full-layer KIVI decode uses the direct
    # packed backend whenever KIVI quantization is enabled.
    enable_full_layer_kivi_dense_decode: bool = False
    pool_kernel_size: int = 1
    # Legacy symmetric compressor controls (for backward compatibility).
    use_nonlinear_compressor: bool = True
    compressor_intermediate_size: int = 2048
    compressor_linear_bias: bool = True
    # New directional compressor controls (match latest DeltaKV training code).
    # Values: auto|linear|mlp_gelu|mlp_swiglu
    compressor_down_type: str = "auto"
    compressor_up_type: str = "auto"
    compressor_down_intermediate_size: int = -1
    compressor_up_intermediate_size: int = -1
    # DeltaKV memory split: reserve a fraction of available KV memory for the sparse full-KV pool
    # (centers + buffer + temp reconstruction slots). Larger values reduce full/latent capacity
    # but improve robustness at large batch sizes / long contexts.
    deltakv_full_pool_reserve_ratio: float = 0.1
    # Cap DeltaKV packed-cache capacity to the configured resident
    # token budget instead of consuming all available memory. This keeps long
    # context experiments reproducible and leaves workspace memory for kernels.
    deltakv_cache_capacity_margin: float = 1.05
    # Extra center slots above the exact fixed-stride center-policy estimate.
    # This avoids allocating cluster_ratio * max_tokens plus no scheduling margin.
    deltakv_center_capacity_margin: float = 1.5
    # Triton kernels: group multiple KV heads per program to reduce redundant loads.
    deltakv_triton_gather_heads_per_program: int = 4
    deltakv_triton_reconstruct_heads_per_program: int = 4
    deltakv_triton_materialize_block_tokens: int = 16
    deltakv_sparse_decode_backend: str = "auto"
    deltakv_cluster_gather_chunk_size: int = 16384
    
    enable_profiler: bool = False
    throughput_log_interval_s: float = 10.0
    allow_missing_deltakv_path: bool = False
    allow_unknown_config_keys: bool = False

    @property
    def world_size(self) -> int:
        return (
            int(self.tensor_parallel_size)
            * int(self.expert_parallel_size)
            * int(self.data_parallel_size)
        )

    @property
    def weight_loading_workers_per_rank(self) -> int:
        return max(1, self.weight_loading_workers // self.world_size)

    def _normalize_platform_aliases(self):
        if self.device_memory_utilization is not None:
            self.gpu_memory_utilization = float(self.device_memory_utilization)
        self.device_memory_utilization = float(self.gpu_memory_utilization)

        if self.decode_graph is not None:
            self.decode_cuda_graph = _coerce_bool_config("decode_graph", self.decode_graph)
        else:
            self.decode_cuda_graph = _coerce_bool_config("decode_cuda_graph", self.decode_cuda_graph)
        self.decode_graph = bool(self.decode_cuda_graph)

        if self.decode_graph_capture_sampling is not None:
            self.decode_cuda_graph_capture_sampling = _coerce_bool_config(
                "decode_graph_capture_sampling",
                self.decode_graph_capture_sampling,
            )
        else:
            self.decode_cuda_graph_capture_sampling = _coerce_bool_config(
                "decode_cuda_graph_capture_sampling",
                self.decode_cuda_graph_capture_sampling,
            )
        self.decode_graph_capture_sampling = bool(self.decode_cuda_graph_capture_sampling)

        if self.decode_graph_capture_sizes is not None:
            self.decode_cuda_graph_capture_sizes = self.decode_graph_capture_sizes

    def __post_init__(self):
        if os.getenv("PROFILER_SVLLM"):
            self.enable_profiler = True

        raw_sparse_method = self.vllm_sparse_method
        raw_sparse_method_normalized = "" if raw_sparse_method is None else str(raw_sparse_method).strip().lower()
        legacy_deltakv_graph_method = raw_sparse_method_normalized in {
            "deltakv-less-memory-cudagraph",
            "deltakv_less_memory_cudagraph",
        }

        self.vllm_sparse_method = normalize_sparse_method(self.vllm_sparse_method)
        if self.vllm_sparse_method not in SUPPORTED_SPARSE_METHODS:
            supported = ", ".join(repr(method) for method in sorted(SUPPORTED_SPARSE_METHODS) if method)
            raise ValueError(
                f"Unsupported vllm_sparse_method={self.vllm_sparse_method!r}. "
                f"Supported methods: '', {supported}."
            )
        self.enable_prefix_caching = _coerce_bool_config("enable_prefix_caching", self.enable_prefix_caching)
        self.max_num_seqs_in_batch = int(self.max_num_seqs_in_batch)
        if self.max_num_seqs_in_batch <= 0:
            raise ValueError(
                "max_num_seqs_in_batch must be > 0, "
                f"got {self.max_num_seqs_in_batch}."
            )
        self.max_decoding_seqs = int(self.max_decoding_seqs)
        if self.max_decoding_seqs <= 0:
            raise ValueError(
                f"max_decoding_seqs must be > 0, got {self.max_decoding_seqs}."
            )
        configured_max_num_seqs_in_gpu = _coerce_optional_positive_int(
            "max_num_seqs_in_gpu",
            self.max_num_seqs_in_gpu,
        )
        if configured_max_num_seqs_in_gpu is None:
            configured_max_num_seqs_in_gpu = max(
                self.max_num_seqs_in_batch * REDUNDANCY_BATCH_SIZE_FACTOR,
                self.max_decoding_seqs,
            )
        if configured_max_num_seqs_in_gpu < self.max_num_seqs_in_batch:
            raise ValueError(
                "max_num_seqs_in_gpu must be >= max_num_seqs_in_batch: "
                f"{configured_max_num_seqs_in_gpu} < {self.max_num_seqs_in_batch}."
            )
        if configured_max_num_seqs_in_gpu < self.max_decoding_seqs:
            raise ValueError(
                "max_num_seqs_in_gpu must be >= max_decoding_seqs: "
                f"{configured_max_num_seqs_in_gpu} < {self.max_decoding_seqs}."
            )
        self.max_num_seqs_in_gpu = int(configured_max_num_seqs_in_gpu)
        self.prefix_cache_block_size = _coerce_optional_positive_int(
            "prefix_cache_block_size",
            self.prefix_cache_block_size,
        )
        self.prefix_cache_max_blocks = _coerce_optional_positive_int(
            "prefix_cache_max_blocks",
            self.prefix_cache_max_blocks,
        )
        recurrent_state_max_bytes = _coerce_optional_positive_int(
            "recurrent_state_max_bytes",
            self.recurrent_state_max_bytes,
        )
        self.prefix_cache_max_recurrent_bytes = _coerce_optional_positive_int(
            "prefix_cache_max_recurrent_bytes",
            self.prefix_cache_max_recurrent_bytes,
        )
        if self.prefix_cache_max_recurrent_bytes is not None:
            log_once(
                "prefix_cache_max_recurrent_bytes is deprecated; use "
                "recurrent_state_max_bytes instead. The budget is an explicit "
                "hard limit for the live recurrent-state pool.",
                level="WARNING",
            )
            if (
                recurrent_state_max_bytes is not None
                and recurrent_state_max_bytes != self.prefix_cache_max_recurrent_bytes
            ):
                raise ValueError(
                    "conflicting recurrent state budgets: "
                    f"recurrent_state_max_bytes={recurrent_state_max_bytes} and "
                    "prefix_cache_max_recurrent_bytes="
                    f"{self.prefix_cache_max_recurrent_bytes}."
                )
            recurrent_state_max_bytes = self.prefix_cache_max_recurrent_bytes
        self.recurrent_state_max_bytes = recurrent_state_max_bytes
        if self.enable_prefix_caching and self.vllm_sparse_method not in PREFIX_CACHE_SUPPORTED_METHODS:
            raise ValueError("prefix caching only supports vanilla, omnikv, quest.")
        self.prefix_cache_salt = str(self.prefix_cache_salt or "")
        self.prefill_schedule_policy = resolve_prefill_schedule_policy(
            self.vllm_sparse_method,
            self.prefill_schedule_policy,
        )
        self.max_num_batched_tokens = int(self.max_num_batched_tokens)
        if self.max_num_batched_tokens <= 0:
            raise ValueError(
                "max_num_batched_tokens must be > 0, "
                f"got {self.max_num_batched_tokens}."
            )
        configured_chunk_prefill_size = (
            None if self.chunk_prefill_size is None else int(self.chunk_prefill_size)
        )
        if self.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
            self.long_prefill_offload_threshold = _resolve_long_prefill_offload_threshold(
                self.long_prefill_offload_threshold
            )
            if (
                configured_chunk_prefill_size is not None
                and configured_chunk_prefill_size != self.long_prefill_offload_threshold
            ):
                log_once(
                    "long_bs1full_short_batch derives chunk_prefill_size from "
                    "long_prefill_offload_threshold; ignoring "
                    f"chunk_prefill_size={configured_chunk_prefill_size} and using "
                    f"{self.long_prefill_offload_threshold}.",
                    level="WARNING",
                )
            self.chunk_prefill_size = self.long_prefill_offload_threshold
        else:
            resolved_offload_threshold = _coerce_optional_positive_int(
                "long_prefill_offload_threshold",
                self.long_prefill_offload_threshold,
            )
            if resolved_offload_threshold is None:
                raise ValueError("long_prefill_offload_threshold must be a positive integer.")
            self.long_prefill_offload_threshold = int(resolved_offload_threshold)
            self.chunk_prefill_size = (
                8192
                if configured_chunk_prefill_size is None
                else configured_chunk_prefill_size
            )
        if self.chunk_prefill_size <= 0:
            raise ValueError(
                f"chunk_prefill_size must be > 0, got {self.chunk_prefill_size}."
            )
        if (
            self.prefill_schedule_policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
            and self.max_num_batched_tokens < self.chunk_prefill_size
        ):
            log_once(
                "long_bs1full_short_batch requires one short-boundary prefill to fit; "
                f"raising max_num_batched_tokens from {self.max_num_batched_tokens} "
                f"to {self.chunk_prefill_size}.",
                level="WARNING",
            )
            self.max_num_batched_tokens = self.chunk_prefill_size
        
        if int(self.mlp_chunk_size) <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}.")
        self.mlp_chunk_size = int(self.mlp_chunk_size)
        if int(self.deltakv_cluster_gather_chunk_size) <= 0:
            raise ValueError(
                "deltakv_cluster_gather_chunk_size must be > 0, "
                f"got {self.deltakv_cluster_gather_chunk_size}."
            )
        self.deltakv_cluster_gather_chunk_size = int(self.deltakv_cluster_gather_chunk_size)
        self.sparse_attn_score_dtype = str(self.sparse_attn_score_dtype or "float32").strip().lower()
        if self.sparse_attn_score_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                "sparse_attn_score_dtype must be 'float32', 'bfloat16', or 'float16', "
                f"got {self.sparse_attn_score_dtype!r}."
            )
        self.full_layer_kv_quant_bits = int(self.full_layer_kv_quant_bits or 0)
        if self.full_layer_kv_quant_bits not in (0, 2, 4):
            raise ValueError(
                "full_layer_kv_quant_bits must be 0, 2, or 4, "
                f"got {self.full_layer_kv_quant_bits}."
            )
        self.full_layer_cluster_ratio = float(self.full_layer_cluster_ratio or 0.0)
        if self.full_layer_cluster_ratio < 0.0:
            raise ValueError(f"full_layer_cluster_ratio must be >= 0, got {self.full_layer_cluster_ratio}.")
        self.kv_quant_bits = int(self.kv_quant_bits or 0)
        if self.kv_quant_bits not in (0, 2, 4):
            raise ValueError(f"kv_quant_bits must be 0, 2, or 4, got {self.kv_quant_bits}.")
        self.kv_quant_group_size = int(self.kv_quant_group_size or 0)
        if self.kv_quant_group_size < 0:
            raise ValueError(f"kv_quant_group_size must be >= 0, got {self.kv_quant_group_size}.")
        self.full_layer_kivi_group_size = int(self.full_layer_kivi_group_size or 32)
        if self.full_layer_kivi_group_size <= 0:
            raise ValueError(
                "full_layer_kivi_group_size must be > 0, "
                f"got {self.full_layer_kivi_group_size}."
            )
        self.full_layer_kivi_residual_length = int(
            self.full_layer_kivi_residual_length or self.full_layer_kivi_group_size
        )
        if self.full_layer_kivi_residual_length <= 0:
            raise ValueError(
                "full_layer_kivi_residual_length must be > 0, "
                f"got {self.full_layer_kivi_residual_length}."
            )
        self.full_layer_kivi_decode_block_seq = int(self.full_layer_kivi_decode_block_seq or 256)
        if self.full_layer_kivi_decode_block_seq <= 0 or self.full_layer_kivi_decode_block_seq % 16 != 0:
            raise ValueError(
                "full_layer_kivi_decode_block_seq must be a positive multiple of 16, "
                f"got {self.full_layer_kivi_decode_block_seq}."
            )
        self.full_layer_kivi_decode_block_n = int(self.full_layer_kivi_decode_block_n or 16)
        if self.full_layer_kivi_decode_block_n <= 0 or self.full_layer_kivi_decode_block_n % 16 != 0:
            raise ValueError(
                "full_layer_kivi_decode_block_n must be a positive multiple of 16, "
                f"got {self.full_layer_kivi_decode_block_n}."
            )
        self.full_layer_kivi_decode_num_warps = int(self.full_layer_kivi_decode_num_warps or 2)
        if self.full_layer_kivi_decode_num_warps not in {1, 2, 4, 8}:
            raise ValueError(
                "full_layer_kivi_decode_num_warps must be one of 1, 2, 4, or 8, "
                f"got {self.full_layer_kivi_decode_num_warps}."
            )
        self.full_layer_kivi_decode_num_stages = int(self.full_layer_kivi_decode_num_stages or 3)
        if self.full_layer_kivi_decode_num_stages <= 0:
            raise ValueError(
                "full_layer_kivi_decode_num_stages must be > 0, "
                f"got {self.full_layer_kivi_decode_num_stages}."
            )
        self.enable_full_layer_kivi_fused_decode = bool(self.enable_full_layer_kivi_fused_decode)
        self.enable_full_layer_kivi_grouped_decode = bool(self.enable_full_layer_kivi_grouped_decode)
        self.enable_full_layer_kivi_dense_decode = bool(self.enable_full_layer_kivi_dense_decode)
        if self.enable_full_layer_kivi_fused_decode:
            raise ValueError(
                "enable_full_layer_kivi_fused_decode was removed; full-layer KIVI decode now "
                "uses the direct packed backend."
            )
        if self.enable_full_layer_kivi_grouped_decode:
            raise ValueError(
                "enable_full_layer_kivi_grouped_decode was removed; full-layer KIVI decode now "
                "uses the direct packed backend."
            )
        self.deltakv_full_pool_reserve_ratio = float(self.deltakv_full_pool_reserve_ratio or 0.0)
        if self.deltakv_full_pool_reserve_ratio < 0.0 or self.deltakv_full_pool_reserve_ratio >= 1.0:
            raise ValueError(
                "deltakv_full_pool_reserve_ratio must be in [0, 1), "
                f"got {self.deltakv_full_pool_reserve_ratio}."
            )
        self.deltakv_cache_capacity_margin = float(self.deltakv_cache_capacity_margin or 1.0)
        if self.deltakv_cache_capacity_margin < 1.0:
            raise ValueError(
                "deltakv_cache_capacity_margin must be >= 1.0, "
                f"got {self.deltakv_cache_capacity_margin}."
            )
        self.deltakv_center_capacity_margin = float(self.deltakv_center_capacity_margin or 1.0)
        if self.deltakv_center_capacity_margin < 1.0:
            raise ValueError(
                "deltakv_center_capacity_margin must be >= 1.0, "
                f"got {self.deltakv_center_capacity_margin}."
            )

        if not os.path.isdir(self.model):
            raise FileNotFoundError(f"Model directory does not exist: {self.model}")
        if self.vllm_sparse_method == "skipkv":
            model_name = _model_path_basename(self.model)
            if model_name not in SUPPORTED_SKIPKV_MODEL_NAMES:
                supported = ", ".join(sorted(SUPPORTED_SKIPKV_MODEL_NAMES))
                raise ValueError(
                    "SkipKV is supported only for the official models with released steering vectors: "
                    f"{supported}. Got model basename {model_name!r} from model path {self.model!r}."
                )
        self.tensor_parallel_size = int(self.tensor_parallel_size)
        self.expert_parallel_size = int(self.expert_parallel_size)
        self.data_parallel_size = int(self.data_parallel_size)
        self.weight_loading_workers = int(self.weight_loading_workers)
        self.moe_backend = str(self.moe_backend or "").strip().lower()
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError(f"tensor_parallel_size must be in [1, 8], got {self.tensor_parallel_size}.")
        if self.expert_parallel_size <= 0:
            raise ValueError(
                f"expert_parallel_size must be positive, got {self.expert_parallel_size}."
            )
        if self.data_parallel_size <= 0:
            raise ValueError(
                f"data_parallel_size must be positive, got {self.data_parallel_size}."
            )
        if self.weight_loading_workers <= 0:
            raise ValueError(
                "weight_loading_workers must be positive, "
                f"got {self.weight_loading_workers}."
            )
        if self.moe_backend not in {"pytorch", "native", "triton"}:
            raise ValueError(
                "moe_backend must be 'pytorch', 'native', or 'triton', "
                f"got {self.moe_backend!r}."
            )
        self._normalize_platform_aliases()
        if legacy_deltakv_graph_method:
            self.decode_cuda_graph = True
            self.decode_graph = True
        if self.decode_cuda_graph_max_cached_graphs is not None:
            self.decode_cuda_graph_max_cached_graphs = int(self.decode_cuda_graph_max_cached_graphs)
            if self.decode_cuda_graph_max_cached_graphs <= 0:
                raise ValueError(
                    "decode_cuda_graph_max_cached_graphs must be a positive integer or None, "
                    f"got {self.decode_cuda_graph_max_cached_graphs}."
                )
        if self.decode_cuda_graph_capture_sampling and not self.decode_cuda_graph:
            raise ValueError("decode_cuda_graph_capture_sampling requires decode_cuda_graph=True.")
        self.decode_cuda_graph_context_policy = _normalize_decode_cuda_graph_context_policy(
            self.decode_cuda_graph_context_policy
        )
        if self.decode_cuda_graph:
            if self.enable_prefix_caching:
                if self.decode_cuda_graph_capture_sampling:
                    raise ValueError(
                        "prefix caching with decode_cuda_graph does not support "
                        "decode_cuda_graph_capture_sampling=True yet."
                    )
            if self.tensor_parallel_size > 1:
                if self.decode_cuda_graph_capture_sampling:
                    raise ValueError(
                        "decode_cuda_graph_capture_sampling is disabled when tensor_parallel_size > 1 "
                        "because TP workers do not materialize rank-0 gathered logits."
                    )
                if not is_tp_decode_cuda_graph_supported(self.vllm_sparse_method):
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
            elif not is_decode_cuda_graph_supported(self.vllm_sparse_method):
                supported = ", ".join(
                    repr(method) for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS) if method
                )
                raise ValueError(f"decode_cuda_graph supports these methods only: '', {supported}.")
            self.decode_cuda_graph_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
                self.decode_cuda_graph_capture_sizes,
                self.max_decoding_seqs,
            )
            self.decode_cuda_graph_context_sizes = _resolve_decode_cuda_graph_context_sizes(
                self.decode_cuda_graph_context_sizes,
                self.max_model_len,
            )
        self.decode_graph = bool(self.decode_cuda_graph)
        self.decode_graph_capture_sampling = bool(self.decode_cuda_graph_capture_sampling)
        self.decode_graph_capture_sizes = self.decode_cuda_graph_capture_sizes
        if isinstance(self.deltakv_path, str):
            deltakv_path = self.deltakv_path.strip()
            self.deltakv_path = None if deltakv_path.lower() in {"", "none", "null"} else deltakv_path
        try:
            self.outer_hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        except Exception as e:
            self.outer_hf_config = _load_raw_qwen35_config(self.model, e)
        is_qwen35 = _is_qwen35_outer_config(self.outer_hf_config)
        self.hf_config = _extract_text_config(self.outer_hf_config)
        if is_qwen35:
            setattr(self.hf_config, "model_type", "qwen3_5")
        model_type = str(getattr(self.hf_config, "model_type", "") or "")
        is_minimax_m2 = model_type == "minimax_m2"
        if self.moe_backend == "native" and not is_minimax_m2:
            raise ValueError(
                "moe_backend='native' is supported only for MiniMax M2.7, "
                f"got model_type={model_type!r}."
            )
        raw_quantization_config = _config_get(
            self.hf_config,
            "quantization_config",
            _config_get(self.outer_hf_config, "quantization_config", None),
        )
        self.quantization_config = QuantizationConfig.from_hf_config(
            raw_quantization_config,
            required_fp8=is_qwen35 or is_minimax_m2,
            model_name="MiniMax M2.7" if is_minimax_m2 else "qwen3_5",
        )
        setattr(self.hf_config, "quantization_config", self.quantization_config)
        if is_minimax_m2:
            _validate_minimax_m2_checkpoint_config(
                self.hf_config,
                raw_quantization_config,
            )
        if getattr(self.hf_config, "model_type", "") in {"deepseek_v2", "deepseek_v32"}:
            raise NotImplementedError(
                f"Unsupported Sparse-vLLM model_type={self.hf_config.model_type!r}. "
                "Supported model types: qwen2, qwen3, qwen3_5, llama."
            )
        if model_type == "qwen3_moe":
            if self.tensor_parallel_size != 1 or self.data_parallel_size != 1:
                raise ValueError(
                    "Qwen3MoE v1 only supports TP=1 and DP=1, got "
                    f"TP={self.tensor_parallel_size}, EP={self.expert_parallel_size}, "
                    f"DP={self.data_parallel_size}."
                )
            num_experts = int(getattr(self.hf_config, "num_experts", 0) or 0)
            if num_experts <= 0:
                raise ValueError(f"Qwen3MoE requires a positive num_experts, got {num_experts}.")
            if self.expert_parallel_size > num_experts:
                raise ValueError(
                    "expert_parallel_size must not exceed num_experts, "
                    f"got EP={self.expert_parallel_size}, num_experts={num_experts}."
                )
            if num_experts % self.expert_parallel_size != 0:
                raise ValueError(
                    "Qwen3MoE requires num_experts divisible by expert_parallel_size, "
                    f"got num_experts={num_experts}, EP={self.expert_parallel_size}."
                )
            top_k = int(getattr(self.hf_config, "num_experts_per_tok", 0) or 0)
            if not 1 <= top_k <= num_experts:
                raise ValueError(
                    "Qwen3MoE num_experts_per_tok must be in [1, num_experts], "
                    f"got top_k={top_k}, num_experts={num_experts}."
                )
            decoder_sparse_step = int(
                getattr(self.hf_config, "decoder_sparse_step", 1)
            )
            mlp_only_layers = tuple(
                int(layer_idx)
                for layer_idx in (getattr(self.hf_config, "mlp_only_layers", ()) or ())
            )
            if decoder_sparse_step != 1 or mlp_only_layers:
                raise NotImplementedError(
                    "Qwen3MoE v1 requires every decoder layer to be MoE, got "
                    f"decoder_sparse_step={decoder_sparse_step}, "
                    f"mlp_only_layers={list(mlp_only_layers)}."
                )
            shared_intermediate_size = int(
                getattr(self.hf_config, "shared_expert_intermediate_size", 0) or 0
            )
            if shared_intermediate_size != 0:
                raise NotImplementedError(
                    "Qwen3MoE v1 does not support shared experts, got "
                    f"shared_expert_intermediate_size={shared_intermediate_size}."
                )
            if self.quantization_config.enabled:
                raise NotImplementedError(
                    "Qwen3MoE v1 supports BF16/FP16 expert weights only; quantized MoE is unsupported."
                )
            model_dtype = getattr(self.hf_config, "torch_dtype", None)
            if model_dtype not in {torch.bfloat16, torch.float16}:
                raise NotImplementedError(
                    "Qwen3MoE v1 supports BF16/FP16 checkpoints only, "
                    f"got torch_dtype={model_dtype}."
                )
            validate_model_runtime_compatibility(
                model_type=model_type,
                sparse_method=self.vllm_sparse_method,
                tensor_parallel_size=self.tensor_parallel_size,
                expert_parallel_size=self.expert_parallel_size,
                data_parallel_size=self.data_parallel_size,
                enforce_eager=self.enforce_eager,
                decode_cuda_graph=self.decode_cuda_graph,
                enable_prefix_caching=self.enable_prefix_caching,
            )
        elif model_type == "minimax_m2":
            if self.tensor_parallel_size != 1 or self.data_parallel_size != 1:
                raise ValueError(
                    "MiniMax M2.7 v1 requires TP=1 and DP=1, got "
                    f"TP={self.tensor_parallel_size}, EP={self.expert_parallel_size}, "
                    f"DP={self.data_parallel_size}."
                )
            num_experts = int(getattr(self.hf_config, "num_local_experts"))
            if self.expert_parallel_size > num_experts:
                raise ValueError(
                    "MiniMax M2.7 expert_parallel_size must not exceed "
                    f"num_local_experts={num_experts}, got {self.expert_parallel_size}."
                )
            if num_experts % self.expert_parallel_size != 0:
                raise ValueError(
                    "MiniMax M2.7 requires num_local_experts divisible by "
                    f"expert_parallel_size, got {num_experts} and "
                    f"{self.expert_parallel_size}."
                )
            if self.decode_cuda_graph and self.moe_backend != "triton":
                raise ValueError(
                    "MiniMax M2.7 decode CUDA Graph requires "
                    "moe_backend='triton'; the pytorch and native backends use "
                    "data-dependent eager expert dispatch."
                )
            validate_model_runtime_compatibility(
                model_type=model_type,
                sparse_method=self.vllm_sparse_method,
                tensor_parallel_size=self.tensor_parallel_size,
                expert_parallel_size=self.expert_parallel_size,
                data_parallel_size=self.data_parallel_size,
                enforce_eager=self.enforce_eager,
                decode_cuda_graph=self.decode_cuda_graph,
                enable_prefix_caching=self.enable_prefix_caching,
            )
        elif self.expert_parallel_size != 1 or self.data_parallel_size != 1:
            raise ValueError(
                f"Dense model_type={model_type!r} requires EP=1 and DP=1, got "
                f"TP={self.tensor_parallel_size}, EP={self.expert_parallel_size}, "
                f"DP={self.data_parallel_size}."
            )
        if (
            self.vllm_sparse_method == "deltakv"
            and not is_qwen35
            and self.deltakv_path is None
            and not self.allow_missing_deltakv_path
        ):
            raise ValueError(
                "DeltaKV requires deltakv_path for compressor sparse layers. "
                "Set allow_missing_deltakv_path=True only for construction-only tests."
            )
        self.runtime_layout = RuntimeLayout.from_config(self.hf_config, require_mixed=is_qwen35)
        if self.max_model_len > self.hf_config.max_position_embeddings:
            logger.warning('max_model_len > model.max_position_embeddings 输出可能不正常')
            self.hf_config.max_position_embeddings = self.max_model_len

        if self.max_num_seqs_in_batch > 32:
            logger.warning('max_num_seqs_in_batch 过大或许会占用太多显存')

        if isinstance(self.full_attn_layers, str):
            layers = self.full_attn_layers.strip()
            self.full_attn_layers = [] if not layers else [int(x) for x in layers.split(",")]

        if self.quest_chunk_size <= 0:
            raise ValueError("quest_chunk_size 必须 > 0")
        if self.quest_token_budget <= 0:
            raise ValueError("quest_token_budget 必须 > 0")
        if self.quest_skip_layers < 0:
            raise ValueError("quest_skip_layers 不能 < 0")
        self.rkv_compression_interval = int(self.rkv_compression_interval or 0)
        if self.rkv_compression_interval <= 0:
            raise ValueError(
                f"rkv_compression_interval must be > 0, got {self.rkv_compression_interval}."
            )
        self.rkv_observation_tokens = int(self.rkv_observation_tokens or 0)
        if self.rkv_observation_tokens <= 0:
            raise ValueError(
                f"rkv_observation_tokens must be > 0, got {self.rkv_observation_tokens}."
            )
        if self.rkv_observation_tokens > 128:
            raise ValueError(
                "rkv_observation_tokens must be <= 128 because the prefill score kernel "
                f"supports at most 128 query tokens, got {self.rkv_observation_tokens}."
            )
        if self.rkv_observation_tokens > self.rkv_compression_interval:
            raise ValueError(
                "rkv_observation_tokens must be <= rkv_compression_interval so the query cache "
                "can be refreshed between decode evictions, "
                f"got observation={self.rkv_observation_tokens} interval={self.rkv_compression_interval}."
            )
        self.rkv_alpha = float(self.rkv_alpha)
        if not 0.0 <= self.rkv_alpha <= 1.0:
            raise ValueError(f"rkv_alpha must be in [0, 1], got {self.rkv_alpha}.")
        self.rkv_similarity_threshold = float(self.rkv_similarity_threshold)
        if not 0.0 <= self.rkv_similarity_threshold <= 1.0:
            raise ValueError(
                "rkv_similarity_threshold must be in [0, 1], "
                f"got {self.rkv_similarity_threshold}."
            )
        self.rkv_recent_similar_keep = int(self.rkv_recent_similar_keep)
        if self.rkv_recent_similar_keep < 0:
            raise ValueError(
                f"rkv_recent_similar_keep must be >= 0, got {self.rkv_recent_similar_keep}."
            )
        self.rkv_max_redundancy_tokens = int(self.rkv_max_redundancy_tokens or 0)
        if self.rkv_max_redundancy_tokens <= 0:
            raise ValueError(
                f"rkv_max_redundancy_tokens must be > 0, got {self.rkv_max_redundancy_tokens}."
            )
        self.rkv_redundancy_window = int(self.rkv_redundancy_window or 0)
        if self.rkv_redundancy_window < 0:
            raise ValueError(
                f"rkv_redundancy_window must be >= 0, got {self.rkv_redundancy_window}."
            )
        if 0 < self.rkv_redundancy_window > self.rkv_max_redundancy_tokens:
            raise ValueError(
                "rkv_redundancy_window must be <= rkv_max_redundancy_tokens, "
                f"got window={self.rkv_redundancy_window} max={self.rkv_max_redundancy_tokens}."
            )
        if self.vllm_sparse_method == "rkv":
            log_once(
                "R-KV support is an approximation of the official implementation: "
                "Sparse-VLLM uses one shared physical token index set across KV heads, "
                "so official per-KV-head token selection is not fully reproduced. "
                f"rkv_redundancy_window={self.rkv_redundancy_window}; values > 0 score "
                "redundancy only over the trailing candidate tokens.",
                level="WARNING",
            )
        self.skipkv_compression_interval = int(self.skipkv_compression_interval or 0)
        if self.skipkv_compression_interval <= 0:
            raise ValueError(
                "skipkv_compression_interval must be > 0, "
                f"got {self.skipkv_compression_interval}."
            )
        self.skipkv_alpha = float(self.skipkv_alpha)
        if self.skipkv_alpha < 0.0:
            raise ValueError(f"skipkv_alpha must be >= 0, got {self.skipkv_alpha}.")
        self.skipkv_similarity_threshold = float(self.skipkv_similarity_threshold)
        if not 0.0 <= self.skipkv_similarity_threshold <= 1.0:
            raise ValueError(
                "skipkv_similarity_threshold must be in [0, 1], "
                f"got {self.skipkv_similarity_threshold}."
            )
        self.skipkv_segment_size = int(self.skipkv_segment_size or 0)
        if self.skipkv_segment_size <= 0:
            raise ValueError(f"skipkv_segment_size must be > 0, got {self.skipkv_segment_size}.")
        self.skipkv_max_redundancy_tokens = int(self.skipkv_max_redundancy_tokens or 0)
        if self.skipkv_max_redundancy_tokens <= 0:
            raise ValueError(
                "skipkv_max_redundancy_tokens must be > 0, "
                f"got {self.skipkv_max_redundancy_tokens}."
            )
        self.skipkv_redundancy_window = int(self.skipkv_redundancy_window or 0)
        if self.skipkv_redundancy_window <= 0:
            raise ValueError(
                "skipkv_redundancy_window must be > 0, "
                f"got {self.skipkv_redundancy_window}."
            )
        if self.skipkv_redundancy_window > self.skipkv_max_redundancy_tokens:
            raise ValueError(
                "skipkv_redundancy_window must be <= skipkv_max_redundancy_tokens, "
                f"got window={self.skipkv_redundancy_window} max={self.skipkv_max_redundancy_tokens}."
            )
        self.skipkv_enable_sentence_scoring = bool(self.skipkv_enable_sentence_scoring)
        self.skipkv_sentence_score_weight = float(self.skipkv_sentence_score_weight)
        if self.skipkv_sentence_score_weight < 0.0:
            raise ValueError(
                "skipkv_sentence_score_weight must be >= 0, "
                f"got {self.skipkv_sentence_score_weight}."
            )
        self.skipkv_sentence_min_tokens = int(self.skipkv_sentence_min_tokens or 0)
        if self.skipkv_sentence_min_tokens <= 0:
            raise ValueError(
                "skipkv_sentence_min_tokens must be > 0, "
                f"got {self.skipkv_sentence_min_tokens}."
            )
        self.skipkv_sentence_max_tokens = int(self.skipkv_sentence_max_tokens or 0)
        if self.skipkv_sentence_max_tokens < self.skipkv_sentence_min_tokens:
            raise ValueError(
                "skipkv_sentence_max_tokens must be >= skipkv_sentence_min_tokens, "
                f"got max={self.skipkv_sentence_max_tokens} min={self.skipkv_sentence_min_tokens}."
            )
        self.skipkv_sentence_embedding_layer = int(self.skipkv_sentence_embedding_layer)
        self.skipkv_max_tracked_sentences = int(self.skipkv_max_tracked_sentences or 0)
        if self.skipkv_max_tracked_sentences <= 0:
            raise ValueError(
                "skipkv_max_tracked_sentences must be > 0, "
                f"got {self.skipkv_max_tracked_sentences}."
            )
        self.skipkv_enable_activation_steering = bool(self.skipkv_enable_activation_steering)
        self.skipkv_steering_layer = int(self.skipkv_steering_layer)
        self.skipkv_steering_alpha = float(self.skipkv_steering_alpha)
        self.skipkv_steering_alpha_increment = float(self.skipkv_steering_alpha_increment)
        self.skipkv_steering_alpha_max = float(self.skipkv_steering_alpha_max)
        if self.skipkv_enable_activation_steering and not self.skipkv_steering_vector_path:
            raise ValueError(
                "skipkv_enable_activation_steering=True requires skipkv_steering_vector_path. "
                "Official SkipKV support is limited to the released steering vectors for "
                f"{', '.join(sorted(SUPPORTED_SKIPKV_MODEL_NAMES))}."
            )
        if is_qwen35 and self.enable_prefix_caching and self.prefix_cache_block_size is None:
            self.prefix_cache_block_size = 4096
        self.prefix_cache_block_size = resolve_prefix_cache_block_size(self)
        if is_qwen35 and self.enable_prefix_caching:
            if self.prefix_cache_block_size < 4096 or self.prefix_cache_block_size % 4096 != 0:
                raise ValueError(
                    "qwen3_5 mixed prefix cache requires prefix_cache_block_size to be "
                    f"4096*N, got {self.prefix_cache_block_size}."
                )

        # Normalize compressor type strings.
        for attr in ("compressor_down_type", "compressor_up_type"):
            v = getattr(self, attr, "auto")
            if v is None:
                v = "auto"
            v = str(v).strip().lower()
            setattr(self, attr, v if v else "auto")

        if self.vllm_sparse_method == "deltakv":
            log_once(
                "DeltaKV support in Sparse-vLLM is still experimental and not fully mature; "
                "verify results carefully before treating them as final.",
                level="WARNING",
            )
            if is_qwen35 and not _is_qwen35_deltakv_checkpoint(self.deltakv_path):
                raise ValueError(_qwen35_deltakv_message())
            if not bool(getattr(self, "use_compression", True)):
                raise ValueError("DeltaKV runtime is compressor-only; set use_compression=True.")
            if bool(getattr(self, "enable_sparse_ref_fp8", False)):
                raise ValueError("enable_sparse_ref_fp8 was removed from the slim DeltaKV runtime.")
            if self.deltakv_path is None and not self.allow_missing_deltakv_path:
                raise ValueError(
                    "DeltaKV requires deltakv_path for compressor sparse layers. "
                    "Set allow_missing_deltakv_path=True only for construction-only tests."
                )
            if self.kv_quant_bits not in (0, 4):
                raise ValueError(
                    "DeltaKV slim runtime supports sparse compressor residual bits 0 or 4 only, "
                    f"got kv_quant_bits={self.kv_quant_bits}."
                )
            if self.full_layer_kv_quant_bits not in (0, 4):
                raise ValueError(
                    "DeltaKV slim runtime supports full-layer storage bits 0 or 4 only, "
                    f"got full_layer_kv_quant_bits={self.full_layer_kv_quant_bits}."
                )
            if self.kv_quant_bits == 4 and self.kv_quant_group_size == 0:
                self.kv_quant_group_size = 32
            self.deltakv_triton_materialize_block_tokens = int(
                self.deltakv_triton_materialize_block_tokens or 16
            )
            if (
                self.deltakv_triton_materialize_block_tokens <= 0
                or self.deltakv_triton_materialize_block_tokens % 8 != 0
            ):
                raise ValueError(
                    "deltakv_triton_materialize_block_tokens must be a positive multiple of 8, "
                    f"got {self.deltakv_triton_materialize_block_tokens}."
                )
            self.deltakv_sparse_decode_backend = _resolve_deltakv_sparse_decode_backend(
                self.deltakv_sparse_decode_backend
            )
            is_bf16_full_compressor_sparse = (
                self.full_layer_kv_quant_bits == 0 and self.kv_quant_bits == 0
            )
            is_bf16_full_int4_compressor_sparse = (
                self.full_layer_kv_quant_bits == 0 and self.kv_quant_bits == 4
            )
            is_kivi4_full_int4_compressor_sparse = (
                self.full_layer_kv_quant_bits == 4
                and self.kv_quant_bits == 4
                and bool(getattr(self, "enable_full_layer_kivi_quant", True))
            )
            if not (
                is_bf16_full_compressor_sparse
                or is_bf16_full_int4_compressor_sparse
                or is_kivi4_full_int4_compressor_sparse
            ):
                raise ValueError(
                    "DeltaKV slim runtime supports exactly three paths: "
                    "(full_layer_kv_quant_bits=0, kv_quant_bits=0) and "
                    "(full_layer_kv_quant_bits=0, kv_quant_bits=4) and "
                    "(full_layer_kv_quant_bits=4, kv_quant_bits=4, enable_full_layer_kivi_quant=True)."
                )

        configured_full_layers = {int(layer) for layer in self.full_attn_layers}
        kv_layers = tuple(int(layer) for layer in self.runtime_layout.kv_idx_to_layer_idx)
        kv_positions = {layer: index for index, layer in enumerate(kv_layers)}
        unknown_full_layers = sorted(configured_full_layers - set(kv_layers))
        if unknown_full_layers and self.vllm_sparse_method in {"omnikv", "deltakv"}:
            raise ValueError(
                "full_attn_layers must contain KV/full-attention layer indices for "
                f"{self.vllm_sparse_method}; non-KV layers={unknown_full_layers}."
            )
        self.obs_layer_ids = []
        for layer in self.full_attn_layers:
            layer = int(layer)
            kv_position = kv_positions.get(layer)
            if kv_position is None or kv_position + 1 >= len(kv_layers):
                continue
            if kv_layers[kv_position + 1] not in configured_full_layers:
                self.obs_layer_ids.append(layer)
        
        # PyramidKV 配置验证与智能生成
        if 'pyramidkv' == self.vllm_sparse_method:
            num_layers = int(self.runtime_layout.num_layers)
            num_kv_layers = int(self.runtime_layout.num_kv_layers)
            if self.pyramid_layer_ratios is None:
                start_l = int(self.pyramidkv_start_layer)
                least_l = (
                    int(self.pyramidkv_least_layer)
                    if self.pyramidkv_least_layer is not None
                    else num_kv_layers - 1
                )
                start_r = float(self.pyramidkv_start_ratio)
                least_r = float(self.pyramidkv_least_ratio)
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
                self.pyramid_layer_ratios = ratios
                logger.info(f"PyramidKV 自动生成 KV layer_ratios = {[f'{r:.3f}' for r in ratios]}")
            else:
                ratios = [float(ratio) for ratio in self.pyramid_layer_ratios]
                if len(ratios) == num_layers and num_layers != num_kv_layers:
                    ratios = [ratios[layer_idx] for layer_idx in self.runtime_layout.kv_idx_to_layer_idx]
                self.pyramid_layer_ratios = ratios
        
        if self.pyramid_layer_ratios is not None:
            # PyramidKV 模式自动启用 SnapKV 逻辑
            if 'pyramidkv' != self.vllm_sparse_method:
                raise ValueError('vllm_sparse_method 应为 pyramidkv')

            num_kv_layers = int(self.runtime_layout.num_kv_layers)
            if len(self.pyramid_layer_ratios) != num_kv_layers:
                raise ValueError(
                    f"pyramid_layer_ratios length ({len(self.pyramid_layer_ratios)}) must equal "
                    f"the number of KV/full-attention layers ({num_kv_layers})."
                )

            if any(r <= 0 or r > 1.0 for r in self.pyramid_layer_ratios):
                raise ValueError("pyramid_layer_ratios 的所有值必须在 (0, 1.0] 范围内")
        
        logger.info(f"LLM Config: {self}".replace('\n', ' '))
        setattr(self.hf_config, "runtime_layout", self.runtime_layout)
