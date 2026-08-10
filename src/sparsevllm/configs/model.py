"""Model metadata loading, layout construction, and checkpoint validation."""

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig

from sparsevllm.method_registry import (
    H2O_SUPPORTED_MODEL_TYPES,
    validate_model_runtime_compatibility,
)
from sparsevllm.utils.log import logger, log_once

try:
    from transformers import Qwen3Config
except ImportError:
    Qwen3Config = AutoConfig

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

        return cls(
            enabled=True,
            quant_method="fp8",
            weight_dtype="e4m3",
            activation_scheme="dynamic",
            weight_block_size=block_tuple,
            model_name=model_name,
        )


def _validate_qwen35_checkpoint_precision(
    hf_config: Any,
    raw_quantization_config: Any,
    quantization_config: QuantizationConfig,
) -> None:
    if quantization_config.enabled:
        return

    quant_method = str(
        _config_get(
            raw_quantization_config,
            "quant_method",
            _config_get(raw_quantization_config, "method", ""),
        )
        or ""
    ).strip().lower()
    if quant_method:
        raise NotImplementedError(
            "qwen3_5 supports unquantized BF16 or block FP8 checkpoints only, "
            f"got quant_method={quant_method!r}."
        )

    configured_dtype = _config_get(hf_config, "torch_dtype", None)
    if configured_dtype not in {torch.bfloat16, "bfloat16"}:
        raise NotImplementedError(
            "Unquantized qwen3_5 checkpoints require BF16 weights, "
            f"got torch_dtype={configured_dtype!r}."
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


def _validate_qwen3_moe_fp8_checkpoint_config(
    hf_config: Any,
    raw_quantization_config: Any,
) -> None:
    architectures = tuple(_config_get(hf_config, "architectures", ()) or ())
    if architectures != ("Qwen3MoeForCausalLM",):
        raise ValueError(
            "Qwen3MoE FP8 requires architectures=['Qwen3MoeForCausalLM'], "
            f"got {list(architectures)}."
        )
    configured_dtype = _config_get(hf_config, "torch_dtype", None)
    if configured_dtype is None:
        configured_dtype = _config_get(hf_config, "dtype", None)
    if configured_dtype not in {torch.bfloat16, "bfloat16"}:
        raise ValueError(
            "Qwen3MoE FP8 requires BF16 non-quantized parameters, "
            f"got dtype={configured_dtype!r}."
        )

    hidden_size = int(_config_get(hf_config, "hidden_size", 0) or 0)
    intermediate_size = int(
        _config_get(hf_config, "moe_intermediate_size", 0) or 0
    )
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "Qwen3MoE FP8 requires hidden_size and moe_intermediate_size "
            f"aligned to 128, got {hidden_size}/{intermediate_size}."
        )

    excluded_modules = {
        str(name)
        for name in (
            _config_get(raw_quantization_config, "modules_to_not_convert", ()) or ()
        )
    }
    num_layers = int(_config_get(hf_config, "num_hidden_layers", 0) or 0)
    required_exclusions = {"lm_head"}
    required_exclusions.update(
        f"model.layers.{layer_idx}.mlp.gate"
        for layer_idx in range(num_layers)
    )
    missing_exclusions = sorted(required_exclusions - excluded_modules)
    if missing_exclusions:
        raise ValueError(
            "Qwen3MoE FP8 quantization_config must exclude lm_head and every "
            f"router gate; missing {missing_exclusions[:8]}."
        )


def _validate_qwen3_fp8_checkpoint_config(
    hf_config: Any,
    *,
    tensor_parallel_size: int,
) -> None:
    architectures = tuple(_config_get(hf_config, "architectures", ()) or ())
    if architectures != ("Qwen3ForCausalLM",):
        raise ValueError(
            "Qwen3 FP8 requires architectures=['Qwen3ForCausalLM'], "
            f"got {list(architectures)}."
        )
    configured_dtype = _config_get(hf_config, "torch_dtype", None)
    if configured_dtype is None:
        configured_dtype = _config_get(hf_config, "dtype", None)
    if configured_dtype not in {torch.bfloat16, "bfloat16"}:
        raise ValueError(
            "Qwen3 FP8 requires BF16 non-quantized parameters, "
            f"got dtype={configured_dtype!r}."
        )

    tp_size = int(tensor_parallel_size)
    head_dim = int(_config_get(hf_config, "head_dim", 0) or 0)
    dimensions = {
        "hidden_size": int(_config_get(hf_config, "hidden_size", 0) or 0),
        "intermediate_size": int(
            _config_get(hf_config, "intermediate_size", 0) or 0
        ),
        "query_size": int(
            _config_get(hf_config, "num_attention_heads", 0) or 0
        )
        * head_dim,
        "key_value_size": int(
            _config_get(hf_config, "num_key_value_heads", 0) or 0
        )
        * head_dim,
    }
    invalid_dimensions = {
        name: size
        for name, size in dimensions.items()
        if size <= 0 or size % (128 * tp_size)
    }
    if invalid_dimensions:
        raise ValueError(
            "Qwen3 FP8 requires every TP-local dense projection dimension to be "
            "128-aligned; "
            f"TP={tp_size}, invalid={invalid_dimensions}."
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




def _validate_runtime_compatibility(config, *, model_type: str) -> None:
    validate_model_runtime_compatibility(
        model_type=model_type,
        sparse_method=config.vllm_sparse_method,
        tensor_parallel_size=config.tensor_parallel_size,
        expert_parallel_size=config.expert_parallel_size,
        data_parallel_size=config.data_parallel_size,
        enforce_eager=config.enforce_eager,
        decode_cuda_graph=config.decode_cuda_graph,
        enable_prefix_caching=config.enable_prefix_caching,
    )


def _validate_qwen3_moe_runtime(config, *, model_type: str) -> None:
    tp_size = int(config.tensor_parallel_size)
    ep_size = int(config.expert_parallel_size)
    if config.data_parallel_size != 1:
        raise ValueError(
            "Qwen3MoE requires DP=1, got "
            f"TP={config.tensor_parallel_size}, EP={config.expert_parallel_size}, "
            f"DP={config.data_parallel_size}."
        )
    if tp_size > 1 and tp_size % ep_size:
        raise ValueError(
            "Qwen3MoE outer tensor_parallel_size must be divisible by "
            f"expert_parallel_size, got outer TP={tp_size}, MoE EP={ep_size}."
        )
    num_experts = int(getattr(config.hf_config, "num_experts", 0) or 0)
    if num_experts <= 0:
        raise ValueError(f"Qwen3MoE requires a positive num_experts, got {num_experts}.")
    if ep_size > num_experts:
        raise ValueError(
            "expert_parallel_size must not exceed num_experts, "
            f"got EP={config.expert_parallel_size}, num_experts={num_experts}."
        )
    if num_experts % ep_size != 0:
        raise ValueError(
            "Qwen3MoE requires num_experts divisible by expert_parallel_size, "
            f"got num_experts={num_experts}, EP={config.expert_parallel_size}."
        )
    if tp_size > 1:
        divisible_fields = {
            "num_attention_heads": int(config.hf_config.num_attention_heads),
            "num_key_value_heads": int(config.hf_config.num_key_value_heads),
            "vocab_size": int(config.hf_config.vocab_size),
        }
        for field, value in divisible_fields.items():
            if value % tp_size:
                raise ValueError(
                    f"Qwen3MoE {field} must be divisible by tensor_parallel_size, "
                    f"got {value} and {tp_size}."
                )
        moe_tp_size = tp_size // ep_size
        moe_intermediate_size = int(config.hf_config.moe_intermediate_size)
        if moe_intermediate_size % moe_tp_size:
            raise ValueError(
                "Qwen3MoE moe_intermediate_size must be divisible by MoE TP size, "
                f"got {moe_intermediate_size} and {moe_tp_size}."
            )
    top_k = int(getattr(config.hf_config, "num_experts_per_tok", 0) or 0)
    if not 1 <= top_k <= num_experts:
        raise ValueError(
            "Qwen3MoE num_experts_per_tok must be in [1, num_experts], "
            f"got top_k={top_k}, num_experts={num_experts}."
        )
    decoder_sparse_step = int(
        getattr(config.hf_config, "decoder_sparse_step", 1)
    )
    mlp_only_layers = tuple(
        int(layer_idx)
        for layer_idx in (getattr(config.hf_config, "mlp_only_layers", ()) or ())
    )
    if decoder_sparse_step != 1 or mlp_only_layers:
        raise NotImplementedError(
            "Qwen3MoE v1 requires every decoder layer to be MoE, got "
            f"decoder_sparse_step={decoder_sparse_step}, "
            f"mlp_only_layers={list(mlp_only_layers)}."
        )
    shared_intermediate_size = int(
        getattr(config.hf_config, "shared_expert_intermediate_size", 0) or 0
    )
    if shared_intermediate_size != 0:
        raise NotImplementedError(
            "Qwen3MoE v1 does not support shared experts, got "
            f"shared_expert_intermediate_size={shared_intermediate_size}."
        )
    model_dtype = getattr(config.hf_config, "torch_dtype", None)
    if model_dtype not in {torch.bfloat16, torch.float16}:
        raise NotImplementedError(
            "Qwen3MoE v1 supports BF16/FP16 checkpoints only, "
            f"got torch_dtype={model_dtype}."
        )
    if tp_size > 1 and model_dtype != torch.bfloat16:
        raise NotImplementedError(
            "Qwen3MoE outer TP supports BF16 checkpoints only, "
            f"got torch_dtype={model_dtype}."
        )
    _validate_runtime_compatibility(config, model_type=model_type)


def _validate_minimax_runtime(config, *, model_type: str) -> None:
    tp_size = int(config.tensor_parallel_size)
    ep_size = int(config.expert_parallel_size)
    if config.data_parallel_size != 1:
        raise ValueError(
            "MiniMax M2.7 requires DP=1, got "
            f"TP={config.tensor_parallel_size}, EP={config.expert_parallel_size}, "
            f"DP={config.data_parallel_size}."
        )
    if tp_size > 1 and tp_size % ep_size:
        raise ValueError(
            "MiniMax M2.7 outer tensor_parallel_size must be divisible by "
            f"expert_parallel_size, got outer TP={tp_size}, MoE EP={ep_size}."
        )
    num_experts = int(getattr(config.hf_config, "num_local_experts"))
    if ep_size > num_experts:
        raise ValueError(
            "MiniMax M2.7 expert_parallel_size must not exceed "
            f"num_local_experts={num_experts}, got {config.expert_parallel_size}."
        )
    if num_experts % ep_size != 0:
        raise ValueError(
            "MiniMax M2.7 requires num_local_experts divisible by "
            f"expert_parallel_size, got {num_experts} and "
            f"{config.expert_parallel_size}."
        )
    if tp_size > 1:
        divisible_fields = {
            "num_attention_heads": int(config.hf_config.num_attention_heads),
            "num_key_value_heads": int(config.hf_config.num_key_value_heads),
            "vocab_size": int(config.hf_config.vocab_size),
        }
        for field, value in divisible_fields.items():
            if value % tp_size:
                raise ValueError(
                    f"MiniMax M2.7 {field} must be divisible by "
                    f"tensor_parallel_size, got {value} and {tp_size}."
                )
        moe_tp_size = tp_size // ep_size
        intermediate_size = int(config.hf_config.intermediate_size)
        if intermediate_size % moe_tp_size:
            raise ValueError(
                "MiniMax M2.7 intermediate_size must be divisible by MoE TP size, "
                f"got {intermediate_size} and {moe_tp_size}."
            )
    _validate_runtime_compatibility(config, model_type=model_type)


def _validate_dense_parallelism(config, *, model_type: str) -> None:
    if config.expert_parallel_size != 1 or config.data_parallel_size != 1:
        raise ValueError(
            f"Dense model_type={model_type!r} requires EP=1 and DP=1, got "
            f"TP={config.tensor_parallel_size}, EP={config.expert_parallel_size}, "
            f"DP={config.data_parallel_size}."
        )


def _finalize_model_config(config, *, is_qwen35: bool) -> None:
    if (
        config.vllm_sparse_method == "deltakv"
        and not is_qwen35
        and config.deltakv_path is None
        and not config.allow_missing_deltakv_path
    ):
        raise ValueError(
            "DeltaKV requires deltakv_path for compressor sparse layers. "
            "Set allow_missing_deltakv_path=True only for construction-only tests."
        )
    config.runtime_layout = RuntimeLayout.from_config(config.hf_config, require_mixed=is_qwen35)
    if config.tiny_random:
        if config.hf_config.num_attention_heads % config.tensor_parallel_size != 0:
            raise ValueError(
                "Tiny random num_attention_heads must be divisible by tensor_parallel_size."
            )
        if config.hf_config.num_key_value_heads % config.tensor_parallel_size != 0:
            raise ValueError(
                "Tiny random num_key_value_heads must be divisible by tensor_parallel_size."
            )
        if config.hf_config.vocab_size % config.tensor_parallel_size != 0:
            raise ValueError(
                "Tiny random vocab_size must be divisible by tensor_parallel_size."
            )
    if config.max_model_len > config.hf_config.max_position_embeddings:
        logger.warning('max_model_len > model.max_position_embeddings 输出可能不正常')
        config.hf_config.max_position_embeddings = config.max_model_len

    if config.max_num_seqs_in_batch > 32:
        logger.warning('max_num_seqs_in_batch 过大或许会占用太多显存')


def load_and_validate_model(config) -> bool:
    if isinstance(config.deltakv_path, str):
        deltakv_path = config.deltakv_path.strip()
        config.deltakv_path = None if deltakv_path.lower() in {"", "none", "null"} else deltakv_path
    if config.tiny_random and config.vllm_sparse_method == "deltakv":
        raise NotImplementedError(
            "Tiny random mode does not support DeltaKV compressor weights yet."
        )
    try:
        config.outer_hf_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
    except Exception as e:
        config.outer_hf_config = _load_raw_qwen35_config(config.model, e)
    is_qwen35 = _is_qwen35_outer_config(config.outer_hf_config)
    config.hf_config = _extract_text_config(config.outer_hf_config)
    if is_qwen35:
        setattr(config.hf_config, "model_type", "qwen3_5")
    model_type = str(getattr(config.hf_config, "model_type", "") or "")
    is_minimax_m2 = model_type == "minimax_m2"
    is_qwen3 = model_type == "qwen3"
    is_qwen3_moe = model_type == "qwen3_moe"

    if config.vllm_sparse_method == "h2o":
        if model_type not in H2O_SUPPORTED_MODEL_TYPES:
            supported = ", ".join(
                repr(value) for value in sorted(H2O_SUPPORTED_MODEL_TYPES)
            )
            raise NotImplementedError(
                "H2O v1 supports the model types already implemented by Sparse-vLLM: "
                f"{supported}; got model_type={model_type!r}."
            )

    if config.tiny_random:
        from sparsevllm.debug.tiny_random import apply_tiny_random_overrides

        if is_qwen35:
            raise NotImplementedError("Tiny random mode does not support qwen3_5 yet.")
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

    raw_quantization_config = _config_get(
        config.hf_config,
        "quantization_config",
        _config_get(config.outer_hf_config, "quantization_config", None),
    )
    quantized_model_name = "qwen3_5"
    if is_minimax_m2:
        quantized_model_name = "MiniMax M2.7"
    elif is_qwen3:
        quantized_model_name = "Qwen3"
    elif is_qwen3_moe:
        quantized_model_name = "Qwen3MoE"
    config.quantization_config = QuantizationConfig.from_hf_config(
        raw_quantization_config,
        required_fp8=is_minimax_m2,
        model_name=quantized_model_name,
    )
    if is_qwen35:
        _validate_qwen35_checkpoint_precision(
            config.hf_config,
            raw_quantization_config,
            config.quantization_config,
        )
    if config.tiny_random and config.quantization_config.enabled:
        raise NotImplementedError("Tiny random mode does not support quantized model weights.")
    setattr(config.hf_config, "quantization_config", config.quantization_config)
    if is_minimax_m2:
        _validate_minimax_m2_checkpoint_config(
            config.hf_config,
            raw_quantization_config,
        )
    if is_qwen3 and config.quantization_config.enabled:
        _validate_qwen3_fp8_checkpoint_config(
            config.hf_config,
            tensor_parallel_size=config.tensor_parallel_size,
        )
    if is_qwen3_moe and config.quantization_config.enabled:
        _validate_qwen3_moe_fp8_checkpoint_config(
            config.hf_config,
            raw_quantization_config,
        )

    if getattr(config.hf_config, "model_type", "") in {"deepseek_v2", "deepseek_v32"}:
        raise NotImplementedError(
            f"Unsupported Sparse-vLLM model_type={config.hf_config.model_type!r}. "
            "Supported model types: qwen2, qwen3, qwen3_5, llama."
        )

    if model_type == "qwen3_moe":
        _validate_qwen3_moe_runtime(config, model_type=model_type)
    elif model_type == "minimax_m2":
        _validate_minimax_runtime(config, model_type=model_type)
    else:
        _validate_dense_parallelism(config, model_type=model_type)
    _finalize_model_config(config, is_qwen35=is_qwen35)
    return is_qwen35
