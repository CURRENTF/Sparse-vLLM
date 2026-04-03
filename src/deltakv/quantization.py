from __future__ import annotations

from typing import Any, Sequence

import torch

DEFAULT_SKIP_MODULES = [
    "compress_down",
    "compress_up",
    "k_compress_down",
    "k_compress_up",
    "v_compress_down",
    "v_compress_up",
    "cluster",
    "transform",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_str_list(value: Any, *, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"{key} must contain strings, got {type(item).__name__}")
            item = item.strip()
            if item:
                items.append(item)
        return items
    raise TypeError(f"{key} must be a string or a sequence of strings, got {type(value).__name__}")


def resolve_torch_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        aliases = {
            "auto": default,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if key in aliases:
            return aliases[key]
        attr = getattr(torch, key, None)
        if isinstance(attr, torch.dtype):
            return attr
    raise ValueError(f"Unsupported torch dtype: {value!r}")


def build_model_load_kwargs(
    infer_config: dict[str, Any] | None,
    *,
    default_torch_dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any], torch.dtype]:
    runtime_infer_config = dict(infer_config or {})
    target_torch_dtype = resolve_torch_dtype(
        runtime_infer_config.pop("torch_dtype", default_torch_dtype),
        default_torch_dtype,
    )

    load_in_4bit = _as_bool(runtime_infer_config.pop("load_in_4bit", False))
    load_in_8bit = _as_bool(runtime_infer_config.pop("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of load_in_4bit/load_in_8bit can be enabled at a time.")

    if not (load_in_4bit or load_in_8bit):
        return runtime_infer_config, {}, target_torch_dtype

    from transformers import BitsAndBytesConfig

    extra_skip_modules = _as_str_list(
        runtime_infer_config.pop("quant_skip_modules", None),
        key="quant_skip_modules",
    )
    extra_skip_modules.extend(
        _as_str_list(
            runtime_infer_config.pop("llm_int8_skip_modules", None),
            key="llm_int8_skip_modules",
        )
    )
    skip_modules = list(dict.fromkeys([*DEFAULT_SKIP_MODULES, *extra_skip_modules]))

    quant_kwargs: dict[str, Any] = {
        "llm_int8_skip_modules": skip_modules,
    }
    llm_int8_threshold = runtime_infer_config.pop("llm_int8_threshold", None)
    if llm_int8_threshold is not None:
        quant_kwargs["llm_int8_threshold"] = float(llm_int8_threshold)

    llm_int8_enable_fp32_cpu_offload = runtime_infer_config.pop(
        "llm_int8_enable_fp32_cpu_offload", None
    )
    if llm_int8_enable_fp32_cpu_offload is not None:
        quant_kwargs["llm_int8_enable_fp32_cpu_offload"] = _as_bool(
            llm_int8_enable_fp32_cpu_offload
        )

    llm_int8_has_fp16_weight = runtime_infer_config.pop("llm_int8_has_fp16_weight", None)
    if llm_int8_has_fp16_weight is not None:
        quant_kwargs["llm_int8_has_fp16_weight"] = _as_bool(llm_int8_has_fp16_weight)

    if load_in_4bit:
        quant_kwargs["load_in_4bit"] = True
        quant_kwargs["bnb_4bit_compute_dtype"] = resolve_torch_dtype(
            runtime_infer_config.pop("bnb_4bit_compute_dtype", target_torch_dtype),
            target_torch_dtype,
        )
        quant_kwargs["bnb_4bit_use_double_quant"] = _as_bool(
            runtime_infer_config.pop("bnb_4bit_use_double_quant", True)
        )
        quant_kwargs["bnb_4bit_quant_type"] = runtime_infer_config.pop(
            "bnb_4bit_quant_type", "nf4"
        )
        bnb_4bit_quant_storage = runtime_infer_config.pop("bnb_4bit_quant_storage", None)
        if bnb_4bit_quant_storage is not None:
            quant_kwargs["bnb_4bit_quant_storage"] = resolve_torch_dtype(
                bnb_4bit_quant_storage,
                torch.uint8,
            )
    else:
        quant_kwargs["load_in_8bit"] = True

    return (
        runtime_infer_config,
        {"quantization_config": BitsAndBytesConfig(**quant_kwargs)},
        target_torch_dtype,
    )


def _module_name_matches(current_name: str, candidate: str) -> bool:
    return (
        current_name == candidate
        or current_name.endswith(f".{candidate}")
        or f".{candidate}." in current_name
    )


def restore_modules_to_dtype(
    model: torch.nn.Module,
    target_torch_dtype: torch.dtype,
    *,
    module_names: Sequence[str] | None = None,
) -> list[str]:
    restored: list[str] = []
    names = module_names or DEFAULT_SKIP_MODULES
    for name, module in model.named_modules():
        if not name:
            continue
        if not any(_module_name_matches(name, candidate) for candidate in names):
            continue
        module.to(dtype=target_torch_dtype)
        restored.append(name)
    return restored
