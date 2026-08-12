from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sparsevllm.utils.config import config_get


def resolve_attention_qk_head_dim(hf_config: Any) -> int:
    dimensions = (
        config_get(hf_config, "qk_nope_head_dim", None),
        config_get(hf_config, "qk_rope_head_dim", None),
    )
    if all(value is not None for value in dimensions):
        head_dim = sum(map(int, dimensions))
    elif (value := config_get(hf_config, "head_dim", None)) is not None:
        head_dim = int(value)
    else:
        hidden_size = int(config_get(hf_config, "hidden_size", 0) or 0)
        num_heads = int(config_get(hf_config, "num_attention_heads", 0) or 0)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError(
                "Attention QK head dimension requires valid head_dim or divisible "
                "hidden_size/num_attention_heads."
            )
        head_dim = hidden_size // num_heads
    if head_dim <= 0:
        raise ValueError(f"Attention QK head dimension must be positive, got {head_dim}.")
    return head_dim


def _coerce_int_list(
    name: str,
    value: Any,
    *,
    allow_none: bool = False,
) -> list[int] | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required.")
    if isinstance(value, str):
        return [int(part) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(
        f"{name} must be a list/tuple of ints or a comma-separated string, "
        f"got {value!r}."
    )


def _attention_type(value: Any) -> str:
    value = str(value).strip().lower()
    if value in {
        "full",
        "full_attention",
        "attention",
        "self_attention",
        "sliding_attention",
    }:
        return "full"
    if value in {
        "linear",
        "linear_attention",
        "recurrent",
        "recurrent_attention",
        "gated_delta",
        "gated_delta_net",
    }:
        return "linear"
    raise ValueError(f"Unsupported attention layer type {value!r}.")


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
        if num_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_layers}.")
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
    def from_config(
        cls,
        hf_config: Any,
        *,
        require_mixed: bool = False,
    ) -> "RuntimeLayout":
        num_layers = int(config_get(hf_config, "num_hidden_layers"))
        if num_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_layers}.")
        layer_types = config_get(hf_config, "layer_types", None)
        full_layers = _coerce_int_list(
            "full_attention_layer_indices",
            config_get(
                hf_config,
                "full_attention_layer_indices",
                config_get(hf_config, "attention_layer_indices", None),
            ),
            allow_none=True,
        )
        linear_layers = _coerce_int_list(
            "linear_attention_layer_indices",
            config_get(hf_config, "linear_attention_layer_indices", None),
            allow_none=True,
        )

        if layer_types is not None:
            if len(layer_types) != num_layers:
                raise ValueError(
                    "layer_types length must equal num_hidden_layers: "
                    f"{len(layer_types)} != {num_layers}."
                )
            inferred_full, inferred_linear = [], []
            for layer_idx, layer_type in enumerate(layer_types):
                target = (
                    inferred_full
                    if _attention_type(layer_type) == "full"
                    else inferred_linear
                )
                target.append(layer_idx)
            full_layers = inferred_full if full_layers is None else full_layers
            linear_layers = inferred_linear if linear_layers is None else linear_layers

        if full_layers is None and linear_layers is None:
            if require_mixed:
                raise ValueError(
                    "Mixed-attention models require layer_types or explicit "
                    "full/linear attention layer indices."
                )
            return cls.dense(num_layers)
        if full_layers is None:
            linear_set = set(linear_layers or ())
            full_layers = [idx for idx in range(num_layers) if idx not in linear_set]
        if linear_layers is None:
            full_set = set(full_layers or ())
            linear_layers = [idx for idx in range(num_layers) if idx not in full_set]

        full_tuple = tuple(sorted(int(idx) for idx in full_layers))
        linear_tuple = tuple(sorted(int(idx) for idx in linear_layers))
        full_set, linear_set = set(full_tuple), set(linear_tuple)
        expected = set(range(num_layers))
        if full_set & linear_set:
            raise ValueError(
                "RuntimeLayout full and linear layer sets overlap: "
                f"{sorted(full_set & linear_set)}."
            )
        if full_set | linear_set != expected:
            raise ValueError(
                "RuntimeLayout layer map is incomplete: "
                f"missing={sorted(expected - (full_set | linear_set))}, "
                f"extra={sorted((full_set | linear_set) - expected)}."
            )

        raw_layer_to_kv = config_get(hf_config, "layer_idx_to_kv_idx", None)
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
            layer_to_kv = [
                None if value is None or int(value) < 0 else int(value)
                for value in raw_layer_to_kv
            ]
            invalid_linear = [
                layer_idx
                for layer_idx in linear_tuple
                if layer_to_kv[layer_idx] is not None
            ]
            if invalid_linear:
                raise ValueError(
                    "Linear-attention layers must not have KV indices: "
                    f"{invalid_linear}."
                )

        kv_pairs = sorted(
            (kv_idx, layer_idx)
            for layer_idx, kv_idx in enumerate(layer_to_kv)
            if kv_idx is not None
        )
        if len(kv_pairs) != len(full_tuple):
            raise ValueError(
                "RuntimeLayout must assign one KV index to each full-attention "
                f"layer: full={len(full_tuple)}, assigned={len(kv_pairs)}."
            )
        kv_indices = [kv_idx for kv_idx, _ in kv_pairs]
        if kv_indices != list(range(len(kv_pairs))):
            raise ValueError(f"KV layer indices must be contiguous, got {kv_indices}.")
        kv_tuple = tuple(layer_idx for _, layer_idx in kv_pairs)
        configured_num_kv_layers = config_get(hf_config, "num_kv_layers", None)
        if (
            configured_num_kv_layers is not None
            and int(configured_num_kv_layers) != len(kv_tuple)
        ):
            raise ValueError(
                f"num_kv_layers={configured_num_kv_layers} does not match "
                f"full-attention layers={len(kv_tuple)}."
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
            raise RuntimeError(
                f"layer_idx={layer_idx} is linear_attention and has no KV cache."
            )
        return int(kv_idx)
