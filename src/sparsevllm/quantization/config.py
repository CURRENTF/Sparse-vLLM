from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sparsevllm.utils.config import config_get


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = False
    quant_method: str = ""
    weight_dtype: str = ""
    activation_scheme: str = ""
    weight_block_size: tuple[int, int] | None = None
    model_name: str = "qwen3_5"

    @classmethod
    def disabled(cls, *, model_name: str = "qwen3_5") -> "QuantizationConfig":
        return cls(model_name=model_name)

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
            return cls.disabled(model_name=model_name)

        quant_method = str(
            config_get(value, "quant_method", config_get(value, "method", ""))
            or ""
        ).strip().lower()
        if quant_method not in {"fp8", "fbgemm_fp8"}:
            if required_fp8:
                raise ValueError(
                    f"{model_name} requires quantization_config.quant_method='fp8', "
                    f"got {quant_method!r}."
                )
            if quant_method:
                raise NotImplementedError(
                    f"Sparse-vLLM does not support quant_method={quant_method!r} "
                    f"for {model_name}."
                )
            return cls.disabled(model_name=model_name)

        weight_dtype = str(
            config_get(
                value,
                "weight_dtype",
                config_get(value, "fmt", config_get(value, "format", "e4m3")),
            )
            or ""
        ).strip().lower()
        if "e4m3" not in weight_dtype:
            raise ValueError(
                f"Sparse-vLLM {model_name} FP8 supports e4m3 weights only, "
                f"got weight_dtype={weight_dtype!r}."
            )

        activation_scheme = str(
            config_get(
                value,
                "activation_scheme",
                config_get(value, "activation", "dynamic"),
            )
            or ""
        ).strip().lower()
        if activation_scheme != "dynamic":
            raise ValueError(
                f"Sparse-vLLM {model_name} FP8 supports dynamic activation only, "
                f"got activation_scheme={activation_scheme!r}."
            )

        block_size = config_get(
            value,
            "weight_block_size",
            config_get(
                value,
                "weight_block_shape",
                config_get(value, "block_size", (128, 128)),
            ),
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
