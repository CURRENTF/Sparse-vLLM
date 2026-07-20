from __future__ import annotations

from typing import Any

from sparsevllm.quantization.fp8 import Fp8BlockScaledLinearBackend


class QuantizationRegistry:
    """Construct quantized local linear backends from validated config objects."""

    @staticmethod
    def create_linear_backend(quantization: Any):
        if not bool(getattr(quantization, "enabled", False)):
            return None
        quant_method = str(getattr(quantization, "quant_method", "") or "").strip().lower()
        if quant_method != "fp8":
            raise ValueError(f"Unsupported quantized Linear method={quant_method!r}.")
        return Fp8BlockScaledLinearBackend(
            block_size=tuple(quantization.weight_block_size),
            backend=getattr(quantization, "backend", "auto"),
            model_name=getattr(quantization, "model_name", "qwen3_5"),
        )
