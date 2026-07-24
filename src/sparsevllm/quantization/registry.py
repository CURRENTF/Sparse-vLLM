from __future__ import annotations

from typing import Any

from sparsevllm.operators.fp8_linear import resolve_fp8_linear_provider


class QuantizationRegistry:
    """Resolve quantized Linear providers from validated config objects."""

    @staticmethod
    def resolve_linear_provider(quantization: Any):
        if not bool(getattr(quantization, "enabled", False)):
            return None
        quant_method = str(getattr(quantization, "quant_method", "") or "").strip().lower()
        if quant_method != "fp8":
            raise ValueError(f"Unsupported quantized Linear method={quant_method!r}.")
        return resolve_fp8_linear_provider(
            tuple(quantization.weight_block_size),
        )
