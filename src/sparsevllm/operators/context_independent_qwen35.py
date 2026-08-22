"""Validation gate for the Qwen3.5 linear-GDN decode chain.

The linear path has no context-proportional launch dimension.  Its existing
decode kernels already consume a static token batch and device-resident state
indices, so the experimental policy only marks instances after checking the
required chain.  No stable kernel is replaced when no code change is needed.
"""

from __future__ import annotations

import torch


_REQUIRED_LINEAR_METHODS = (
    "_decode_gdn",
    "_project_qkvzba",
    "_repeat_qk_for_value_heads",
)


def bind_context_independent_qwen35_linear_attention(
    model: torch.nn.Module,
) -> int:
    bound = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen35LinearAttention":
            continue
        missing = [
            name
            for name in _REQUIRED_LINEAR_METHODS
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Qwen3.5 context-independent linear attention is missing "
                f"required decode hooks: {missing}"
            )
        module.cuda_graph_context_independent = True
        bound += 1
    return bound


__all__ = ["bind_context_independent_qwen35_linear_attention"]
