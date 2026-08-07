from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _SwiGLUCompressor(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(input_size, intermediate_size * 2, bias=bias)
        self.w3 = nn.Linear(intermediate_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _normalize_compressor_type(kind: str) -> str:
    if kind is None:
        return "auto"
    aliases = {
        "": "auto",
        "auto": "auto",
        "linear": "linear",
        "mlp": "mlp_gelu",
        "gelu": "mlp_gelu",
        "mlp_gelu": "mlp_gelu",
        "swiglu": "mlp_swiglu",
        "mlp_swiglu": "mlp_swiglu",
    }
    normalized = str(kind).lower().strip()
    if normalized not in aliases:
        raise ValueError(f"Unknown compressor type: {kind}. Use auto|linear|mlp_gelu|mlp_swiglu.")
    return aliases[normalized]


def create_compressor(is_down: bool, config):
    """Build the learned token-level DeltaKV compressor used at HF inference time."""
    if not getattr(config, "use_compression", False):
        return None

    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    kv_factor = 1 if getattr(config, "split_kv", False) else 2
    kv_dim = head_dim * config.num_key_value_heads * kv_factor
    input_size = kv_dim if is_down else config.kv_compressed_size
    output_size = config.kv_compressed_size if is_down else kv_dim
    bias = bool(getattr(config, "compressor_linear_bias", True))

    kind_attr = "compressor_down_type" if is_down else "compressor_up_type"
    kind = _normalize_compressor_type(getattr(config, kind_attr, "auto"))
    if kind == "auto":
        kind = "mlp_gelu" if getattr(config, "use_nonlinear_compressor", False) else "linear"

    if kind == "linear":
        return nn.Linear(input_size, output_size, bias=bias)

    inter_attr = "compressor_down_intermediate_size" if is_down else "compressor_up_intermediate_size"
    intermediate_size = int(getattr(config, inter_attr, 0) or 0)
    if intermediate_size <= 0:
        intermediate_size = int(getattr(config, "compressor_intermediate_size", 0) or 0)
    if intermediate_size <= 0:
        intermediate_size = (input_size + output_size) // 2

    if kind == "mlp_gelu":
        return nn.Sequential(
            nn.Linear(input_size, intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, output_size, bias=bias),
        )
    if kind == "mlp_swiglu":
        return _SwiGLUCompressor(input_size, intermediate_size, output_size, bias=bias)
    raise AssertionError(f"Unhandled compressor type: {kind}")


def reshape_and_apply_qk_norm(
    attn: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    query_hidden_shape: tuple[int, ...],
    key_hidden_shape: tuple[int, ...] | None = None,
):
    query_states = query_states.view(query_hidden_shape)
    key_states = key_states.view(key_hidden_shape or query_hidden_shape)
    if hasattr(attn, "q_norm"):
        query_states = attn.q_norm(query_states)
    if hasattr(attn, "k_norm"):
        key_states = attn.k_norm(key_states)
    return query_states.transpose(1, 2), key_states.transpose(1, 2)
