import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def reverse_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对已应用 RoPE 的向量执行逆操作，恢复到位置无关状态。
    
    RoPE 公式:     y1 = x1*cos - x2*sin,  y2 = x2*cos + x1*sin
    De-RoPE 公式:  x1 = y1*cos + y2*sin,  x2 = y2*cos - y1*sin
    """
    y1, y2 = torch.chunk(x.float(), 2, dim=-1)
    x1 = y1 * cos + y2 * sin
    x2 = y2 * cos - y1 * sin
    return torch.cat((x1, x2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        inv_freq: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        if inv_freq is None:
            inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


_ROPE_CACHE = {}


def _rope_cache_key(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None,
):
    try:
        default_device = str(torch.get_default_device())
    except Exception:
        default_device = "cpu"
    if default_device.startswith("cuda") and torch.cuda.is_available():
        default_device = f"cuda:{torch.cuda.current_device()}"

    if rope_scaling is None:
        scaling_key = None
    else:
        scaling_key = tuple(sorted((str(k), str(v)) for k, v in rope_scaling.items()))
    return (head_size, rotary_dim, max_position, float(base), scaling_key, default_device)


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    key = _rope_cache_key(head_size, rotary_dim, max_position, base, rope_scaling)
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]

    inv_freq = None
    if rope_scaling is not None:
        rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", ""))).lower()
        if rope_type != "llama3":
            raise NotImplementedError(f"Unsupported RoPE scaling type for Sparse-vLLM: {rope_scaling!r}")
        factor = float(rope_scaling["factor"])
        low_freq_factor = float(rope_scaling["low_freq_factor"])
        high_freq_factor = float(rope_scaling["high_freq_factor"])
        old_context_len = float(rope_scaling["original_max_position_embeddings"])

        default_inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        wavelen = 2 * math.pi / default_inv_freq
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, default_inv_freq / factor, default_inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, inv_freq=inv_freq)
    _ROPE_CACHE[key] = rotary_emb
    return rotary_emb
