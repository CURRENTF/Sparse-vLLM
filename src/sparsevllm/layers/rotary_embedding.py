import math
from functools import lru_cache
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
        rope_scaling: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if rope_scaling is not None:
            scaling = dict(rope_scaling)
            rope_type = scaling.get("rope_type", scaling.get("type"))
            if rope_type != "llama3":
                raise NotImplementedError(f"Unsupported rope_scaling={scaling!r}.")
            factor = float(scaling["factor"])
            low_freq_factor = float(scaling["low_freq_factor"])
            high_freq_factor = float(scaling["high_freq_factor"])
            old_context_len = float(scaling["original_max_position_embeddings"])

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile(fullgraph=True, dynamic=True)
    def compiled_forward(
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

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compiled_forward(positions, query, key)


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: tuple[tuple[str, object], ...] | None = None,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling=rope_scaling)
    return rotary_emb
