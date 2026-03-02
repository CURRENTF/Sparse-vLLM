from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class FlashMLA:
    """Thin optional wrapper around the `flash_mla` package (DeepSeek FlashMLA kernels).

    This repo does not vendor FlashMLA. All calls must be guarded because importing the
    package requires a compiled CUDA extension.
    """

    flash_mla_sparse_fwd: Callable[..., torch.Tensor]
    flash_mla_with_kvcache: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    get_mla_metadata: Callable[..., Any]


_FLASH_MLA: FlashMLA | None = None
_FLASH_MLA_ERR: Exception | None = None


def try_get_flash_mla() -> FlashMLA | None:
    """Return FlashMLA bindings if installed; otherwise return None.

    Never raises on import failures to keep Sparse-vLLM usable without FlashMLA.
    """
    global _FLASH_MLA, _FLASH_MLA_ERR
    if _FLASH_MLA is not None:
        return _FLASH_MLA
    if _FLASH_MLA_ERR is not None:
        return None

    try:
        import flash_mla  # type: ignore

        _FLASH_MLA = FlashMLA(
            flash_mla_sparse_fwd=flash_mla.flash_mla_sparse_fwd,
            flash_mla_with_kvcache=flash_mla.flash_mla_with_kvcache,
            get_mla_metadata=flash_mla.get_mla_metadata,
        )
        return _FLASH_MLA
    except Exception as e:  # pragma: no cover - depends on external install
        _FLASH_MLA_ERR = e
        return None


def require_flash_mla() -> FlashMLA:
    """Return FlashMLA bindings or raise a helpful error."""
    flash = try_get_flash_mla()
    if flash is not None:
        return flash
    err = _FLASH_MLA_ERR
    if err is None:
        raise RuntimeError("flash_mla is not available.")
    raise RuntimeError(
        "flash_mla is not available (missing/failed CUDA extension import). "
        f"Original error: {type(err).__name__}: {err}"
    )


def flash_mla_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    # Backward-compat alias (some early FlashMLA builds used `softmax_scale`).
    softmax_scale: float | None = None,
    d_v: int = 512,
    attn_sink: float | None = None,
    topk_length: int | None = None,
) -> torch.Tensor:
    """Convenience wrapper for `flash_mla.flash_mla_sparse_fwd`.

    Shapes (FlashMLA convention):
      - q: [s_q, h_q, d_qk] (bf16, CUDA)
      - kv: [s_kv, h_kv, d_qk] (bf16, CUDA; for DeepSeek MQA typically h_kv=1)
      - indices: [s_q, h_kv, topk] (int32, CUDA; -1 indicates invalid)
      - returns: [s_q, h_q, d_v]
    """
    flash = require_flash_mla()
    if not (q.is_cuda and kv.is_cuda and indices.is_cuda):
        raise ValueError("flash_mla_sparse_attn expects CUDA tensors.")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("flash_mla_sparse_attn expects bf16 q/kv tensors.")
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    if sm_scale is None:
        sm_scale = softmax_scale
    if sm_scale is None:
        sm_scale = float(q.shape[-1] ** -0.5)
    # flash_mla_sparse_fwd returns (out, max_logits, lse).
    try:
        out, _, _ = flash.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            float(sm_scale),
            d_v=int(d_v),
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
    except TypeError:  # pragma: no cover - depends on external FlashMLA version
        out, _, _ = flash.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            softmax_scale=float(sm_scale),
            d_v=int(d_v),
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
    return out
