"""Experimental batch-only decode-attention provider.

This module is deliberately separate from the stable Triton backend so the
bucketed CUDA Graph path remains an unchanged correctness and performance
baseline while the fixed-split implementation is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm.engine.cache_manager import DecodeComputeView
from sparsevllm.kernels.triton.context_independent_flash_decoding import (
    context_independent_flash_decode,
)
from sparsevllm.layers.attention_backend import (
    TritonAttentionBackend,
    _fake_attention_output,
    _fake_decode_attention_enabled,
    _fill_fake_attention_score,
    _require_explicit_payload,
)
from sparsevllm.utils.profiler import profiler


@dataclass(frozen=True)
class ContextIndependentDecodeTuning:
    max_kv_splits: int = 16
    target_tokens_per_split: int = 1024
    block_n: int = 64
    num_warps: int = 2

    def __post_init__(self) -> None:
        if self.max_kv_splits <= 0 or self.target_tokens_per_split <= 0:
            raise ValueError("context-independent split settings must be positive")


class ContextIndependentTritonAttentionBackend(TritonAttentionBackend):
    """Fixed-workspace MHA/GQA provider selected only by batch-only policy."""

    name = "triton_context_independent"
    cuda_graph_context_independent = True

    def __init__(
        self,
        *,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        tuning: ContextIndependentDecodeTuning,
    ) -> None:
        super().__init__()
        self.max_batch_size = int(max_batch_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.tuning = tuning
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self._mid_o = torch.empty(
            (
                self.max_batch_size,
                self.num_heads,
                tuning.max_kv_splits,
                self.head_dim,
            ),
            dtype=torch.float32,
            device=device,
        )
        self._mid_lse = torch.empty(
            (self.max_batch_size, self.num_heads, tuning.max_kv_splits),
            dtype=torch.float32,
            device=device,
        )

    @property
    def workspace_bytes(self) -> int:
        return (
            self._mid_o.numel() * self._mid_o.element_size()
            + self._mid_lse.numel() * self._mid_lse.element_size()
        )

    def get_decode_workspace(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            int(batch_size) > self.max_batch_size
            or int(num_heads) != self.num_heads
            or int(head_dim) != self.head_dim
            or device != self._mid_o.device
        ):
            raise RuntimeError(
                "context-independent decode workspace mismatch: "
                f"requested={(batch_size, num_heads, head_dim, device)} "
                f"prepared={(self.max_batch_size, self.num_heads, self.head_dim, self._mid_o.device)}"
            )
        return self._mid_o[:batch_size], self._mid_lse[:batch_size]

    def run_decode(
        self,
        q: torch.Tensor,
        view: DecodeComputeView,
        *,
        mid_o: torch.Tensor,
        mid_o_logexpsum: torch.Tensor,
        max_len_in_batch: int,
        block_seq: int,
        num_heads: int,
        num_kv_heads: int,
        gqa_block_n: int = 16,
        gqa_num_warps: int = 2,
    ) -> torch.Tensor:
        del max_len_in_batch, block_seq, num_heads, num_kv_heads, gqa_block_n, gqa_num_warps
        payload = _require_explicit_payload(
            view,
            operation="context-independent Triton decode",
        )
        if _fake_decode_attention_enabled():
            _fill_fake_attention_score(view.meta.attn_score)
            return _fake_attention_output(q)
        if payload.backend != "dense":
            raise RuntimeError(
                "context-independent Triton decode currently requires the dense explicit-KV "
                f"payload, got backend={payload.backend!r}"
            )
        self._debug_check_decode_bounds(view)
        with profiler.record("decode_attention_context_independent"):
            return context_independent_flash_decode(
                q,
                payload.k_cache,
                payload.v_cache,
                view.meta.active_slots,
                view.meta.req_indices,
                view.meta.context_lens,
                mid_o,
                mid_o_logexpsum,
                attn_score=view.meta.attn_score,
                target_tokens_per_split=self.tuning.target_tokens_per_split,
                block_n=self.tuning.block_n,
                num_warps=self.tuning.num_warps,
            )


def bind_context_independent_triton_attention(
    model: torch.nn.Module,
    *,
    max_batch_size: int,
    device: torch.device,
    tuning: ContextIndependentDecodeTuning | None = None,
) -> tuple[int, int]:
    """Bind one shared fixed workspace per attention shape.

    Returns ``(bound_layer_count, workspace_bytes)``. Specialized Gemma and MLA
    modules are intentionally not matched here and are handled by their own
    experimental providers.
    """
    default_tuning = tuning or ContextIndependentDecodeTuning()
    providers: dict[tuple[int, int], ContextIndependentTritonAttentionBackend] = {}
    bound = 0
    for module in model.modules():
        backend = getattr(module, "attention_backend", None)
        if type(backend) is not TritonAttentionBackend:
            continue
        num_heads = int(getattr(module, "num_heads"))
        head_dim = int(getattr(module, "head_dim"))
        signature = (num_heads, head_dim)
        provider = providers.get(signature)
        if provider is None:
            shape_tuning = (
                ContextIndependentDecodeTuning(
                    max_kv_splits=default_tuning.max_kv_splits,
                    target_tokens_per_split=default_tuning.target_tokens_per_split,
                    block_n=128,
                    num_warps=4,
                )
                if tuning is None and head_dim == 256
                else default_tuning
            )
            provider = ContextIndependentTritonAttentionBackend(
                max_batch_size=max_batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                device=device,
                tuning=shape_tuning,
            )
            providers[signature] = provider
        module.attention_backend = provider
        bound += 1
    return bound, sum(provider.workspace_bytes for provider in providers.values())
