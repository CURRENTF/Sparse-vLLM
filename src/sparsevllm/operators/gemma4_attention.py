from __future__ import annotations

import torch

from sparsevllm.layers.attention_backend import (
    TritonAttentionBackend,
    _require_explicit_payload,
)


class Gemma4AttentionBackend(TritonAttentionBackend):
    """Gemma 4 attention semantics isolated from the tuned generic kernels."""

    name = "triton_gemma4"

    def __init__(self, *, sliding_window: int | None) -> None:
        super().__init__()
        self.sliding_window = None if sliding_window is None else int(sliding_window)

    def run_prefill(
        self,
        q: torch.Tensor,
        view,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_input_len: int,
    ) -> torch.Tensor:
        payload = _require_explicit_payload(view, operation="Gemma 4 prefill")
        output = torch.empty_like(q)
        from sparsevllm.utils.context import get_context

        image_groups = getattr(get_context(), "multimodal_image_groups", None)
        if self.sliding_window is not None and isinstance(image_groups, torch.Tensor):
            from sparsevllm.kernels.triton.gemma4_multimodal_context_attention import (
                gemma4_multimodal_context_attention,
            )

            gemma4_multimodal_context_attention(
                q,
                payload.k_cache,
                payload.v_cache,
                output,
                view.meta.req_indices,
                b_start_loc,
                view.meta.context_lens,
                view.meta.context_lens - chunk_lens,
                max_input_len,
                view.meta.active_slots,
                image_groups,
                sliding_window=self.sliding_window,
                attn_score=view.meta.attn_score,
            )
            return output
        from sparsevllm.kernels.triton.gemma4_context_attention import (
            gemma4_context_attention,
        )

        gemma4_context_attention(
            q,
            payload.k_cache,
            payload.v_cache,
            output,
            view.meta.req_indices,
            b_start_loc,
            view.meta.context_lens,
            view.meta.context_lens - chunk_lens,
            max_input_len,
            view.meta.active_slots,
            sliding_window=self.sliding_window,
            attn_score=view.meta.attn_score,
        )
        return output

    def run_decode(
        self,
        q: torch.Tensor,
        view,
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
        del max_len_in_batch, num_heads, num_kv_heads, gqa_block_n, gqa_num_warps
        payload = _require_explicit_payload(view, operation="Gemma 4 decode")
        from sparsevllm.kernels.triton.gemma4_decode_attention import (
            gemma4_decode_stage1,
            gemma4_decode_stage2,
        )

        group_size = int(q.shape[1]) // int(payload.k_cache.shape[1])
        if mid_o.shape[2] == 1 and view.meta.attn_score is None and group_size in {2, 4, 8}:
            from sparsevllm.kernels.triton.gemma4_single_block_decode_attention import (
                gemma4_single_block_decode,
            )

            output = torch.empty_like(q)
            gemma4_single_block_decode(
                q, payload.k_cache, payload.v_cache, view.meta.active_slots,
                view.meta.req_indices, view.meta.context_lens, output,
                block_seq=block_seq, sliding_window=self.sliding_window,
            )
            return output
        gemma4_decode_stage1(
            q, payload.k_cache, payload.v_cache, view.meta.active_slots,
            view.meta.req_indices, view.meta.context_lens, mid_o,
            mid_o_logexpsum, block_seq=block_seq,
            sliding_window=self.sliding_window,
            attn_score=view.meta.attn_score,
        )
        output = torch.empty_like(q)
        gemma4_decode_stage2(
            mid_o,
            mid_o_logexpsum,
            view.meta.context_lens,
            output,
            block_seq=block_seq,
            sliding_window=self.sliding_window,
        )
        return output


__all__ = ["Gemma4AttentionBackend"]
