from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm.layers.attention_backend import (
    TritonAttentionBackend,
    _require_explicit_payload,
)


@dataclass
class _FlashInferState:
    wrapper: object
    plan_key: tuple[object, int, int, int] | None = None


class Gemma4FlashInferPrefill:
    """Shared FlashInfer plans for Gemma 4 text-prefill head shapes."""

    def __init__(self) -> None:
        self._states: dict[tuple[int, int, int, int], _FlashInferState] = {}

    @staticmethod
    def _page_metadata(view, max_context_len: int):
        meta = view.meta
        rows = meta.active_slots.index_select(0, meta.req_indices.to(torch.long))[
            :, :max_context_len
        ]
        positions = torch.arange(
            max_context_len,
            device=meta.context_lens.device,
            dtype=meta.context_lens.dtype,
        )
        indices = rows.masked_select(
            positions.unsqueeze(0) < meta.context_lens.unsqueeze(1)
        ).to(torch.int32).contiguous()
        indptr = torch.cat(
            (
                torch.zeros(1, device=indices.device, dtype=torch.int32),
                meta.context_lens.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        return indices, indptr, torch.ones_like(meta.context_lens, dtype=torch.int32)

    def run(
        self,
        q: torch.Tensor,
        view,
        *,
        q_start: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_context_len: int,
        sliding_window: int | None,
    ) -> torch.Tensor:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
        from sparsevllm.utils.context import get_context

        payload = _require_explicit_payload(view, operation="Gemma 4 prefill")
        meta = view.meta
        if meta.active_slots.dtype != torch.int32 or meta.active_slots.ndim != 2:
            raise TypeError("Gemma 4 FlashInfer prefill requires an int32 page table.")
        q_heads, kv_heads, head_dim = map(
            int, (q.shape[1], payload.k_cache.shape[1], q.shape[2])
        )
        window_left = -1 if sliding_window is None else int(sliding_window) - 1
        key = q_heads, kv_heads, head_dim, window_left
        state = self._states.get(key)
        if state is None:
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
            state = _FlashInferState(
                BatchPrefillWithPagedKVCacheWrapper(
                    workspace, kv_layout="NHD", backend="auto"
                )
            )
            self._states[key] = state
        context = get_context()
        plan_key = (
            context.attention_validation_scope,
            meta.active_slots.data_ptr(),
            meta.req_indices.data_ptr(),
            meta.context_lens.data_ptr(),
        )
        if state.plan_key != plan_key:
            indices, kv_indptr, last_page_len = self._page_metadata(
                view, int(max_context_len)
            )
            qo_indptr = torch.cat((q_start, q_start[-1:] + chunk_lens[-1:]))
            state.wrapper.plan(
                qo_indptr,
                kv_indptr,
                indices,
                last_page_len,
                q_heads,
                kv_heads,
                head_dim,
                1,
                causal=True,
                sm_scale=1.0,
                window_left=window_left,
                q_data_type=q.dtype,
                kv_data_type=payload.k_cache.dtype,
                non_blocking=True,
            )
            state.plan_key = plan_key
        output = torch.empty_like(q)
        state.wrapper.run(
            q,
            (payload.k_cache.unsqueeze(1), payload.v_cache.unsqueeze(1)),
            out=output,
        )
        return output


class Gemma4AttentionBackend(TritonAttentionBackend):
    """Gemma 4 attention semantics isolated from the tuned generic kernels."""

    name = "triton_gemma4"

    def __init__(
        self,
        *,
        sliding_window: int | None,
        flashinfer_prefill: Gemma4FlashInferPrefill | None = None,
        use_window_decode: bool = False,
        global_decode_heads_per_program: int | None = None,
    ) -> None:
        super().__init__()
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self.flashinfer_prefill = flashinfer_prefill
        self.use_window_decode = bool(use_window_decode)
        self.global_decode_heads_per_program = global_decode_heads_per_program

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
        from sparsevllm.utils.context import get_context

        image_groups = getattr(get_context(), "multimodal_image_groups", None)
        if self.sliding_window is not None and isinstance(image_groups, torch.Tensor):
            from sparsevllm.kernels.triton.gemma4_multimodal_context_attention import (
                gemma4_multimodal_context_attention,
            )

            output = torch.empty_like(q)
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
        if self.flashinfer_prefill is not None and view.meta.attn_score is None:
            return self.flashinfer_prefill.run(
                q,
                view,
                q_start=b_start_loc,
                chunk_lens=chunk_lens,
                max_context_len=max_input_len,
                sliding_window=self.sliding_window,
            )
        output = torch.empty_like(q)
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
        if (
            self.use_window_decode
            and self.sliding_window is not None
            and view.meta.attn_score is None
            and int(q.shape[-1]) == 256
            and group_size in {2, 4}
            and mid_o.shape[2]
            >= (self.sliding_window + block_seq - 1) // block_seq
        ):
            from sparsevllm.kernels.triton.gemma4_window_decode_attention import (
                gemma4_window_decode,
            )

            output = torch.empty_like(q)
            window_blocks = (self.sliding_window + block_seq - 1) // block_seq
            gemma4_window_decode(
                q,
                payload.k_cache,
                payload.v_cache,
                view.meta.active_slots,
                view.meta.req_indices,
                view.meta.context_lens,
                mid_o[:, :, :window_blocks],
                mid_o_logexpsum[:, :, :window_blocks],
                output,
                block_seq=block_seq,
                sliding_window=self.sliding_window,
            )
            return output
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
        if (
            self.sliding_window is None
            and view.meta.attn_score is None
            and int(q.shape[-1]) == 512
            and self.global_decode_heads_per_program is not None
            and group_size % self.global_decode_heads_per_program == 0
        ):
            from sparsevllm.kernels.triton.gemma4_global_decode_attention import (
                gemma4_global_decode_stage1,
            )

            gemma4_global_decode_stage1(
                q,
                payload.k_cache,
                payload.v_cache,
                view.meta.active_slots,
                view.meta.req_indices,
                view.meta.context_lens,
                mid_o,
                mid_o_logexpsum,
                block_seq=block_seq,
                heads_per_program=self.global_decode_heads_per_program,
            )
            output = torch.empty_like(q)
            gemma4_decode_stage2(
                mid_o,
                mid_o_logexpsum,
                view.meta.context_lens,
                output,
                block_seq=block_seq,
                sliding_window=None,
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


__all__ = ["Gemma4AttentionBackend", "Gemma4FlashInferPrefill"]
