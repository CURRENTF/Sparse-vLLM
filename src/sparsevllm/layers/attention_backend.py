import os

import torch

from sparsevllm.engine.cache_manager import DecodeComputeView, PrefillComputeView
from sparsevllm.utils.context import get_context
from sparsevllm.triton_kernel.context_flashattention_nopad import context_attention_fwd
from sparsevllm.triton_kernel.flash_decoding_stage1 import flash_decode_stage1 as mha_flash_decode_stage1
from sparsevllm.triton_kernel.flash_decoding_stage1 import flash_decode_stage1_with_score as mha_flash_decode_stage1_with_score
from sparsevllm.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.triton_kernel.flash_decoding_stage2_hd256 import flash_decode_stage2 as flash_decode_stage2_hd256
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1 as gqa_flash_decode_stage1
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1_with_score as gqa_flash_decode_stage1_with_score
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1_hd256 import flash_decode_stage1 as gqa_flash_decode_stage1_hd256
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1_hd256 import (
    flash_decode_stage1_with_score as gqa_flash_decode_stage1_hd256_with_score,
)
from sparsevllm.utils.profiler import profiler


def _fake_attention_enabled() -> bool:
    value = os.environ.get("SPARSEVLLM_FAKE_ATTENTION", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _fake_prefill_attention_enabled() -> bool:
    value = os.environ.get("SPARSEVLLM_FAKE_PREFILL_ATTENTION", "")
    return value.lower() in {"1", "true", "yes", "on"} or _fake_attention_enabled()


def _fake_decode_attention_enabled() -> bool:
    value = os.environ.get("SPARSEVLLM_FAKE_DECODE_ATTENTION", "")
    return value.lower() in {"1", "true", "yes", "on"} or _fake_attention_enabled()


def _fake_attention_output(q: torch.Tensor) -> torch.Tensor:
    mode = os.environ.get("SPARSEVLLM_FAKE_ATTENTION_MODE", "zero").strip().lower()
    if mode in {"zero", "zeros"}:
        return torch.zeros_like(q)
    if mode == "copy":
        return q.clone()
    if mode == "empty":
        return torch.empty_like(q)
    raise ValueError(
        "SPARSEVLLM_FAKE_ATTENTION_MODE must be one of 'zero', 'copy', or 'empty', "
        f"got {mode!r}."
    )


def _fill_fake_attention_score(attn_score: torch.Tensor | None) -> None:
    if attn_score is not None:
        attn_score.zero_()


class TritonAttentionBackend:
    """Thin backend wrapper around the existing Sparse-vLLM Triton attention kernels."""

    def run_prefill(
        self,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        max_input_len: int,
    ) -> torch.Tensor:
        b_seq_len = view.context_lens
        b_prompt_cache_len = b_seq_len - chunk_lens
        self._debug_check_prefill_bounds(q, view, chunk_lens=chunk_lens)
        if _fake_prefill_attention_enabled():
            _fill_fake_attention_score(view.attn_score)
            return _fake_attention_output(q)
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            view.k_cache,
            view.v_cache,
            o,
            view.req_indices,
            b_start_loc,
            b_seq_len,
            b_prompt_cache_len,
            max_input_len,
            view.active_slots,
            attn_score=view.attn_score,
        )
        return o

    def _debug_check_prefill_bounds(
        self,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        chunk_lens: torch.Tensor,
    ):
        if os.environ.get("SVLLM_DEBUG_PREFILL_BOUNDS", "0") != "1":
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if view.active_slots.dim() != 2:
            raise RuntimeError(
                f"prefill bounds check expects 2D active_slots, got shape={tuple(view.active_slots.shape)}"
            )
        rows = view.req_indices.to(torch.long)
        row_min = int(rows.min().item()) if rows.numel() > 0 else 0
        row_max = int(rows.max().item()) if rows.numel() > 0 else -1
        if row_min < 0 or row_max >= int(view.active_slots.shape[0]):
            raise RuntimeError(
                "prefill req row index out of bounds: "
                f"row_min={row_min} row_max={row_max} num_rows={int(view.active_slots.shape[0])}"
            )
        if int(chunk_lens.sum().item()) != int(q.shape[0]):
            raise RuntimeError(
                "prefill q/chunk length mismatch: "
                f"q_tokens={int(q.shape[0])} chunk_tokens={int(chunk_lens.sum().item())}"
            )
        if bool((view.context_lens < chunk_lens).any().item()):
            raise RuntimeError(
                "prefill context_lens shorter than chunk_lens: "
                f"context_lens={view.context_lens.detach().cpu().tolist()} "
                f"chunk_lens={chunk_lens.detach().cpu().tolist()}"
            )
        visible_len = int(view.context_lens.max().item()) if view.context_lens.numel() > 0 else 0
        if visible_len > int(view.active_slots.shape[1]):
            raise RuntimeError(
                "prefill visible length exceeds active slot table width: "
                f"visible_len={visible_len} active_slots_width={int(view.active_slots.shape[1])}"
            )
        visible_slots = view.active_slots.index_select(0, rows)[:, :visible_len]
        pos = torch.arange(visible_len, device=visible_slots.device)[None, :]
        valid_pos = pos < view.context_lens[:, None]
        slot_cap = int(view.k_cache.shape[0])
        bad = ((visible_slots < 0) | (visible_slots >= slot_cap)) & valid_pos
        if bool(bad.any().item()):
            layer_idx = getattr(get_context(), "now_layer_idx", None)
            loc = bad.nonzero(as_tuple=False)[0]
            bad_b = int(loc[0].item())
            bad_pos = int(loc[1].item())
            bad_slot = int(visible_slots[bad_b, bad_pos].item())
            bad_req_row = int(rows[bad_b].item())
            raise RuntimeError(
                "prefill physical slot out of bounds before attention: "
                f"layer={layer_idx} batch={bad_b} req_row={bad_req_row} pos={bad_pos} "
                f"slot={bad_slot} slot_cap={slot_cap} context_len={int(view.context_lens[bad_b].item())} "
                f"k_shape={tuple(view.k_cache.shape)} v_shape={tuple(view.v_cache.shape)} "
                f"active_slots_shape={tuple(view.active_slots.shape)}"
            )

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
    ) -> torch.Tensor:
        if _fake_decode_attention_enabled():
            _fill_fake_attention_score(view.attn_score)
            return _fake_attention_output(q)
        if view.backend == "full_layer_kivi":
            self._run_full_layer_kivi_decode_stage1(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_o_logexpsum,
                max_len_in_batch=max_len_in_batch,
                block_seq=block_seq,
            )
            o = torch.empty_like(q)
            flash_decode_stage2(mid_o, mid_o_logexpsum, view.context_lens, o, block_seq)
            return o
        if view.backend == "flash_attn_contiguous":
            from flash_attn import flash_attn_with_kvcache

            if view.active_slots.dim() != 2:
                raise RuntimeError("flash_attn_contiguous decode expects a 2D active slot table.")
            batch, width = int(view.active_slots.shape[0]), int(view.active_slots.shape[1])
            expected = batch * width
            if int(view.k_cache.shape[0]) < expected or int(view.v_cache.shape[0]) < expected:
                raise RuntimeError(
                    "flash_attn_contiguous decode got a cache smaller than the materialized active view: "
                    f"cache={int(view.k_cache.shape[0])}/{int(view.v_cache.shape[0])} expected={expected}."
                )
            k_cache = view.k_cache[:expected].view(batch, width, int(view.k_cache.shape[1]), int(view.k_cache.shape[2]))
            v_cache = view.v_cache[:expected].view(batch, width, int(view.v_cache.shape[1]), int(view.v_cache.shape[2]))
            with profiler.record("decode_attention_flash_attn_sparse"):
                # Decode uses q_len=1 and the materialized KV view contains no future tokens.
                out = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=view.context_lens.to(torch.int32),
                    causal=False,
                )
            return out.squeeze(1)

        self._debug_check_decode_bounds(view)
        profile_kind = "full" if int(max_len_in_batch) > 8192 else "sparse"
        is_gqa = int(num_heads) > int(num_kv_heads)
        use_gqa_hd256 = is_gqa and int(q.shape[-1]) == 256
        with profiler.record(f"decode_attention_stage1_{profile_kind}"):
            if view.attn_score is not None:
                if use_gqa_hd256:
                    gqa_flash_decode_stage1_hd256_with_score(
                        q,
                        view.k_cache,
                        view.v_cache,
                        view.active_slots,
                        view.req_indices,
                        view.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        view.attn_score,
                        block_seq,
                    )
                elif is_gqa:
                    gqa_flash_decode_stage1_with_score(
                        q,
                        view.k_cache,
                        view.v_cache,
                        view.active_slots,
                        view.req_indices,
                        view.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        view.attn_score,
                        block_seq,
                    )
                else:
                    mha_flash_decode_stage1_with_score(
                        q,
                        view.k_cache,
                        view.v_cache,
                        view.active_slots,
                        view.req_indices,
                        view.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        view.attn_score,
                        block_seq,
                    )
            else:
                if use_gqa_hd256:
                    gqa_flash_decode_stage1_hd256(
                        q,
                        view.k_cache,
                        view.v_cache,
                        view.active_slots,
                        view.req_indices,
                        view.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        block_seq,
                    )
                elif is_gqa:
                    gqa_flash_decode_stage1(
                        q,
                        view.k_cache,
                        view.v_cache,
                        view.active_slots,
                        view.req_indices,
                        view.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        block_seq,
                    )
                else:
                    mha_flash_decode_stage1(
                        q,
                        view.k_cache,
                        view.v_cache,
                        view.active_slots,
                        view.req_indices,
                        view.context_lens,
                        max_len_in_batch,
                        mid_o,
                        mid_o_logexpsum,
                        block_seq,
                    )

        o = torch.empty_like(q)
        with profiler.record(f"decode_attention_stage2_{profile_kind}"):
            if use_gqa_hd256:
                flash_decode_stage2_hd256(mid_o, mid_o_logexpsum, view.context_lens, o, block_seq)
            else:
                flash_decode_stage2(mid_o, mid_o_logexpsum, view.context_lens, o, block_seq)
        return o

    def _run_full_layer_kivi_decode_stage1(
        self,
        q: torch.Tensor,
        view: DecodeComputeView,
        *,
        mid_o: torch.Tensor,
        mid_o_logexpsum: torch.Tensor,
        max_len_in_batch: int,
        block_seq: int,
    ):
        meta = view.metadata
        if meta is None:
            raise RuntimeError("full_layer_kivi decode view is missing metadata.")
        from sparsevllm.triton_kernel.deltakv_kernels import full_layer_kivi_flash_decode_stage1

        full_layer_kivi_flash_decode_stage1(
            q=q,
            raw_k=view.k_cache,
            raw_v=view.v_cache,
            raw_slots_map=view.active_slots,
            kivi_block_slots_map=meta["kivi_block_slots_map"],
            kivi_block_start_pos=meta["kivi_block_start_pos"],
            key_packed=meta["key_packed"],
            key_scales=meta["key_scales"],
            key_mins=meta["key_mins"],
            value_packed=meta["value_packed"],
            value_scales=meta["value_scales"],
            value_mins=meta["value_mins"],
            req_indices=view.req_indices,
            context_lens=view.context_lens,
            max_len_in_batch=max_len_in_batch,
            mid_out=mid_o,
            mid_out_logsumexp=mid_o_logexpsum,
            group_size=int(meta["group_size"]),
            block_seq=block_seq,
            block_n=int(meta.get("block_n", 16)),
            num_warps=int(meta.get("num_warps", 2)),
            num_stages=int(meta.get("num_stages", 3)),
            attn_score=view.attn_score,
        )

    def _debug_check_decode_bounds(self, view: DecodeComputeView):
        if os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") != "1":
            return
        if view.backend != "dense":
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if view.active_slots.dim() != 2:
            raise RuntimeError(
                f"debug slot bounds check expects 2D active_slots, got shape={tuple(view.active_slots.shape)}"
            )
        rows = view.req_indices.to(torch.long)
        row_min = int(rows.min().item()) if rows.numel() > 0 else 0
        row_max = int(rows.max().item()) if rows.numel() > 0 else -1
        if row_min < 0 or row_max >= int(view.active_slots.shape[0]):
            raise RuntimeError(
                "decode req row index out of bounds: "
                f"row_min={row_min} row_max={row_max} num_rows={int(view.active_slots.shape[0])}"
            )
        visible_len = int(view.context_lens.max().item()) if view.context_lens.numel() > 0 else 0
        if visible_len > int(view.active_slots.shape[1]):
            raise RuntimeError(
                "decode visible length exceeds Req_to_tokens width: "
                f"visible_len={visible_len} req_to_tokens_width={int(view.active_slots.shape[1])}"
            )
        visible_slots = view.active_slots.index_select(0, rows)[:, :visible_len]
        pos = torch.arange(visible_len, device=visible_slots.device)[None, :]
        valid_pos = pos < view.context_lens[:, None]
        slot_cap = int(view.k_cache.shape[0])
        bad = ((visible_slots < 0) | (visible_slots >= slot_cap)) & valid_pos
        if bool(bad.any().item()):
            loc = bad.nonzero(as_tuple=False)[0]
            bad_b = int(loc[0].item())
            bad_pos = int(loc[1].item())
            bad_slot = int(visible_slots[bad_b, bad_pos].item())
            bad_req_row = int(rows[bad_b].item())
            raise RuntimeError(
                "decode physical slot out of bounds before stage1: "
                f"batch={bad_b} req_row={bad_req_row} pos={bad_pos} "
                f"slot={bad_slot} slot_cap={slot_cap} context_len={int(view.context_lens[bad_b].item())}"
            )
