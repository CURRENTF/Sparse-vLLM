import os

import torch
from torch import nn

from sparsevllm.layers.attention_backend import TritonAttentionBackend
from sparsevllm.utils.context import get_context

from sparsevllm.engine.sparse_controller import SparseController


def get_decode_workspace(
    context,
    batch_size: int,
    num_heads: int,
    num_blocks: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_o = (batch_size, num_heads, num_blocks, head_dim)
    shape_lse = (batch_size, num_heads, num_blocks)
    mid_o = context.decode_mid_o
    if (
        mid_o is None
        or mid_o.device != device
        or mid_o.shape[0] < batch_size
        or mid_o.shape[1] < num_heads
        or mid_o.shape[2] < num_blocks
        or mid_o.shape[3] < head_dim
    ):
        mid_o = torch.empty(shape_o, dtype=torch.float32, device=device)
        context.decode_mid_o = mid_o

    mid_lse = context.decode_mid_o_logexpsum
    if (
        mid_lse is None
        or mid_lse.device != device
        or mid_lse.shape[0] < batch_size
        or mid_lse.shape[1] < num_heads
        or mid_lse.shape[2] < num_blocks
    ):
        mid_lse = torch.empty(shape_lse, dtype=torch.float32, device=device)
        context.decode_mid_o_logexpsum = mid_lse

    return (
        mid_o[:batch_size, :num_heads, :num_blocks, :head_dim],
        mid_lse[:batch_size, :num_heads, :num_blocks],
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.attention_backend = TritonAttentionBackend()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        context = get_context()
        cache_manager = context.cache_manager
        sparse_controller: SparseController = context.sparse_controller
        layer_idx = context.now_layer_idx

        temp_slots = None
        try:
            if context.is_prefill:
                selection = sparse_controller.get_prefill_selection(layer_idx)
                prefill_view = cache_manager.build_prefill_compute_view(
                    layer_idx,
                    k,
                    v,
                    selection,
                )
                temp_slots = prefill_view.temp_slots

                if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() <= 1:
                    return torch.empty_like(q)

                b_start_loc = context.cu_seqlens_q[:-1]
                chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                max_context_len = prefill_view.max_context_len
                if max_context_len is not None:
                    max_input_len = int(max_context_len)
                elif torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    max_input_len = int(prefill_view.active_slots.shape[1])
                else:
                    max_input_len = prefill_view.context_lens.max().item()

                o = self.attention_backend.run_prefill(
                    q,
                    prefill_view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                    max_input_len=max_input_len,
                )
                cache_manager.collect_prefill_attention_score(
                    layer_idx,
                    q,
                    prefill_view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
                cache_manager.record_prefill_query(
                    layer_idx,
                    q,
                    prefill_view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
            else:    # decode
                batch_size = q.shape[0]
                selection = sparse_controller.get_decode_selection(
                    layer_idx,
                    q,
                )
                decode_view = cache_manager.build_decode_compute_view(
                    layer_idx,
                    q,
                    selection,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                )
                temp_slots = decode_view.temp_slots

                max_context_len = decode_view.max_context_len
                static_cap = getattr(cache_manager, "_decode_static_max_context_len", None)
                if static_cap is not None:
                    max_context_len = max(
                        int(max_context_len) if max_context_len is not None else 0,
                        int(static_cap),
                    )
                if max_context_len is None:
                    raise RuntimeError(f"static decode requires max_context_len, got None at layer={layer_idx}")
                max_len_in_batch = int(max_context_len)
                if decode_view.active_slots.dim() == 2:
                    slot_table_len = int(decode_view.active_slots.shape[1])
                    if (
                        os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") == "1"
                        and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
                    ):
                        actual_max_len = int(decode_view.context_lens.max().item()) if decode_view.context_lens.numel() > 0 else 0
                        if actual_max_len > slot_table_len:
                            raise RuntimeError(
                                "decode context length exceeds active slot table width: "
                                f"layer={layer_idx} context_lens_max={actual_max_len} "
                                f"slot_table_len={slot_table_len}"
                            )
                    if max_len_in_batch > slot_table_len:
                        max_len_in_batch = slot_table_len
                    if max_len_in_batch <= 0:
                        raise RuntimeError(
                            f"decode requires a positive context length, got {max_len_in_batch} at layer={layer_idx}"
                        )
                BLOCK_SEQ = cache_manager.get_decode_block_seq(layer_idx, 256)
                num_seq_blocks = (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ

                mid_o, mid_o_logexpsum = get_decode_workspace(
                    context,
                    batch_size,
                    self.num_heads,
                    num_seq_blocks,
                    self.head_dim,
                    q.device,
                )

                o = self.attention_backend.run_decode(
                    q,
                    decode_view,
                    mid_o=mid_o,
                    mid_o_logexpsum=mid_o_logexpsum,
                    max_len_in_batch=max_len_in_batch,
                    block_seq=BLOCK_SEQ,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                )
                cache_manager.record_decode_query(layer_idx, q)

            sparse_controller.on_layer_attention_end(layer_idx)
            cache_manager.on_layer_attention_end(layer_idx)
            return o
        finally:
            if temp_slots is not None and temp_slots.numel() > 0:
                cache_manager.release_layer_temp_slots(layer_idx, temp_slots)
