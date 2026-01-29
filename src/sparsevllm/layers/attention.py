import torch
from torch import nn
import triton
import triton.language as tl

from sparsevllm.triton_kernel.context_flashattention_nopad import context_attention_fwd
from sparsevllm.triton_kernel.flash_decoding_stage1 import flash_decode_stage1 as mha_flash_decode_stage1
from sparsevllm.triton_kernel.flash_decoding_stage1 import flash_decode_stage1_with_score as mha_flash_decode_stage1_with_score
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1 as gqa_flash_decode_stage1
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1_with_score as gqa_flash_decode_stage1_with_score
from sparsevllm.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger

from sparsevllm.engine.sparse_controller import SparseController


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(-1) == 1
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        cache_manager = context.cache_manager
        sparse_controller: SparseController = context.sparse_controller
        k_cache, v_cache, slot_mapping = cache_manager.get_layer_store_view(context.now_layer_idx)

        # 1. 写入 KV Cache (物理行为)
        # 无论是 DeltaKV 还是全量/SnapKV，均先将当前 KV 写入物理槽位 (对于 DeltaKV，是写入 Base Pool 作为 Recent)
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # 2. 获取逻辑视图
        layer_active_slots, layer_active_indices, layer_req_indices, layer_context_lens, layer_attn_score, deltakv_temp_slots = \
            sparse_controller.get_read_view(context.now_layer_idx)

        assert layer_active_slots is not None
        b_req_idx = layer_req_indices

        # --- 通用稀疏/全量路径 (使用 Triton) ---
        try:
            if context.is_prefill:
                if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() <= 1:
                    return torch.empty_like(q)

                b_start_loc = context.cu_seqlens_q[:-1]
                chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                b_seq_len = layer_context_lens
                b_prompt_cache_len = b_seq_len - chunk_lens
                max_input_len = b_seq_len.max().item()

                # Triton 路径需要物理槽位 layer_active_slots 用于 Req_to_tokens 寻址
                # 它内部通过 prompt_cache_len 实现因果掩码，目前不需要显式的 pos_ids
                o = torch.empty_like(q)
                context_attention_fwd(
                    q, k_cache, v_cache, o,
                    b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len,
                    layer_active_slots,
                    attn_score=layer_attn_score,
                )
            else:    # decode
                batch_size = q.shape[0]
                max_len_in_batch = layer_context_lens.max().item()
                BLOCK_SEQ = 256

                mid_o = torch.empty(
                    (batch_size, self.num_heads, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ, self.head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                mid_o_logexpsum = torch.empty(
                    (batch_size, self.num_heads, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ),
                    dtype=torch.float32,
                    device=q.device,
                )

                is_gqa = self.num_heads > self.num_kv_heads
                if layer_attn_score is not None:
                    if is_gqa:
                        gqa_flash_decode_stage1_with_score(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, layer_attn_score, BLOCK_SEQ,
                        )
                    else:
                        mha_flash_decode_stage1_with_score(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, layer_attn_score, BLOCK_SEQ,
                        )
                else:
                    if is_gqa:
                        gqa_flash_decode_stage1(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, BLOCK_SEQ,
                        )
                    else:
                        mha_flash_decode_stage1(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, BLOCK_SEQ,
                        )

                o = torch.empty_like(q)
                flash_decode_stage2(mid_o, mid_o_logexpsum, layer_context_lens, o, BLOCK_SEQ)

            return o
        finally:
            # DeltaKV reconstructs some KV into scratch slots; recycle them immediately after use.
            if deltakv_temp_slots is not None and deltakv_temp_slots.numel() > 0:
                cache_manager.free_temp_deltakv_full(deltakv_temp_slots)
