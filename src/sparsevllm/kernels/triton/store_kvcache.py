from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


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


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n_tokens, num_heads, head_dim = key.shape
    d_model = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(-1) == 1
    assert slot_mapping.numel() == n_tokens
    max_launch_tokens = int(os.getenv("SPARSEVLLM_STORE_KVCACHE_CHUNK_TOKENS", "524288") or 0)
    if max_launch_tokens <= 0 or n_tokens <= max_launch_tokens:
        store_kvcache_kernel[(n_tokens,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            d_model,
        )
        return

    for start in range(0, n_tokens, max_launch_tokens):
        end = min(n_tokens, start + max_launch_tokens)
        store_kvcache_kernel[(end - start,)](
            key[start:end],
            key.stride(0),
            value[start:end],
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping[start:end],
            d_model,
        )
