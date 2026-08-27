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


@triton.jit
def store_kvcache_with_quest_metadata_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    page_max_ptr,
    page_min_ptr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

    page_slot = slot // PAGE_SIZE
    page_offset = slot % PAGE_SIZE
    metadata_offsets = page_slot * D + offsets
    old_max = tl.load(
        page_max_ptr + metadata_offsets,
        mask=page_offset != 0,
        other=-float("inf"),
    )
    old_min = tl.load(
        page_min_ptr + metadata_offsets,
        mask=page_offset != 0,
        other=float("inf"),
    )
    tl.store(page_max_ptr + metadata_offsets, tl.maximum(old_max, key))
    tl.store(page_min_ptr + metadata_offsets, tl.minimum(old_min, key))


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


def store_kvcache_with_quest_metadata(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Store decode KV and update the owning QuEST page bounds atomically."""

    n_tokens, num_heads, head_dim = key.shape
    d_model = num_heads * head_dim
    if page_max.shape != page_min.shape or page_max.shape[1:] != key.shape[1:]:
        raise ValueError(
            "QuEST page metadata must have matching [pages, heads, dim] shapes"
        )
    if page_max.dtype != key.dtype or page_min.dtype != key.dtype:
        raise TypeError("QuEST page metadata must use the key dtype")
    if page_max.device != key.device or page_min.device != key.device:
        raise ValueError("QuEST page metadata must share the key device")
    if not page_max.is_contiguous() or not page_min.is_contiguous():
        raise ValueError("QuEST page metadata must be contiguous")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("KV innermost dimensions must be contiguous")
    if key.stride(1) != head_dim or value.stride(1) != head_dim:
        raise ValueError("KV heads must be densely packed")
    if k_cache.stride(-1) != 1 or slot_mapping.numel() != n_tokens:
        raise ValueError("KV cache layout or slot_mapping shape is invalid")
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    expected_pages = (int(k_cache.shape[0]) + page_size - 1) // page_size
    if int(page_max.shape[0]) != expected_pages:
        raise ValueError(
            "QuEST page metadata capacity must cover the KV cache exactly: "
            f"expected_pages={expected_pages} got={int(page_max.shape[0])}"
        )
    if n_tokens == 0:
        return
    store_kvcache_with_quest_metadata_kernel[(n_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        page_max,
        page_min,
        D=d_model,
        PAGE_SIZE=page_size,
    )
