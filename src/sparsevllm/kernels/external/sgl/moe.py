from __future__ import annotations

import importlib

import torch
import triton

from sparsevllm.kernels.external.sgl.support import sgl_kernel_support
from sparsevllm.kernels.moe import MoeAlignment


def sgl_moe_alignment_support() -> tuple[bool, str]:
    """Check the SGL expert-alignment API used by the Triton MoE provider."""

    supported, reason = sgl_kernel_support("MoE alignment")
    if not supported:
        return supported, reason
    try:
        alignment = importlib.import_module("sgl_kernel").moe_align_block_size
    except Exception as error:
        return False, (
            "sglang-kernel MoE alignment failed to load: "
            f"{type(error).__name__}: {error}"
        )
    return (True, reason) if callable(alignment) else (
        False,
        "sglang-kernel moe_align_block_size is not callable",
    )


def sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
) -> MoeAlignment:
    """Group local expert assignments with the SGL CUDA kernel."""

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError(f"SGL MoE alignment requires experts, got {num_experts}.")
    supported, reason = sgl_moe_alignment_support()
    if not supported:
        raise RuntimeError(reason)
    num_assignments = int(topk_ids.numel())
    max_num_tokens_padded = triton.cdiv(
        num_assignments + num_experts * (int(block_size) - 1),
        int(block_size),
    ) * int(block_size)
    sorted_token_ids = torch.empty(
        max_num_tokens_padded,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        max_num_tokens_padded // int(block_size),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum_buffer = torch.empty(
        num_experts + 1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    from sgl_kernel import moe_align_block_size

    # The extra empty logical expert makes the complete [0, num_experts)
    # range participate; hardware tests cover assignments to the final expert.
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        int(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )
    return MoeAlignment(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size=int(block_size),
        naive=False,
    )


__all__ = [
    "sgl_moe_align_block_size",
    "sgl_moe_alignment_support",
]
