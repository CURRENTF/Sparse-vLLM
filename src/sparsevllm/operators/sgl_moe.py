from __future__ import annotations

import torch
import triton

from sparsevllm.operators.sgl_kernel import sgl_kernel_support
from sparsevllm.triton_kernel.moe import MoeAlignment


def sgl_moe_alignment_support() -> tuple[bool, str]:
    """Check the SGL expert-alignment API used by the Triton MoE provider."""

    return sgl_kernel_support("MoE alignment")


def sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> MoeAlignment:
    """Build a local MoE assignment with the SGL CUDA kernel."""

    local_expert_start = int(local_expert_start)
    local_expert_end = int(local_expert_end)
    num_local_experts = local_expert_end - local_expert_start
    if not 0 <= local_expert_start < local_expert_end <= int(num_experts):
        raise ValueError(
            "Invalid local expert range "
            f"[{local_expert_start}, {local_expert_end}) for {num_experts} experts."
        )
    supported, reason = sgl_moe_alignment_support()
    if not supported:
        raise RuntimeError(reason)
    local_topk_ids = topk_ids
    has_remote_experts = num_local_experts != int(num_experts)
    if has_remote_experts:
        from sparsevllm.triton_kernel.moe import localize_expert_ids

        local_topk_ids = localize_expert_ids(
            topk_ids,
            local_expert_start=local_expert_start,
            local_expert_end=local_expert_end,
            remote_expert_id=num_local_experts,
        )
    num_assignments = int(local_topk_ids.numel())
    padding_experts = num_local_experts + int(has_remote_experts)
    max_num_tokens_padded = triton.cdiv(
        num_assignments + padding_experts * (int(block_size) - 1),
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
        num_local_experts + 1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    from sgl_kernel import moe_align_block_size

    # sgl-kernel 0.3.x iterates to num_experts - 1. The extra empty logical
    # expert makes the complete [0, num_experts) range participate.
    moe_align_block_size(
        local_topk_ids,
        num_local_experts + 1,
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
    "sgl_moe_alignment_support",
    "sgl_moe_align_block_size",
]
