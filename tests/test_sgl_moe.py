from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from sparsevllm.kernels.external.sgl.moe import (
    sgl_moe_align_block_size,
    sgl_moe_alignment_support,
)
from sparsevllm.kernels.triton.moe import fused_moe, moe_align_block_size
from sparsevllm.operators.moe import _sgl_moe_align_block_size


def test_sgl_moe_support_rejects_missing_package() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        assert sgl_moe_alignment_support() == (
            False,
            "sgl-kernel is not installed",
        )


def test_sgl_moe_support_rejects_04_package() -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value="0.4.5"),
    ):
        supported, reason = sgl_moe_alignment_support()

    assert not supported
    assert ">=0.3.21,<0.4" in reason


def test_sgl_moe_support_accepts_declared_package() -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value="0.3.21"),
    ):
        supported, reason = sgl_moe_alignment_support()

    assert supported
    assert "0.3.21" in reason


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sgl-kernel are required",
)
def test_sgl_moe_alignment_matches_grouped_assignments() -> None:
    torch.manual_seed(20260808)
    topk_ids = torch.randint(
        0,
        64,
        (32, 4),
        dtype=torch.int32,
        device="cuda",
    )
    topk_ids[0].fill_(63)
    reference_sorted, reference_experts, reference_count = moe_align_block_size(
        topk_ids,
        16,
        64,
    )
    actual = sgl_moe_align_block_size(
        topk_ids,
        block_size=16,
        num_experts=64,
    )
    torch.cuda.synchronize()
    count = int(reference_count.item())
    assert int(actual.num_tokens_post_padded.item()) == count
    assert torch.equal(
        actual.expert_ids[: count // 16],
        reference_experts[: count // 16],
    )
    num_assignments = int(topk_ids.numel())
    for block_index in range(count // 16):
        block = slice(block_index * 16, (block_index + 1) * 16)
        reference_ids = sorted(
            value
            for value in reference_sorted[block].tolist()
            if value < num_assignments
        )
        actual_ids = sorted(
            value
            for value in actual.sorted_token_ids[block].tolist()
            if value < num_assignments
        )
        assert actual_ids == reference_ids


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sgl-kernel are required",
)
@pytest.mark.parametrize(("local_start", "local_end"), [(0, 32), (32, 64)])
def test_sgl_moe_alignment_matches_ep_shard(
    local_start: int,
    local_end: int,
) -> None:
    torch.manual_seed(20260810)
    topk_ids = torch.randint(
        0,
        64,
        (32, 4),
        dtype=torch.int32,
        device="cuda",
    )
    topk_ids[0] = torch.tensor([0, 31, 32, 63], device="cuda")
    reference_sorted, reference_experts, reference_count = moe_align_block_size(
        topk_ids,
        16,
        64,
        local_expert_start=local_start,
        local_expert_end=local_end,
    )
    actual = _sgl_moe_align_block_size(
        topk_ids,
        block_size=16,
        num_experts=64,
        local_expert_start=local_start,
        local_expert_end=local_end,
    )
    torch.cuda.synchronize()
    num_assignments = int(topk_ids.numel())

    def grouped_assignments(sorted_ids, expert_ids, count):
        grouped = {}
        for block_index in range(int(count.item()) // 16):
            expert_id = int(expert_ids[block_index].item())
            if expert_id < 0:
                continue
            block = slice(block_index * 16, (block_index + 1) * 16)
            grouped.setdefault(expert_id, []).extend(
                value
                for value in sorted_ids[block].tolist()
                if value < num_assignments
            )
        return {
            expert_id: sorted(assignment_ids)
            for expert_id, assignment_ids in grouped.items()
        }

    assert grouped_assignments(
        actual.sorted_token_ids,
        actual.expert_ids,
        actual.num_tokens_post_padded,
    ) == grouped_assignments(
        reference_sorted,
        reference_experts,
        reference_count,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sgl-kernel are required",
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 5, 64])
@pytest.mark.parametrize("all_remote", [False, True])
@pytest.mark.parametrize("local_start", [0, 4])
def test_sgl_ep_alignment_preserves_full_fused_moe_output(
    num_tokens: int,
    all_remote: bool,
    local_start: int,
) -> None:
    torch.manual_seed(20260810 + num_tokens + int(all_remote))
    device = torch.device("cuda")
    hidden_size = 64
    intermediate_size = 32
    num_experts = 8
    num_local_experts = 4
    top_k = 2
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w13_weight = torch.randn(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_weight = torch.randn(
        num_local_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    if all_remote:
        remote_start = num_local_experts if local_start == 0 else 0
        topk_ids = torch.randint(
            remote_start,
            remote_start + num_local_experts,
            (num_tokens, top_k),
            dtype=torch.int64,
            device=device,
        )
    else:
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            dtype=torch.int64,
            device=device,
        )
        remote_id = num_local_experts if local_start == 0 else 0
        topk_ids[0] = torch.tensor(
            [local_start, remote_id],
            device=device,
        )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    kwargs = {
        "num_experts": num_experts,
        "local_expert_start": local_start,
    }

    expected = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        **kwargs,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        alignment_impl=_sgl_moe_align_block_size,
        **kwargs,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)

    if num_tokens == 2:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_actual = fused_moe(
                hidden_states,
                w13_weight,
                w2_weight,
                topk_ids,
                topk_weights,
                alignment_impl=_sgl_moe_align_block_size,
                **kwargs,
            )
        hidden_states.copy_(torch.randn_like(hidden_states))
        if all_remote:
            replay_ids = torch.randint(
                remote_start,
                remote_start + num_local_experts,
                topk_ids.shape,
                dtype=topk_ids.dtype,
                device=device,
            )
        else:
            replay_ids = torch.randint(
                0,
                num_experts,
                topk_ids.shape,
                dtype=topk_ids.dtype,
                device=device,
            )
            remote_id = num_local_experts if local_start == 0 else 0
            replay_ids[0] = torch.tensor(
                [local_start + 1, remote_id],
                device=device,
            )
        topk_ids.copy_(replay_ids)
        replay_weights = torch.rand_like(topk_weights)
        replay_weights /= replay_weights.sum(dim=-1, keepdim=True)
        topk_weights.copy_(replay_weights)
        replay_expected = fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            **kwargs,
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_actual, replay_expected)
