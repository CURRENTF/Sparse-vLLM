from __future__ import annotations

import importlib.metadata
from unittest.mock import patch

import pytest
import torch

from sparsevllm.operators.sgl_moe import (
    SglGlmFusedMoeGate,
    _parse_version,
    sgl_fused_moe_gate_support,
    sgl_moe_alignment_support,
    sgl_moe_align_block_size,
    sgl_moe_ep_alignment_support,
)
from sparsevllm.triton_kernel.moe import moe_align_block_size
from sparsevllm.triton_kernel.moe import fused_moe


def test_sgl_moe_version_parser_accepts_post_release() -> None:
    assert _parse_version("0.3.14.post1+cu128") == (0, 3, 14)


def test_sgl_moe_support_rejects_missing_package() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        assert sgl_fused_moe_gate_support() == (
            False,
            "sgl-kernel is not installed",
        )


def test_sgl_moe_support_accepts_sglang_kernel_04_package() -> None:
    def package_version(distribution: str) -> str:
        if distribution == "sglang-kernel":
            return "0.4.5"
        raise importlib.metadata.PackageNotFoundError(distribution)

    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", side_effect=package_version),
    ):
        gate_supported, gate_reason = sgl_fused_moe_gate_support()
        align_supported, align_reason = sgl_moe_alignment_support()
        ep_supported, ep_reason = sgl_moe_ep_alignment_support()

    assert gate_supported
    assert align_supported
    assert ep_supported
    assert "0.4.5" in gate_reason
    assert "0.4.5" in align_reason
    assert "0.4.5" in ep_reason


def test_sgl_moe_ep_support_accepts_canonical_03_package() -> None:
    def package_version(distribution: str) -> str:
        if distribution == "sgl-kernel":
            return "0.3.21"
        raise importlib.metadata.PackageNotFoundError(distribution)

    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", side_effect=package_version),
    ):
        supported, reason = sgl_moe_ep_alignment_support()

    assert supported
    assert "0.3.21" in reason


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fused_moe_gate_support()[0],
    reason="CUDA and a validated sgl-kernel are required",
)
def test_sgl_glm_fused_gate_matches_reference_and_replays_graph() -> None:
    torch.manual_seed(20260807)
    logits = torch.randn(32, 64, dtype=torch.float32, device="cuda") * 3
    correction_bias = (
        torch.randn(64, dtype=torch.float32, device="cuda") * 0.1
    )
    scaling = 1.8
    scores = torch.sigmoid(logits)
    reference_ids = torch.topk(
        scores + correction_bias,
        4,
        dim=-1,
        sorted=False,
    ).indices
    reference_weights = scores.gather(1, reference_ids)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True) + 1e-20
    reference_weights *= scaling
    gate = SglGlmFusedMoeGate(num_experts=64, top_k=4)

    actual_weights, actual_ids = gate(
        logits,
        correction_bias,
        top_k=4,
        routed_scaling_factor=scaling,
    )

    reference_order = reference_ids.argsort(dim=-1)
    actual_order = actual_ids.argsort(dim=-1)
    sorted_reference_ids = reference_ids.gather(1, reference_order)
    sorted_actual_ids = actual_ids.gather(1, actual_order)
    sorted_reference_weights = reference_weights.gather(1, reference_order)
    sorted_actual_weights = actual_weights.gather(1, actual_order)
    assert torch.equal(sorted_actual_ids, sorted_reference_ids)
    torch.testing.assert_close(
        sorted_actual_weights,
        sorted_reference_weights,
        rtol=2e-6,
        atol=2e-7,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = gate(
            logits,
            correction_bias,
            top_k=4,
            routed_scaling_factor=scaling,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_ids, actual_ids)
    torch.testing.assert_close(graph_weights, actual_weights, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fused_moe_gate_support()[0],
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
        local_expert_start=0,
        local_expert_end=64,
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
    not torch.cuda.is_available() or not sgl_moe_ep_alignment_support()[0],
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
    actual = sgl_moe_align_block_size(
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
    not torch.cuda.is_available() or not sgl_moe_ep_alignment_support()[0],
    reason="CUDA and a validated sgl-kernel are required",
)
@pytest.mark.parametrize("num_tokens", [5, 64])
@pytest.mark.parametrize("all_remote", [False, True])
def test_sgl_ep_alignment_preserves_full_fused_moe_output(
    num_tokens: int,
    all_remote: bool,
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
        topk_ids = torch.randint(
            num_local_experts,
            num_experts,
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
        topk_ids[0] = torch.tensor([0, num_local_experts], device=device)
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    kwargs = {
        "num_experts": num_experts,
        "local_expert_start": 0,
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
        alignment_impl=sgl_moe_align_block_size,
        **kwargs,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
