import unittest

import pytest
import torch
import torch.nn.functional as F

from sparsevllm.triton_kernel.moe import fused_moe, moe_align_block_size
from sparsevllm.triton_kernel.moe_topk import topk_softmax


def _pytorch_topk_reference(
    logits: torch.Tensor,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(probabilities, 8, dim=-1)
    if norm_topk_prob:
        weights /= weights.sum(dim=-1, keepdim=True)
    return weights, ids


def _oracle_local_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_start: int,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    local_expert_end = local_expert_start + int(w13_weight.shape[0])
    for local_expert_id in range(int(w13_weight.shape[0])):
        global_expert_id = local_expert_start + local_expert_id
        token_ids, topk_slots = torch.where(topk_ids == global_expert_id)
        if token_ids.numel() == 0:
            continue
        assert local_expert_start <= global_expert_id < local_expert_end
        gate_up = F.linear(hidden_states[token_ids], w13_weight[local_expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, w2_weight[local_expert_id])
        expert_output *= topk_weights[token_ids, topk_slots, None]
        output.index_add_(0, token_ids, expert_output.to(output.dtype))
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_moe_align_block_size_filters_ep_experts_and_pads_blocks():
    topk_ids = torch.tensor(
        [[2, 0], [3, 2], [5, 2], [0, 1]],
        dtype=torch.int64,
        device="cuda",
    )
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        4,
        6,
        local_expert_start=2,
        local_expert_end=4,
    )
    torch.cuda.synchronize()

    invalid = topk_ids.numel()
    assert int(num_tokens_post_padded.item()) == 8
    assert expert_ids[:2].tolist() == [0, 1]
    assert all(expert_id == -1 for expert_id in expert_ids[2:].tolist())
    assert sorted(sorted_ids[:4].tolist()) == sorted([0, 3, 5, invalid])
    assert sorted(sorted_ids[4:8].tolist()) == sorted([2, invalid, invalid, invalid])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_moe_alignment_covers_hotspot_and_empty_rank(dtype):
    for topk_ids, local_start, local_end in (
        (torch.zeros((128, 8), dtype=dtype, device="cuda"), 0, 64),
        (torch.zeros((16, 8), dtype=dtype, device="cuda"), 64, 128),
    ):
        sorted_ids, expert_ids, num_padded = moe_align_block_size(
            topk_ids,
            16,
            128,
            local_expert_start=local_start,
            local_expert_end=local_end,
        )
        torch.cuda.synchronize()
        expected = 1024 if local_start == 0 else 0
        assert int(num_padded.item()) == expected
        assert int((expert_ids >= 0).sum().item()) == expected // 16
        valid = sorted_ids[sorted_ids < topk_ids.numel()]
        assert int(valid.numel()) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_matches_pytorch(dtype, norm_topk_prob):
    torch.manual_seed(21)
    base = torch.arange(128, dtype=dtype, device="cuda") / 16 - 4
    logits = torch.stack(
        [base[torch.randperm(128, device="cuda")] for _ in range(257)]
    )
    expected_probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_ids = torch.topk(expected_probs, 8, dim=-1)
    if norm_topk_prob:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    actual_weights, actual_ids = topk_softmax(
        logits,
        top_k=8,
        norm_topk_prob=norm_topk_prob,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_ids, expected_ids.to(torch.int32))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 4e-3
    assert torch.allclose(
        actual_weights.float(),
        expected_weights,
        atol=tolerance,
        rtol=tolerance,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_uses_lowest_expert_id_for_ties():
    logits = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    _, ids = topk_softmax(logits, top_k=8, norm_topk_prob=True)
    torch.cuda.synchronize()
    assert ids.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_is_stable_for_extreme_finite_logits():
    logits = torch.full((2, 128), -100, dtype=torch.bfloat16, device="cuda")
    logits[:, :8] = torch.arange(100, 92, -1, dtype=torch.bfloat16, device="cuda")
    expected_weights, expected_ids = _pytorch_topk_reference(logits, True)

    weights, ids = topk_softmax(logits, top_k=8, norm_topk_prob=True)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.allclose(weights.float(), expected_weights, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_nonfinite_inputs_keep_ids_in_range_and_propagate_nan():
    logits = torch.zeros(2, 128, dtype=torch.bfloat16, device="cuda")
    logits[0, 3] = float("nan")
    logits[1, 7] = float("inf")

    weights, ids = topk_softmax(logits, top_k=8, norm_topk_prob=False)
    torch.cuda.synchronize()

    assert bool(((ids >= 0) & (ids < 128)).all())
    assert not bool(torch.isfinite(weights).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_rejects_unsupported_shape_and_layout():
    with pytest.raises(ValueError, match="num_experts=128"):
        topk_softmax(
            torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda"),
            top_k=8,
            norm_topk_prob=True,
        )
    non_contiguous = torch.zeros(128, 2, dtype=torch.bfloat16, device="cuda").T
    with pytest.raises(ValueError, match="contiguous"):
        topk_softmax(non_contiguous, top_k=8, norm_topk_prob=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_naive_decode_matches_oracle(dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_experts = 16
    hidden_size = 37
    intermediate_size = 23
    hidden_states = torch.randn(1, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.tensor([[3, 12]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor([[0.65, 0.35]], dtype=dtype, device=device)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_aligned_prefill_matches_oracle_with_padding():
    torch.manual_seed(12)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 19
    num_experts = 7
    top_k = 3
    hidden_size = 45
    intermediate_size = 29
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_ids[:5] = 0
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_ep_local_output_matches_oracle_and_ignores_remote_experts():
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 13
    num_experts = 8
    local_expert_start = 2
    num_local_experts = 4
    top_k = 2
    hidden_size = 32
    intermediate_size = 17
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        local_expert_start,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_can_preserve_fp32_topk_sum():
    torch.manual_seed(15)
    device = torch.device("cuda")
    hidden_states = torch.randn(5, 31, device=device, dtype=torch.bfloat16)
    w13_weight = torch.randn(4, 38, 31, device=device, dtype=torch.bfloat16)
    w2_weight = torch.randn(4, 31, 19, device=device, dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 3], [1, 2], [2, 0], [3, 1], [0, 2]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(5, 2, device=device, dtype=torch.bfloat16)

    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=4,
        local_expert_start=0,
        output_dtype=torch.float32,
    )
    torch.cuda.synchronize()

    assert actual.dtype == torch.float32


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_returns_zero_when_all_assignments_are_remote():
    torch.manual_seed(14)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_states = torch.randn(9, 24, device=device, dtype=dtype)
    w13_weight = torch.randn(4, 26, 24, device=device, dtype=dtype)
    w2_weight = torch.randn(4, 24, 13, device=device, dtype=dtype)
    topk_ids = torch.zeros(9, 2, dtype=torch.int64, device=device)
    topk_weights = torch.full((9, 2), 0.5, dtype=dtype, device=device)

    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=8,
        local_expert_start=4,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual).item() == 0
