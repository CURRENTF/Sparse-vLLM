import unittest

import torch
import torch.nn.functional as F

from sparsevllm.quantization.fp8 import (
    Fp8BlockScaledLinearBackend,
    fp8_blockwise_dequantize,
    fp8_blockwise_linear_reference,
)
from sparsevllm.triton_kernel.minimax_m2_moe_fp8 import fused_moe_fp8


def _inputs():
    torch.manual_seed(17)
    num_tokens = 12
    num_experts = 4
    top_k = 2
    hidden_size = 128
    intermediate_size = 128
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    w13_scale_inv = (
        torch.rand(num_experts, 2, 1, device="cuda", dtype=torch.float32) * 0.1
        + 0.9
    )
    w2_scale_inv = (
        torch.rand(num_experts, 1, 1, device="cuda", dtype=torch.float32) * 0.1
        + 0.9
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        device="cuda",
        dtype=torch.float32,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return (
        hidden_states,
        w13_weight,
        w13_scale_inv,
        w2_weight,
        w2_scale_inv,
        topk_ids,
        topk_weights,
    )


def _explicit_oracle(
    hidden_states,
    w13_weight,
    w13_scale_inv,
    w2_weight,
    w2_scale_inv,
    topk_ids,
    topk_weights,
):
    intermediate_size = int(w2_weight.shape[-1])
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for token_id in range(hidden_states.shape[0]):
        for topk_slot in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, topk_slot])
            w1 = fp8_blockwise_dequantize(
                w13_weight[expert_id, :intermediate_size],
                w13_scale_inv[expert_id, :1],
            )
            w3 = fp8_blockwise_dequantize(
                w13_weight[expert_id, intermediate_size:],
                w13_scale_inv[expert_id, 1:],
            )
            w2 = fp8_blockwise_dequantize(
                w2_weight[expert_id],
                w2_scale_inv[expert_id],
            )
            expert_input = hidden_states[token_id].float()
            expert_output = F.linear(
                F.silu(F.linear(expert_input, w1))
                * F.linear(expert_input, w3),
                w2,
            )
            output[token_id] += expert_output * topk_weights[
                token_id, topk_slot
            ]
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_generic_native_fp8_linear_matches_reference_intermediate():
    hidden_states, w13_weight, w13_scale_inv, *_ = _inputs()
    native = Fp8BlockScaledLinearBackend(
        block_size=(128, 128),
        backend="auto",
        model_name="MiniMax M2.7",
    )
    actual = native(
        hidden_states,
        w13_weight[0],
        w13_scale_inv[0],
    )
    expected = fp8_blockwise_linear_reference(
        hidden_states,
        w13_weight[0],
        w13_scale_inv[0],
    )
    torch.cuda.synchronize()
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float())
    relative_l2 /= torch.linalg.vector_norm(expected.float())

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=1.0e-2,
        rtol=1.2e-1,
    )
    assert float(relative_l2) < 0.06


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_routed_fp8_matches_explicit_dequant_oracle():
    inputs = _inputs()
    actual = fused_moe_fp8(
        *inputs,
        num_experts=4,
        local_expert_start=0,
    )
    torch.cuda.synchronize()
    expected = _explicit_oracle(*inputs)
    error = actual.float() - expected

    torch.testing.assert_close(
        actual.float(),
        expected,
        atol=1.0e-3,
        rtol=1.2e-1,
    )
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)
    assert float(relative_l2) < 0.06


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_routed_fp8_ep_partition_and_cuda_graph_replay():
    inputs = _inputs()
    hidden_states, w13_weight, w13_scale_inv, w2_weight, w2_scale_inv, ids, weights = (
        inputs
    )
    full_output = fused_moe_fp8(
        *inputs,
        num_experts=4,
        local_expert_start=0,
    )
    rank0_output = fused_moe_fp8(
        hidden_states,
        w13_weight[:2].contiguous(),
        w13_scale_inv[:2].contiguous(),
        w2_weight[:2].contiguous(),
        w2_scale_inv[:2].contiguous(),
        ids,
        weights,
        num_experts=4,
        local_expert_start=0,
    )
    rank1_output = fused_moe_fp8(
        hidden_states,
        w13_weight[2:].contiguous(),
        w13_scale_inv[2:].contiguous(),
        w2_weight[2:].contiguous(),
        w2_scale_inv[2:].contiguous(),
        ids,
        weights,
        num_experts=4,
        local_expert_start=2,
    )
    torch.testing.assert_close(
        rank0_output + rank1_output,
        full_output,
        atol=2.0e-4,
        rtol=2.0e-3,
    )

    static_hidden = hidden_states.clone()
    static_ids = ids.clone()
    static_weights = weights.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fused_moe_fp8(
            static_hidden,
            w13_weight,
            w13_scale_inv,
            w2_weight,
            w2_scale_inv,
            static_ids,
            static_weights,
            num_experts=4,
            local_expert_start=0,
        )

    static_hidden.copy_(hidden_states * 0.5)
    graph.replay()
    torch.cuda.synchronize()
    expected_replay = fused_moe_fp8(
        static_hidden,
        w13_weight,
        w13_scale_inv,
        w2_weight,
        w2_scale_inv,
        static_ids,
        static_weights,
        num_experts=4,
        local_expert_start=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(graph_output, expected_replay)
