import unittest

import torch
import torch.nn.functional as F

from sparsevllm.quantization.fp8 import (
    Fp8BlockScaledLinearBackend,
    fp8_blockwise_linear_reference,
    load_finegrained_fp8_kernel,
)
from sparsevllm.triton_kernel.minimax_m2_moe_fp8 import (
    _expert_order_moe_sum,
    fused_moe_fp8,
)


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
            gate = fp8_blockwise_linear_reference(
                hidden_states[token_id],
                w13_weight[expert_id, :intermediate_size],
                w13_scale_inv[expert_id, :1],
            )
            up = fp8_blockwise_linear_reference(
                hidden_states[token_id],
                w13_weight[expert_id, intermediate_size:],
                w13_scale_inv[expert_id, 1:],
            )
            expert_output = fp8_blockwise_linear_reference(
                F.silu(gate) * up,
                w2_weight[expert_id],
                w2_scale_inv[expert_id],
            )
            output[token_id] += expert_output * topk_weights[
                token_id, topk_slot
            ]
    return output


def _native_expert_oracle(
    hidden_states,
    w13_weight,
    w13_scale_inv,
    w2_weight,
    w2_scale_inv,
    topk_ids,
    topk_weights,
):
    kernel = load_finegrained_fp8_kernel()
    output = torch.zeros_like(hidden_states)
    for expert_id in range(w13_weight.shape[0]):
        token_ids, topk_slots = torch.where(topk_ids == expert_id)
        if token_ids.numel() == 0:
            continue
        expert_input = hidden_states[token_ids].contiguous()
        gate_up = kernel.matmul(
            expert_input,
            w13_weight[expert_id],
            w13_scale_inv[expert_id],
            [128, 128],
            hidden_states.dtype,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate) * up).contiguous()
        down = kernel.matmul(
            activated,
            w2_weight[expert_id],
            w2_scale_inv[expert_id],
            [128, 128],
            hidden_states.dtype,
        )
        output.index_add_(
            0,
            token_ids,
            down * topk_weights[token_ids, topk_slots, None].to(down.dtype),
        )
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
        actual.float(), expected.float(), atol=1.0e-5, rtol=1.0e-3
    )
    assert float(relative_l2) < 1.0e-3


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_routed_fp8_matches_dynamic_w8a8_oracle():
    inputs = _inputs()
    actual = fused_moe_fp8(
        *inputs,
        num_experts=4,
        local_expert_start=0,
    )
    torch.cuda.synchronize()
    expected = _explicit_oracle(*inputs)
    error = actual.float() - expected

    torch.testing.assert_close(actual.float(), expected, atol=5.0e-5, rtol=5.0e-3)
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)
    assert float(relative_l2) < 5.0e-3


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_routed_fp8_matches_native_real_shapes():
    torch.manual_seed(29)
    max_tokens = 100
    hidden_size = 3072
    intermediate_size = 1536
    hidden_states = torch.randn(
        max_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w13_weight = (
        torch.randn(
            1,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
        )
        * 0.02
    ).to(torch.float8_e4m3fn)
    w2_weight = (
        torch.randn(
            1,
            hidden_size,
            intermediate_size,
            device="cuda",
        )
        * 0.02
    ).to(torch.float8_e4m3fn)
    w13_scale_inv = torch.rand(
        1,
        2 * intermediate_size // 128,
        hidden_size // 128,
        device="cuda",
        dtype=torch.float32,
    )
    w2_scale_inv = torch.rand(
        1,
        hidden_size // 128,
        intermediate_size // 128,
        device="cuda",
        dtype=torch.float32,
    )

    for num_tokens in (8, 24, 58, 100):
        topk_ids = torch.zeros(
            num_tokens,
            1,
            device="cuda",
            dtype=torch.int64,
        )
        topk_weights = torch.ones(
            num_tokens,
            1,
            device="cuda",
            dtype=torch.float32,
        )
        inputs = (
            hidden_states[:num_tokens].contiguous(),
            w13_weight,
            w13_scale_inv,
            w2_weight,
            w2_scale_inv,
            topk_ids,
            topk_weights,
        )
        expected = _native_expert_oracle(*inputs)
        actual = fused_moe_fp8(
            *inputs,
            num_experts=1,
            local_expert_start=0,
        )
        torch.cuda.synchronize()

        assert torch.equal(actual, expected), f"token count {num_tokens} differs"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_expert_order_sum_matches_official_bf16_accumulation():
    torch.manual_seed(23)
    num_tokens = 5
    num_experts = 8
    hidden_size = 128
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device="cuda")
            for _ in range(num_tokens)
        ]
    ).contiguous()
    slot_scales = torch.tensor(
        [1.0, 0.01, -1.0, 0.005, 0.5, -0.5, 0.02, -0.02],
        device="cuda",
        dtype=torch.bfloat16,
    )
    inputs = (
        torch.randn(
            num_tokens,
            num_experts,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * slot_scales[None, :, None]
    ).contiguous()

    expected = torch.zeros(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expert_id in range(num_experts):
        token_ids, topk_slots = torch.where(topk_ids == expert_id)
        expected.index_add_(0, token_ids, inputs[token_ids, topk_slots])

    actual = _expert_order_moe_sum(
        inputs,
        topk_ids,
        num_experts=num_experts,
        local_expert_start=0,
        local_expert_end=num_experts,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


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

    replay_ids = torch.zeros_like(ids)
    replay_ids[:, 1] = 3
    replay_weights = torch.flip(weights, dims=(1,)).contiguous()
    static_hidden.copy_(hidden_states * 0.5)
    static_ids.copy_(replay_ids)
    static_weights.copy_(replay_weights)
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

    static_ids.copy_((ids + 1) % 4)
    static_weights.copy_(weights.flip(-1))
    graph.replay()
    torch.cuda.synchronize()
    expected_reroute = fused_moe_fp8(
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
    assert torch.equal(graph_output, expected_reroute)
