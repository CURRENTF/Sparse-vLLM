import pytest
import torch
import torch.nn.functional as F

from sparsevllm.operators.fp8_linear import resolve_fp8_linear_provider
from sparsevllm.quantization.fp8 import fp8_blockwise_linear_reference
from sparsevllm.triton_kernel.fp8_blockwise import fp8_blockwise_matmul
from sparsevllm.triton_kernel.moe import fused_moe_fp8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _fp8_weight(shape, device):
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(-3.0, 3.0)
        .to(torch.float8_e4m3fn)
    )


@pytest.mark.parametrize(
    ("tokens", "out_features", "in_features"),
    [(1, 128, 128), (7, 256, 384), (19, 129, 257)],
)
def test_fp8_blockwise_matmul_matches_reference(
    tokens,
    out_features,
    in_features,
):
    torch.manual_seed(tokens + out_features + in_features)
    device = torch.device("cuda")
    inputs = torch.randn(
        tokens,
        in_features,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = _fp8_weight((out_features, in_features), device)
    scales = (
        torch.rand(
            (out_features + 127) // 128,
            (in_features + 127) // 128,
            device=device,
        )
        + 0.25
    )

    actual = fp8_blockwise_matmul(inputs, weight, scales)
    expected = fp8_blockwise_linear_reference(inputs, weight, scales).to(
        torch.bfloat16
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1)


def test_resolved_fp8_linear_provider_matches_reference():
    torch.manual_seed(29)
    device = torch.device("cuda")
    inputs = torch.randn(3, 128, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((128, 128), device)
    scales = torch.rand(1, 1, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=weight.shape[1],
        output_features=weight.shape[0],
    )

    actual = provider(inputs, weight, scales)
    expected = fp8_blockwise_linear_reference(inputs, weight, scales).to(
        torch.bfloat16
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1)


def test_resolver_uses_triton_for_non_sm90_aligned_shape():
    torch.manual_seed(31)
    device = torch.device("cuda")
    inputs = torch.randn(3, 257, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((129, 257), device)
    scales = torch.rand(2, 3, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=weight.shape[1],
        output_features=weight.shape[0],
    )

    assert provider.name == "triton"
    actual = provider(inputs, weight, scales)
    expected = fp8_blockwise_linear_reference(inputs, weight, scales).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1)


def _reference_moe(
    hidden_states,
    w13_weight,
    w2_weight,
    w13_scale,
    w2_scale,
    topk_ids,
    topk_weights,
    gate_up_order,
):
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate = w2_weight.shape[-1]
    for token in range(hidden_states.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            packed = fp8_blockwise_linear_reference(
                hidden_states[token : token + 1],
                w13_weight[expert],
                w13_scale[expert],
            )
            first, second = packed.split(intermediate, dim=-1)
            gate, up = (
                (first, second)
                if gate_up_order == "gate_up"
                else (second, first)
            )
            activated = F.silu(gate) * up
            expert_output = fp8_blockwise_linear_reference(
                activated.to(hidden_states.dtype),
                w2_weight[expert],
                w2_scale[expert],
            )
            output[token].add_(
                expert_output[0] * topk_weights[token, slot].float()
            )
    return output.to(hidden_states.dtype)


@pytest.mark.parametrize("gate_up_order", ["gate_up", "up_gate"])
def test_fp8_moe_matches_reference(gate_up_order):
    torch.manual_seed(17)
    device = torch.device("cuda")
    tokens, experts, top_k = 5, 4, 2
    hidden, intermediate = 128, 128
    hidden_states = torch.randn(
        tokens,
        hidden,
        device=device,
        dtype=torch.bfloat16,
    )
    w13_weight = _fp8_weight((experts, 2 * intermediate, hidden), device)
    w2_weight = _fp8_weight((experts, hidden, intermediate), device)
    w13_scale = torch.rand(experts, 2, 1, device=device) + 0.25
    w2_scale = torch.rand(experts, 1, 1, device=device) + 0.25
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(
        tokens,
        top_k,
        device=device,
        dtype=torch.float32,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    actual = fused_moe_fp8(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        num_experts=experts,
        local_expert_start=0,
        gate_up_order=gate_up_order,
    )
    expected = _reference_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        gate_up_order,
    )

    torch.testing.assert_close(actual, expected, rtol=4.0e-2, atol=5.0)
