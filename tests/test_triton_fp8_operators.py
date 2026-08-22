import pytest
import torch
import torch.nn.functional as F

from sparsevllm.operators.fp8_linear import (
    FlashInferGroupwiseSm120Fp8LinearProvider,
    Fp8LinearSpec,
    _sm120_activation_workspace,
    resolve_fp8_linear_provider,
)
from sparsevllm.platforms import current_platform
from sparsevllm.quantization.fp8 import fp8_blockwise_linear_reference
from sparsevllm.kernels.triton.fp8_blockwise import fp8_blockwise_matmul
from sparsevllm.kernels.triton.moe import fused_moe_fp8
from sparsevllm.kernels.triton.minimax_m2_moe import fused_minimax_m2_moe_fp8


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


def _assert_fp8_pipeline_close(actual, expected):
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    relative_l2 = torch.linalg.vector_norm(
        actual_fp32 - expected_fp32
    ) / torch.linalg.vector_norm(expected_fp32)
    cosine = F.cosine_similarity(
        actual_fp32.flatten(),
        expected_fp32.flatten(),
        dim=0,
    )
    assert relative_l2.item() < 2.0e-2
    assert cosine.item() > 0.9998


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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="profiled FP8 Linear dispatch requires SM120",
)
def test_resolver_binds_profiled_sm120_fp8_linear_dispatch_plan():
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    weight = _fp8_weight((5120, 2048), device)
    scales = torch.rand(40, 16, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=2048,
        output_features=5120,
    )

    assert provider.name == "sm120_fp8_linear_dispatch_plan"
    assert provider._route(511).provider.name == "triton"
    assert provider._route(512).provider.name == "flashinfer_groupwise_sm120"

    for tokens in (1, 512):
        inputs = torch.randn(
            tokens,
            2048,
            device=device,
            dtype=torch.bfloat16,
        )
        actual = provider(inputs, weight, scales)
        expected = fp8_blockwise_linear_reference(inputs, weight, scales)
        _assert_fp8_pipeline_close(actual, expected)

    assert provider.runtime_kernel_stats()["kernel_paths"] == {
        "flashinfer_groupwise_sm120": {
            "cuda_graph_capture_dispatches": 0,
            "eager_dispatches": 1,
        },
        "triton": {
            "cuda_graph_capture_dispatches": 0,
            "eager_dispatches": 1,
        },
    }


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="FlashInfer groupwise FP8 Linear requires SM120",
)
def test_flashinfer_sm120_fp8_linear_cuda_graph_replay():
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    inputs = torch.randn(3, 384, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((256, 384), device)
    scales = torch.rand(2, 3, device=device) + 0.25
    spec = Fp8LinearSpec(
        block_shape=(128, 128),
        input_features=int(weight.shape[1]),
        output_features=int(weight.shape[0]),
    )
    caps = current_platform.get_device_caps(torch.cuda.current_device())
    assert FlashInferGroupwiseSm120Fp8LinearProvider.supports(
        spec, caps
    ).supported
    provider = FlashInferGroupwiseSm120Fp8LinearProvider.bind(
        spec,
        caps,
    )
    assert provider.name == "flashinfer_groupwise_sm120"

    provider(inputs, weight, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider(inputs, weight, scales)
    inputs.copy_(torch.randn_like(inputs))
    graph.replay()
    torch.cuda.synchronize()

    expected = fp8_blockwise_linear_reference(inputs, weight, scales)
    _assert_fp8_pipeline_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="FlashInfer groupwise FP8 Linear requires SM120",
)
def test_sm120_fp8_linear_reuses_geometric_activation_workspace():
    device = torch.device("cuda")
    first = torch.empty((513, 640), device=device, dtype=torch.bfloat16)
    second = torch.empty((600, 640), device=device, dtype=torch.bfloat16)

    first_quantized, first_scales = _sm120_activation_workspace(first)
    second_quantized, second_scales = _sm120_activation_workspace(second)

    assert first_quantized.untyped_storage().data_ptr() == (
        second_quantized.untyped_storage().data_ptr()
    )
    assert first_scales.untyped_storage().data_ptr() == (
        second_scales.untyped_storage().data_ptr()
    )
    assert first_quantized.shape == (513, 640)
    assert second_quantized.shape == (600, 640)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="FlashInfer block-scale FP8 Linear requires SM90",
)
def test_resolved_sm90_fp8_linear_cuda_graph_replay():
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    inputs = torch.randn(3, 384, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((256, 384), device)
    scales = torch.rand(2, 3, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=int(weight.shape[1]),
        output_features=int(weight.shape[0]),
    )

    assert provider.name == "flashinfer_sm90"
    provider(inputs, weight, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider(inputs, weight, scales)
    inputs.copy_(torch.randn_like(inputs))
    graph.replay()
    torch.cuda.synchronize()

    expected = fp8_blockwise_linear_reference(inputs, weight, scales)
    _assert_fp8_pipeline_close(actual, expected)


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
    w13_scale = torch.rand(
        experts,
        2 * intermediate // 128,
        hidden // 128,
        device=device,
    ) + 0.25
    w2_scale = torch.rand(
        experts,
        hidden // 128,
        intermediate // 128,
        device=device,
    ) + 0.25
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

    _assert_fp8_pipeline_close(actual, expected)


@pytest.mark.parametrize(
    ("hidden", "intermediate"),
    [(128, 128), (3072, 384)],
)
def test_minimax_fused_gate_up_matches_generic_fp8_pipeline(hidden, intermediate):
    torch.manual_seed(71)
    device = torch.device("cuda")
    tokens, experts, top_k = 4, 4, 2
    hidden_states = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w13_weight = _fp8_weight((experts, 2 * intermediate, hidden), device)
    w2_weight = _fp8_weight((experts, hidden, intermediate), device)
    w13_scale = torch.rand(
        experts,
        2 * intermediate // 128,
        hidden // 128,
        device=device,
    ) + 0.25
    w2_scale = torch.rand(
        experts,
        hidden // 128,
        intermediate // 128,
        device=device,
    ) + 0.25
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    arguments = (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
    )
    kwargs = {"num_experts": experts, "local_expert_start": 0}

    expected = fused_moe_fp8(*arguments, **kwargs, gate_up_order="gate_up")
    actual = fused_minimax_m2_moe_fp8(*arguments, **kwargs)
    torch.cuda.synchronize()

    _assert_fp8_pipeline_close(actual, expected)
