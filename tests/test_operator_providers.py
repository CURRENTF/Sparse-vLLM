import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.operators.all_reduce import HopperTp2FlashInferAllReduceProvider
from sparsevllm.operators.activation import (
    SILU_AND_MUL_REGISTRY,
    SiluAndMulSpec,
    TorchSiluAndMulProvider,
    TritonSiluAndMulProvider,
)
from sparsevllm.operators.fp8_linear import (
    FP8_LINEAR_REGISTRY,
    FlashInferSm90Fp8LinearProvider,
    Fp8LinearSpec,
    TritonFp8LinearProvider,
    resolve_fp8_linear_provider,
)
from sparsevllm.operators.gate_up_swiglu import (
    GATE_UP_SWIGLU_REGISTRY,
    GateUpSwiGLUOpSpec,
    NativeGateUpSwiGLUProvider,
)
from sparsevllm.operators.moe import (
    MOE_REGISTRY,
    FlashInferCutlassFp8MoeProvider,
    HopperQwen36HybridFp8MoeProvider,
    MoeOpSpec,
    SglAlignedTritonGlmMoeProvider,
    resolve_moe_provider,
)
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _cuda_caps(
    capability: tuple[int, int],
    *,
    native_fp8: bool = True,
    runtime_version: str | None = "13.0",
    device_name: str | None = None,
) -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name=device_name or f"SM{capability[0]}{capability[1]}",
        compute_capability=capability,
        runtime_version=runtime_version,
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        supports_native_fp8=native_fp8,
    )


def _non_cuda_caps(platform: PlatformEnum) -> DeviceCaps:
    return DeviceCaps(
        platform=platform,
        device_type="cuda" if platform == PlatformEnum.ROCM else "cpu",
        device_index=0,
        device_name=platform.name,
        supports_triton=platform == PlatformEnum.ROCM,
        supports_bfloat16=True,
        supports_native_fp8=platform == PlatformEnum.ROCM,
    )


def _moe_spec(
    *,
    activation_dtype=torch.bfloat16,
    weight_dtype=torch.float8_e4m3fn,
    block_shape=(128, 128),
    hidden_size=256,
    intermediate_size=128,
    num_local_experts=4,
    num_experts=8,
    top_k=2,
    ep_size=2,
    tp_size=1,
    routing_method="softmax",
    scale_dtype=None,
    cuda_graph=True,
) -> MoeOpSpec:
    return MoeOpSpec(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        block_shape=block_shape,
        ep_size=ep_size,
        cuda_graph=cuda_graph,
        tp_size=tp_size,
        routing_method=routing_method,
        scale_dtype=scale_dtype,
    )


def _linear_spec(
    *,
    input_features=256,
    output_features=128,
    activation_dtype=torch.bfloat16,
    block_shape=(128, 128),
) -> Fp8LinearSpec:
    return Fp8LinearSpec(
        block_shape=block_shape,
        input_features=input_features,
        output_features=output_features,
        activation_dtype=activation_dtype,
    )


def _gate_up_spec(**overrides) -> GateUpSwiGLUOpSpec:
    values = {
        "hidden_size": 2048,
        "intermediate_size": 512,
        "tp_size": 1,
        "activation_dtype": torch.bfloat16,
        "weight_dtype": torch.bfloat16,
        "cuda_graph": True,
    }
    values.update(overrides)
    return GateUpSwiGLUOpSpec(**values)


def test_flashinfer_all_reduce_falls_back_before_unsupported_shape_launch():
    provider = HopperTp2FlashInferAllReduceProvider.__new__(
        HopperTp2FlashInferAllReduceProvider
    )
    provider.fallback = Mock()
    tensor = torch.randn(1, 248320, dtype=torch.bfloat16)
    provider.fallback.run.return_value = tensor

    assert provider.run(tensor) is tensor
    provider.fallback.run.assert_called_once_with(tensor)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_h20_gate_up_provider_accepts_profiled_qwen36_shape(tp_size):
    resolved = OpResolver(GATE_UP_SWIGLU_REGISTRY).resolve(
        _gate_up_spec(tp_size=tp_size),
        _cuda_caps((9, 0), device_name="NVIDIA H20"),
    )

    assert resolved.provider.name == "h20_triton_decode"


@pytest.mark.parametrize(
    "spec,caps",
    [
        (
            _gate_up_spec(weight_dtype=torch.float8_e4m3fn),
            _cuda_caps((9, 0), device_name="NVIDIA H20"),
        ),
        (
            _gate_up_spec(intermediate_size=768),
            _cuda_caps((9, 0), device_name="NVIDIA H20"),
        ),
        (
            _gate_up_spec(),
            _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        ),
        (_gate_up_spec(), _cuda_caps((8, 9), device_name="NVIDIA H20")),
    ],
)
def test_unprofiled_gate_up_shape_uses_native_provider(spec, caps):
    resolved = OpResolver(GATE_UP_SWIGLU_REGISTRY).resolve(spec, caps)

    assert resolved.provider.name == "native"


def test_native_gate_up_provider_matches_swiglu_semantics():
    torch.manual_seed(0)
    inputs = torch.randn(3, 8)
    projection = torch.nn.Linear(8, 12, bias=False)
    with torch.inference_mode():
        projected = projection(inputs)
        gate, up = projected.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        actual = NativeGateUpSwiGLUProvider().run(
            _gate_up_spec(
                hidden_size=8,
                intermediate_size=6,
                activation_dtype=torch.float32,
                weight_dtype=torch.float32,
                cuda_graph=False,
            ),
            inputs,
            projection,
        )

    torch.testing.assert_close(actual, expected)


def test_silu_and_mul_provider_respects_platform_capability():
    cuda_provider = OpResolver(SILU_AND_MUL_REGISTRY).resolve(
        SiluAndMulSpec(activation_dtype=torch.bfloat16),
        _cuda_caps((9, 0)),
        op_spec=SiluAndMulSpec(activation_dtype=torch.bfloat16),
    ).provider
    cpu_provider = OpResolver(SILU_AND_MUL_REGISTRY).resolve(
        SiluAndMulSpec(activation_dtype=torch.float32),
        _non_cuda_caps(PlatformEnum.CPU),
        op_spec=SiluAndMulSpec(activation_dtype=torch.float32),
    ).provider

    assert isinstance(cuda_provider, TritonSiluAndMulProvider)
    assert isinstance(cpu_provider, TorchSiluAndMulProvider)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_experts": 0},
        {"num_local_experts": 3},
        {"ep_size": 0},
        {"tp_size": 0},
        {"hidden_size": 0},
        {"intermediate_size": -1},
        {"top_k": 0},
        {"top_k": 9},
        {"block_shape": (128, 0)},
        {"routing_method": "unknown"},
    ],
)
def test_moe_spec_rejects_inconsistent_semantics(overrides):
    values = {
        "num_experts": 8,
        "num_local_experts": 4,
        "hidden_size": 256,
        "intermediate_size": 128,
        "top_k": 2,
        "activation_dtype": torch.bfloat16,
        "weight_dtype": torch.float8_e4m3fn,
        "block_shape": (128, 128),
        "ep_size": 2,
        "cuda_graph": True,
        "tp_size": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        MoeOpSpec(**values)


@pytest.mark.parametrize(
    ("capability", "activation_dtype"),
    [
        ((8, 9), torch.bfloat16),
        ((8, 9), torch.float16),
        ((9, 0), torch.float16),
        ((10, 0), torch.bfloat16),
    ],
)
def test_fp8_linear_uses_generic_triton_when_specialization_does_not_match(
    capability,
    activation_dtype,
):
    with patch("sparsevllm.operators.fp8_linear.find_spec", return_value=object()):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(activation_dtype=activation_dtype),
            _cuda_caps(capability),
        )

    assert resolved.provider.name == "triton"


def test_fp8_linear_prefers_flashinfer_on_sm90():
    with patch(
        "sparsevllm.operators.fp8_linear.find_spec",
        return_value=object(),
    ):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "flashinfer_sm90"


def test_fp8_linear_uses_triton_when_flashinfer_is_missing_on_sm90():
    with patch("sparsevllm.operators.fp8_linear.find_spec", return_value=None):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "triton"
    assert resolved.rejected == (("flashinfer_sm90", "flashinfer is not installed"),)


def test_fp8_linear_resolution_does_not_require_nvcc():
    with patch(
        "sparsevllm.operators.fp8_linear.find_spec",
        return_value=object(),
    ):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "flashinfer_sm90"


def test_flashinfer_linear_does_not_mask_missing_jit_artifact():
    flashinfer_call = Mock(
        side_effect=RuntimeError(
            "Assertion failed: !cubin.empty() || isPathValid(path_)"
        )
    )
    provider = FlashInferSm90Fp8LinearProvider()
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)

    with (
        patch.dict(
            sys.modules,
            {
                "flashinfer.gemm": SimpleNamespace(
                    fp8_blockscale_gemm_sm90=flashinfer_call
                )
            },
        ),
        pytest.raises(RuntimeError, match="cubin.empty"),
    ):
        provider(x, weight, scale)

    assert flashinfer_call.call_count == 1


def test_flashinfer_linear_does_not_mask_other_runtime_failures():
    flashinfer_call = Mock(side_effect=RuntimeError("invalid scale layout"))
    provider = FlashInferSm90Fp8LinearProvider()
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)

    with (
        patch.dict(
            sys.modules,
            {
                "flashinfer.gemm": SimpleNamespace(
                    fp8_blockscale_gemm_sm90=flashinfer_call
                )
            },
        ),
        pytest.raises(RuntimeError, match="invalid scale layout"),
    ):
        provider(x, weight, scale)

def test_fp8_linear_reports_unsupported_pre_fp8_device():
    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _cuda_caps((8, 0), native_fp8=False),
        )


@pytest.mark.parametrize(
    "spec",
    [
        _linear_spec(block_shape=(64, 128)),
        _linear_spec(activation_dtype=torch.float32),
    ],
)
def test_fp8_linear_rejects_unsupported_operation_specs(spec):
    with pytest.raises(RuntimeError, match="No block-scaled FP8 Linear provider"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(spec, _cuda_caps((9, 0)))


@pytest.mark.parametrize("platform", [PlatformEnum.CPU, PlatformEnum.ROCM])
def test_fp8_linear_rejects_non_cuda_platforms(platform):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _non_cuda_caps(platform),
        )


def test_fp8_linear_rejects_cuda_without_triton_or_flashinfer():
    caps = _cuda_caps((9, 0))
    caps = DeviceCaps(**{**caps.__dict__, "supports_triton": False})
    with (
        patch("sparsevllm.operators.fp8_linear.find_spec", return_value=None),
        pytest.raises(RuntimeError, match="does not support Triton"),
    ):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            caps,
        )


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_linear_spec(input_features=255), "input_features divisible by 128"),
        (_linear_spec(output_features=127), "output_features divisible by 64"),
    ],
)
def test_fp8_linear_uses_triton_for_unsupported_flashinfer_shapes(spec, reason):
    with patch("sparsevllm.operators.fp8_linear.find_spec", return_value=object()):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "triton"
    assert resolved.rejected[0][0] == "flashinfer_sm90"
    assert reason in resolved.rejected[0][1]


@pytest.mark.parametrize("runtime_version", [None, "12.7", "invalid"])
def test_flashinfer_providers_require_cuda_12_8(runtime_version):
    caps = _cuda_caps((9, 0), runtime_version=runtime_version)
    with (
        patch("sparsevllm.operators.fp8_linear.find_spec", return_value=object()),
        patch("sparsevllm.operators.moe.find_spec", return_value=object()),
    ):
        linear = OpResolver(FP8_LINEAR_REGISTRY).resolve(_linear_spec(), caps)
        moe = OpResolver(MOE_REGISTRY).resolve(_moe_spec(), caps)

    assert linear.provider.name == "triton"
    assert moe.provider.name == "triton"
    assert "requires CUDA runtime >= 12.8" in linear.rejected[0][1]
    assert any("requires CUDA runtime >= 12.8" in reason for _, reason in moe.rejected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (10, 0)])
def test_unquantized_moe_uses_triton_on_supported_cuda(dtype, capability):
    spec = _moe_spec(
        activation_dtype=dtype,
        weight_dtype=dtype,
        block_shape=None,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps(capability, native_fp8=False),
    )

    assert resolved.provider.name == "triton"


def test_hopper_fused_moe_uses_profiled_tp_ep_shape():
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=384,
        num_local_experts=64,
        num_experts=128,
        top_k=8,
        ep_size=2,
        tp_size=2,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps(
            (9, 0),
            native_fp8=False,
            device_name="NVIDIA H100 80GB HBM3",
        ),
    )

    assert resolved.provider.name == "triton_hopper_fused"


def test_glm_tp2_moe_uses_sgl_aligned_provider():
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=768,
        num_local_experts=64,
        num_experts=64,
        top_k=4,
        ep_size=1,
        tp_size=2,
        routing_method="biased_sigmoid",
    )

    with patch(
        "sparsevllm.operators.sgl_moe.sgl_moe_alignment_support",
        return_value=(True, "available"),
    ):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps(
                (9, 0),
                native_fp8=False,
                device_name="NVIDIA H100 80GB HBM3",
            ),
        )

    assert resolved.provider.name == "sgl_aligned_triton_glm"


def test_glm_tp2_fused_shared_decode_uses_sgl_aligned_provider():
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=768,
        num_local_experts=65,
        num_experts=65,
        top_k=5,
        ep_size=1,
        tp_size=2,
        routing_method="biased_sigmoid",
    )

    with patch(
        "sparsevllm.operators.sgl_moe.sgl_moe_alignment_support",
        return_value=(True, "available"),
    ):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps(
                (9, 0),
                native_fp8=False,
                device_name="NVIDIA H100 80GB HBM3",
            ),
        )

    assert resolved.provider.name == "sgl_aligned_triton_glm"


def test_glm_tp2_ep2_moe_uses_sgl_aligned_provider():
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=1536,
        num_local_experts=32,
        num_experts=64,
        top_k=4,
        ep_size=2,
        tp_size=1,
        routing_method="biased_sigmoid",
    )

    with patch(
        "sparsevllm.operators.sgl_moe.sgl_moe_ep_alignment_support",
        return_value=(True, "supported"),
    ):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps(
                (9, 0),
                native_fp8=False,
                device_name="NVIDIA H100 80GB HBM3",
            ),
        )

    assert resolved.provider.name == "sgl_aligned_triton_glm"


@pytest.mark.parametrize(
    ("num_tokens", "expects_sgl_alignment"),
    [(4, False), (5, True), (64, True), (65, False)],
)
def test_glm_tp2_ep2_sgl_alignment_is_bounded(
    num_tokens,
    expects_sgl_alignment,
):
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=1536,
        num_local_experts=32,
        num_experts=64,
        top_k=4,
        ep_size=2,
        tp_size=1,
        routing_method="biased_sigmoid",
    )
    hidden_states = torch.empty(num_tokens, 2048)
    topk_ids = torch.empty(num_tokens, 4, dtype=torch.int64)
    topk_weights = torch.empty(num_tokens, 4)
    weights = torch.empty(0)
    provider = SglAlignedTritonGlmMoeProvider()

    with patch("sparsevllm.triton_kernel.moe.fused_moe") as fused_moe:
        provider.run(
            spec,
            hidden_states,
            topk_ids,
            topk_weights,
            weights,
            weights,
            None,
            None,
            local_expert_start=0,
            ep_rank=0,
        )

    alignment_impl = fused_moe.call_args.kwargs["alignment_impl"]
    assert (alignment_impl is not None) is expects_sgl_alignment


@pytest.mark.parametrize(
    ("tp_size", "ep_size", "intermediate_size", "num_local_experts"),
    [(1, 1, 512, 256), (2, 1, 256, 256), (1, 2, 512, 128)],
)
def test_qwen36_bf16_moe_uses_profiled_hopper_provider(
    tp_size,
    ep_size,
    intermediate_size,
    num_local_experts,
):
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts=256,
        top_k=8,
        ep_size=ep_size,
        tp_size=tp_size,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps((9, 0), native_fp8=False, device_name="NVIDIA H20"),
    )

    assert resolved.provider.name == "h20_qwen36_fused_bf16"
    h100 = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps(
            (9, 0), native_fp8=False, device_name="NVIDIA H100 80GB HBM3"
        ),
    )
    assert h100.provider.name == "triton_hopper_fused"


@pytest.mark.parametrize(
    ("tp_size", "ep_size", "intermediate_size", "num_local_experts"),
    [(4, 1, 384, 256), (2, 2, 768, 128), (1, 4, 1536, 64)],
)
def test_minimax_m2_fused_moe_uses_dedicated_provider(
    tp_size,
    ep_size,
    intermediate_size,
    num_local_experts,
):
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts=256,
        top_k=8,
        ep_size=ep_size,
        tp_size=tp_size,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
    )

    assert resolved.provider.name == "triton_minimax_m2_fused"


def test_minimax_m2_fused_moe_rejects_graph_on_unsupported_device():
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )
    caps = _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3")
    caps = DeviceCaps(**{**caps.__dict__, "supports_graph_capture": False})

    resolved = OpResolver(MOE_REGISTRY).resolve(spec, caps)

    assert resolved.provider.name == "triton"
    assert (
        "triton_minimax_m2_fused",
        "device does not support CUDA Graph capture",
    ) in resolved.rejected


def test_minimax_m2_fused_moe_requires_native_fp8():
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps(
                (9, 0),
                native_fp8=False,
                device_name="NVIDIA H100 80GB HBM3",
            ),
        )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"routing_method": "softmax"}, "biased-sigmoid routing"),
        ({"scale_dtype": torch.bfloat16}, "FP32 expert scales"),
        ({"hidden_size": 2048}, "MiniMax M2.7 expert shape"),
    ],
)
def test_minimax_m2_fused_moe_falls_back_locally(overrides, reason):
    values = dict(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )
    values.update(overrides)
    resolved = OpResolver(MOE_REGISTRY).resolve(
        _moe_spec(**values),
        _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
    )

    assert resolved.provider.name == "triton"
    rejection = dict(resolved.rejected)["triton_minimax_m2_fused"]
    assert reason in rejection


def test_hopper_fused_moe_falls_back_for_missing_graph_support():
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=384,
        num_local_experts=64,
        num_experts=128,
        top_k=8,
        ep_size=2,
        tp_size=2,
    )
    caps = _cuda_caps(
        (9, 0),
        native_fp8=False,
        device_name="NVIDIA H100 80GB HBM3",
    )
    caps = DeviceCaps(**{**caps.__dict__, "supports_graph_capture": False})

    resolved = OpResolver(MOE_REGISTRY).resolve(spec, caps)

    assert resolved.provider.name == "triton"
    assert (
        "triton_hopper_fused",
        "device does not support CUDA Graph capture",
    ) in resolved.rejected


def test_fp8_moe_prefers_flashinfer_only_on_sm90():
    spec = _moe_spec()
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        hopper = OpResolver(MOE_REGISTRY).resolve(spec, _cuda_caps((9, 0)))
        blackwell = OpResolver(MOE_REGISTRY).resolve(spec, _cuda_caps((10, 0)))

    assert hopper.provider.name == "flashinfer_cutlass_fp8_sm90"
    assert blackwell.provider.name == "triton"


def test_qwen36_hybrid_moe_uses_profiled_graph_shape_on_h100():
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=128,
        num_experts=256,
        top_k=8,
        ep_size=2,
        tp_size=1,
    )
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        )

    assert resolved.provider.name == "hopper_qwen36_hybrid_fp8"
    assert resolved.provider.gate_up_order == "up_gate"


def test_qwen36_hybrid_moe_uses_profiled_single_gpu_shape_on_h100():
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=1,
    )
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        )

    assert resolved.provider.name == "hopper_qwen36_hybrid_fp8"
    assert resolved.provider.gate_up_order == "up_gate"


def test_qwen36_pure_tp_uses_triton_for_sharded_experts():
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=256,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=2,
    )
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        )

    assert resolved.provider.name == "triton"


@pytest.mark.parametrize(
    ("spec_overrides", "caps_overrides", "reason"),
    [
        (
            {},
            {"supports_graph_capture": False},
            "device does not support CUDA Graph capture",
        ),
        ({"intermediate_size": 640}, {}, "requires profiled Qwen3.6"),
    ],
)
def test_qwen36_hybrid_moe_rejects_unprofiled_execution(
    spec_overrides,
    caps_overrides,
    reason,
):
    values = dict(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=128,
        num_experts=256,
        top_k=8,
        ep_size=2,
        tp_size=1,
    )
    values.update(spec_overrides)
    spec = _moe_spec(**values)
    caps = _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3")
    caps = DeviceCaps(**{**caps.__dict__, **caps_overrides})

    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(spec, caps)

    assert resolved.provider.name == "flashinfer_cutlass_fp8_sm90"
    assert reason in dict(resolved.rejected)["hopper_qwen36_hybrid_fp8"]


def test_qwen36_hybrid_moe_supports_eager_execution():
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=1,
        cuda_graph=False,
    )
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        )

    assert resolved.provider.name == "hopper_qwen36_hybrid_fp8"


def test_h20_qwen36_hybrid_moe_uses_profiled_provider():
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=1,
    )
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name="NVIDIA H20"),
        )

    assert resolved.provider.name == "h20_qwen36_hybrid_fp8"


def test_qwen36_hybrid_moe_dispatches_by_token_bucket():
    provider = HopperQwen36HybridFp8MoeProvider()
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=128,
        num_experts=256,
        top_k=8,
        ep_size=2,
        tp_size=1,
    )
    small_output = torch.ones(4, 2)
    large_output = torch.ones(5, 2)
    triton_call = Mock(return_value=small_output)
    weights = torch.empty(1)

    with patch.dict(
        sys.modules,
        {
            "sparsevllm.triton_kernel.moe": SimpleNamespace(
                fused_moe_fp8=triton_call
            )
        },
    ):
        actual_small = provider.run(
            spec,
            torch.empty(4, 2),
            torch.empty(4, 8, dtype=torch.int32),
            torch.empty(4, 8),
            weights,
            weights,
            weights,
            weights,
            local_expert_start=128,
            ep_rank=1,
        )

    with patch.object(
        FlashInferCutlassFp8MoeProvider,
        "run",
        return_value=large_output,
    ) as flashinfer_call:
        actual_large = provider.run(
            spec,
            torch.empty(5, 2),
            torch.empty(5, 8, dtype=torch.int32),
            torch.empty(5, 8),
            weights,
            weights,
            weights,
            weights,
            local_expert_start=128,
            ep_rank=1,
        )

    assert actual_small is small_output
    assert actual_large is large_output
    assert triton_call.call_args.kwargs["gate_up_order"] == "up_gate"
    flashinfer_call.assert_called_once()


def test_qwen36_hybrid_moe_uses_larger_triton_bucket_on_single_gpu():
    provider = HopperQwen36HybridFp8MoeProvider()
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=512,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=1,
    )
    triton_output = torch.ones(8, 2)
    flashinfer_output = torch.ones(9, 2)
    triton_call = Mock(return_value=triton_output)
    weights = torch.empty(1)

    with patch.dict(
        sys.modules,
        {
            "sparsevllm.triton_kernel.moe": SimpleNamespace(
                fused_moe_fp8=triton_call
            )
        },
    ):
        actual_decode = provider.run(
            spec,
            torch.empty(8, 2),
            torch.empty(8, 8, dtype=torch.int32),
            torch.empty(8, 8),
            weights,
            weights,
            weights,
            weights,
            local_expert_start=0,
            ep_rank=0,
        )

    with patch.object(
        FlashInferCutlassFp8MoeProvider,
        "run",
        return_value=flashinfer_output,
    ) as flashinfer_call:
        actual_prefill = provider.run(
            spec,
            torch.empty(9, 2),
            torch.empty(9, 8, dtype=torch.int32),
            torch.empty(9, 8),
            weights,
            weights,
            weights,
            weights,
            local_expert_start=0,
            ep_rank=0,
        )

    assert actual_decode is triton_output
    assert actual_prefill is flashinfer_output
    assert triton_call.call_args.kwargs["gate_up_order"] == "up_gate"
    flashinfer_call.assert_called_once()


def test_fp8_moe_uses_triton_when_flashinfer_is_missing_on_sm90():
    with patch("sparsevllm.operators.moe.find_spec", return_value=None):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "triton"
    assert (
        "flashinfer_cutlass_fp8_sm90",
        "flashinfer is not installed",
    ) in resolved.rejected
    assert (
        "triton_hopper_fused",
        "requires unquantized BF16 expert weights",
    ) in resolved.rejected


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_moe_spec(block_shape=(64, 128)), "block_shape"),
        (_moe_spec(hidden_size=129), "128-aligned"),
        (
            _moe_spec(activation_dtype=torch.float32),
            "BF16 or FP16 activations",
        ),
    ],
)
def test_fp8_moe_rejects_unsupported_operation_specs(spec, reason):
    with pytest.raises(RuntimeError, match=reason):
        OpResolver(MOE_REGISTRY).resolve(spec, _cuda_caps((10, 0)))


def test_fp8_moe_rejects_pre_fp8_cuda():
    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(),
            _cuda_caps((8, 0), native_fp8=False),
        )


def test_fp8_moe_uses_triton_for_tensor_parallel_expert_shards():
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(tp_size=2, ep_size=1, num_local_experts=8),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "triton"
    assert (
        "flashinfer_cutlass_fp8_sm90",
        "does not support tensor-parallel expert shards",
    ) in resolved.rejected
    assert (
        "triton_hopper_fused",
        "requires unquantized BF16 expert weights",
    ) in resolved.rejected


@pytest.mark.parametrize("platform", [PlatformEnum.CPU, PlatformEnum.ROCM])
def test_moe_rejects_non_cuda_platforms(platform):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(
                activation_dtype=torch.bfloat16,
                weight_dtype=torch.bfloat16,
                block_shape=None,
            ),
            _non_cuda_caps(platform),
        )


def test_provider_owns_packed_gate_up_layout():
    triton_spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
    )
    triton = OpResolver(MOE_REGISTRY).resolve(
        triton_spec,
        _cuda_caps((9, 0)),
    ).provider
    flashinfer_spec = _moe_spec()
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        flashinfer = OpResolver(MOE_REGISTRY).resolve(
            flashinfer_spec,
            _cuda_caps((9, 0)),
        ).provider

    triton_w13 = torch.zeros(4, 256, 256, dtype=torch.bfloat16)
    triton_w2 = torch.zeros(4, 256, 128, dtype=torch.bfloat16)
    gate = torch.full((128, 256), 1, dtype=torch.bfloat16)
    up = torch.full((128, 256), 2, dtype=torch.bfloat16)
    for projection, weight in (("gate", gate), ("up", up)):
        triton.load_expert_projection(
            triton_spec,
            local_expert_id=0,
            projection=projection,
            loaded_weight=weight,
            loaded_scale=None,
            w13_weight=triton_w13,
            w2_weight=triton_w2,
            w13_scale_inv=None,
            w2_scale_inv=None,
        )
    assert torch.equal(triton_w13[0, :128], gate)
    assert torch.equal(triton_w13[0, 128:], up)

    flashinfer_w13 = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
    flashinfer_w2 = torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn)
    flashinfer_s13 = torch.zeros(4, 2, 2)
    flashinfer_s2 = torch.zeros(4, 2, 1)
    gate_fp8 = torch.full((128, 256), 1).to(torch.float8_e4m3fn)
    up_fp8 = torch.full((128, 256), 2).to(torch.float8_e4m3fn)
    gate_scale = torch.full((1, 2), 3.0)
    up_scale = torch.full((1, 2), 4.0)
    for projection, weight, scale in (
        ("gate", gate_fp8, gate_scale),
        ("up", up_fp8, up_scale),
    ):
        flashinfer.load_expert_projection(
            flashinfer_spec,
            local_expert_id=0,
            projection=projection,
            loaded_weight=weight,
            loaded_scale=scale,
            w13_weight=flashinfer_w13,
            w2_weight=flashinfer_w2,
            w13_scale_inv=flashinfer_s13,
            w2_scale_inv=flashinfer_s2,
        )
    assert torch.equal(flashinfer_w13[0, :128], up_fp8)
    assert torch.equal(flashinfer_w13[0, 128:], gate_fp8)
    assert torch.equal(flashinfer_s13[0, :1], up_scale)
    assert torch.equal(flashinfer_s13[0, 1:], gate_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_public_resolvers_match_live_device_capabilities():
    from sparsevllm.platforms import current_platform

    caps = current_platform.get_device_caps(torch.cuda.current_device())
    linear = resolve_fp8_linear_provider(
        (128, 128),
        input_features=256,
        output_features=128,
    )
    unquantized_moe = resolve_moe_provider(
        _moe_spec(
            activation_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
            block_shape=None,
        )
    )

    assert unquantized_moe.name == "triton"
    assert linear.name in {"flashinfer_sm90", "triton"}
    if linear.name == "flashinfer_sm90":
        assert caps.compute_capability == (9, 0)
