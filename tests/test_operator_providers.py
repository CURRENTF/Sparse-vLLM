import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.operators.fp8_linear import (
    FP8_LINEAR_REGISTRY,
    FlashInferSm90Fp8LinearProvider,
    Fp8LinearSpec,
    TritonFp8LinearProvider,
    resolve_fp8_linear_provider,
)
from sparsevllm.operators.moe import MOE_REGISTRY, MoeOpSpec, resolve_moe_provider
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _cuda_caps(
    capability: tuple[int, int],
    *,
    native_fp8: bool = True,
) -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name=f"SM{capability[0]}{capability[1]}",
        compute_capability=capability,
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
) -> MoeOpSpec:
    return MoeOpSpec(
        num_experts=8,
        num_local_experts=4,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=2,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        block_shape=block_shape,
        ep_size=2,
        cuda_graph=True,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_experts": 0},
        {"num_local_experts": 3},
        {"ep_size": 0},
        {"hidden_size": 0},
        {"intermediate_size": -1},
        {"top_k": 0},
        {"top_k": 9},
        {"block_shape": (128, 0)},
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
            Fp8LinearSpec((128, 128), activation_dtype),
            _cuda_caps(capability),
        )

    assert resolved.provider.name == "triton"


def test_fp8_linear_prefers_flashinfer_on_sm90():
    with (
        patch("sparsevllm.operators.fp8_linear.find_spec", return_value=object()),
        patch("sparsevllm.operators.fp8_linear.which", return_value="/cuda/bin/nvcc"),
    ):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            Fp8LinearSpec((128, 128)),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "flashinfer_sm90"


def test_fp8_linear_uses_triton_when_flashinfer_is_missing_on_sm90():
    with patch("sparsevllm.operators.fp8_linear.find_spec", return_value=None):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            Fp8LinearSpec((128, 128)),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "triton"
    assert resolved.rejected == (("flashinfer_sm90", "flashinfer is not installed"),)


def test_fp8_linear_resolution_does_not_require_nvcc():
    with (
        patch("sparsevllm.operators.fp8_linear.find_spec", return_value=object()),
        patch("sparsevllm.operators.fp8_linear.which", return_value=None),
    ):
        resolved = OpResolver(FP8_LINEAR_REGISTRY).resolve(
            Fp8LinearSpec((128, 128)),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "flashinfer_sm90"


def test_flashinfer_linear_binds_triton_for_missing_uncached_kernel():
    flashinfer_call = Mock(
        side_effect=RuntimeError(
            "Assertion failed: !cubin.empty() || isPathValid(path_)"
        )
    )
    fallback_output = torch.ones(2, 128, dtype=torch.bfloat16)
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
        patch("sparsevllm.operators.fp8_linear.which", return_value=None),
        patch.object(
            TritonFp8LinearProvider,
            "__call__",
            return_value=fallback_output,
        ) as triton_call,
    ):
        first = provider(x, weight, scale)
        second = provider(x, weight, scale)

    assert first is fallback_output
    assert second is fallback_output
    assert flashinfer_call.call_count == 1
    assert triton_call.call_count == 2


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
        patch("sparsevllm.operators.fp8_linear.which", return_value=None),
        pytest.raises(RuntimeError, match="invalid scale layout"),
    ):
        provider(x, weight, scale)

    assert provider._fallback is None


def test_fp8_linear_reports_unsupported_pre_fp8_device():
    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            Fp8LinearSpec((128, 128)),
            _cuda_caps((8, 0), native_fp8=False),
        )


@pytest.mark.parametrize(
    "spec",
    [
        Fp8LinearSpec((64, 128)),
        Fp8LinearSpec((128, 128), torch.float32),
    ],
)
def test_fp8_linear_rejects_unsupported_operation_specs(spec):
    with pytest.raises(RuntimeError, match="No block-scaled FP8 Linear provider"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(spec, _cuda_caps((9, 0)))


@pytest.mark.parametrize("platform", [PlatformEnum.CPU, PlatformEnum.ROCM])
def test_fp8_linear_rejects_non_cuda_platforms(platform):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            Fp8LinearSpec((128, 128)),
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
            Fp8LinearSpec((128, 128)),
            caps,
        )


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


def test_fp8_moe_prefers_flashinfer_only_on_sm90():
    spec = _moe_spec()
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        hopper = OpResolver(MOE_REGISTRY).resolve(spec, _cuda_caps((9, 0)))
        blackwell = OpResolver(MOE_REGISTRY).resolve(spec, _cuda_caps((10, 0)))

    assert hopper.provider.name == "flashinfer_cutlass_fp8_sm90"
    assert blackwell.provider.name == "triton"


def test_fp8_moe_uses_triton_when_flashinfer_is_missing_on_sm90():
    with patch("sparsevllm.operators.moe.find_spec", return_value=None):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "triton"
    assert resolved.rejected == (
        ("flashinfer_cutlass_fp8_sm90", "flashinfer is not installed"),
    )


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
    triton = OpResolver(MOE_REGISTRY).resolve(
        _moe_spec(
            activation_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
            block_shape=None,
        ),
        _cuda_caps((9, 0)),
    ).provider
    with patch("sparsevllm.operators.moe.find_spec", return_value=object()):
        flashinfer = OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(),
            _cuda_caps((9, 0)),
        ).provider

    assert triton.packed_projection_offset("gate", 128) == 0
    assert triton.packed_projection_offset("up", 128) == 128
    assert flashinfer.packed_projection_offset("up", 128) == 0
    assert flashinfer.packed_projection_offset("gate", 128) == 128
    with pytest.raises(ValueError, match="Unknown packed MoE projection"):
        triton.packed_projection_offset("down", 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_public_resolvers_match_live_device_capabilities():
    from sparsevllm.platforms import current_platform

    caps = current_platform.get_device_caps(torch.cuda.current_device())
    linear = resolve_fp8_linear_provider((128, 128))
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
