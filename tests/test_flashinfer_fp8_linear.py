from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.kernels.external.flashinfer.fp8_linear import (
    _sm90_fp8_linear_op,
    _sm120_groupwise_fp8_linear_op,
    flashinfer_fp8_blockscale_gemm_sm90,
    flashinfer_fp8_nt_groupwise_sm120,
    flashinfer_sm120_groupwise_fp8_linear_support,
)
from sparsevllm.kernels.external.flashinfer.moe import (
    _cutlass_fp8_moe_op,
    flashinfer_cutlass_fp8_moe_support,
    flashinfer_cutlass_fused_moe,
)
from sparsevllm.kernels.external.flashinfer.support import (
    flashinfer_kernel_health,
)
from sparsevllm.kernels.external.support import (
    ExternalKernelContractError,
    ExternalKernelFamilyError,
    KernelFamilyState,
)


def test_flashinfer_health_distinguishes_missing_package() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        health = flashinfer_kernel_health()

    assert health.state is KernelFamilyState.ABSENT
    assert "flashinfer-python is not installed" in health.reason


@pytest.mark.parametrize("version", ["0.6.14", "0.7.0"])
def test_flashinfer_health_rejects_outside_declared_range(version: str) -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
    ):
        health = flashinfer_kernel_health()

    assert health.state is KernelFamilyState.BROKEN
    assert "flashinfer-python>=0.6.15,<0.7" in health.reason


def test_flashinfer_feature_does_not_hide_broken_family() -> None:
    _sm120_groupwise_fp8_linear_op.cache_clear()
    try:
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ExternalKernelFamilyError) as exc_info:
                flashinfer_sm120_groupwise_fp8_linear_support()
    finally:
        _sm120_groupwise_fp8_linear_op.cache_clear()

    assert exc_info.value.health.state is KernelFamilyState.ABSENT


def test_flashinfer_groupwise_feature_accepts_public_contract() -> None:
    def gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode=None,
        mma_sm=1,
        scale_granularity_mnk=(1, 128, 128),
        out=None,
        out_dtype=None,
        backend="cutlass",
    ):
        pass

    module = SimpleNamespace(gemm_fp8_nt_groupwise=gemm_fp8_nt_groupwise)
    _sm120_groupwise_fp8_linear_op.cache_clear()
    try:
        with (
            patch(
                "sparsevllm.kernels.external.flashinfer.fp8_linear.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
        ):
            assert flashinfer_sm120_groupwise_fp8_linear_support() == (
                True,
                "available",
            )
    finally:
        _sm120_groupwise_fp8_linear_op.cache_clear()


def test_flashinfer_groupwise_feature_rejects_schema_drift() -> None:
    module = SimpleNamespace(gemm_fp8_nt_groupwise=lambda a, b: None)
    _sm120_groupwise_fp8_linear_op.cache_clear()
    try:
        with (
            patch(
                "sparsevllm.kernels.external.flashinfer.fp8_linear.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
        ):
            with pytest.raises(ExternalKernelContractError, match="unsupported schema"):
                flashinfer_sm120_groupwise_fp8_linear_support()
    finally:
        _sm120_groupwise_fp8_linear_op.cache_clear()


def test_flashinfer_groupwise_adapter_fixes_layout_contract() -> None:
    gemm = Mock(return_value=object())
    _sm120_groupwise_fp8_linear_op.cache_clear()
    try:
        with patch(
            "sparsevllm.kernels.external.flashinfer.fp8_linear._sm120_groupwise_fp8_linear_op",
            return_value=(gemm, "available"),
        ):
            activation = torch.empty(2, 128)
            weight = torch.empty(128, 128)
            activation_scale = torch.empty(2, 1)
            weight_scale = torch.empty(1, 1)
            result = flashinfer_fp8_nt_groupwise_sm120(
                activation,
                weight,
                activation_scale,
                weight_scale,
                out_dtype=torch.bfloat16,
            )
    finally:
        _sm120_groupwise_fp8_linear_op.cache_clear()

    assert result is gemm.return_value
    gemm.assert_called_once_with(
        activation,
        weight,
        activation_scale,
        weight_scale,
        scale_major_mode="K",
        scale_granularity_mnk=(1, 128, 128),
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )


def test_flashinfer_sm90_adapter_fixes_scale_contract() -> None:
    gemm = Mock(return_value=object())
    _sm90_fp8_linear_op.cache_clear()
    try:
        with patch(
            "sparsevllm.kernels.external.flashinfer.fp8_linear._sm90_fp8_linear_op",
            return_value=(gemm, "available"),
        ):
            inputs = torch.empty(2, 128)
            weight = torch.empty(128, 128)
            weight_scale = torch.empty(1, 1)
            result = flashinfer_fp8_blockscale_gemm_sm90(
                inputs,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
            )
    finally:
        _sm90_fp8_linear_op.cache_clear()

    assert result is gemm.return_value
    gemm.assert_called_once_with(
        inputs,
        weight,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
    )


def test_flashinfer_moe_feature_accepts_public_contract() -> None:
    function = Mock()
    activation_type = object()

    def import_module(name):
        if name == "flashinfer.fused_moe":
            return SimpleNamespace(cutlass_fused_moe=function)
        if name == "flashinfer.tllm_enums":
            return SimpleNamespace(
                ActivationType=SimpleNamespace(Swiglu=activation_type)
            )
        raise AssertionError(name)

    _cutlass_fp8_moe_op.cache_clear()
    try:
        with (
            patch(
                "sparsevllm.kernels.external.flashinfer.moe.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", side_effect=import_module),
        ):
            assert flashinfer_cutlass_fp8_moe_support() == (True, "available")
    finally:
        _cutlass_fp8_moe_op.cache_clear()


def test_flashinfer_moe_feature_rejects_missing_contract() -> None:
    _cutlass_fp8_moe_op.cache_clear()
    try:
        with (
            patch(
                "sparsevllm.kernels.external.flashinfer.moe.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch(
                "importlib.import_module",
                return_value=SimpleNamespace(),
            ),
            pytest.raises(ExternalKernelContractError, match="cutlass_fused_moe"),
        ):
            flashinfer_cutlass_fp8_moe_support()
    finally:
        _cutlass_fp8_moe_op.cache_clear()


def test_flashinfer_moe_adapter_fixes_execution_contract() -> None:
    function = Mock()
    activation_type = object()
    hidden_states = torch.empty(2, 128, dtype=torch.bfloat16)
    topk_ids = torch.empty(2, 2, dtype=torch.int64)
    topk_weights = torch.empty(2, 2, dtype=torch.bfloat16)
    w13_weight = torch.empty(4, 256, 128)
    w2_weight = torch.empty(4, 128, 128)
    w13_scale_inv = torch.empty(4, 2, 1)
    w2_scale_inv = torch.empty(4, 1, 1)
    output = torch.empty_like(hidden_states)

    with patch(
        "sparsevllm.kernels.external.flashinfer.moe._cutlass_fp8_moe_op",
        return_value=(function, activation_type, "available"),
    ):
        flashinfer_cutlass_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            w13_weight,
            w2_weight,
            w13_scale_inv,
            w2_scale_inv,
            ep_size=2,
            ep_rank=1,
            output=output,
        )

    function.assert_called_once()
    args, kwargs = function.call_args
    assert args[0] is hidden_states
    assert torch.equal(args[1], topk_ids.to(dtype=torch.int32))
    assert torch.equal(args[2], topk_weights.to(dtype=torch.float32))
    assert args[3] is w13_weight
    assert args[4] is w2_weight
    assert args[5] is hidden_states.dtype
    assert kwargs["quant_scales"][0] is w13_scale_inv
    assert kwargs["quant_scales"][1] is w2_scale_inv
    assert kwargs["ep_size"] == 2
    assert kwargs["ep_rank"] == 1
    assert kwargs["output"] is output
    assert kwargs["use_deepseek_fp8_block_scale"] is True
    assert kwargs["use_fused_finalize"] is False
    assert kwargs["enable_pdl"] is None
    assert kwargs["activation_type"] is activation_type
