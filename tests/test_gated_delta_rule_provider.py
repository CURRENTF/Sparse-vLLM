from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sparsevllm.models.gdn_runtime import (
    bind_gated_delta_rule_op,
    build_gated_delta_rule_op,
)
from sparsevllm.kernels.external.flashinfer.gdn import (
    flashinfer_sm90_gdn_prefill_support,
)
from sparsevllm.kernels.external.support import (
    KernelFamilyHealth,
    KernelFamilyState,
)
from sparsevllm.operators.gated_delta_rule import (
    GATED_DELTA_RULE_REGISTRY,
    FlashInferSm90GatedDeltaRuleProvider,
    GatedDeltaRuleOpSpec,
    PreparedGatedDeltaRuleOp,
    TritonGatedDeltaRuleProvider,
)
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _caps(*, compute_capability=(9, 0)) -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=compute_capability,
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        supports_native_fp8=True,
    )


def _spec(**overrides) -> GatedDeltaRuleOpSpec:
    values = {
        "num_key_heads": 2,
        "num_value_heads": 4,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "recurrent_state_dtype": torch.float32,
        "cuda_graph_decode": True,
    }
    values.update(overrides)
    return GatedDeltaRuleOpSpec(**values)


@patch(
    "sparsevllm.operators.gated_delta_rule.flashinfer_sm90_gdn_prefill_support",
    return_value=(True, "validated"),
)
def test_h100_selects_fixed_flashinfer_prefill_triton_decode_plan(_support):
    resolved = OpResolver(GATED_DELTA_RULE_REGISTRY).resolve(_spec(), _caps())

    assert isinstance(resolved.provider, FlashInferSm90GatedDeltaRuleProvider)
    metadata = dict(resolved.report.provider_metadata.items)
    assert metadata["implementation_kind"] == "atomic_provider"
    assert "flashinfer.gdn_prefill" in metadata["prefill_kernel_path"]
    assert "fused_recurrent" in metadata["decode_kernel_path"]
    assert metadata["runtime_state_layout"] == "k_major_hkv"
    assert metadata["prefill_state_layout"] == "v_major_hvk"
    assert metadata["state_layout_adapter"] == "transpose_last_two_dims"


def test_non_sm90_selects_repo_triton_plan_without_probing_flashinfer():
    with patch(
        "sparsevllm.operators.gated_delta_rule.flashinfer_sm90_gdn_prefill_support"
    ) as support:
        resolved = OpResolver(GATED_DELTA_RULE_REGISTRY).resolve(
            _spec(),
            _caps(compute_capability=(12, 0)),
        )

    support.assert_not_called()
    assert isinstance(resolved.provider, TritonGatedDeltaRuleProvider)


def test_sm90_old_flashinfer_version_is_a_local_provider_rejection():
    health = KernelFamilyHealth(
        "flashinfer-python",
        KernelFamilyState.READY,
        "0.6.15.post1",
        "package family is ready",
    )
    with (
        patch(
            "sparsevllm.kernels.external.flashinfer.gdn.flashinfer_kernel_health",
            return_value=health,
        ),
        patch(
            "sparsevllm.kernels.external.flashinfer.gdn._gdn_prefill_op"
        ) as operation,
    ):
        supported, reason = flashinfer_sm90_gdn_prefill_support()

    assert not supported
    assert ">= 0.6.17" in reason
    operation.assert_not_called()


def test_sm90_flashinfer_plan_rejects_old_cuda_without_feature_probe():
    caps = _caps()
    caps = DeviceCaps(
        **{
            **caps.__dict__,
            "runtime_version": "12.7",
        }
    )
    with patch(
        "sparsevllm.operators.gated_delta_rule.flashinfer_sm90_gdn_prefill_support"
    ) as support:
        result = FlashInferSm90GatedDeltaRuleProvider.supports(_spec(), caps)

    support.assert_not_called()
    assert not result.supported
    assert "runtime >= 12.8" in result.reason


def test_flashinfer_prefill_adapter_converts_log_gate_and_state_contract():
    provider = FlashInferSm90GatedDeltaRuleProvider()
    packed_qk = torch.randn(1, 3, 2, 256, dtype=torch.bfloat16)
    q = packed_qk[..., :128]
    k = packed_qk[..., 128:]
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    v = torch.randn(1, 3, 4, 128, dtype=torch.bfloat16)
    g = torch.randn(1, 3, 4, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn_like(g))
    state = torch.randn(2, 4, 128, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    output = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    final_state = torch.randn(2, 4, 128, 128, dtype=torch.float32)

    with (
        patch(
            "sparsevllm.operators.gated_delta_rule.l2norm_fwd",
            side_effect=lambda tensor: tensor,
        ) as normalize,
        patch(
            "sparsevllm.operators.gated_delta_rule.flashinfer_chunk_gated_delta_rule_sm90",
            return_value=(output, final_state),
        ) as kernel,
    ):
        actual_output, actual_state = provider.run_prefill(
            _spec(recurrent_state_dtype=torch.bfloat16),
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            cu_seqlens=cu_seqlens,
        )

    assert actual_output.shape == (1, 3, 4, 128)
    torch.testing.assert_close(actual_state, final_state.transpose(-1, -2))
    assert actual_state.is_contiguous()
    assert all(call.args[0].is_contiguous() for call in normalize.call_args_list)
    call = kernel.call_args
    torch.testing.assert_close(call.args[3], torch.exp(g.squeeze(0)))
    assert call.args[5].dtype == torch.float32
    torch.testing.assert_close(
        call.args[5],
        state.to(torch.float32).transpose(-1, -2),
    )
    assert call.args[5].is_contiguous()
    assert call.args[6] is cu_seqlens


def test_bound_decode_calls_fused_raw_gating_kernel_without_dispatch():
    provider = TritonGatedDeltaRuleProvider()
    q = torch.randn(4, 1, 2, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(4, 1, 4, 128, dtype=torch.bfloat16)
    state = torch.randn(9, 4, 128, 128, dtype=torch.float32)
    state_indices = torch.tensor([1, 3, 5, 8], dtype=torch.int32)
    A_log = torch.randn(4, dtype=torch.float32)
    dt_bias = torch.randn(4, dtype=torch.float32)
    a = torch.randn(4, 4, dtype=torch.bfloat16)
    b = torch.randn(4, 4, dtype=torch.bfloat16)
    output = torch.randn(4, 1, 4, 128, dtype=torch.bfloat16)

    with patch(
        "sparsevllm.operators.gated_delta_rule.fused_recurrent_gated_delta_rule",
        return_value=(output, state),
    ) as kernel:
        actual = provider.run_decode(
            _spec(),
            q=q,
            k=k,
            v=v,
            initial_state=state,
            state_indices=state_indices,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
        )

    assert actual is output
    assert kernel.call_args.kwargs == {
        "q": q,
        "k": k,
        "v": v,
        "initial_state": state,
        "inplace_final_state": True,
        "ssm_state_indices": state_indices,
        "use_qk_l2norm_in_kernel": True,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a_raw": a,
        "b_raw": b,
    }


def test_triton_prefill_provider_expands_qk_to_value_heads():
    provider = TritonGatedDeltaRuleProvider()
    q = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    k = q + 100
    v = torch.randn(1, 4, 6, 3)
    g = torch.randn(1, 4, 6)
    beta = torch.randn(1, 4, 6)
    state = torch.randn(1, 6, 3, 3)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    output = torch.randn_like(v)

    with patch(
        "sparsevllm.operators.gated_delta_rule.chunk_gated_delta_rule",
        return_value=(output, state),
    ) as kernel:
        actual = provider.run_prefill(
            _spec(
                num_key_heads=2,
                num_value_heads=6,
                key_head_dim=3,
                value_head_dim=3,
            ),
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            cu_seqlens=cu_seqlens,
        )

    assert actual[0] is output
    assert actual[1] is state
    repeated_q = kernel.call_args.kwargs["q"]
    repeated_k = kernel.call_args.kwargs["k"]
    assert repeated_q.shape == (1, 4, 6, 3)
    assert repeated_k.shape == (1, 4, 6, 3)
    assert torch.equal(repeated_q[:, :, 0], q[:, :, 0])
    assert torch.equal(repeated_q[:, :, 1], q[:, :, 0])
    assert torch.equal(repeated_q[:, :, 2], q[:, :, 0])
    assert torch.equal(repeated_q[:, :, 3], q[:, :, 1])
    assert torch.equal(repeated_k[:, :, 5], k[:, :, 1])


def test_prepared_gdn_operator_rejects_calls_after_close():
    provider = Mock(name="provider")
    provider.name = "test"
    prepared = PreparedGatedDeltaRuleOp(_spec(), provider)

    prepared.close()

    provider.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.run_decode()


def test_model_runtime_builds_and_binds_one_shared_gdn_operator():
    config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
    )
    prepared = Mock(name="prepared_gdn")
    root = nn.Module()
    root.first = nn.Module()
    root.first.is_gated_delta_rule_layer = True
    root.second = nn.Module()
    root.second.is_gated_delta_rule_layer = True

    with patch(
        "sparsevllm.models.gdn_runtime.prepare_gated_delta_rule_op",
        return_value=prepared,
    ) as prepare:
        actual = build_gated_delta_rule_op(
            config,
            attention_tp_size=2,
            device=torch.device("cuda", 3),
            cuda_graph=True,
        )

    assert actual is prepared
    spec = prepare.call_args.args[0]
    assert (spec.num_key_heads, spec.num_value_heads) == (2, 4)
    assert spec.recurrent_state_dtype == torch.bfloat16
    assert prepare.call_args.kwargs == {"device_index": 3}
    assert bind_gated_delta_rule_op(root, prepared) == 2
    assert root.first.gated_delta_rule_op is prepared
    assert root.second.gated_delta_rule_op is prepared
