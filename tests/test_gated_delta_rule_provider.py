from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sparsevllm.kernels.external.flashinfer.gdn import (
    flashinfer_sm90_gdn_prefill_support,
)
from sparsevllm.models.gdn_runtime import (
    bind_gated_delta_rule_op,
    build_gated_delta_rule_op,
)
from sparsevllm.operators.gated_delta_rule import (
    FlashInferSm90GatedDeltaRuleProvider,
    GatedDeltaRuleOpSpec,
    PreparedGatedDeltaRuleOp,
    TritonGatedDeltaRuleProvider,
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
    normalized_q = q.squeeze(0).contiguous()
    normalized_k = k.squeeze(0).contiguous()

    with (
        patch(
            "sparsevllm.operators.gated_delta_rule.fused_qk_l2norm_fwd",
            return_value=(normalized_q, normalized_k),
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
    normalize.assert_called_once_with(q, k)
    call = kernel.call_args
    assert call.args[0] is normalized_q
    assert call.args[1] is normalized_k
    torch.testing.assert_close(call.args[3], torch.exp(g.squeeze(0)))
    assert call.args[5].dtype == torch.float32
    torch.testing.assert_close(
        call.args[5],
        state.to(torch.float32).transpose(-1, -2),
    )
    assert call.args[5].is_contiguous()
    assert call.args[6] is cu_seqlens


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires CUDA SM90",
)
def test_flashinfer_prefill_matches_independent_triton_provider():
    supported, reason = flashinfer_sm90_gdn_prefill_support()
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(20260824)
    token_count, num_key_heads, num_value_heads, head_dim = 48, 2, 4, 128
    q = torch.randn(
        1,
        token_count,
        num_key_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(
        1,
        token_count,
        num_value_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    g = -(
        torch.rand(
            1,
            token_count,
            num_value_heads,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.2
        + 0.01
    )
    beta = torch.sigmoid(torch.randn_like(g))
    initial_state = torch.randn(
        2,
        num_value_heads,
        head_dim,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    cu_seqlens = torch.tensor(
        [0, 17, token_count], device="cuda", dtype=torch.int32
    )
    spec = _spec(
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
    )

    actual_output, actual_state = (
        FlashInferSm90GatedDeltaRuleProvider().run_prefill(
            spec,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state.clone(),
            cu_seqlens=cu_seqlens,
        )
    )
    expected_output, expected_state = TritonGatedDeltaRuleProvider().run_prefill(
        spec,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), rtol=0.05, atol=2e-4
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), rtol=0.02, atol=2e-3
    )


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
    normalized_q = q.squeeze(0) / 10
    normalized_k = k.squeeze(0) / 10

    with (
        patch(
            "sparsevllm.operators.gated_delta_rule.fused_qk_l2norm_fwd",
            return_value=(normalized_q, normalized_k),
        ) as normalize,
        patch(
            "sparsevllm.operators.gated_delta_rule.chunk_gated_delta_rule",
            return_value=(output, state),
        ) as kernel,
    ):
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
    normalize.assert_called_once_with(q, k)
    repeated_q = kernel.call_args.kwargs["q"]
    repeated_k = kernel.call_args.kwargs["k"]
    assert repeated_q.shape == (1, 4, 6, 3)
    assert repeated_k.shape == (1, 4, 6, 3)
    assert torch.equal(repeated_q[:, :, 0], normalized_q[None, :, 0])
    assert torch.equal(repeated_q[:, :, 1], normalized_q[None, :, 0])
    assert torch.equal(repeated_q[:, :, 2], normalized_q[None, :, 0])
    assert torch.equal(repeated_q[:, :, 3], normalized_q[None, :, 1])
    assert torch.equal(repeated_k[:, :, 5], normalized_k[None, :, 1])
    assert kernel.call_args.kwargs["use_qk_l2norm_in_kernel"] is False


def test_prepared_gdn_operator_rejects_calls_after_close():
    provider = Mock(name="provider")
    provider.name = "test"
    prepared = PreparedGatedDeltaRuleOp(_spec(), provider)

    prepared.close()

    provider.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.run_decode()


def test_prepared_gdn_operator_forwards_auxiliary_pipeline_without_dispatch():
    provider = Mock(name="provider")
    provider.name = "test"
    provider.run_gating.return_value = ("g", "beta")
    provider.run_prefill_conv.return_value = "conv"
    provider.prepare_decode_inputs.return_value = ("q", "k", "v", "z", "a", "b")
    provider.run_gated_rmsnorm.return_value = "norm"
    prepared = PreparedGatedDeltaRuleOp(_spec(), provider)

    assert prepared.run_gating(A_log=1, a=2, b=3, dt_bias=4) == ("g", "beta")
    assert prepared.run_prefill_conv(mixed_qkv=1) == "conv"
    assert prepared.prepare_decode_inputs(mixed_qkv=1) == (
        "q", "k", "v", "z", "a", "b"
    )
    assert prepared.run_gated_rmsnorm(x=1) == "norm"
    provider.run_gating.assert_called_once_with(A_log=1, a=2, b=3, dt_bias=4)


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
    root.second.bind_gated_delta_rule_op = Mock()

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
    root.second.bind_gated_delta_rule_op.assert_called_once_with(prepared)
