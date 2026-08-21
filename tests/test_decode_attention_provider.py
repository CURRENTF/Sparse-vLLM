from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.method_registry import sparse_decode_attention_requires_scores
from sparsevllm.operators.decode_attention import (
    DECODE_ATTENTION_REGISTRY,
    DecodeAttentionOpSpec,
    PreparedDecodeAttentionOp,
    SglFa3PagedDecodeAttentionProvider,
    TritonPagedDecodeAttentionProvider,
)


@pytest.mark.parametrize(
    ("method", "requires_scores"),
    [
        (None, False),
        ("vanilla", False),
        ("streamingllm", False),
        ("quest", False),
        ("rkv", False),
        ("snapkv", True),
        ("h2o", True),
        ("pyramidkv", True),
        ("omnikv", True),
        ("skipkv", True),
        ("deltakv", True),
    ],
)
def test_sparse_decode_score_contract_is_method_specific(
    method,
    requires_scores,
):
    assert sparse_decode_attention_requires_scores(method) is requires_scores
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _h100_caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        supports_native_fp8=True,
    )


def _spec(**overrides) -> DecodeAttentionOpSpec:
    values = {
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "softmax_scale": 128**-0.5,
        "max_batch_size": 64,
        "causal": True,
        "page_size": 1,
        "may_require_attention_scores": False,
        "layer_varying_page_table": False,
        "cuda_graph": True,
    }
    values.update(overrides)
    return DecodeAttentionOpSpec(**values)


@patch(
    "sparsevllm.operators.decode_attention.sgl_fa3_device_support",
    return_value=(True, "ready"),
)
def test_h100_score_free_decode_selects_sgl_fa3(_support):
    resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(
        _spec(),
        _h100_caps(),
    )

    assert isinstance(resolved.provider, SglFa3PagedDecodeAttentionProvider)
    assert resolved.report.provider_metadata.items


@patch(
    "sparsevllm.operators.decode_attention.sgl_fa3_device_support",
    return_value=(True, "ready"),
)
def test_h100_head_dim_256_decode_selects_sgl_fa3(_support):
    resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(
        _spec(num_query_heads=16, num_kv_heads=2, head_dim=256),
        _h100_caps(),
    )

    assert isinstance(resolved.provider, SglFa3PagedDecodeAttentionProvider)


@patch(
    "sparsevllm.operators.decode_attention.sgl_fa3_device_support",
    return_value=(True, "ready"),
)
def test_score_capable_decode_binds_triton_before_execution(_support):
    spec = _spec(
        may_require_attention_scores=True,
        layer_varying_page_table=True,
    )
    sgl_support = SglFa3PagedDecodeAttentionProvider.supports(
        spec,
        _h100_caps(),
    )
    triton_support = TritonPagedDecodeAttentionProvider.supports(
        spec,
        _h100_caps(),
    )
    resolved = OpResolver(DECODE_ATTENTION_REGISTRY).resolve(
        spec,
        _h100_caps(),
    )

    assert not sgl_support.supported
    assert "attention score" in sgl_support.reason
    assert triton_support.supported
    assert isinstance(resolved.provider, TritonPagedDecodeAttentionProvider)


def test_prepared_score_free_decode_rejects_late_score_request():
    provider = Mock(name="provider")
    provider.name = "score_free"
    prepared = PreparedDecodeAttentionOp(_spec(), provider)
    view = SimpleNamespace(meta=SimpleNamespace(attn_score=torch.empty(1)))

    with pytest.raises(RuntimeError, match="score-free provider"):
        prepared.run(torch.empty(1, 32, 128), view)

    provider.run.assert_not_called()


def test_sgl_decode_provider_uses_prepared_explicit_kv_adapter():
    provider = SglFa3PagedDecodeAttentionProvider()
    kernel = Mock(name="fa3_kernel")
    provider._kernel = kernel
    q = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    k_cache = torch.randn(16, 8, 128, dtype=torch.bfloat16)
    v_cache = torch.randn(16, 8, 128, dtype=torch.bfloat16)
    active_slots = torch.arange(16, dtype=torch.int32).view(2, 8)
    req_indices = torch.arange(2, dtype=torch.int32)
    context_lens = torch.full((2,), 8, dtype=torch.int32)
    view = SimpleNamespace(
        payload=SimpleNamespace(k_cache=k_cache, v_cache=v_cache),
        meta=SimpleNamespace(
            active_slots=active_slots,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=None,
        ),
    )
    scope = object()
    decode_launch_op = Mock(name="unused_triton_launch")
    kernel.run_explicit.side_effect = lambda *args, **kwargs: args[6]

    with patch(
        "sparsevllm.operators.decode_attention.get_context",
        return_value=SimpleNamespace(attention_validation_scope=scope),
    ):
        output = provider.run(
            _spec(),
            q,
            view,
            decode_launch_op=decode_launch_op,
        )

    assert output.shape == q.shape
    call = kernel.run_explicit.call_args
    expected_inputs = (
        q,
        k_cache,
        v_cache,
        active_slots,
        req_indices,
        context_lens,
    )
    assert all(actual is expected for actual, expected in zip(call.args[:6], expected_inputs))
    assert call.args[6] is output
    assert call.kwargs == {"validation_scope": scope}
    decode_launch_op.launch_config.assert_not_called()


def test_triton_provider_owns_launch_config_and_workspace_preparation():
    provider = TritonPagedDecodeAttentionProvider()
    provider._backend = Mock(name="triton_backend")
    q = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    view = SimpleNamespace(
        meta=SimpleNamespace(
            active_slots=torch.arange(16, dtype=torch.int32).view(2, 8),
            context_lens=torch.full((2,), 8, dtype=torch.int32),
            max_context_len=8,
            attn_score=torch.empty(2, 32, 8),
        ),
    )
    cache_manager = SimpleNamespace(
        _decode_static_max_context_len=None,
        get_decode_block_seq=Mock(return_value=4),
    )
    context = SimpleNamespace(
        cache_manager=cache_manager,
        now_layer_idx=3,
        decode_mid_o=None,
        decode_mid_o_logexpsum=None,
    )
    decode_launch_op = Mock(name="decode_launch")
    decode_launch_op.launch_config.return_value = (4, 8, 4)
    output = torch.empty_like(q)
    provider._backend.run_decode.return_value = output

    with patch(
        "sparsevllm.operators.decode_attention.get_context",
        return_value=context,
    ):
        actual = provider.run(
            _spec(may_require_attention_scores=True),
            q,
            view,
            decode_launch_op=decode_launch_op,
        )

    assert actual is output
    decode_launch_op.launch_config.assert_called_once_with(
        block_seq=4,
        max_context_len=8,
        requires_attention_scores=True,
    )
    kwargs = provider._backend.run_decode.call_args.kwargs
    assert kwargs["mid_o"].shape == (2, 32, 2, 128)
    assert kwargs["mid_o_logexpsum"].shape == (2, 32, 2)
    assert kwargs["max_len_in_batch"] == 8
    assert kwargs["block_seq"] == 4
    assert kwargs["num_heads"] == 32
    assert kwargs["num_kv_heads"] == 8
    assert kwargs["gqa_block_n"] == 8
    assert kwargs["gqa_num_warps"] == 4
