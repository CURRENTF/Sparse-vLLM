import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.kernels.external.flashinfer.decode import (
    flashinfer_paged_decode_support,
)
from sparsevllm.kernels.external.sgl.fa3 import sgl_fa3_device_support
from sparsevllm.method_registry import sparse_decode_attention_requires_scores
from sparsevllm.models.attention_runtime import build_mha_decode_attention_spec
from sparsevllm.operators.decode_attention import (
    DecodeAttentionRunResult,
    DecodeAttentionOpSpec,
    FlashInferPagedDecodeAttentionProvider,
    PreparedDecodeAttentionOp,
    SglFa3PagedDecodeAttentionProvider,
    TritonPagedDecodeAttentionProvider,
)
from sparsevllm.platforms import DeviceCaps, PlatformEnum


@pytest.mark.parametrize(
    ("method", "requires_scores"),
    [
        (None, False),
        ("vanilla", False),
        ("streamingllm", False),
        ("quest", False),
        ("rkv", False),
        ("snapkv", False),
        ("h2o", False),
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


def test_h2o_runtime_decode_spec_is_score_free_while_eviction_is_disabled():
    config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        torch_dtype=torch.bfloat16,
    )

    spec = build_mha_decode_attention_spec(
        config,
        sparse_method="h2o",
        attention_tp_size=1,
        max_batch_size=8,
        cuda_graph=True,
    )

    assert not spec.may_require_attention_scores
    assert not spec.h2o_layerwise_probability_scores
    assert not spec.kernel_request.requires_softmax_lse


def _cuda_caps(
    *,
    device_name: str = "NVIDIA H100 80GB HBM3",
    compute_capability: tuple[int, int] = (9, 0),
    supports_triton: bool = True,
) -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name=device_name,
        compute_capability=compute_capability,
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=supports_triton,
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


def test_flashinfer_lse_decode_rejects_cuda_graph_before_dependency_probe():
    spec = _spec(
        may_require_attention_scores=True,
        layer_varying_page_table=True,
        h2o_layerwise_probability_scores=True,
    )
    caps = _cuda_caps(
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        compute_capability=(12, 0),
    )

    with patch(
        "sparsevllm.operators.decode_attention.flashinfer_paged_decode_support"
    ) as support:
        result = FlashInferPagedDecodeAttentionProvider.supports(spec, caps)

    assert not result.supported
    assert "CUDA Graph" in result.reason
    support.assert_not_called()


def test_prepared_h2o_decode_applies_fixed_probability_scorer():
    spec = _spec(
        may_require_attention_scores=True,
        layer_varying_page_table=True,
        h2o_layerwise_probability_scores=True,
    )
    provider = Mock(name="lse_provider")
    provider.name = "lse_provider"
    q = torch.empty(2, 32, 128, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    softmax_lse = torch.empty(32, 2, dtype=torch.float32)
    provider.run.return_value = DecodeAttentionRunResult(output, softmax_lse)
    score = torch.empty(2, 8, dtype=torch.float32)
    view = SimpleNamespace(
        payload=SimpleNamespace(k_cache=torch.empty(16, 8, 128)),
        meta=SimpleNamespace(
            active_slots=torch.empty(2, 8, dtype=torch.int32),
            req_indices=torch.empty(2, dtype=torch.int32),
            context_lens=torch.empty(2, dtype=torch.int32),
            attn_score=score,
        ),
    )

    with patch(
        "sparsevllm.kernels.triton.h2o_decode_score.h2o_probability_from_lse"
    ) as scorer:
        actual = PreparedDecodeAttentionOp(spec, provider).run(q, view)

    assert actual is output
    scorer.assert_called_once_with(
        q,
        view.payload.k_cache,
        softmax_lse,
        view.meta.active_slots,
        view.meta.req_indices,
        view.meta.context_lens,
        score,
        softmax_scale=spec.softmax_scale,
    )


def test_flashinfer_decode_normalizes_log2_lse_for_repo_scorer():
    spec = _spec(
        activation_dtype=torch.bfloat16,
        head_dim=128,
        may_require_attention_scores=True,
        layer_varying_page_table=True,
        cuda_graph=False,
        h2o_layerwise_probability_scores=True,
    )
    provider = FlashInferPagedDecodeAttentionProvider()
    wrapper = Mock(name="flashinfer_wrapper")
    state = SimpleNamespace(plan=Mock(name="plan"), wrapper=wrapper)
    provider._state = state
    q = torch.empty(2, 32, 128, dtype=torch.bfloat16)
    view = SimpleNamespace(
        payload=SimpleNamespace(
            k_cache=torch.empty(16, 8, 128, dtype=torch.bfloat16),
            v_cache=torch.empty(16, 8, 128, dtype=torch.bfloat16),
        ),
        meta=SimpleNamespace(
            active_slots=torch.empty(2, 8, dtype=torch.int32),
            req_indices=torch.empty(2, dtype=torch.int32),
            context_lens=torch.empty(2, dtype=torch.int32),
            attn_score=torch.empty(2, 8, dtype=torch.float32),
        ),
    )

    def run(_q, _kv, *, out, return_lse):
        assert return_lse
        return out, torch.full((2, 32), 2.0, dtype=torch.float32)

    wrapper.run.side_effect = run
    result = provider.run(spec, q, view)

    assert isinstance(result, DecodeAttentionRunResult)
    assert result.output.shape == q.shape
    torch.testing.assert_close(
        result.softmax_lse,
        torch.full((32, 2), 2.0 * math.log(2.0)),
    )
    state.plan.assert_called_once_with(
        spec,
        active_slots=view.meta.active_slots,
        req_indices=view.meta.req_indices,
        context_lens=view.meta.context_lens,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("activation_dtype", "head_dim"),
    [
        (torch.bfloat16, 128),
    ],
)
@pytest.mark.parametrize(
    "provider_type",
    [SglFa3PagedDecodeAttentionProvider, FlashInferPagedDecodeAttentionProvider],
)
def test_prepared_h2o_probability_score_matches_paged_torch(
    activation_dtype,
    head_dim,
    provider_type,
):
    if provider_type is SglFa3PagedDecodeAttentionProvider:
        supported, reason = sgl_fa3_device_support(torch.cuda.current_device())
        if not supported:
            pytest.skip(reason)
    else:
        flashinfer_paged_decode_support()

    torch.manual_seed(17)
    batch, query_heads, kv_heads, width = 2, 16, 2, 17
    capacity = batch * width
    q = torch.randn(
        batch,
        query_heads,
        head_dim,
        dtype=activation_dtype,
        device="cuda",
    )
    k_cache = torch.randn(
        capacity,
        kv_heads,
        head_dim,
        dtype=activation_dtype,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        capacity,
        dtype=torch.int64,
        device="cuda",
    ).to(torch.int32).view(batch, width)
    req_indices = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([17, 13], dtype=torch.int32, device="cuda")
    score = torch.empty(batch, width, dtype=torch.float32, device="cuda")
    view = SimpleNamespace(
        payload=SimpleNamespace(k_cache=k_cache, v_cache=v_cache),
        meta=SimpleNamespace(
            active_slots=page_table,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=score,
        ),
    )
    spec = _spec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        may_require_attention_scores=True,
        layer_varying_page_table=True,
        cuda_graph=(provider_type is SglFa3PagedDecodeAttentionProvider),
        h2o_layerwise_probability_scores=True,
    )
    provider = provider_type()
    provider.prepare(spec, device_index=torch.cuda.current_device())
    prepared = PreparedDecodeAttentionOp(spec, provider)

    try:
        with patch(
            "sparsevllm.operators.decode_attention.get_context",
            return_value=SimpleNamespace(attention_validation_scope=object()),
        ):
            output = prepared.run(q, view)
    finally:
        prepared.close()

    expected_output = []
    expected_score = []
    group = query_heads // kv_heads
    for batch_idx in range(batch):
        length = int(context_lens[batch_idx].item())
        row = int(req_indices[batch_idx].item())
        active = page_table[row, :length].long()
        keys = k_cache[active].repeat_interleave(group, dim=1)
        values = v_cache[active].repeat_interleave(group, dim=1)
        logits = torch.einsum(
            "hd,lhd->hl",
            q[batch_idx].float(),
            keys.float(),
        ) * spec.softmax_scale
        probabilities = torch.softmax(logits, dim=-1)
        expected_output.append(
            torch.einsum("hl,lhd->hd", probabilities, values.float()).to(
                activation_dtype
            )
        )
        expected_score.append(probabilities.sum(dim=0))

    torch.testing.assert_close(
        output,
        torch.stack(expected_output),
        rtol=3e-2,
        atol=3e-2,
    )
    for batch_idx, expected in enumerate(expected_score):
        length = int(context_lens[batch_idx].item())
        torch.testing.assert_close(
            score[batch_idx, :length],
            expected,
            rtol=3e-3,
            atol=3e-3,
        )
        assert torch.all(score[batch_idx, length:] == 0)


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
    assert call.kwargs == {
        "validation_scope": scope,
        "return_softmax_lse": False,
    }
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
