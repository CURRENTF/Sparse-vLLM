from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsevllm.operators.prefill_attention import (
    PREFILL_ATTENTION_REGISTRY,
    FlashInferPagedPrefillAttentionProvider,
    PrefillAttentionOpSpec,
)
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _spec(**overrides) -> PrefillAttentionOpSpec:
    values = {
        "num_query_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "softmax_scale": 128**-0.5,
        "causal": True,
        "page_size": 1,
        "requires_attention_scores": False,
        "layer_invariant_page_table": True,
    }
    values.update(overrides)
    return PrefillAttentionOpSpec(**values)


def _h100_caps(**overrides) -> DeviceCaps:
    values = {
        "platform": PlatformEnum.CUDA,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "runtime_version": "13.0",
        "supports_graph_capture": True,
        "supports_triton": True,
        "supports_bfloat16": True,
        "supports_native_fp8": True,
    }
    values.update(overrides)
    return DeviceCaps(**values)


@patch(
    "sparsevllm.operators.prefill_attention.version",
    return_value="0.6.15",
)
@patch("sparsevllm.operators.prefill_attention.find_spec", return_value=object())
def test_resolver_prefers_flashinfer_for_profiled_minimax_shape(_find, _version):
    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
        _spec(), _h100_caps()
    )
    assert resolved.provider.name == "flashinfer_paged_prefill_fa3_sm90"


@pytest.mark.parametrize(
    ("spec", "caps", "reason"),
    [
        (_spec(head_dim=64), _h100_caps(), "profiled local"),
        (_spec(activation_dtype=torch.float16), _h100_caps(), "BF16"),
        (_spec(page_size=16), _h100_caps(), "page_size=1"),
        (
            _spec(requires_attention_scores=True),
            _h100_caps(),
            "attention scores",
        ),
        (
            _spec(layer_invariant_page_table=False),
            _h100_caps(),
            "shared across model layers",
        ),
        (
            _spec(),
            _h100_caps(compute_capability=(8, 0), device_name="NVIDIA A100"),
            "SM90",
        ),
    ],
)
def test_flashinfer_provider_rejects_unsupported_contracts(spec, caps, reason):
    result = FlashInferPagedPrefillAttentionProvider.supports(spec, caps)
    assert not result.supported
    assert reason in result.reason


@patch(
    "sparsevllm.operators.prefill_attention.version",
    return_value="0.6.14",
)
@patch("sparsevllm.operators.prefill_attention.find_spec", return_value=object())
def test_resolver_falls_back_when_flashinfer_is_too_old(_find, _version):
    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
        _spec(), _h100_caps()
    )
    assert resolved.provider.name == "triton_paged_prefill"
    assert (
        "flashinfer_paged_prefill_fa3_sm90",
        "requires flashinfer-python >= 0.6.15, got 0.6.14",
    ) in resolved.rejected


def _torch_prefill_oracle(q, logical_k, logical_v, q_lens, kv_lens):
    outputs = []
    q_cursor = 0
    for q_len, kv_len, k, v in zip(q_lens, kv_lens, logical_k, logical_v):
        q_seq = q[q_cursor : q_cursor + q_len].transpose(0, 1).float()
        q_cursor += q_len
        k = k.transpose(0, 1).float().repeat_interleave(6, dim=0)
        v = v.transpose(0, 1).float().repeat_interleave(6, dim=0)
        q_positions = kv_len - q_len + torch.arange(q_len, device=q.device)
        k_positions = torch.arange(kv_len, device=q.device)
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores = torch.matmul(q_seq, k.transpose(-1, -2)) * (128**-0.5)
        scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
        output = torch.matmul(torch.softmax(scores, dim=-1), v)
        outputs.append(output.transpose(0, 1).to(torch.bfloat16))
    return torch.cat(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_page_size_one_matches_noncontiguous_torch_oracle():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The specialized provider requires SM90.")
    pytest.importorskip("flashinfer")
    torch.manual_seed(20260809)
    q_lens = [3, 2]
    kv_lens = [5, 6]
    q = torch.randn(5, 12, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(23, 2, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    pages = torch.randperm(23, device="cuda")[:11]
    rows = torch.zeros(2, 6, device="cuda", dtype=torch.int32)
    rows[0, :5] = pages[:5].to(torch.int32)
    rows[1, :6] = pages[5:].to(torch.int32)
    logical_k = [k_cache[pages[:5]], k_cache[pages[5:]]]
    logical_v = [v_cache[pages[:5]], v_cache[pages[5:]]]
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=rows,
        req_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        context_lens=torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        attn_score=None,
    )
    provider = FlashInferPagedPrefillAttentionProvider()
    spec = _spec()
    provider.prepare(spec)
    actual = provider.run(
        spec,
        q,
        view,
        qo_indptr=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        chunk_lens=torch.tensor(q_lens, device="cuda", dtype=torch.int32),
        max_context_len=6,
        layer_idx=0,
    )
    expected = _torch_prefill_oracle(
        q, logical_k, logical_v, q_lens, kv_lens
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
