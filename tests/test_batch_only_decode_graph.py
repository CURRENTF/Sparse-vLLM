from types import SimpleNamespace

import pytest
import torch

from sparsevllm.configs.cuda_graph import (
    _normalize_decode_graph_shape_policy,
    build_decode_cuda_graph_batch_only_startup_plan,
)
from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparsevllm.kernels.triton.context_independent_flash_decoding import (
    context_independent_flash_decode,
)
from sparsevllm.kernels.triton.sglang_gemma4_decode_attention import (
    sglang_gemma4_decode,
)
from sparsevllm.operators.context_independent_gemma4_attention import (
    ContextIndependentGemma4OperatorProvider,
)
from sparsevllm.operators.decode_attention import (
    ContextIndependentTritonDecodeAttentionProvider,
    DecodeAttentionOpSpec,
    TritonPagedDecodeAttentionProvider,
)
from sparsevllm.operators.gemma4 import Gemma4OpSpec, TritonGemma4OperatorProvider
from sparsevllm.operators.mla_attention import (
    ContextIndependentMlaTritonProvider,
    MlaAttentionOpSpec,
    MlaTritonProvider,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


def _cuda_caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="test sm90",
        compute_capability=(9, 0),
        runtime_version="12.8",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        multiprocessor_count=120,
    )


def test_batch_only_policy_aliases_and_rejects_unknown_values() -> None:
    assert _normalize_decode_graph_shape_policy("batch") == "batch_only"
    assert _normalize_decode_graph_shape_policy("context-independent") == "batch_only"
    assert _normalize_decode_graph_shape_policy(None) == "bucketed"
    with pytest.raises(ValueError, match="shape_policy"):
        _normalize_decode_graph_shape_policy("sequence_only")


def test_batch_only_startup_plan_has_one_graph_per_batch_and_path() -> None:
    config = SimpleNamespace(
        decode_graph_capture_sizes=[1, 4],
        decode_graph_startup_capture_limit=8,
        decode_graph_max_cached_graphs=8,
        sparse_method="quest",
        max_model_len=32768,
        sink_keep_tokens=64,
        decode_keep_tokens=4096,
        recent_keep_tokens=512,
    )
    assert build_decode_cuda_graph_batch_only_startup_plan(config) == [
        (4, 32768, True),
        (4, 4672, False),
        (1, 32768, True),
        (1, 4672, False),
    ]


def test_batch_only_state_identity_omits_context_capacity() -> None:
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.shape_policy = "batch_only"
    runner.startup_plan_sealed = False
    runner._graphs = {}
    runner.max_cached_graphs = None
    runner.cache_manager = SimpleNamespace(device=torch.device("cpu"))
    runner.eviction_count = 0

    state = runner._select_state(
        method="quest",
        batch_size=4,
        context_capacity=32768,
        is_long_text=True,
        capture_sampling=False,
        graph_path_id="long",
    )
    reused = runner._select_state(
        method="quest",
        batch_size=4,
        context_capacity=8192,
        is_long_text=True,
        capture_sampling=False,
        graph_path_id="long",
    )
    assert reused is state
    assert state.key.context_capacity == 0
    assert state.capture_context_capacity == 32768

    with pytest.raises(RuntimeError, match="exceeded captured path capacity"):
        runner._select_state(
            method="quest",
            batch_size=4,
            context_capacity=65536,
            is_long_text=True,
            capture_sampling=False,
            graph_path_id="long",
        )


def test_mha_resolver_contract_selects_only_context_independent_provider() -> None:
    spec = DecodeAttentionOpSpec(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        activation_dtype=torch.bfloat16,
        softmax_scale=128**-0.5,
        max_batch_size=8,
        context_independent_cuda_graph=True,
    )
    caps = _cuda_caps()
    assert ContextIndependentTritonDecodeAttentionProvider.supports(spec, caps).supported
    assert not TritonPagedDecodeAttentionProvider.supports(spec, caps).supported

    h2o_spec = DecodeAttentionOpSpec(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        activation_dtype=torch.bfloat16,
        softmax_scale=128**-0.5,
        max_batch_size=8,
        may_require_attention_scores=True,
        h2o_layerwise_probability_scores=True,
        context_independent_cuda_graph=True,
    )
    assert ContextIndependentTritonDecodeAttentionProvider.supports(
        h2o_spec, caps
    ).supported


def test_mla_resolver_contract_selects_fixed_launch_provider() -> None:
    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=2,
        cuda_graph=True,
        context_independent_cuda_graph=True,
    )
    caps = _cuda_caps()
    assert ContextIndependentMlaTritonProvider.supports(spec, caps).supported
    assert not MlaTritonProvider.supports(spec, caps).supported


def test_gemma4_resolver_contract_selects_fixed_grid_provider() -> None:
    spec = Gemma4OpSpec(
        activation_dtype=torch.bfloat16,
        head_dims=(256, 512),
        cuda_graph=True,
        attention_contracts=((8, 2, 256, 1023), (8, 1, 512, -1)),
        max_batch_size=8,
        context_independent_cuda_graph=True,
    )
    caps = _cuda_caps()
    assert ContextIndependentGemma4OperatorProvider.supports(spec, caps).supported
    assert not TritonGemma4OperatorProvider.supports(spec, caps).supported


def _decode_reference(
    q, k, v, slots, req_indices, lengths, window=None, *, scale=True
):
    output = torch.empty_like(q)
    lse = torch.empty(
        q.shape[1], q.shape[0], dtype=torch.float32, device=q.device
    )
    group_size = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths.tolist()):
        start = max(0, length - int(window or length))
        indices = slots[req_indices[batch], start:length].long()
        keys = k[indices].repeat_interleave(group_size, dim=1)
        values = v[indices].repeat_interleave(group_size, dim=1)
        logits = torch.einsum("hd,lhd->hl", q[batch].float(), keys.float())
        if scale:
            logits = logits / q.shape[-1] ** 0.5
        probabilities = logits.softmax(-1)
        output[batch] = torch.einsum(
            "hl,lhd->hd", probabilities, values.float()
        ).to(q.dtype)
        lse[:, batch] = logits.logsumexp(-1)
    return output, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
def test_context_independent_mha_matches_reference_and_replays_new_lengths() -> None:
    torch.manual_seed(11)
    batch, heads, kv_heads, head_dim, capacity = 2, 8, 2, 128, 257
    device = torch.device("cuda")
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        batch * capacity, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)
    slots = torch.arange(
        batch * capacity, dtype=torch.int32, device=device
    ).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([129, 257], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch, heads, 8, head_dim, dtype=torch.float32, device=device
    )
    mid_lse = torch.empty(batch, heads, 8, dtype=torch.float32, device=device)
    output_lse = torch.empty(heads, batch, dtype=torch.float32, device=device)

    def run():
        return context_independent_flash_decode(
            q,
            k,
            v,
            slots,
            req_indices,
            lengths,
            mid_o,
            mid_lse,
            target_tokens_per_split=64,
            return_softmax_lse=True,
            output_lse=output_lse,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_lse = run()
    lengths.copy_(torch.tensor([17, 201], dtype=torch.int32, device=device))
    q.copy_(torch.randn_like(q))
    graph.replay()
    expected_output, expected_lse = _decode_reference(
        q, k, v, slots, req_indices, lengths
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(graph_lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
@pytest.mark.parametrize("window", [None, 8])
def test_context_independent_gemma4_matches_reference_and_graph(window) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    batch, heads, kv_heads, head_dim, capacity = 2, 4, 2, 256, 33
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        batch * capacity, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)
    slots = torch.arange(
        batch * capacity, dtype=torch.int32, device=device
    ).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.tensor([33, 21], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch, heads, 8, head_dim, dtype=torch.float32, device=device
    )
    mid_lse = torch.empty(batch, heads, 8, dtype=torch.float32, device=device)
    splits = torch.empty(batch, dtype=torch.int32, device=device)

    def run():
        return sglang_gemma4_decode(
            q,
            k,
            v,
            slots,
            req_indices,
            lengths,
            mid_o,
            mid_lse,
            splits,
            sliding_window=window,
            device_core_count=120,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    lengths.copy_(torch.tensor([17, 29], dtype=torch.int32, device=device))
    q.copy_(torch.randn_like(q))
    graph.replay()
    expected, _ = _decode_reference(
        q, k, v, slots, req_indices, lengths, window, scale=False
    )
    torch.cuda.synchronize()
    cosine = torch.nn.functional.cosine_similarity(
        graph_output.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine > 0.999
