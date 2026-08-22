from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sparsevllm.layers.mla_attention as mla_attention_module
from sparsevllm.engine.cache_manager import (
    AttentionKeyComputeView,
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparsevllm.layers.mla_attention import (
    MLAAttention,
    estimate_mla_prefill_workspace_bytes,
)
from sparsevllm.operators.mla_attention import (
    MlaAttentionOpSpec,
    MlaAttentionProvider,
)
from sparsevllm.utils.context import get_context, reset_context, set_context


class _TestProvider(MlaAttentionProvider):
    name = "test"

    def __init__(
        self,
        spec: MlaAttentionOpSpec,
        *,
        device: torch.device | str,
        max_batch_size: int,
    ) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.max_batch_size = int(max_batch_size)


class _ExplicitPrefillProvider(_TestProvider):
    supports_explicit_prefill = True

    def run_explicit_prefill(
        self,
        q,
        view,
        output,
        *,
        cu_seqlens_q,
        max_seqlen_q,
        validation_scope=None,
    ):
        self.prefill_call = {
            "q": q,
            "view": view,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": max_seqlen_q,
            "validation_scope": validation_scope,
        }
        output.copy_(q)
        return output


def _spec(tp_size: int = 4) -> MlaAttentionOpSpec:
    return MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=tp_size,
        cuda_graph=False,
    )


def _attention(
    *,
    device: torch.device | str = "cpu",
    tp_size: int = 4,
    max_batch_size: int = 4,
    budget: int = 64 * 1024 * 1024,
) -> MLAAttention:
    spec = _spec(tp_size)
    resolved_device = torch.device("cuda:0") if str(device) == "cuda" else device
    return MLAAttention(
        spec=spec,
        provider=_TestProvider(
            spec,
            device=resolved_device,
            max_batch_size=max_batch_size,
        ),
        prefill_workspace_bytes=budget,
        hidden_size=64,
        projection_chunk_size=8,
    )


def _view(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
) -> PrefillComputeView:
    return PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=active_slots,
            req_indices=request_indices,
            context_lens=context_lens,
            max_context_len=int(context_lens.max().item()),
        ),
        payload=MlaLatentPayload(
            latent_cache=latent_cache,
            rope_cache=rope_cache,
        ),
    )


def test_mla_binds_key_materializer_once_per_manager_and_layer() -> None:
    attention = _attention()
    project_latent = Mock()
    first_manager = SimpleNamespace(register_attention_key_materializer=Mock())
    second_manager = SimpleNamespace(register_attention_key_materializer=Mock())

    attention._ensure_key_materializer(first_manager, 0, project_latent)
    attention._ensure_key_materializer(first_manager, 0, project_latent)
    attention._ensure_key_materializer(first_manager, 1, project_latent)
    attention._ensure_key_materializer(second_manager, 0, project_latent)

    assert first_manager.register_attention_key_materializer.call_count == 2
    second_manager.register_attention_key_materializer.assert_called_once()


def _expand_history(attention: MLAAttention, history):
    heads = attention.spec.local_q_heads
    latent = history.gathered_latent
    rope = history.gathered_rope[:, None, :].expand(-1, heads, -1)
    k_nope = latent[:, None, :192].expand(-1, heads, -1)
    expanded_k = torch.cat((k_nope, rope), dim=-1)
    expanded_v = latent[:, None, 192:448].expand(-1, heads, -1).contiguous()
    return attention.bind_prefill_kv(
        history,
        expanded_k=expanded_k,
        expanded_v=expanded_v,
    )


def _torch_prefill(
    q: torch.Tensor,
    workset,
    chunk_lens: torch.Tensor,
) -> torch.Tensor:
    history = workset.history
    outputs = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens.tolist()):
        context_len = int(history.context_lens[batch_index].item())
        prefix_len = context_len - int(chunk_len)
        history_start = int(history.packed_offsets[batch_index].item())
        keys = workset.expanded_k[
            history_start : history_start + context_len
        ].float()
        values = workset.expanded_v[
            history_start : history_start + context_len
        ].float()
        queries = q[query_start : query_start + chunk_len].float()
        for query_offset, query in enumerate(queries):
            visible = prefix_len + query_offset + 1
            logits = torch.einsum("hd,thd->ht", query, keys[:visible])
            probabilities = torch.softmax(logits * (256**-0.5), dim=-1)
            outputs.append(
                torch.einsum(
                    "ht,thd->hd",
                    probabilities,
                    values[:visible],
                ).to(torch.bfloat16)
            )
        query_start += int(chunk_len)
    return torch.stack(outputs)


def test_mla_prefill_workspace_estimate_accounts_for_full_history() -> None:
    actual = estimate_mla_prefill_workspace_bytes(
        total_visible_tokens=11,
        query_tokens=5,
        batch_size=2,
        max_context_len=7,
        local_q_heads=5,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        hidden_size=64,
        projection_chunk_size=4,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
    )

    gathered = 11 * (512 + 64) * 2
    projected = 11 * 5 * (192 + 256) * 2
    projection_scratch = 4 * 5 * (192 + 256) * 2
    expanded_k = 11 * 5 * 256 * 2
    attention_output = 5 * 5 * 256 * 2
    output_projection_scratch = 4 * 64 * 2
    metadata = (2 * 7 + 2 * 2 + 7) * 4
    assert actual == max(
        gathered + projected + projection_scratch,
        gathered + projected + expanded_k + attention_output,
        attention_output + output_projection_scratch,
    ) + metadata


def test_mla_prefill_rejects_wrong_payload_before_gather() -> None:
    attention = _attention()
    view = PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0]], dtype=torch.int32),
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([1], dtype=torch.int32),
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty(1, 1, 256, dtype=torch.bfloat16),
            v_cache=torch.empty(1, 1, 256, dtype=torch.bfloat16),
        ),
    )

    with (
        patch("sparsevllm.layers.mla_attention.gather_latent_history") as gather,
        pytest.raises(TypeError, match="MlaLatentPayload"),
    ):
        attention.prepare_prefill_history(view, query_tokens=1)
    gather.assert_not_called()


def test_mla_prefill_budget_fails_before_allocation_or_gather() -> None:
    attention = _attention(budget=1)
    view = _view(
        torch.empty(2, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 1, 64, dtype=torch.bfloat16),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )

    with (
        patch("sparsevllm.layers.mla_attention.gather_latent_history") as gather,
        pytest.raises(MemoryError, match="exceeds its configured budget"),
    ):
        attention.prepare_prefill_history(view, query_tokens=2)
    gather.assert_not_called()


def test_mla_attention_bind_resolves_provider_once() -> None:
    spec = _spec()
    provider = _TestProvider(spec, device="cpu", max_batch_size=8)

    with patch(
        "sparsevllm.layers.mla_attention.resolve_mla_attention_provider",
        return_value=provider,
    ) as resolve:
        attention = MLAAttention.bind(
            spec=spec,
            device="cpu",
            max_batch_size=8,
            prefill_workspace_bytes=1024,
            hidden_size=64,
            projection_chunk_size=8,
        )

    assert attention.provider is provider
    resolve.assert_called_once_with(
        spec,
        device="cpu",
        max_batch_size=8,
    )


def test_set_context_starts_a_new_attention_validation_scope() -> None:
    reset_context()
    initial_scope = get_context().attention_validation_scope
    set_context(False)
    first_step_scope = get_context().attention_validation_scope
    set_context(False)
    second_step_scope = get_context().attention_validation_scope

    assert first_step_scope is not initial_scope
    assert second_step_scope is not first_step_scope


def test_mla_decode_passes_valid_batch_size_to_provider() -> None:
    class _RecordingProvider(_TestProvider):
        def run(
            self,
            q_nope_absorbed,
            q_rope,
            view,
            output,
            *,
            validation_scope=None,
            valid_batch_size=None,
        ):
            self.valid_batch_size = valid_batch_size
            self.output_is_contiguous = output.is_contiguous()
            return output

    spec = _spec(tp_size=4)
    provider = _RecordingProvider(spec, device="cpu", max_batch_size=4)
    attention = MLAAttention(
        spec=spec,
        provider=provider,
        prefill_workspace_bytes=64 * 1024 * 1024,
        hidden_size=64,
        projection_chunk_size=8,
    )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.arange(16, dtype=torch.int32).view(4, 4),
            req_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
            context_lens=torch.tensor([4, 4, 4, 4], dtype=torch.int32),
        ),
        payload=MlaLatentPayload(
            latent_cache=torch.empty(16, 1, 512, dtype=torch.bfloat16),
            rope_cache=torch.empty(16, 1, 64, dtype=torch.bfloat16),
        ),
    )
    q_nope_absorbed = torch.empty(5, 4, 512, dtype=torch.bfloat16).transpose(0, 1)
    q_rope = torch.empty(5, 4, 64, dtype=torch.bfloat16).transpose(0, 1)

    set_context(False, seqs=[object(), object(), object()])
    try:
        output = attention.run_decode(q_nope_absorbed, q_rope, view)
    finally:
        reset_context()

    assert output.shape == q_nope_absorbed.shape
    assert not q_nope_absorbed.is_contiguous()
    assert output.is_contiguous()
    assert provider.output_is_contiguous
    assert provider.valid_batch_size == 3


def test_mla_prefill_reuses_validated_packing_across_layers() -> None:
    attention = _attention()
    active_slots = torch.tensor([[3, 1]], dtype=torch.int32)
    request_indices = torch.tensor([0], dtype=torch.int32)
    context_lens = torch.tensor([2], dtype=torch.int32)
    first_view = _view(
        torch.empty(4, 1, 512, dtype=torch.bfloat16),
        torch.empty(4, 1, 64, dtype=torch.bfloat16),
        active_slots,
        request_indices,
        context_lens,
    )
    second_view = _view(
        torch.empty(4, 1, 512, dtype=torch.bfloat16),
        torch.empty(4, 1, 64, dtype=torch.bfloat16),
        active_slots,
        request_indices,
        context_lens,
    )

    with (
        patch(
            "sparsevllm.layers.mla_attention.validate_gather_metadata"
        ) as validate,
        patch(
            "sparsevllm.layers.mla_attention.gather_latent_history"
        ) as gather,
    ):
        first = attention.prepare_prefill_history(first_view, query_tokens=2)
        second = attention.prepare_prefill_history(second_view, query_tokens=2)
        reset_context()
        attention.prepare_prefill_history(second_view, query_tokens=2)

    assert validate.call_count == 2
    assert gather.call_count == 3
    assert first.packed_offsets is second.packed_offsets
    assert first.packed_cu_seqlens is second.packed_cu_seqlens
    assert first.packed_slots is second.packed_slots


def test_mla_prefill_reuses_query_validation_across_layers() -> None:
    attention = _attention()
    view = _view(
        torch.empty(2, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 1, 64, dtype=torch.bfloat16),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    with patch("sparsevllm.layers.mla_attention.gather_latent_history"):
        history = attention.prepare_prefill_history(view, query_tokens=2)
    workset = _expand_history(attention, history)
    q = torch.empty(2, 5, 256, dtype=torch.bfloat16)
    starts = torch.tensor([0], dtype=torch.int32)
    chunks = torch.tensor([2], dtype=torch.int32)

    with (
        patch(
            "sparsevllm.layers.mla_attention._host_int_values",
            wraps=mla_attention_module._host_int_values,
        ) as host_values,
        patch.object(
            attention.prefill_backend,
            "run_prefill",
            return_value=torch.empty_like(q),
        ),
    ):
        attention.run_prefill(
            q,
            workset,
            b_start_loc=starts,
            chunk_lens=chunks,
        )
        attention.run_prefill(
            q,
            workset,
            b_start_loc=starts,
            chunk_lens=chunks,
        )
        assert host_values.call_count == 2

        reset_context()
        attention.run_prefill(
            q,
            workset,
            b_start_loc=starts,
            chunk_lens=chunks,
        )
        assert host_values.call_count == 4


def test_mla_prefill_exposes_expanded_explicit_score_view() -> None:
    attention = _attention()
    view = _view(
        torch.empty(2, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 1, 64, dtype=torch.bfloat16),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    with patch("sparsevllm.layers.mla_attention.gather_latent_history"):
        history = attention.prepare_prefill_history(view, query_tokens=2)
    workset = _expand_history(attention, history)

    score_view = attention.build_prefill_explicit_view(workset)

    assert isinstance(score_view.payload, ExplicitKVPayload)
    assert score_view.payload.k_cache is workset.expanded_k
    assert score_view.payload.v_cache is workset.expanded_v
    assert score_view.payload.metadata == {
        "layout": "mla_packed_varlen",
        "cu_seqlens_k": history.packed_cu_seqlens,
    }
    torch.testing.assert_close(
        history.packed_cu_seqlens,
        torch.tensor([0, 2], dtype=torch.int32),
    )
    assert score_view.meta.active_slots is history.packed_slots
    assert score_view.meta.req_indices is history.local_req_indices
    assert score_view.meta.context_lens is history.context_lens
    assert score_view.meta.max_context_len == history.max_context_len


def test_mla_prefill_dispatches_explicit_provider_with_shared_cu_seqlens() -> None:
    spec = _spec()
    provider = _ExplicitPrefillProvider(spec, device="cpu", max_batch_size=4)
    attention = MLAAttention(
        spec=spec,
        provider=provider,
        prefill_workspace_bytes=64 * 1024 * 1024,
        hidden_size=64,
        projection_chunk_size=8,
    )
    view = _view(
        torch.empty(2, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 1, 64, dtype=torch.bfloat16),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    with patch("sparsevllm.layers.mla_attention.gather_latent_history"):
        history = attention.prepare_prefill_history(view, query_tokens=2)
    workset = _expand_history(attention, history)
    q = torch.randn(2, 5, 256, dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
    set_context(True, cu_seqlens_q=cu_seqlens_q)
    try:
        output = attention.run_prefill(
            q,
            workset,
            b_start_loc=cu_seqlens_q[:-1],
            chunk_lens=cu_seqlens_q[1:] - cu_seqlens_q[:-1],
        )
    finally:
        reset_context()

    torch.testing.assert_close(output, q)
    assert provider.prefill_call["view"].payload.k_cache is workset.expanded_k
    assert provider.prefill_call["cu_seqlens_q"] is cu_seqlens_q
    assert provider.prefill_call["max_seqlen_q"] == 2


def test_mla_materializes_actual_keys_for_permuted_slots() -> None:
    torch.manual_seed(47)
    attention = _attention(tp_size=4)
    latent_cache = torch.randn(7, 1, 512, dtype=torch.bfloat16)
    rope_cache = torch.randn(7, 1, 64, dtype=torch.bfloat16)
    slots = torch.tensor([[5, 1], [6, 2]], dtype=torch.int32)
    view = AttentionKeyComputeView(
        active_slots=slots,
        payload=MlaLatentPayload(
            latent_cache=latent_cache,
            rope_cache=rope_cache,
        ),
    )
    weights = torch.randn(
        attention.spec.local_q_heads,
        448,
        512,
        dtype=torch.bfloat16,
    )

    def project_latent(latent: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "tr,hor->tho",
            latent.float(),
            weights.float(),
        ).to(torch.bfloat16).flatten(1)

    actual = attention.materialize_expanded_keys(
        view,
        project_latent=project_latent,
    )

    flat_slots = slots.long().flatten()
    latent = latent_cache[flat_slots, 0]
    projected = torch.einsum(
        "tr,hor->tho",
        latent.float(),
        weights.float(),
    ).to(torch.bfloat16)
    expected = torch.cat(
        (
            projected[..., :192],
            rope_cache[flat_slots, 0][:, None, :].expand(
                -1,
                attention.spec.local_q_heads,
                -1,
            ),
        ),
        dim=-1,
    ).view(2, 2, attention.spec.local_q_heads, 256)

    torch.testing.assert_close(actual, expected)


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for MLA full-history prefill tests",
)


@CUDA_REQUIRED
def test_mla_prefill_matches_ragged_full_history_oracle() -> None:
    torch.manual_seed(41)
    attention = _attention(device="cuda", tp_size=4)
    latent_cache = torch.randn(20, 1, 512, dtype=torch.bfloat16, device="cuda")
    rope_cache = torch.randn(20, 1, 64, dtype=torch.bfloat16, device="cuda")
    active_slots = torch.full((3, 7), -1, dtype=torch.int32, device="cuda")
    active_slots[2, :5] = torch.tensor([13, 2, 17, 5, 11], device="cuda")
    active_slots[0, :7] = torch.tensor(
        [19, 1, 7, 15, 3, 9, 6],
        device="cuda",
    )
    context_lens = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    view = _view(
        latent_cache,
        rope_cache,
        active_slots,
        torch.tensor([2, 0], dtype=torch.int32, device="cuda"),
        context_lens,
    )
    history = attention.prepare_prefill_history(view, query_tokens=5)
    workset = _expand_history(attention, history)
    chunk_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q = torch.randn(5, 5, 256, dtype=torch.bfloat16, device="cuda")

    output = attention.run_prefill(
        q,
        workset,
        b_start_loc=b_start_loc,
        chunk_lens=chunk_lens,
    )
    torch.cuda.synchronize()
    expected = _torch_prefill(q, workset, chunk_lens)

    assert history.visible_tokens == 12
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        rtol=3e-2,
        atol=3e-2,
    )


@CUDA_REQUIRED
def test_mla_prefill_is_invariant_to_chunk_boundary() -> None:
    torch.manual_seed(43)
    attention = _attention(device="cuda", tp_size=4)
    latent_cache = torch.randn(12, 1, 512, dtype=torch.bfloat16, device="cuda")
    rope_cache = torch.randn(12, 1, 64, dtype=torch.bfloat16, device="cuda")
    active_slots = torch.tensor(
        [[9, 1, 11, 3, 7, 5]],
        dtype=torch.int32,
        device="cuda",
    )
    request_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    q = torch.randn(6, 5, 256, dtype=torch.bfloat16, device="cuda")

    full_history = attention.prepare_prefill_history(
        _view(
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            torch.tensor([6], dtype=torch.int32, device="cuda"),
        ),
        query_tokens=6,
    )
    full_workset = _expand_history(attention, full_history)
    full_output = attention.run_prefill(
        q,
        full_workset,
        b_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        chunk_lens=torch.tensor([6], dtype=torch.int32, device="cuda"),
    )

    first_history = attention.prepare_prefill_history(
        _view(
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            torch.tensor([4], dtype=torch.int32, device="cuda"),
        ),
        query_tokens=4,
    )
    first_output = attention.run_prefill(
        q[:4],
        _expand_history(attention, first_history),
        b_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        chunk_lens=torch.tensor([4], dtype=torch.int32, device="cuda"),
    )
    second_output = attention.run_prefill(
        q[4:],
        full_workset,
        b_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        chunk_lens=torch.tensor([2], dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(first_output, full_output[:4], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(second_output, full_output[4:], rtol=3e-2, atol=3e-2)
