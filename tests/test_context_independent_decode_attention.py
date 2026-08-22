from __future__ import annotations

import pytest
import torch

from sparsevllm.kernels.triton.context_independent_flash_decoding import (
    context_independent_flash_decode,
)
from sparsevllm.layers.attention import Attention
from sparsevllm.operators.context_independent_decode_attention import (
    ContextIndependentTritonAttentionBackend,
    bind_context_independent_triton_attention,
)


class _AttentionContainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = Attention(8, 64, 0.125, 2)
        self.second = Attention(8, 64, 0.125, 2)
        self.other_shape = Attention(8, 128, 0.125, 2)
        self.hd256 = Attention(8, 256, 0.125, 2)


def test_context_independent_binding_shares_workspace_by_shape() -> None:
    model = _AttentionContainer()
    bound, workspace_bytes = bind_context_independent_triton_attention(
        model,
        max_batch_size=4,
        device=torch.device("cpu"),
    )

    assert bound == 4
    assert isinstance(model.first.attention_backend, ContextIndependentTritonAttentionBackend)
    assert model.first.attention_backend is model.second.attention_backend
    assert model.first.attention_backend is not model.other_shape.attention_backend
    assert model.first.attention_backend.tuning.block_n == 64
    assert model.hd256.attention_backend.tuning.block_n == 128
    assert model.hd256.attention_backend.tuning.num_warps == 4
    assert workspace_bytes > 0


def _reference_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads, head_dim = q.shape
    group_size = num_heads // k.shape[1]
    output = torch.empty_like(q)
    scores = torch.full(
        (batch, num_heads, active_slots.shape[1]),
        -1e20,
        dtype=torch.float32,
        device=q.device,
    )
    for batch_id in range(batch):
        length = int(context_lens[batch_id].item())
        slots = active_slots[req_indices[batch_id], :length].long()
        keys = k.index_select(0, slots).repeat_interleave(group_size, dim=1)
        values = v.index_select(0, slots).repeat_interleave(group_size, dim=1)
        raw_scores = torch.einsum("hd,lhd->hl", q[batch_id].float(), keys.float())
        probs = torch.softmax(raw_scores / (head_dim**0.5), dim=-1)
        output[batch_id] = torch.einsum("hl,lhd->hd", probs, values.float()).to(q.dtype)
        scores[batch_id, :, :length] = raw_scores
    return output, scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_dim"),
    [(8, 8, 64), (16, 2, 128), (8, 2, 256)],
)
def test_context_independent_decode_matches_reference(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, capacity, max_splits = 3, 4608, 8
    q = torch.randn((batch, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((capacity * batch, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    active_slots = torch.arange(capacity * batch, device=device, dtype=torch.int64).view(batch, capacity)
    req_indices = torch.tensor([2, 0, 1], dtype=torch.int64, device=device)
    context_lens = torch.tensor([1, 1025, 4607], dtype=torch.int32, device=device)
    mid_o = torch.empty((batch, num_heads, max_splits, head_dim), dtype=torch.float32, device=device)
    mid_lse = torch.empty((batch, num_heads, max_splits), dtype=torch.float32, device=device)
    score = torch.full((batch, num_heads, capacity), -1e20, dtype=torch.float32, device=device)

    actual = context_independent_flash_decode(
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_o,
        mid_lse,
        attn_score=score,
        target_tokens_per_split=1024,
    )
    expected, expected_score = _reference_decode(
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for batch_id, length in enumerate(context_lens.tolist()):
        torch.testing.assert_close(
            score[batch_id, :, :length],
            expected_score[batch_id, :, :length],
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("score_dims", [2, 3])
def test_context_independent_decode_reduces_2d_scores_across_heads(
    score_dims: int,
) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, num_heads, num_kv_heads, head_dim = 2, 16, 2, 128
    capacity, max_splits = 257, 8
    q = torch.randn(batch, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        batch * capacity,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    slots = torch.arange(
        batch * capacity, dtype=torch.int64, device=device
    ).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int64, device=device)
    context_lens = torch.tensor([129, 257], dtype=torch.int32, device=device)
    mid_o = torch.empty(
        batch, num_heads, max_splits, head_dim, dtype=torch.float32, device=device
    )
    mid_lse = torch.empty(
        batch, num_heads, max_splits, dtype=torch.float32, device=device
    )
    score_shape = (
        (batch, capacity)
        if score_dims == 2
        else (batch, num_heads, capacity)
    )
    score = torch.full(score_shape, -1e20, dtype=torch.float32, device=device)

    context_independent_flash_decode(
        q,
        k,
        v,
        slots,
        req_indices,
        context_lens,
        mid_o,
        mid_lse,
        attn_score=score,
        target_tokens_per_split=64,
    )
    _, expected_3d = _reference_decode(
        q,
        k,
        v,
        slots,
        req_indices,
        context_lens,
    )
    expected = expected_3d.amax(dim=1) if score_dims == 2 else expected_3d
    for batch_id, length in enumerate(context_lens.tolist()):
        torch.testing.assert_close(
            score[batch_id, ..., :length],
            expected[batch_id, ..., :length],
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_context_independent_decode_graph_replays_different_lengths() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch, num_heads, num_kv_heads, head_dim = 2, 8, 2, 128
    capacity, max_splits = 4096, 8
    q = torch.randn((batch, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((batch * capacity, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    active_slots = torch.arange(batch * capacity, dtype=torch.int64, device=device).view(batch, capacity)
    req_indices = torch.arange(batch, dtype=torch.int64, device=device)
    context_lens = torch.tensor([512, 2048], dtype=torch.int32, device=device)
    mid_o = torch.empty((batch, num_heads, max_splits, head_dim), dtype=torch.float32, device=device)
    mid_lse = torch.empty((batch, num_heads, max_splits), dtype=torch.float32, device=device)

    def run() -> torch.Tensor:
        return context_independent_flash_decode(
            q,
            k,
            v,
            active_slots,
            req_indices,
            context_lens,
            mid_o,
            mid_lse,
            target_tokens_per_split=1024,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()

    context_lens.copy_(torch.tensor([4095, 1025], dtype=torch.int32, device=device))
    q.copy_(torch.randn_like(q))
    graph.replay()
    expected, _ = _reference_decode(q, k, v, active_slots, req_indices, context_lens)
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_output, expected, atol=2e-2, rtol=2e-2)
