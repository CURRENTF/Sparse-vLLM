from __future__ import annotations

import pytest
import torch

from sparsevllm.kernels.triton.gemma4_context_attention import gemma4_context_attention
from sparsevllm.kernels.triton.gemma4_decode_attention import (
    gemma4_decode_stage1,
    gemma4_decode_stage2,
)
from sparsevllm.kernels.triton.gemma4_single_block_decode_attention import (
    gemma4_single_block_decode,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _slots_and_lengths():
    lengths = torch.tensor([21, 13], dtype=torch.int32, device="cuda")
    slots = torch.zeros((2, 21), dtype=torch.int32, device="cuda")
    slots[0, :21] = torch.arange(21, dtype=torch.int32, device="cuda")
    slots[1, :13] = torch.arange(21, 34, dtype=torch.int32, device="cuda")
    return slots, lengths


def _decode_reference(q, k, v, slots, lengths, window):
    output = torch.empty_like(q)
    for batch, length in enumerate(lengths.tolist()):
        start = max(0, length - (window or length))
        indices = slots[batch, start:length].long()
        for head in range(q.shape[1]):
            logits = q[batch, head] @ k[indices, head // (q.shape[1] // k.shape[1])].T
            output[batch, head] = logits.softmax(-1) @ v[
                indices, head // (q.shape[1] // v.shape[1])
            ]
    return output


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("sliding_window", [None, 4])
def test_gemma4_prefill_matches_torch(head_dim, sliding_window):
    torch.manual_seed(3)
    prefix = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    chunks = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    lengths = prefix + chunks
    starts = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    slots = torch.zeros((2, 9), dtype=torch.int32, device="cuda")
    slots[0, :9] = torch.arange(9, dtype=torch.int32, device="cuda")
    slots[1, :5] = torch.arange(9, 14, dtype=torch.int32, device="cuda")
    key = torch.randn(14, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(8, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)

    gemma4_context_attention(
        query,
        key,
        value,
        output,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        starts,
        lengths,
        prefix,
        5,
        slots,
        sliding_window=sliding_window,
    )
    reference = torch.empty_like(output)
    for batch, (prefix_len, chunk_len, start) in enumerate(
        zip(prefix.tolist(), chunks.tolist(), starts.tolist())
    ):
        for offset in range(chunk_len):
            end = prefix_len + offset + 1
            begin = max(0, end - (sliding_window or end))
            indices = slots[batch, begin:end].long()
            for head in range(query.shape[1]):
                logits = query[start + offset, head] @ key[indices, head // 2].T
                reference[start + offset, head] = logits.softmax(-1) @ value[indices, head // 2]
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert cosine > 0.999


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("sliding_window", [None, 4])
def test_gemma4_decode_matches_torch(head_dim, sliding_window):
    torch.manual_seed(7)
    slots, lengths = _slots_and_lengths()
    key = torch.randn(34, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(2, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    block_seq = 8
    blocks = (int(lengths.max()) + block_seq - 1) // block_seq
    mid = torch.empty(2, 4, blocks, head_dim, dtype=torch.float32, device="cuda")
    lse = torch.empty(2, 4, blocks, dtype=torch.float32, device="cuda")
    output = torch.empty_like(query)

    gemma4_decode_stage1(
        query,
        key,
        value,
        slots,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        lengths,
        mid,
        lse,
        block_seq=block_seq,
        sliding_window=sliding_window,
    )
    gemma4_decode_stage2(
        mid,
        lse,
        lengths,
        output,
        block_seq=block_seq,
        sliding_window=sliding_window,
    )

    reference = _decode_reference(query, key, value, slots, lengths, sliding_window)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert cosine > 0.999


@pytest.mark.parametrize("group_size", [2, 4, 8])
@pytest.mark.parametrize("head_dim", [256, 512])
def test_gemma4_single_block_decode_matches_torch(group_size, head_dim):
    torch.manual_seed(11)
    slots, lengths = _slots_and_lengths()
    key = torch.randn(34, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(2, 2 * group_size, head_dim, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    gemma4_single_block_decode(
        query,
        key,
        value,
        slots,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        lengths,
        output,
        block_seq=256,
        sliding_window=None,
    )
    reference = _decode_reference(query, key, value, slots, lengths, None)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert cosine > 0.999


@pytest.mark.parametrize("head_dim", [256, 512])
def test_gemma4_decode_supports_cuda_graph(head_dim):
    slots, lengths = _slots_and_lengths()
    query = torch.randn(2, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    key = torch.randn(34, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    mid = torch.empty(2, 4, 1, head_dim, dtype=torch.float32, device="cuda")
    lse = torch.empty(2, 4, 1, dtype=torch.float32, device="cuda")
    output = torch.empty_like(query)
    request_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    def run():
        gemma4_decode_stage1(
            query,
            key,
            value,
            slots,
            request_indices,
            lengths,
            mid,
            lse,
            block_seq=256,
            sliding_window=1024,
        )
        gemma4_decode_stage2(
            mid,
            lse,
            lengths,
            output,
            block_seq=256,
            sliding_window=1024,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    query.copy_(torch.randn_like(query))
    graph.replay()
    first = output.clone()
    graph.replay()
    assert torch.equal(first, output)


@pytest.mark.parametrize("score_dims", [2, 3])
def test_gemma4_decode_collects_raw_qk_scores(score_dims):
    torch.manual_seed(13)
    slots, lengths = _slots_and_lengths()
    head_dim = 256
    query = torch.randn(2, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    key = torch.randn(34, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    mid = torch.empty(2, 4, 3, head_dim, dtype=torch.float32, device="cuda")
    lse = torch.empty(2, 4, 3, dtype=torch.float32, device="cuda")
    score = torch.full(
        (2, 4, 21) if score_dims == 3 else (2, 21),
        -1e20,
        dtype=torch.float32,
        device="cuda",
    )
    gemma4_decode_stage1(
        query,
        key,
        value,
        slots,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        lengths,
        mid,
        lse,
        block_seq=8,
        sliding_window=None,
        attn_score=score,
    )
    expected = torch.empty(2, 4, 21, dtype=torch.float32, device="cuda")
    expected.fill_(-1e20)
    for batch, length in enumerate(lengths.tolist()):
        for head in range(4):
            indices = slots[batch, :length].long()
            expected[batch, head, :length] = (
                query[batch, head].float()
                @ key[indices, head // 2].float().T
            )
    expected = expected if score_dims == 3 else expected.max(1).values
    torch.testing.assert_close(score, expected, rtol=2e-2, atol=1.0)


@pytest.mark.parametrize("score_dims", [2, 3])
def test_gemma4_prefill_collects_raw_qk_scores(score_dims):
    torch.manual_seed(17)
    head_dim = 256
    prefix = torch.tensor([0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    slots = torch.arange(4, dtype=torch.int32, device="cuda").unsqueeze(0)
    key = torch.randn(4, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(4, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    score = torch.zeros(
        (1, 4, 4) if score_dims == 3 else (1, 4),
        dtype=torch.float32,
        device="cuda",
    )
    gemma4_context_attention(
        query,
        key,
        value,
        output,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        starts,
        lengths,
        prefix,
        4,
        slots,
        sliding_window=None,
        attn_score=score,
    )
    expected = torch.zeros(1, 4, 4, dtype=torch.float32, device="cuda")
    for head in range(4):
        logits = query[:, head].float() @ key[:, head // 2].float().T
        expected[0, head] = logits.tril().sum(0)
    expected = (
        expected
        if score_dims == 3
        else (expected / 4).max(1).values.clamp_min_(0)
    )
    torch.testing.assert_close(score, expected, rtol=2e-2, atol=1.0)
