import pytest
import torch

from sparsevllm.kernels.triton.h2o_score import h2o_softmax_accumulate
from sparsevllm.kernels.triton.h2o_decode_score import h2o_probability_from_lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("batch_stride", [False, True])
def test_h2o_softmax_accumulate_matches_torch(batch, batch_stride):
    torch.manual_seed(0)
    layers, capacity, width = 3, 192, 129
    previous_width = width - 1
    allocated_batch = batch + 3 if batch_stride else batch
    logits_storage = torch.randn(layers, allocated_batch, capacity, device="cuda")
    logits = logits_storage[:, :batch]
    initial = torch.randn(layers, batch, capacity, device="cuda")
    actual = initial.clone()
    expected = initial.clone()
    scale = 128**-0.5

    probabilities = torch.softmax(logits[:, :, :width] * scale, dim=-1)
    expected[:, :, :previous_width].add_(
        probabilities[:, :, :previous_width]
    )
    expected[:, :, previous_width:width].copy_(
        probabilities[:, :, previous_width:width]
    )
    h2o_softmax_accumulate(
        logits,
        actual,
        width=width,
        previous_width=previous_width,
        softmax_scale=scale,
    )

    assert torch.allclose(
        actual[:, :, :width], expected[:, :, :width], atol=1e-6, rtol=1e-5
    )
    assert torch.equal(actual[:, :, width:], initial[:, :, width:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_h2o_decode_probability_matches_paged_torch(batch):
    torch.manual_seed(11)
    query_heads, kv_heads, head_dim, width = 16, 2, 128, 129
    capacity = batch * width
    q = torch.randn(
        batch,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        capacity,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_table = torch.randperm(capacity, device="cuda", dtype=torch.int64).to(
        torch.int32
    ).view(batch, width)
    request_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    context_lens = torch.arange(
        width - batch + 1,
        width + 1,
        device="cuda",
        dtype=torch.int32,
    )
    scale = head_dim**-0.5
    attention_lse = torch.empty(
        (query_heads, batch), dtype=torch.float32, device="cuda"
    )
    expected_rows = []
    group = query_heads // kv_heads
    for batch_idx in range(batch):
        length = int(context_lens[batch_idx].item())
        keys = k.index_select(0, page_table[batch_idx, :length].long())
        expanded_keys = keys.repeat_interleave(group, dim=1)
        logits = torch.einsum(
            "hd,khd->hk",
            q[batch_idx].float(),
            expanded_keys.float(),
        ) * scale
        attention_lse[:, batch_idx] = torch.logsumexp(logits, dim=-1)
        expected_rows.append(torch.softmax(logits, dim=-1).sum(dim=0))
    actual = torch.full((batch, width), torch.nan, device="cuda")

    h2o_probability_from_lse(
        q,
        k,
        attention_lse,
        page_table,
        request_indices,
        context_lens,
        actual,
        softmax_scale=scale,
    )

    for batch_idx in range(batch):
        length = int(context_lens[batch_idx].item())
        assert torch.allclose(
            actual[batch_idx, :length],
            expected_rows[batch_idx],
            atol=2e-3,
            rtol=2e-3,
        )
        assert torch.all(actual[batch_idx, length:] == 0)
