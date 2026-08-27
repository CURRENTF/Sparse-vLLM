from __future__ import annotations

import pytest
import torch

from sparsevllm.kernels.external.flashinfer.topk import (
    flashinfer_top_k_page_table_transform_support,
)
from sparsevllm.kernels.triton.mla.copy_latent import (
    copy_latent_to_cache_with_quest_metadata,
)
from sparsevllm.kernels.triton.quest_decode_view import (
    finalize_quest_decode_view,
    score_quest_pages,
)
from sparsevllm.kernels.triton.store_kvcache import (
    store_kvcache_with_quest_metadata,
)
from sparsevllm.operators.quest_selection import (
    FlashInferQuestPageSelectionProvider,
    QuestPageSelectionOpSpec,
)


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for QuEST kernel tests",
)


def _stable_small_index_topk(
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> torch.Tensor:
    rows = []
    for row in range(int(scores.shape[0])):
        length = int(lengths[row].item())
        indices = torch.argsort(
            scores[row, :length].float().cpu(),
            descending=True,
            stable=True,
        )[:k]
        indices = indices.sort().values
        rows.append(page_table[row].cpu().index_select(0, indices))
    return torch.stack(rows).to(scores.device)


@CUDA_REQUIRED
def test_flashinfer_quest_page_selection_matches_stable_oracle_and_graph() -> None:
    supported, reason = flashinfer_top_k_page_table_transform_support()
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(20260827)
    batch_size, width, k = 3, 67, 7
    scores = torch.randn(
        batch_size,
        width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    scores.clamp_max_(2)
    scores[:, : k + 1] = 3
    page_table = torch.randperm(
        batch_size * width,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, width)
    lengths = torch.tensor([67, 53, 31], device="cuda", dtype=torch.int32)
    provider = FlashInferQuestPageSelectionProvider(
        op_spec=QuestPageSelectionOpSpec(
            score_dtype=torch.bfloat16,
            cuda_graph=True,
        )
    )

    actual = provider.select(scores, page_table, lengths, k)
    expected = _stable_small_index_topk(scores, page_table, lengths, k)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = provider.select(scores, page_table, lengths, k)
    scores.copy_(torch.flip(scores, dims=(1,)))
    graph.replay()
    expected = _stable_small_index_topk(scores, page_table, lengths, k)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("num_query_heads", "num_metadata_heads", "head_dim"),
    [(8, 2, 128), (1, 1, 576)],
)
def test_fused_quest_page_score_matches_tensor_oracle(
    num_query_heads: int,
    num_metadata_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(29 + head_dim)
    batch_size, physical_pages, logical_pages = 2, 11, 7
    query = torch.randn(
        batch_size,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    keys = torch.randn(
        physical_pages,
        4,
        num_metadata_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_min = keys.amin(dim=1)
    page_max = keys.amax(dim=1)
    row_pages = torch.stack(
        (
            torch.randperm(physical_pages, device="cuda")[:logical_pages],
            torch.randperm(physical_pages, device="cuda")[:logical_pages],
        )
    ).to(torch.int32)

    actual = score_quest_pages(query, page_max, page_min, row_pages)
    selected_max = page_max.index_select(
        0,
        row_pages.to(torch.long).reshape(-1),
    ).view(batch_size, logical_pages, num_metadata_heads, head_dim)
    selected_min = page_min.index_select(
        0,
        row_pages.to(torch.long).reshape(-1),
    ).view(batch_size, logical_pages, num_metadata_heads, head_dim)
    group_size = num_query_heads // num_metadata_heads
    grouped_query = query.view(
        batch_size,
        num_metadata_heads,
        group_size,
        head_dim,
    )
    query_positive = grouped_query.clamp_min(0).reshape(
        batch_size * num_metadata_heads,
        group_size,
        head_dim,
    )
    query_negative = grouped_query.clamp_max(0).reshape(
        batch_size * num_metadata_heads,
        group_size,
        head_dim,
    )
    max_transposed = selected_max.permute(0, 2, 3, 1).reshape(
        batch_size * num_metadata_heads,
        head_dim,
        logical_pages,
    )
    min_transposed = selected_min.permute(0, 2, 3, 1).reshape(
        batch_size * num_metadata_heads,
        head_dim,
        logical_pages,
    )
    expected = torch.bmm(query_positive, max_transposed)
    expected += torch.bmm(query_negative, min_transposed)
    expected = expected.view(
        batch_size,
        num_metadata_heads,
        group_size,
        logical_pages,
    ).amax(dim=2).amax(dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = score_quest_pages(
            query,
            page_max,
            page_min,
            row_pages,
        )
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@CUDA_REQUIRED
@pytest.mark.parametrize("use_dense_fallback", [False, True])
def test_quest_decode_view_finalizer_matches_reference(
    use_dense_fallback: bool,
) -> None:
    page_size, prev_budget = 4, 3
    selected = torch.tensor(
        [[7, 2, 5], [4, 1, 6]],
        dtype=torch.int32,
        device="cuda",
    )
    page_table = torch.tensor(
        [[7, 2, 5, 9, 8], [4, 1, 6, 3, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    num_pages = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([18, 10], dtype=torch.int32, device="cuda")
    sparse_width = (prev_budget + 1) * page_size
    output_width = 20 if use_dense_fallback else sparse_width
    dense = (
        torch.arange(40, dtype=torch.int32, device="cuda").view(2, 20)
        if use_dense_fallback
        else None
    )

    actual = finalize_quest_decode_view(
        selected,
        page_table,
        num_pages,
        context_lens,
        dense,
        page_size=page_size,
        token_budget=12,
        output_width=output_width,
    )
    expected = finalize_quest_decode_view(
        selected.cpu(),
        page_table.cpu(),
        num_pages.cpu(),
        context_lens.cpu(),
        None if dense is None else dense.cpu(),
        page_size=page_size,
        token_budget=12,
        output_width=output_width,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.cpu(),
            expected_tensor,
            rtol=0,
            atol=0,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = finalize_quest_decode_view(
            selected,
            page_table,
            num_pages,
            context_lens,
            dense,
            page_size=page_size,
            token_budget=12,
            output_width=output_width,
        )
    graph.replay()
    for captured_tensor, expected_tensor in zip(captured, expected):
        torch.testing.assert_close(
            captured_tensor.cpu(),
            expected_tensor,
            rtol=0,
            atol=0,
        )


@CUDA_REQUIRED
def test_explicit_kv_store_updates_quest_bounds_incrementally() -> None:
    page_size, num_pages, num_heads, head_dim = 4, 3, 2, 8
    k_cache = torch.full(
        (page_size * num_pages, num_heads, head_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    page_max = torch.full(
        (num_pages, num_heads, head_dim),
        99,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_min = torch.full_like(page_max, -99)
    keys = torch.randn(
        page_size,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    values = torch.randn_like(keys)

    for offset in range(page_size):
        slots = torch.tensor(
            [page_size + offset],
            dtype=torch.int32,
            device="cuda",
        )
        store_kvcache_with_quest_metadata(
            keys[offset : offset + 1],
            values[offset : offset + 1],
            k_cache,
            v_cache,
            slots,
            page_max,
            page_min,
            page_size=page_size,
        )
        torch.testing.assert_close(
            page_max[1],
            keys[: offset + 1].amax(dim=0),
        )
        torch.testing.assert_close(
            page_min[1],
            keys[: offset + 1].amin(dim=0),
        )
    torch.testing.assert_close(k_cache[page_size : 2 * page_size], keys)
    torch.testing.assert_close(v_cache[page_size : 2 * page_size], values)
    torch.testing.assert_close(page_max[0], torch.full_like(page_max[0], 99))


@CUDA_REQUIRED
def test_explicit_quest_metadata_store_is_cuda_graph_replay_safe() -> None:
    page_size = 4
    key = torch.randn(1, 1, 8, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    k_cache = torch.empty(8, 1, 8, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.empty_like(k_cache)
    slot = torch.tensor([0], dtype=torch.int32, device="cuda")
    page_max = torch.empty(2, 1, 8, dtype=torch.bfloat16, device="cuda")
    page_min = torch.empty_like(page_max)

    store_kvcache_with_quest_metadata(
        key,
        value,
        k_cache,
        v_cache,
        slot,
        page_max,
        page_min,
        page_size=page_size,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        store_kvcache_with_quest_metadata(
            key,
            value,
            k_cache,
            v_cache,
            slot,
            page_max,
            page_min,
            page_size=page_size,
        )
    first_key = key.clone()
    second_key = torch.randn_like(key)
    key.copy_(second_key)
    slot.fill_(1)
    graph.replay()
    torch.testing.assert_close(
        page_max[0],
        torch.maximum(first_key[0], second_key[0]),
    )
    torch.testing.assert_close(
        page_min[0],
        torch.minimum(first_key[0], second_key[0]),
    )


@CUDA_REQUIRED
def test_mla_store_updates_fused_quest_bounds_incrementally() -> None:
    page_size, num_pages = 2, 2
    latent_cache = torch.zeros(
        page_size * num_pages,
        1,
        512,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope_cache = torch.zeros(
        page_size * num_pages,
        1,
        64,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_max = torch.full(
        (num_pages, 1, 576),
        99,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_min = torch.full_like(page_max, -99)
    latent = torch.randn(
        page_size,
        1,
        512,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope = torch.randn(
        page_size,
        1,
        64,
        dtype=torch.bfloat16,
        device="cuda",
    )
    fused = torch.cat((latent, rope), dim=-1)

    for offset in range(page_size):
        copy_latent_to_cache_with_quest_metadata(
            latent[offset : offset + 1],
            rope[offset : offset + 1],
            torch.tensor([offset], dtype=torch.int32, device="cuda"),
            latent_cache,
            rope_cache,
            page_max,
            page_min,
            page_size=page_size,
        )
        torch.testing.assert_close(
            page_max[0],
            fused[: offset + 1].amax(dim=0),
        )
        torch.testing.assert_close(
            page_min[0],
            fused[: offset + 1].amin(dim=0),
        )
    torch.testing.assert_close(latent_cache[:page_size], latent)
    torch.testing.assert_close(rope_cache[:page_size], rope)
