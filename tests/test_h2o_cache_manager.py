from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import ANY, patch

import numpy as np
import pytest
import torch

from sparsevllm.engine.cache_manager.base import (
    CacheManager,
    LayerBatchStates,
    PrefillComputeView,
)
from sparsevllm.engine.cache_manager.h2o import H2OCacheManager
from sparsevllm.engine.cache_manager.snapkv import SnapKVCacheManager
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.method_registry import (
    PREFILL_POLICY_ALL_CHUNKED,
    get_default_prefill_schedule_policy,
)
from sparsevllm.utils.context import reset_context, set_context


@pytest.fixture(autouse=True)
def _reset_runtime_context():
    reset_context()
    yield
    reset_context()


def _layout(layers: int = 1):
    return SimpleNamespace(
        kv_idx_to_layer_idx=tuple(range(layers)),
        kv_layer_index=lambda layer: int(layer),
        is_full_attention=lambda layer: 0 <= int(layer) < layers,
    )


def _manager_with_layer_rows(
    lengths_by_layer: list[list[int]],
    *,
    decode_budget=4,
    prefill_budget=8,
    chunk_prefill_size=4,
):
    if not lengths_by_layer:
        raise ValueError("lengths_by_layer must not be empty")
    batch_size = len(lengths_by_layer[0])
    if any(len(lengths) != batch_size for lengths in lengths_by_layer):
        raise ValueError("all layers must contain the same number of rows")

    manager = object.__new__(H2OCacheManager)
    manager.device = torch.device("cpu")
    manager.num_layers = len(lengths_by_layer)
    manager.num_kv_layers = len(lengths_by_layer)
    manager.runtime_layout = _layout(manager.num_layers)
    manager.config = SimpleNamespace(
        vllm_sparse_method="h2o",
        h2o_decode_budget=decode_budget,
        h2o_prefill_budget=prefill_budget,
        h2o_recent_ratio=0.5,
        h2o_prefill_score_window=4,
        max_model_len=64,
        snapkv_window_size=4,
        snapkv_num_full_layers=0,
        pyramid_layer_ratios=None,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        chunk_prefill_size=chunk_prefill_size,
    )
    manager.max_model_len = 64
    manager.num_kv_heads = 2
    manager.head_dim = 2
    manager.hf_config = SimpleNamespace(torch_dtype=torch.float32)
    manager._h2o_scores = {}
    manager._h2o_recent_cursors = {}
    manager._h2o_counters = {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_evictions": 0,
        "dropped_tokens": 0,
    }
    manager._h2o_ring_counters = {
        "fast_rows": 0,
        "fallback_rows": 0,
    }
    manager._h2o_final_prefill_workspace = None
    manager._uniform_decode_metadata = False
    manager.seq_id_to_row = [
        {idx: idx for idx in range(batch_size)}
        for _ in lengths_by_layer
    ]
    manager.row_seq_lens = [
        np.asarray(lengths, dtype=np.int32) for lengths in lengths_by_layer
    ]
    manager.buffer_req_to_token_slots_tensor = torch.zeros(
        (manager.num_layers, batch_size, 64), dtype=torch.int32
    )
    manager.buffer_req_to_token_slots = [
        manager.buffer_req_to_token_slots_tensor[layer_idx]
        for layer_idx in range(manager.num_layers)
    ]
    for layer_idx, lengths in enumerate(lengths_by_layer):
        for row, length in enumerate(lengths):
            slot_start = row * 100
            manager.buffer_req_to_token_slots[layer_idx][row, :length] = torch.arange(
                slot_start, slot_start + length, dtype=torch.int32
            )
    manager.free_slots_stack_tensor = None
    manager.free_slots_stack = [
        torch.zeros((512,), dtype=torch.int32) for _ in lengths_by_layer
    ]
    manager._num_free_slots = [32 for _ in lengths_by_layer]
    num_slots = max(4096, batch_size * 100 + 64)
    manager.config.num_kvcache_slots = num_slots
    manager.kv_cache = torch.zeros(
        (
            2,
            manager.num_layers,
            num_slots,
            manager.num_kv_heads,
            manager.head_dim,
        ),
        dtype=manager.hf_config.torch_dtype,
    )
    return manager


def _manager_with_rows(
    lengths: list[int],
    *,
    decode_budget=4,
    prefill_budget=8,
    chunk_prefill_size=4,
):
    return _manager_with_layer_rows(
        [lengths],
        decode_budget=decode_budget,
        prefill_budget=prefill_budget,
        chunk_prefill_size=chunk_prefill_size,
    )


def _set_scores_from_slot_rows(manager: H2OCacheManager):
    for layer_idx in manager.kv_transformer_layer_indices():
        for seq_id, row_idx in manager.seq_id_to_row[layer_idx].items():
            row_len = int(manager.row_seq_lens[layer_idx][row_idx])
            manager._h2o_scores[(layer_idx, seq_id)] = (
                manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len]
                .float()
                .clone()
            )


def _set_layer_row_slots(
    manager: H2OCacheManager,
    layer_idx: int,
    rows: list[list[int]],
):
    for row_idx, slots in enumerate(rows):
        row_len = int(manager.row_seq_lens[layer_idx][row_idx])
        assert len(slots) == row_len
        manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len] = torch.tensor(
            slots,
            dtype=torch.int32,
        )


def _fill_kv_by_physical_slot(manager: H2OCacheManager):
    for layer_idx in manager.kv_transformer_layer_indices():
        k_cache, v_cache = manager.get_layer_kv_cache(layer_idx)
        slot_values = torch.arange(k_cache.shape[0], dtype=k_cache.dtype).view(-1, 1, 1)
        offsets = torch.arange(
            manager.num_kv_heads * manager.head_dim,
            dtype=k_cache.dtype,
        ).view(1, manager.num_kv_heads, manager.head_dim)
        k_cache.copy_(slot_values * 10 + offsets)
        v_cache.copy_(-slot_values * 10 - offsets - 1)


def _use_kv_layout(manager: H2OCacheManager, layout: str):
    if layout == "tensor":
        return
    if layout != "list":
        raise ValueError(f"unknown KV layout: {layout}")
    manager.kv_cache = [
        (
            manager.kv_cache[0, layer_idx].clone(),
            manager.kv_cache[1, layer_idx].clone(),
        )
        for layer_idx in manager.kv_transformer_layer_indices()
    ]


def _assert_scores_match_slot_rows(manager: H2OCacheManager):
    for layer_idx in manager.kv_transformer_layer_indices():
        for seq_id, row_idx in manager.seq_id_to_row[layer_idx].items():
            row_len = int(manager.row_seq_lens[layer_idx][row_idx])
            assert torch.equal(
                manager._h2o_scores[(layer_idx, seq_id)],
                manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len].float(),
            )


def _append_ring_token(
    manager: H2OCacheManager,
    layer_idx: int,
    seq_id: int,
    *,
    token_slot: int,
    token_score: float,
):
    row_idx = manager.seq_id_to_row[layer_idx][seq_id]
    budget = manager.h2o_decode_budget
    assert int(manager.row_seq_lens[layer_idx][row_idx]) == budget
    score = manager._h2o_scores[(layer_idx, seq_id)]
    assert int(score.numel()) == budget
    manager.buffer_req_to_token_slots[layer_idx][row_idx, budget] = int(token_slot)
    manager.row_seq_lens[layer_idx][row_idx] = budget + 1
    manager._h2o_scores[(layer_idx, seq_id)] = torch.cat(
        (score, torch.tensor([token_score], dtype=torch.float32))
    )


def _token_score_map(manager: H2OCacheManager, layer_idx: int, seq_id: int):
    row_idx = manager.seq_id_to_row[layer_idx][seq_id]
    row_len = int(manager.row_seq_lens[layer_idx][row_idx])
    slots = manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len].tolist()
    scores = manager._h2o_scores[(layer_idx, seq_id)].tolist()
    assert len(slots) == len(scores)
    assert len(slots) == len(set(slots))
    return {int(slot): float(score) for slot, score in zip(slots, scores)}


def _seq(seq_id: int, prompt_len: int, *, prefilled: int, chunk: int) -> Sequence:
    seq = Sequence(list(range(prompt_len)))
    seq.seq_id = int(seq_id)
    seq.num_prefilled_tokens = int(prefilled)
    seq.current_chunk_size = int(chunk)
    return seq


def test_h2o_registry_defaults_to_chunked_prefill():
    assert get_default_prefill_schedule_policy("h2o") == PREFILL_POLICY_ALL_CHUNKED


def test_h2o_cache_manager_factory_routes_first_class_method():
    expected = object()
    config = SimpleNamespace(
        vllm_sparse_method="h2o",
        hf_config=SimpleNamespace(model_type="qwen2"),
    )
    with patch(
        "sparsevllm.engine.cache_manager.h2o.H2OCacheManager",
        return_value=expected,
    ) as constructor:
        actual = CacheManager.create(config, SimpleNamespace())
    assert actual is expected
    constructor.assert_called_once_with(config, ANY)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"budget": 0, "recent_ratio": 0.5}, "budget must be positive"),
        ({"budget": 4, "recent_ratio": 0.0}, "recent_ratio must be"),
        ({"budget": 4, "recent_ratio": 1.0}, "recent_ratio must be"),
    ],
)
def test_h2o_selection_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        H2OCacheManager.select_h2o_indices(torch.arange(8, dtype=torch.float32), **kwargs)


def test_h2o_selection_keeps_heavy_and_recent_in_logical_order():
    scores = torch.tensor([1.0, 9.0, 2.0, 8.0, 3.0, 0.0, 0.0, 0.0])
    keep = H2OCacheManager.select_h2o_indices(scores, budget=4, recent_ratio=0.5)
    assert keep.tolist() == [1, 3, 6, 7]


def test_h2o_drop_one_batch_matches_general_selection():
    scores = torch.tensor(
        [
            [3.0, 1.0, 2.0, 8.0, 9.0],
            [1.0, 4.0, 2.0, 8.0, 9.0],
            [5.0, 3.0, 4.0, 8.0, 9.0],
            [7.0, 9.0, 8.0, 5.0, 6.0],
        ]
    )

    drop_one = H2OCacheManager.select_h2o_drop_one_indices_batch(
        scores, budget=4, recent_ratio=0.5
    )
    general = H2OCacheManager.select_h2o_indices_batch(
        scores, budget=4, recent_ratio=0.5
    )

    assert torch.equal(drop_one, general)


def test_h2o_general_batch_selection_matches_scalar_multi_drop():
    scores = torch.tensor(
        [
            [9.0, 1.0, 7.0, 3.0, 5.0, 11.0, 13.0],
            [2.0, 10.0, 4.0, 8.0, 6.0, 12.0, 14.0],
            [15.0, 19.0, 17.0, 21.0, 16.0, 23.0, 25.0],
            [30.0, 26.0, 28.0, 24.0, 22.0, 27.0, 29.0],
        ]
    )

    batched = H2OCacheManager.select_h2o_indices_batch(
        scores, budget=4, recent_ratio=0.5
    )
    scalar = torch.stack(
        [
            H2OCacheManager.select_h2o_indices(
                scores[batch_idx],
                budget=4,
                recent_ratio=0.5,
            )
            for batch_idx in range(scores.shape[0])
        ]
    )

    assert torch.equal(batched, scalar)


def test_h2o_score_vector_expands_adds_and_gathers_with_physical_row():
    manager = _manager_with_rows([6])
    seq = _seq(0, 20, prefilled=0, chunk=6)
    manager._h2o_scores[(0, 0)] = torch.arange(6, dtype=torch.float32)

    manager.free_part_slots(0, seq, torch.tensor([1, 3, 4, 5]))

    assert manager.row_seq_lens[0].tolist() == [4]
    assert manager._h2o_scores[(0, 0)].tolist() == [1.0, 3.0, 4.0, 5.0]
    accumulated = manager._accumulate_score(
        manager._h2o_scores[(0, 0)],
        torch.tensor([0.25, 0.25, 0.25, 0.25, 0.5, 1.0]),
        new_len=6,
        weight=2.0,
    )
    assert accumulated.tolist() == [1.5, 3.5, 4.5, 5.5, 1.0, 2.0]


def test_h2o_prefill_score_ranges_use_compressed_physical_coordinates():
    manager = _manager_with_rows([11])
    seq = _seq(0, 100, prefilled=64, chunk=3)

    ranges = manager.prefill_score_ranges(0, [seq])

    assert ranges[0][2:] == (8, 8, 11)


def test_h2o_prefill_score_collection_accumulates_in_physical_coordinates():
    manager = _manager_with_rows([6])
    seq = _seq(0, 20, prefilled=8, chunk=2)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = PrefillComputeView(
        k_cache=torch.empty((16, 1, 1)),
        v_cache=torch.empty((16, 1, 1)),
        active_slots=manager.buffer_req_to_token_slots[0],
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([6], dtype=torch.int32),
        max_context_len=6,
    )
    set_context(is_prefill=True, cache_manager=manager, seqs=[seq])

    def fake_prefill_score_fwd(*args, **kwargs):
        attn_score = args[2]
        prompt_cache_lens = args[6]
        score_starts = args[9]
        score_ends = args[10]
        assert prompt_cache_lens.tolist() == [4]
        assert score_starts.tolist() == [4]
        assert score_ends.tolist() == [6]
        assert kwargs == {"candidate_start": 0, "num_recent_tokens": 0}
        attn_score[0, :6] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    with patch(
        "sparsevllm.engine.cache_manager.h2o.prefill_score_fwd",
        side_effect=fake_prefill_score_fwd,
    ):
        manager.collect_prefill_attention_score(
            0,
            torch.empty((2, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([2], dtype=torch.int32),
        )

    assert manager._h2o_scores[(0, 0)].tolist() == pytest.approx(
        [1.2, 2.4, 3.6, 4.8, 1.0, 1.2]
    )


def test_h2o_prefill_score_collection_rejects_misaligned_physical_view():
    manager = _manager_with_rows([6])
    seq = _seq(0, 20, prefilled=8, chunk=2)
    manager._h2o_scores[(0, 0)] = torch.ones(4)
    view = PrefillComputeView(
        k_cache=torch.empty((16, 1, 1)),
        v_cache=torch.empty((16, 1, 1)),
        active_slots=manager.buffer_req_to_token_slots[0],
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([7], dtype=torch.int32),
        max_context_len=7,
    )
    set_context(is_prefill=True, cache_manager=manager, seqs=[seq])

    with pytest.raises(RuntimeError, match="compressed physical coordinates"):
        manager.collect_prefill_attention_score(
            0,
            torch.empty((2, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([2], dtype=torch.int32),
        )


def test_h2o_missing_score_for_existing_physical_prefix_fails_fast():
    manager = _manager_with_rows([8])
    seq = _seq(0, 100, prefilled=64, chunk=3)
    with pytest.raises(RuntimeError, match="score vector is missing"):
        manager._require_score_length(0, seq, 8)


def test_h2o_intermediate_and_final_prefill_use_distinct_budgets_and_counters():
    manager = _manager_with_rows([10], decode_budget=4, prefill_budget=8)
    seq = _seq(0, 20, prefilled=0, chunk=10)
    manager._h2o_scores[(0, 0)] = torch.arange(10, dtype=torch.float32)

    manager.evict_after_prefill([seq])
    assert manager.row_seq_lens[0][0] == 8
    assert manager._h2o_counters["intermediate_prefill_evictions"] == 1

    manager.buffer_req_to_token_slots[0][0, 8:10] = torch.tensor([20, 21])
    manager.row_seq_lens[0][0] = 10
    manager._h2o_scores[(0, 0)] = manager._expand_score(
        manager._h2o_scores[(0, 0)], 10, device=manager.device
    )
    seq.num_prefilled_tokens = 10
    seq.current_chunk_size = 10
    manager.evict_after_prefill([seq])

    assert manager.row_seq_lens[0][0] == 4
    assert manager._h2o_counters["final_prefill_evictions"] == 1
    assert manager._h2o_counters["dropped_tokens"] == 8
    assert manager._h2o_recent_cursors == {(0, 0): 2}


def test_h2o_decode_score_update_supports_batch_with_different_kv_lengths():
    manager = _manager_with_rows([4, 3])
    seq0 = _seq(0, 10, prefilled=10, chunk=1)
    seq1 = _seq(1, 10, prefilled=10, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0])
    manager._h2o_scores[(0, 1)] = torch.tensor([4.0, 5.0])
    normalized = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, -100.0]], dtype=torch.float32
    )

    manager.update_decode_attention_scores(0, [seq0, seq1], normalized)

    assert manager._h2o_scores[(0, 0)].tolist() == pytest.approx([1.1, 2.2, 3.3, 0.4])
    assert manager._h2o_scores[(0, 1)].tolist() == pytest.approx([4.5, 5.6, 0.7])


def test_h2o_all_layer_score_does_not_alias_graph_scratch_below_budget():
    manager = _manager_with_layer_rows([[3], [3]], decode_budget=4)
    seq = _seq(0, 10, prefilled=10, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0])
    manager._h2o_scores[(1, 0)] = torch.tensor([3.0, 4.0])
    graph_scratch = torch.tensor(
        [
            [[0.1, 0.2, 0.7, -1e20]],
            [[0.3, 0.4, 0.3, -1e20]],
        ],
        dtype=torch.float32,
    )

    used_fast_path = manager.update_decode_attention_scores_all_layers(
        [0, 1],
        [seq],
        graph_scratch,
    )
    expected = {
        (0, 0): torch.tensor([1.1, 2.2, 0.7]),
        (1, 0): torch.tensor([3.3, 4.4, 0.3]),
    }
    assert used_fast_path
    graph_scratch.fill_(-1e20)

    for key, score in expected.items():
        assert torch.allclose(manager._h2o_scores[key], score)
        assert manager._h2o_scores[key].untyped_storage().data_ptr() != (
            graph_scratch.untyped_storage().data_ptr()
        )


def test_h2o_decode_score_update_batches_uniform_lengths():
    manager = _manager_with_rows([4, 4])
    seq0 = _seq(0, 10, prefilled=10, chunk=1)
    seq1 = _seq(1, 10, prefilled=10, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0])
    manager._h2o_scores[(0, 1)] = torch.tensor([4.0, 5.0, 6.0])
    normalized = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32
    )

    with patch.object(
        H2OCacheManager,
        "_accumulate_score",
        side_effect=AssertionError("uniform score update used scalar fallback"),
    ):
        manager.update_decode_attention_scores(0, [seq0, seq1], normalized)

    assert manager._h2o_scores[(0, 0)].tolist() == pytest.approx([1.1, 2.2, 3.3, 0.4])
    assert manager._h2o_scores[(0, 1)].tolist() == pytest.approx([4.5, 5.6, 6.7, 0.8])


def test_h2o_all_layer_score_update_batches_uniform_rows_once():
    manager = _manager_with_layer_rows([[4, 4], [4, 4]])
    seqs = [
        _seq(0, 10, prefilled=10, chunk=1),
        _seq(1, 10, prefilled=10, chunk=1),
    ]
    previous = {}
    for layer_idx in range(2):
        for seq_id in range(2):
            score = torch.tensor(
                [
                    10.0 * layer_idx + 3.0 * seq_id + 1.0,
                    10.0 * layer_idx + 3.0 * seq_id + 2.0,
                    10.0 * layer_idx + 3.0 * seq_id + 3.0,
                ]
            )
            manager._h2o_scores[(layer_idx, seq_id)] = score
            previous[(layer_idx, seq_id)] = score.clone()
    reduced = torch.arange(2 * 2 * 5, dtype=torch.float32).view(2, 2, 5) / 10.0
    expected = reduced[:, :, :4].clone()
    expected[:, :, :3] += torch.stack(
        [previous[(layer_idx, seq_id)] for layer_idx in range(2) for seq_id in range(2)]
    ).view(2, 2, 3)

    with patch.object(
        manager,
        "update_decode_attention_scores",
        side_effect=AssertionError("uniform all-layer update used per-layer fallback"),
    ):
        used_fast_path = manager.update_decode_attention_scores_all_layers(
            [0, 1], seqs, reduced
        )

    assert used_fast_path
    for layer_idx in range(2):
        for seq_id in range(2):
            assert torch.equal(
                manager._h2o_scores[(layer_idx, seq_id)],
                expected[layer_idx, seq_id],
            )
            assert torch.equal(
                previous[(layer_idx, seq_id)],
                torch.tensor(
                    [
                        10.0 * layer_idx + 3.0 * seq_id + 1.0,
                        10.0 * layer_idx + 3.0 * seq_id + 2.0,
                        10.0 * layer_idx + 3.0 * seq_id + 3.0,
                    ]
                ),
            )
    storage_ptrs = {
        score.untyped_storage().data_ptr() for score in manager._h2o_scores.values()
    }
    assert len(storage_ptrs) == 1


def test_h2o_all_layer_score_update_explicitly_falls_back_for_nonuniform_rows():
    manager = _manager_with_layer_rows([[4, 3], [4, 3]])
    seqs = [
        _seq(0, 10, prefilled=10, chunk=1),
        _seq(1, 10, prefilled=10, chunk=1),
    ]
    for layer_idx in range(2):
        manager._h2o_scores[(layer_idx, 0)] = torch.tensor([1.0, 2.0, 3.0])
        manager._h2o_scores[(layer_idx, 1)] = torch.tensor([4.0, 5.0])
    reduced = torch.ones((2, 2, 4), dtype=torch.float32)
    original = manager.update_decode_attention_scores

    with patch.object(
        manager,
        "update_decode_attention_scores",
        wraps=original,
    ) as per_layer:
        used_fast_path = manager.update_decode_attention_scores_all_layers(
            [0, 1], seqs, reduced
        )

    assert not used_fast_path
    assert per_layer.call_count == 2
    for layer_idx in range(2):
        assert manager._h2o_scores[(layer_idx, 0)].tolist() == [2.0, 3.0, 4.0, 1.0]
        assert manager._h2o_scores[(layer_idx, 1)].tolist() == [5.0, 6.0, 1.0]


def test_h2o_decode_ring_fast_path_updates_all_layers_without_ordered_compaction():
    manager = _manager_with_layer_rows(
        [[5, 5] for _ in range(28)], decode_budget=4, prefill_budget=8
    )
    seqs = [
        _seq(0, 5, prefilled=5, chunk=1),
        _seq(1, 5, prefilled=5, chunk=1),
    ]
    _set_scores_from_slot_rows(manager)

    with patch.object(
        SnapKVCacheManager,
        "free_part_slots_batch",
        side_effect=AssertionError("decode ring used ordered batch compaction"),
    ):
        manager.evict_after_decode(seqs)

    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [
        [4, 4] for _ in range(28)
    ]
    _assert_scores_match_slot_rows(manager)
    assert set(manager._h2o_recent_cursors.values()) == {3}
    assert manager._h2o_ring_counters == {"fast_rows": 56, "fallback_rows": 0}
    assert manager._num_free_slots == [34 for _ in range(28)]
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_evictions": 56,
        "dropped_tokens": 56,
    }


def test_h2o_intermediate_prefill_reuses_uniform_batch_fast_path():
    manager = _manager_with_layer_rows(
        [[6, 6], [7, 7]], decode_budget=3, prefill_budget=4
    )
    seqs = [
        _seq(0, 20, prefilled=0, chunk=6),
        _seq(1, 20, prefilled=0, chunk=6),
    ]
    _set_scores_from_slot_rows(manager)
    calls = []
    original = SnapKVCacheManager.free_part_slots_batch

    def tracked_batch_free(self, layer_idx, batch_seqs, keep_indices, **kwargs):
        calls.append((int(layer_idx), keep_indices.clone()))
        return original(self, layer_idx, batch_seqs, keep_indices, **kwargs)

    with patch.object(
        SnapKVCacheManager,
        "free_part_slots_batch",
        new=tracked_batch_free,
    ):
        assert manager._try_batched_evict(seqs, is_prefill=True)

    assert [call[0] for call in calls] == [0, 1]
    assert all(call[1].shape == (2, 4) for call in calls)
    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [[4, 4], [4, 4]]
    _assert_scores_match_slot_rows(manager)
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 4,
        "final_prefill_evictions": 0,
        "decode_evictions": 0,
        "dropped_tokens": 10,
    }


def test_h2o_mixed_prefill_budgets_fall_back_with_exact_counters():
    manager = _manager_with_rows([6, 6], decode_budget=3, prefill_budget=4)
    seqs = [
        _seq(0, 6, prefilled=0, chunk=6),
        _seq(1, 20, prefilled=0, chunk=6),
    ]
    _set_scores_from_slot_rows(manager)
    _fill_kv_by_physical_slot(manager)
    k_cache, v_cache = manager.get_layer_kv_cache(0)
    selected_slots = torch.tensor([3, 4, 5], dtype=torch.long)
    expected_k = k_cache.index_select(0, selected_slots).clone()
    expected_v = v_cache.index_select(0, selected_slots).clone()

    assert not manager._try_batched_evict(seqs, is_prefill=True)
    manager.evict_after_prefill(seqs)

    assert manager.row_seq_lens[0].tolist() == [3, 4]
    assert manager._h2o_scores[(0, 0)].tolist() == [3.0, 4.0, 5.0]
    assert manager._h2o_scores[(0, 1)].tolist() == [102.0, 103.0, 104.0, 105.0]
    final_slots = manager.buffer_req_to_token_slots[0][0, :3].long()
    assert final_slots.tolist() == [0, 1, 2]
    assert torch.equal(k_cache.index_select(0, final_slots), expected_k)
    assert torch.equal(v_cache.index_select(0, final_slots), expected_v)
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 1,
        "final_prefill_evictions": 1,
        "decode_evictions": 0,
        "dropped_tokens": 5,
    }


@pytest.mark.parametrize("kv_layout", ["tensor", "list"])
def test_h2o_final_prefill_dense_batch_preserves_logical_kv_alignment(
    kv_layout: str,
):
    manager = _manager_with_layer_rows(
        [[6, 6], [6, 6]], decode_budget=4, prefill_budget=8
    )
    rows_by_layer = [
        [[9, 2, 7, 1, 6, 4], [29, 22, 27, 21, 26, 24]],
        [[109, 102, 107, 101, 106, 104], [129, 122, 127, 121, 126, 124]],
    ]
    for layer_idx, rows in enumerate(rows_by_layer):
        _set_layer_row_slots(manager, layer_idx, rows)
        for seq_id in range(2):
            manager._h2o_scores[(layer_idx, seq_id)] = torch.tensor(
                [1.0, 9.0, 2.0, 8.0, 0.0, 0.0]
            )
    _use_kv_layout(manager, kv_layout)
    _fill_kv_by_physical_slot(manager)
    seqs = [
        _seq(0, 6, prefilled=0, chunk=6),
        _seq(1, 6, prefilled=0, chunk=6),
    ]
    keep = torch.tensor([1, 3, 4, 5], dtype=torch.long)
    expected = {}
    for layer_idx in range(2):
        k_cache, v_cache = manager.get_layer_kv_cache(layer_idx)
        for seq_id, row_slots in enumerate(rows_by_layer[layer_idx]):
            selected_slots = torch.tensor(row_slots, dtype=torch.long)[keep]
            expected[(layer_idx, seq_id)] = (
                k_cache.index_select(0, selected_slots).clone(),
                v_cache.index_select(0, selected_slots).clone(),
            )

    workspace_ptrs = []
    original_workspace = manager._get_final_prefill_workspace

    def tracked_workspace(**kwargs):
        workspace = original_workspace(**kwargs)
        workspace_ptrs.append(workspace.untyped_storage().data_ptr())
        return workspace

    with patch.object(
        manager,
        "_get_final_prefill_workspace",
        side_effect=tracked_workspace,
    ):
        manager.evict_after_prefill(seqs)

    for layer_idx, rows in enumerate(rows_by_layer):
        k_cache, v_cache = manager.get_layer_kv_cache(layer_idx)
        released = []
        active = []
        for seq_id, row_slots in enumerate(rows):
            sorted_slots = sorted(row_slots)
            destination = sorted_slots[:4]
            released.extend(sorted_slots[4:])
            active.extend(destination)
            actual_slots = manager.buffer_req_to_token_slots[layer_idx][
                seq_id, :4
            ].long()
            assert actual_slots.tolist() == destination
            expected_k, expected_v = expected[(layer_idx, seq_id)]
            assert torch.equal(k_cache.index_select(0, actual_slots), expected_k)
            assert torch.equal(v_cache.index_select(0, actual_slots), expected_v)
            assert manager._h2o_scores[(layer_idx, seq_id)].tolist() == [
                9.0,
                8.0,
                0.0,
                0.0,
            ]
        assert len(active) == len(set(active))
        assert manager.free_slots_stack[layer_idx][32:36].tolist() == released
        assert manager._num_free_slots[layer_idx] == 36

    assert workspace_ptrs[0] == workspace_ptrs[1]
    workspace = manager._h2o_final_prefill_workspace
    assert workspace is not None
    assert list(workspace.shape) == [2, 2, 4, 2, 2]
    accounting = manager.memory_accounting()
    workspace_entries = [
        item
        for item in accounting["tensors"]
        if item["path"] == "_h2o_final_prefill_workspace"
    ]
    assert len(workspace_entries) == 1
    assert workspace_entries[0]["nbytes"] == workspace.untyped_storage().nbytes()


def test_h2o_intermediate_prefill_does_not_move_kv_payloads():
    manager = _manager_with_rows([6], decode_budget=3, prefill_budget=4)
    _set_layer_row_slots(manager, 0, [[9, 2, 7, 1, 6, 4]])
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 9.0, 2.0, 8.0, 0.0, 0.0])
    _fill_kv_by_physical_slot(manager)
    k_cache, v_cache = manager.get_layer_kv_cache(0)
    old_k = k_cache.clone()
    old_v = v_cache.clone()
    seq = _seq(0, 20, prefilled=0, chunk=6)

    manager.evict_after_prefill([seq])

    assert manager.buffer_req_to_token_slots[0][0, :4].tolist() == [2, 1, 6, 4]
    assert torch.equal(k_cache, old_k)
    assert torch.equal(v_cache, old_v)
    assert manager._h2o_final_prefill_workspace is None


def test_h2o_final_prefill_capacity_preflight_prevents_partial_layer_updates():
    manager = _manager_with_layer_rows([[6], [6]], decode_budget=4, prefill_budget=8)
    _set_scores_from_slot_rows(manager)
    _fill_kv_by_physical_slot(manager)
    manager._num_free_slots[1] = 511
    layer0_slots = manager.buffer_req_to_token_slots[0].clone()
    layer0_k, layer0_v = manager.get_layer_kv_cache(0)
    expected_k = layer0_k.clone()
    expected_v = layer0_v.clone()
    seq = _seq(0, 6, prefilled=0, chunk=6)

    with pytest.raises(RuntimeError, match=r"overflow.*layer=1"):
        manager.evict_after_prefill([seq])

    assert torch.equal(manager.buffer_req_to_token_slots[0], layer0_slots)
    assert torch.equal(layer0_k, expected_k)
    assert torch.equal(layer0_v, expected_v)
    assert manager.row_seq_lens[0].tolist() == [6]
    assert manager._num_free_slots[0] == 32
    assert manager._h2o_final_prefill_workspace is None


def test_h2o_decode_ring_rejects_more_than_one_appended_token():
    manager = _manager_with_rows([6], decode_budget=4, prefill_budget=8)
    seqs = [_seq(0, 6, prefilled=6, chunk=1)]
    _set_scores_from_slot_rows(manager)

    with pytest.raises(RuntimeError, match="at most one appended token"):
        manager.evict_after_decode(seqs)


def test_h2o_decode_ring_rejects_cursor_without_appended_token():
    manager = _manager_with_rows([4], decode_budget=4, prefill_budget=8)
    seq = _seq(0, 4, prefilled=4, chunk=1)
    _set_scores_from_slot_rows(manager)
    manager._h2o_recent_cursors[(0, 0)] = 2

    with pytest.raises(RuntimeError, match="cursor exists without one appended"):
        manager.evict_after_decode([seq])


def test_h2o_decode_ring_fallback_preserves_score_slot_alignment():
    manager = _manager_with_layer_rows(
        [[5, 5], [5, 5]], decode_budget=4, prefill_budget=8
    )
    seqs = [
        _seq(0, 5, prefilled=5, chunk=1),
        _seq(1, 5, prefilled=5, chunk=1),
    ]
    _set_scores_from_slot_rows(manager)
    manager.buffer_req_to_token_slots_tensor = None

    manager.evict_after_decode(seqs)

    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [[4, 4], [4, 4]]
    _assert_scores_match_slot_rows(manager)
    assert manager._h2o_ring_counters == {"fast_rows": 0, "fallback_rows": 4}
    assert manager._h2o_counters["decode_evictions"] == 4
    assert manager._h2o_counters["dropped_tokens"] == 4


def test_h2o_decode_evicts_every_over_budget_step_and_counts_drops():
    manager = _manager_with_rows([5], decode_budget=4, prefill_budget=8)
    seq = _seq(0, 5, prefilled=5, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.arange(5, dtype=torch.float32)

    manager.evict_after_decode([seq])

    assert manager.row_seq_lens[0][0] == 4
    assert manager._h2o_counters["decode_evictions"] == 1
    assert manager._h2o_counters["dropped_tokens"] == 1


def test_h2o_decode_ring_matches_ordered_token_sets_across_steps():
    manager = _manager_with_rows([4], decode_budget=4, prefill_budget=8)
    seq = _seq(0, 4, prefilled=0, chunk=4)
    manager._h2o_scores[(0, 0)] = torch.tensor([10.0, 1.0, 5.0, 0.5])
    manager.evict_after_prefill([seq])
    assert manager._h2o_recent_cursors[(0, 0)] == 2

    ordered_tokens = [0, 1, 2, 3]
    score_by_token = {0: 10.0, 1: 1.0, 2: 5.0, 3: 0.5}
    for token_slot, token_score in ((4, 9.0), (5, 8.0), (6, 7.0)):
        score_by_token[token_slot] = token_score
        ordered_with_new = ordered_tokens + [token_slot]
        heavy_candidates = ordered_with_new[:3]
        dropped_token = min(heavy_candidates, key=score_by_token.__getitem__)
        ordered_tokens = [
            token for token in ordered_with_new if token != dropped_token
        ]

        _append_ring_token(
            manager,
            0,
            0,
            token_slot=token_slot,
            token_score=token_score,
        )
        manager.evict_after_decode([seq])

        actual = _token_score_map(manager, 0, 0)
        assert set(actual) == set(ordered_tokens)
        assert actual == pytest.approx(
            {token: score_by_token[token] for token in ordered_tokens}
        )

    assert manager._num_free_slots[0] == 35
    assert manager._h2o_ring_counters == {"fast_rows": 3, "fallback_rows": 0}


def test_h2o_decode_ring_ties_are_deterministic_across_fast_and_fallback():
    retained_sets = []
    for use_fast_path in (True, False):
        manager = _manager_with_rows([5], decode_budget=4, prefill_budget=8)
        seq = _seq(0, 5, prefilled=5, chunk=1)
        manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 1.0, 1.0, 8.0, 9.0])
        if not use_fast_path:
            manager.buffer_req_to_token_slots_tensor = None

        manager.evict_after_decode([seq])
        retained = set(_token_score_map(manager, 0, 0))
        retained_sets.append(retained)

        assert len(retained) == 4
        assert {3, 4}.issubset(retained)
        assert len(retained.intersection({0, 1, 2})) == 2

    assert retained_sets[0] == retained_sets[1]


def test_h2o_decode_ring_handles_mixed_short_and_mature_batch():
    manager = _manager_with_layer_rows(
        [[5, 3], [5, 3]], decode_budget=4, prefill_budget=8
    )
    seqs = [
        _seq(0, 5, prefilled=5, chunk=1),
        _seq(1, 3, prefilled=3, chunk=1),
    ]
    _set_scores_from_slot_rows(manager)

    manager.evict_after_decode(seqs)

    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [[4, 3], [4, 3]]
    _assert_scores_match_slot_rows(manager)
    assert set(manager._h2o_recent_cursors) == {(0, 0), (1, 0)}
    assert manager._h2o_ring_counters == {"fast_rows": 2, "fallback_rows": 0}
    assert manager._num_free_slots == [33, 33]


def test_h2o_decode_ring_survives_sequence_reorder_and_temporary_absence():
    manager = _manager_with_rows([4, 4], decode_budget=4, prefill_budget=8)
    seq0 = _seq(0, 4, prefilled=0, chunk=4)
    seq1 = _seq(1, 4, prefilled=0, chunk=4)
    _set_scores_from_slot_rows(manager)
    manager.evict_after_prefill([seq0, seq1])

    _append_ring_token(manager, 0, 1, token_slot=1_000, token_score=1_000.0)
    manager.evict_after_decode([seq1])
    assert manager._h2o_recent_cursors[(0, 0)] == 2
    assert manager._h2o_recent_cursors[(0, 1)] == 3

    _append_ring_token(manager, 0, 0, token_slot=2_000, token_score=2_000.0)
    manager.evict_after_decode([seq0])
    assert manager._h2o_recent_cursors[(0, 0)] == 3

    _append_ring_token(manager, 0, 1, token_slot=3_000, token_score=3_000.0)
    _append_ring_token(manager, 0, 0, token_slot=4_000, token_score=4_000.0)
    manager.evict_after_decode([seq1, seq0])

    assert manager._h2o_recent_cursors == {(0, 0): 2, (0, 1): 2}
    _assert_scores_match_slot_rows(manager)
    active_slots = []
    for row_idx, row_len in enumerate(manager.row_seq_lens[0]):
        active_slots.extend(
            manager.buffer_req_to_token_slots[0][row_idx, : int(row_len)].tolist()
        )
    assert len(active_slots) == len(set(active_slots))
    assert manager._num_free_slots[0] == 36


def test_h2o_prefill_cursor_lifecycle_initializes_and_clears_by_phase():
    manager = _manager_with_rows([4], decode_budget=4, prefill_budget=8)
    seq = _seq(0, 4, prefilled=0, chunk=4)
    _set_scores_from_slot_rows(manager)

    manager.evict_after_prefill([seq])
    assert manager._h2o_recent_cursors == {(0, 0): 2}

    seq.token_ids.extend(range(4, 10))
    seq.num_prompt_tokens = 10
    seq.num_tokens = 10
    seq.num_prefilled_tokens = 4
    seq.current_chunk_size = 1
    manager.evict_after_prefill([seq])
    assert manager._h2o_recent_cursors == {}


def test_h2o_controller_consumes_snapkv_style_normalized_decode_scores():
    class RecordingManager:
        device = torch.device("cpu")

        def __init__(self):
            self.normalized = None
            self.layer_indices = None
            self.evicted = False

        def update_decode_attention_scores_all_layers(
            self,
            layer_indices,
            seqs,
            normalized,
        ):
            self.layer_indices = list(layer_indices)
            self.normalized = normalized.clone()

        def evict_after_decode(self, seqs):
            self.evicted = True

    manager = RecordingManager()
    config = SimpleNamespace(
        vllm_sparse_method="h2o",
        obs_layer_ids=[],
        full_attn_layers=[],
        runtime_layout=_layout(),
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            torch_dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
    )
    controller = SparseController(config, manager)
    reduced_scores = controller._get_h2o_decode_score_buffer(1, 1, 3)
    reduced_scores[0].copy_(torch.tensor([[0.2, 0.3, 0.5]]))
    controller.layer_batch_sparse_states[0].attn_score = reduced_scores[0]
    controller.layer_batch_sparse_states[0].context_lens = torch.tensor([3])

    controller._h2o_decode_eviction([_seq(0, 3, prefilled=3, chunk=1)])

    assert manager.layer_indices == [0]
    assert manager.normalized.shape == (1, 1, 3)
    assert torch.allclose(manager.normalized, reduced_scores)
    assert manager.evicted


def test_h2o_prepare_uses_one_contiguous_snapkv_style_buffer_for_all_kv_layers():
    manager = SimpleNamespace(device=torch.device("cpu"))
    config = SimpleNamespace(
        vllm_sparse_method="h2o",
        obs_layer_ids=[],
        full_attn_layers=[],
        runtime_layout=_layout(2),
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            torch_dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
        decode_cuda_graph=False,
    )
    controller = SparseController(config, manager)
    for layer_idx in range(2):
        state = controller.layer_batch_sparse_states[layer_idx]
        state.context_lens = torch.tensor([4, 3], dtype=torch.int32)
        state.max_context_len = 4
    seqs = [
        _seq(0, 4, prefilled=4, chunk=1),
        _seq(1, 3, prefilled=3, chunk=1),
    ]
    original = controller._get_h2o_decode_score_buffer

    with patch.object(
        controller,
        "_get_h2o_decode_score_buffer",
        wraps=original,
    ) as allocate:
        controller._prepare_h2o_decode_attn_score_buffer(seqs)

    allocate.assert_called_once()
    assert len(controller._h2o_decode_attn_score_buffers) == 1
    backing = next(iter(controller._h2o_decode_attn_score_buffers.values()))
    assert backing.shape == (2, 2, 4)
    assert torch.all(backing == -1e20)
    for layer_idx in range(2):
        assert (
            controller.layer_batch_sparse_states[layer_idx].attn_score.data_ptr()
            == backing[layer_idx].data_ptr()
        )
    resolved_layers, resolved = controller._resolve_h2o_decode_attn_score_buffer(
        {
            layer_idx: controller.layer_batch_sparse_states[layer_idx].attn_score
            for layer_idx in range(2)
        }
    )
    assert resolved_layers == [0, 1]
    assert resolved.data_ptr() == backing.data_ptr()


def test_h2o_layer_end_normalizes_the_fused_2d_score_in_place():
    manager = SimpleNamespace(device=torch.device("cpu"))
    config = SimpleNamespace(
        vllm_sparse_method="h2o",
        obs_layer_ids=[],
        full_attn_layers=[],
        runtime_layout=_layout(),
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            torch_dtype=torch.float16,
        ),
        tensor_parallel_size=1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float16",
        decode_cuda_graph=False,
    )
    controller = SparseController(config, manager)
    score = controller._get_h2o_decode_score_buffer(1, 1, 3)[0]
    score.copy_(torch.tensor([[0.0, 1.0, 2.0]]))
    controller.layer_batch_sparse_states[0].attn_score = score

    original_ptr = score.data_ptr()
    controller.on_layer_attention_end(0)

    assert score.data_ptr() == original_ptr
    assert score.dtype == torch.float32
    assert torch.allclose(score, torch.softmax(torch.tensor([[0.0, 1.0, 2.0]]), dim=-1))


def test_h2o_static_decode_reuses_uniform_rows_without_generic_metadata_rebuild():
    manager = _manager_with_layer_rows([[4], [4]])
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 5
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 5, prefilled=4, chunk=1)
    input_ids = torch.empty(1, dtype=torch.int64)
    positions = torch.empty(1, dtype=torch.int64)
    slot_mapping = torch.empty(1, dtype=torch.int32)
    context_lens = torch.empty(1, dtype=torch.int32)
    req_indices = torch.empty(1, dtype=torch.int32)

    with patch.object(
        SnapKVCacheManager,
        "_allocate_decode_batch_all_layers",
        side_effect=AssertionError("generic metadata rebuild must not run"),
    ):
        manager.prepare_decode_static(
            [seq],
            input_ids,
            positions,
            slot_mapping,
            context_lens,
            req_indices,
        )

    assert manager._num_free_slots == [31, 31]
    assert [int(lengths[0]) for lengths in manager.row_seq_lens] == [5, 5]
    assert manager.buffer_req_to_token_slots[0][0, 4].item() == 131
    assert manager.buffer_req_to_token_slots[1][0, 4].item() == 231
    assert slot_mapping.tolist() == [131]
    assert context_lens.tolist() == [5]
    assert req_indices.tolist() == [0]
    assert manager.layer_batch_states[0].slot_mapping.data_ptr() != slot_mapping.data_ptr()
    assert manager.layer_batch_states[0].max_context_len == 5
    assert manager.layer_batch_states[1].max_context_len == 5


def test_h2o_static_decode_rejects_divergent_layer_row_lengths_before_allocation():
    manager = _manager_with_layer_rows([[4], [3]])
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 5
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 5, prefilled=4, chunk=1)
    buffers = (
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match="uniform row lengths"):
        manager.prepare_decode_static([seq], *buffers)

    assert manager._num_free_slots == [32, 32]
    assert [int(lengths[0]) for lengths in manager.row_seq_lens] == [4, 3]


def test_h2o_static_decode_rechecks_row_lengths_after_row_cache_reuse():
    manager = _manager_with_layer_rows([[4], [4]])
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 6
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 6, prefilled=4, chunk=1)
    buffers = (
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    )

    manager.prepare_decode_static([seq], *buffers)
    assert getattr(manager, "_h2o_decode_static_rows", None) is not None
    manager.row_seq_lens[1][0] = 4
    manager.buffer_req_to_token_slots[1][0, 4] = 0

    with pytest.raises(RuntimeError, match="uniform row lengths"):
        manager.prepare_decode_static([seq], *buffers)

    assert manager._num_free_slots == [31, 31]
    assert [int(lengths[0]) for lengths in manager.row_seq_lens] == [5, 4]


def test_h2o_multilayer_reduced_decode_context_bounds_are_checked_together():
    class RecordingManager:
        device = torch.device("cpu")

        def __init__(self):
            self.updates = 0
            self.evictions = 0

        def update_decode_attention_scores_all_layers(
            self,
            layer_indices,
            seqs,
            normalized,
        ):
            assert list(layer_indices) == [0, 1]
            assert len(seqs) == 1
            assert normalized.shape == (2, 1, 3)
            self.updates += 1

        def evict_after_decode(self, seqs):
            self.evictions += 1

    manager = RecordingManager()
    config = SimpleNamespace(
        vllm_sparse_method="h2o",
        obs_layer_ids=[],
        full_attn_layers=[],
        runtime_layout=_layout(2),
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=1,
            num_attention_heads=1,
            head_dim=1,
            torch_dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
        decode_cuda_graph=False,
    )
    controller = SparseController(config, manager)
    reduced_scores = controller._get_h2o_decode_score_buffer(2, 1, 3)
    reduced_scores.zero_()
    controller.layer_batch_sparse_states[0].attn_score = reduced_scores[0]
    controller.layer_batch_sparse_states[1].attn_score = reduced_scores[1]
    controller.layer_batch_sparse_states[0].context_lens = torch.tensor([3])
    controller.layer_batch_sparse_states[1].context_lens = torch.tensor([2])
    seqs = [_seq(0, 3, prefilled=3, chunk=1)]

    controller._h2o_decode_eviction(seqs)
    assert manager.updates == 1
    assert manager.evictions == 1

    reduced_scores.zero_()
    controller.layer_batch_sparse_states[1].context_lens = torch.tensor([4])
    with pytest.raises(RuntimeError, match="exceed the reduced score width"):
        controller._h2o_decode_eviction(seqs)
    assert manager.updates == 1
    assert manager.evictions == 1


def test_h2o_contiguous_decode_buffer_handles_padded_graph_batch_and_low_dtype():
    class RecordingManager:
        device = torch.device("cpu")

        def __init__(self):
            self.normalized = None
            self.evicted = False

        def update_decode_attention_scores_all_layers(
            self,
            layer_indices,
            seqs,
            normalized,
        ):
            assert list(layer_indices) == [0, 1]
            assert len(seqs) == 3
            self.normalized = normalized.clone()

        def evict_after_decode(self, seqs):
            self.evicted = True

    manager = RecordingManager()
    config = SimpleNamespace(
        vllm_sparse_method="h2o",
        obs_layer_ids=[],
        full_attn_layers=[],
        runtime_layout=_layout(2),
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            torch_dtype=torch.float16,
        ),
        tensor_parallel_size=1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float16",
        decode_cuda_graph=True,
    )
    controller = SparseController(config, manager)
    reduced_scores = controller._get_h2o_decode_score_buffer(2, 4, 4)
    original_ptr = reduced_scores.data_ptr()
    same_scores = controller._get_h2o_decode_score_buffer(2, 4, 4)
    assert same_scores.data_ptr() == original_ptr
    assert same_scores.dtype == torch.float32

    valid_lens = torch.tensor([4, 3, 2, 0], dtype=torch.int32)
    for layer_idx in range(2):
        for batch_idx, valid_len in enumerate(valid_lens.tolist()):
            if valid_len:
                same_scores[layer_idx, batch_idx, :valid_len] = torch.softmax(
                    torch.arange(valid_len, dtype=torch.float32)
                    + 10 * layer_idx
                    + batch_idx,
                    dim=-1,
                )
        state = controller.layer_batch_sparse_states[layer_idx]
        state.attn_score = same_scores[layer_idx]
        state.context_lens = valid_lens
    expected = same_scores[:, :3].clone()

    controller._h2o_decode_eviction(
        [_seq(seq_id, 4, prefilled=4, chunk=1) for seq_id in range(3)]
    )

    assert manager.normalized.shape == (2, 3, 4)
    assert manager.normalized.dtype == torch.float32
    assert torch.allclose(manager.normalized, expected)
    assert torch.equal(same_scores[:, :3], expected)
    assert manager.evicted
    assert any(
        tensor.data_ptr() == original_ptr
        for tensor in controller.decode_cuda_graph_keepalive_tensors()
    )

    refs = {
        layer_idx: {"attn_score": same_scores[layer_idx]}
        for layer_idx in range(2)
    }
    same_scores.fill_(1.0)
    assert controller.reset_decode_attn_scores_for_graph(refs)
    assert torch.all(same_scores == -1e20)
    controller.clear_decode_attn_score_buffers()
    assert controller._h2o_decode_attn_score_buffers == {}


def test_h2o_free_seq_cleans_score_vectors():
    manager = _manager_with_rows([2])
    manager._h2o_scores[(0, 0)] = torch.ones(2)
    manager._h2o_recent_cursors[(0, 0)] = 1
    with patch.object(SnapKVCacheManager, "free_seq", autospec=True) as parent_free:
        manager.free_seq(0)
    assert manager._h2o_scores == {}
    assert manager._h2o_recent_cursors == {}
    parent_free.assert_called_once_with(manager, 0)


def test_h2o_reset_after_warmup_clears_scores_and_counters():
    manager = _manager_with_rows([2])
    manager._h2o_scores[(0, 0)] = torch.ones(2)
    manager._h2o_recent_cursors[(0, 0)] = 1
    manager._h2o_counters.update(
        {
            "intermediate_prefill_evictions": 1,
            "final_prefill_evictions": 2,
            "decode_evictions": 3,
            "dropped_tokens": 4,
        }
    )
    manager._h2o_ring_counters.update({"fast_rows": 5, "fallback_rows": 6})

    manager.reset_after_warmup()
    assert manager._h2o_scores == {}
    assert manager._h2o_recent_cursors == {}
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_evictions": 0,
        "dropped_tokens": 0,
    }
    assert manager._h2o_ring_counters == {"fast_rows": 0, "fallback_rows": 0}


def test_h2o_capacity_hooks_reserve_prefill_peak_and_gate_chunk_with_real_free_slots():
    manager = _manager_with_rows([8], decode_budget=4, prefill_budget=8)
    manager._num_free_slots = [4]
    seq = _seq(0, 100, prefilled=10, chunk=1)

    assert manager.prompt_admission_cost(seq) == 12
    assert manager.prompt_logical_reservation_cost(seq) == 12
    assert manager.reserved_prefill_slots(deque([seq]), 4) == 4
    assert manager.prefill_step_free_slots_for(seq) == 4
    assert manager.prefill_step_reservation_cost(seq, 4) == 4
    assert manager.decode_step_free_slots_for(seq) == 4
    assert manager.decode_step_reservation_cost(seq) == 1
    assert manager.decode_cuda_graph_context_capacity(
        [seq],
        requested_context_capacity=128,
        current_context_capacity=64,
    ) == (5, False)

    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_batched_tokens=16,
        max_decoding_seqs=4,
        chunk_prefill_size=16,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        vllm_sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, manager)
    scheduler.waiting.append(seq)
    scheduled, is_prefill, _ = scheduler.schedule()
    assert is_prefill
    assert scheduled == [seq]
    assert seq.current_chunk_size == 4


def test_h2o_scheduler_does_not_admit_partial_prefills_that_fill_all_slots():
    manager = _manager_with_rows(
        [0, 0, 0],
        decode_budget=4,
        prefill_budget=8,
        chunk_prefill_size=4,
    )
    manager._num_free_slots = [12]
    seqs = [_seq(seq_id, 20, prefilled=0, chunk=0) for seq_id in range(3)]
    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=3,
        max_num_batched_tokens=12,
        max_decoding_seqs=3,
        chunk_prefill_size=4,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=4,
        vllm_sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, manager)
    scheduler.waiting.extend(seqs)

    scheduled, is_prefill, _ = scheduler.schedule()

    assert is_prefill
    assert scheduled == [seqs[0]]
    assert seqs[0].current_chunk_size == 4

    progress = []
    while not scheduler.decoding:
        for seq in scheduled:
            row_idx = manager.seq_id_to_row[0][seq.seq_id]
            manager.row_seq_lens[0][row_idx] += int(seq.current_chunk_size)
            manager._num_free_slots[0] -= int(seq.current_chunk_size)
            progress.append((seq.seq_id, int(seq.num_prefilled_tokens)))
            budget = (
                manager.h2o_decode_budget
                if seq.is_last_chunk_prefill
                else manager.h2o_prefill_budget
            )
            physical_len = int(manager.row_seq_lens[0][row_idx])
            if physical_len > budget:
                manager.row_seq_lens[0][row_idx] = budget
                manager._num_free_slots[0] += physical_len - budget
        scheduler.postprocess(scheduled, [0] * len(scheduled), is_prefill=True)
        if scheduler.decoding:
            break
        assert manager.reserved_prefill_slots(scheduler.waiting, 4) >= 4
        scheduled, is_prefill, _ = scheduler.schedule()
        assert is_prefill
        assert scheduled

    assert scheduler.decoding[0] is seqs[0]
    assert seqs[0].num_prefilled_tokens == seqs[0].num_prompt_tokens
    assert len(progress) == 5


def test_h2o_scheduler_reserves_until_first_prefill_eviction_peak():
    manager = _manager_with_rows(
        [4, 0],
        decode_budget=50,
        prefill_budget=100,
        chunk_prefill_size=4,
    )
    manager._num_free_slots = [108]
    first = _seq(0, 300, prefilled=4, chunk=0)
    second = _seq(1, 300, prefilled=0, chunk=0)
    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=2,
        max_num_batched_tokens=8,
        max_decoding_seqs=2,
        chunk_prefill_size=4,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        num_sink_tokens=0,
        num_recent_tokens=0,
        decode_keep_tokens=50,
        vllm_sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, manager)
    scheduler.waiting.extend((first, second))

    assert manager.reserved_prefill_slots(scheduler.waiting, 4) == 100
    scheduled, is_prefill, _ = scheduler.schedule()

    assert is_prefill
    assert scheduled == [first]
    assert first.current_chunk_size == 4


def test_h2o_debug_summary_exposes_auditable_eviction_counters():
    manager = _manager_with_rows([2])
    manager._h2o_scores[(0, 0)] = torch.ones(2)
    manager._h2o_recent_cursors[(0, 0)] = 1
    with patch.object(SnapKVCacheManager, "debug_state_summary", return_value={"live_rows": {}}):
        summary = manager.debug_state_summary()
    assert summary["h2o"]["counters"]["dropped_tokens"] == 0
    assert summary["h2o"]["score_lengths"] == {"0:0": 2}
    assert summary["h2o"]["recent_cursors"] == {"0:0": 1}
