from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparsevllm.engine.cache_manager.h2o import H2OCacheManager
from sparsevllm.engine.cache_manager.h2o_retention import H2ORetention
from sparsevllm.engine.cache_manager.storage import ExplicitKVStorage, MlaLatentStorage
from sparsevllm.engine.sparse_methods.h2o_selection import select_h2o_heads


def test_late_max_preserves_head_history_across_chunks():
    # Alternating heads favor token 0 under early max; cumulative max favors 1.
    p = torch.tensor([
        [[.4, .3, .3], [0., .3, .7]],
        [[0., .3, .7], [.4, .3, .3]],
    ])
    cumulative = None
    for chunk in p:
        cumulative = H2OCacheManager._accumulate_score(
            cumulative, chunk, new_len=3, weight=1,
        )
    torch.testing.assert_close(cumulative, p.sum(0))
    keep = select_h2o_heads(cumulative, selection_groups=1, budget=2, recent_ratio=.5, reduction='max')
    assert keep.tolist() == [[1, 2]]
    assert p.amax(1).sum(0)[0] > p.amax(1).sum(0)[1]


def test_groups_select_independently_and_mean_changes_only_ranking():
    cumulative = torch.tensor([[9., 6., 0., 0.], [0., 6., 0., 0.], [0., 0., 8., 0.], [0., 0., 7., 0.]])
    args = dict(selection_groups=2, budget=2, recent_ratio=.5)
    assert select_h2o_heads(cumulative, reduction='max', **args).tolist() == [[0, 3], [2, 3]]
    assert select_h2o_heads(cumulative, reduction='mean', **args).tolist() == [[1, 3], [2, 3]]
    mha = select_h2o_heads(cumulative, selection_groups=4, budget=2, recent_ratio=.5, reduction='max')
    assert mha.tolist() == [[0, 3], [1, 3], [2, 3], [2, 3]]


def test_ties_and_recent_suffix_fill_budget_in_logical_order():
    scores = torch.zeros(3, 9)
    keep = select_h2o_heads(scores, selection_groups=3, budget=4, recent_ratio=.5, reduction='max')
    assert keep.tolist() == [[0, 1, 7, 8]] * 3
    all_tokens = select_h2o_heads(scores, selection_groups=3, budget=12, recent_ratio=.5, reduction='mean')
    assert all_tokens.tolist() == [list(range(9))] * 3


def manager_with_storage(mla=False):
    manager = object.__new__(H2OCacheManager)
    manager.device = torch.device('cpu')
    manager.num_kv_heads = 2
    manager.runtime_layout = SimpleNamespace(kv_layer_index=lambda layer: layer)
    if mla:
        storage = MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
    else:
        storage = ExplicitKVStorage(num_kv_heads=2, head_dim=4, dtype=torch.float32)
    storage.allocate(num_layers=1, num_slots=12, device=manager.device)
    manager.attention_cache_storage = storage
    manager.seq_id_to_row = [{7: 0}]
    manager.row_seq_lens = [np.array([6], dtype=np.int32)]
    manager.buffer_req_to_token_slots = [torch.tensor([[8, 2, 6, 4, 0, 9, 0, 0]], dtype=torch.int32)]
    manager.free_slots_stack = [torch.zeros(12, dtype=torch.int32)]
    manager._num_free_slots = [6]
    manager._h2o_scores = {(0, 7): torch.arange(24).reshape(4, 6).float()}
    manager._h2o_positions = {(0, 7): torch.tensor([[1, 3, 5, 7, 9, 11]]).expand(1 if mla else 2, -1).clone()}
    manager._h2o_counters = dict(final_prefill_evictions=0, intermediate_prefill_evictions=0, dropped_tokens=0)
    return manager


@pytest.mark.parametrize('final', [False, True])
def test_explicit_retention_packs_different_heads_without_union(final):
    manager = manager_with_storage()
    storage = manager.attention_cache_storage
    storage.cache.copy_(torch.arange(storage.cache.numel()).reshape_as(storage.cache))
    before = storage.cache.clone()
    old_slots = manager.buffer_req_to_token_slots[0][0, :6].long().clone()
    old_scores = manager._h2o_scores[(0, 7)].clone()
    keep = torch.tensor([[0, 2, 5], [1, 4, 5]])
    manager.commit_h2o_retention([H2ORetention(0, 7, 6, keep, final)])
    destinations = manager.buffer_req_to_token_slots[0][0, :3].long()
    # Independent scalar oracle: the packed slot can contain different tokens.
    for head in range(2):
        for packed in range(3):
            source = old_slots[keep[head, packed]]
            torch.testing.assert_close(storage.cache[:, 0, destinations[packed], head], before[:, 0, source, head])
            for query_head in range(2 * head, 2 * head + 2):
                assert manager._h2o_scores[(0, 7)][query_head, packed] == old_scores[query_head, keep[head, packed]]
    assert manager.row_seq_lens[0][0] == 3
    assert manager._num_free_slots == [9]
    assert storage.cache.shape == before.shape
    assert manager._h2o_positions[(0, 7)].tolist() == [[1, 5, 11], [3, 9, 11]]
    released = set(manager.free_slots_stack[0][6:9].tolist())
    assert released.isdisjoint(destinations.tolist())
    assert released | set(destinations.tolist()) == set(old_slots.tolist())


def test_mla_retention_keeps_latent_rope_and_all_score_heads_paired():
    manager = manager_with_storage(mla=True)
    storage = manager.attention_cache_storage
    for slot in range(12):
        storage.latent_cache[:, slot].fill_(slot)
        storage.rope_cache[:, slot].fill_(slot + 20)
    score_before = manager._h2o_scores[(0, 7)].clone()
    manager.commit_h2o_retention([H2ORetention(0, 7, 6, torch.tensor([[1, 3, 5]]), True)])
    for dest, source in zip(manager.buffer_req_to_token_slots[0][0, :3], [2, 4, 9]):
        assert (storage.latent_cache[0, dest] == source).all()
        assert (storage.rope_cache[0, dest] == source + 20).all()
    torch.testing.assert_close(manager._h2o_scores[(0, 7)], score_before[:, [1, 3, 5]])
    assert manager._h2o_positions[(0, 7)].tolist() == [[3, 7, 11]]


@pytest.mark.parametrize('keep', [torch.tensor([[0, 0, 5], [1, 4, 5]]), torch.tensor([[0, 2, 6], [1, 4, 5]])])
def test_invalid_retention_does_not_mutate_cache_or_allocator(keep):
    manager = manager_with_storage()
    before = manager.attention_cache_storage.cache.clone()
    slots = manager.buffer_req_to_token_slots[0].clone()
    with pytest.raises(RuntimeError, match='indices'):
        manager.commit_h2o_retention([H2ORetention(0, 7, 6, keep, True)])
    torch.testing.assert_close(manager.attention_cache_storage.cache, before, equal_nan=True)
    torch.testing.assert_close(manager.buffer_req_to_token_slots[0], slots)
    assert manager.row_seq_lens[0][0] == 6
    assert manager._num_free_slots == [6]


def test_chunk_prefill_append_retention_and_final_handoff_follow_logical_positions():
    from sparsevllm.engine.cache_manager.base import LayerBatchStates
    from sparsevllm.engine.sequence import Sequence
    from sparsevllm.engine.sparse_methods.base import SparseStepContext
    from sparsevllm.engine.sparse_methods.h2o import H2ORuntime

    manager = manager_with_storage()
    manager.num_layers = manager.num_kv_layers = 1
    manager.runtime_layout.kv_idx_to_layer_idx = (0,)
    manager.max_model_len = 16
    manager.layer_batch_states = [LayerBatchStates()]
    manager.buffer_req_to_token_slots = [torch.zeros(1, 16, dtype=torch.int32)]
    manager.free_slots_stack = [torch.arange(12, dtype=torch.int32)]
    manager._num_free_slots = [12]
    manager.row_seq_lens[0][0] = 0
    manager._h2o_scores.clear()
    manager._h2o_positions.clear()
    manager.config = SimpleNamespace(h2o_prefill_budget=4, h2o_decode_budget=3,
                                     h2o_prefill_score_window=0, h2o_recent_ratio=.5,
                                     h2o_head_reduction='max')
    runtime = object.__new__(H2ORuntime)
    runtime.config = manager.config
    runtime.cache_manager = manager
    seq = Sequence(list(range(9)))
    seq.seq_id = 7
    histories = [[], []]
    cumulative_reference = [dict() for _ in range(4)]
    for logical_start in (0, 3, 6):
        seq.num_prefilled_tokens = logical_start
        seq.current_chunk_size = 3
        _, positions, _ = manager._prepare_prefill([seq])
        assert positions.tolist() == list(range(logical_start, logical_start + 3))
        slots = manager.layer_batch_states[0].slot_mapping.long()
        for index, slot in enumerate(slots):
            for head in range(2):
                manager.attention_cache_storage.cache[:, 0, slot, head].fill_(logical_start + index + head * 100)
        for history in histories:
            history.extend(range(logical_start, logical_start + 3))
        length = len(histories[0])
        delta = torch.zeros(1, 4, length)
        for head in range(4):
            history = histories[head // 2]
            # Independently generated normalized causal probabilities; this CPU
            # test establishes state/coordinate correctness, not GPU numerics.
            for query_pos in range(logical_start, logical_start + 3):
                visible = [token for token in history if token <= query_pos]
                weights = [float(1 + ((token + head * 3) % 7)) for token in visible]
                denominator = sum(weights)
                for token, weight in zip(visible, weights):
                    mass = weight / denominator
                    delta[0, head, history.index(token)] += mass
                    cumulative_reference[head][token] = cumulative_reference[head].get(token, 0) + mass
        manager.accumulate_prefill_scores(0, [seq], delta)
        runtime.finish_step(SparseStepContext([seq], True, None))
        budget = 3 if logical_start == 6 else 4
        for group in range(2):
            history = histories[group]
            if len(history) > budget:
                recent_count = max(1, int(budget * .5))
                rank = lambda token: max(cumulative_reference[2 * group + h][token] for h in range(2))
                heavy = sorted(history[:-recent_count], key=lambda token: (-rank(token), token))[:budget - recent_count]
                histories[group] = sorted(heavy + history[-recent_count:])
            for head in (2 * group, 2 * group + 1):
                cumulative_reference[head] = {token: cumulative_reference[head][token] for token in histories[group]}
                torch.testing.assert_close(manager._h2o_scores[(0, 7)][head], torch.tensor(list(cumulative_reference[head].values())))
        assert manager._h2o_positions[(0, 7)].tolist() == histories
        resident = int(manager.row_seq_lens[0][0])
        assert resident == min(length, budget)
        for packed, slot in enumerate(manager.buffer_req_to_token_slots[0][0, :resident]):
            for head in range(2):
                assert (manager.attention_cache_storage.cache[:, 0, slot, head] == histories[head][packed] + 100 * head).all()
        assert manager._num_free_slots[0] + resident == 12


def _multi_request_manager():
    manager = manager_with_storage()
    manager.num_layers = manager.num_kv_layers = 2
    manager.runtime_layout.kv_idx_to_layer_idx = (0, 1)
    manager.attention_cache_storage.allocate(num_layers=2, num_slots=24, device=manager.device)
    cache = manager.attention_cache_storage.cache
    cache.copy_(torch.arange(cache.numel()).reshape_as(cache))
    manager.config = SimpleNamespace(h2o_prefill_budget=4, h2o_decode_budget=3,
                                     h2o_recent_ratio=.5, h2o_head_reduction='max')
    manager.seq_id_to_row = [{7: 0, 8: 1}, {7: 0, 8: 1}]
    manager.row_seq_lens = [np.array([6, 5], dtype=np.int32) for _ in range(2)]
    manager.buffer_req_to_token_slots = [torch.zeros(2, 8, dtype=torch.int32) for _ in range(2)]
    manager.free_slots_stack = [torch.zeros(24, dtype=torch.int32) for _ in range(2)]
    manager._num_free_slots = [13, 13]
    manager._h2o_scores.clear()
    manager._h2o_positions.clear()
    for layer in range(2):
        for row, (seq_id, length) in enumerate([(7, 6), (8, 5)]):
            manager.buffer_req_to_token_slots[layer][row, :length] = torch.arange(length).flip(0) + row * 12
            manager._h2o_scores[(layer, seq_id)] = torch.stack([
                torch.arange(length).float().roll(layer + head) for head in range(4)
            ])
            manager._h2o_positions[(layer, seq_id)] = torch.arange(length)[None].expand(2, -1).clone()
    return manager


def _mixed_prefill_seqs():
    from sparsevllm.engine.sequence import Sequence
    seqs = [Sequence(list(range(6))), Sequence(list(range(20)))]
    for seq, seq_id, chunk in zip(seqs, [7, 8], [6, 5]):
        seq.seq_id = seq_id
        seq.current_chunk_size = chunk
    return seqs


def test_multilayer_mixed_prefill_budgets_preserve_independent_head_payloads():
    from sparsevllm.engine.sparse_methods.base import SparseStepContext
    from sparsevllm.engine.sparse_methods.h2o import H2ORuntime

    manager = _multi_request_manager()
    expected = {}
    old_cache = manager.attention_cache_storage.cache.clone()
    for layer in range(2):
        for row, (seq_id, length, budget) in enumerate([(7, 6, 3), (8, 5, 4)]):
            scores = manager._h2o_scores[(layer, seq_id)]
            recent = max(1, int(budget * .5))
            for group in range(2):
                rank = lambda index: max(float(scores[2 * group + head, index]) for head in range(2))
                heavy = sorted(range(length - recent), key=lambda index: (-rank(index), index))[:budget - recent]
                selected = sorted(heavy + list(range(length - recent, length)))
                slots = manager.buffer_req_to_token_slots[layer][row, selected].long()
                expected[layer, seq_id, group] = (selected, old_cache[:, layer, slots, group].clone())
    runtime = object.__new__(H2ORuntime)
    runtime.config, runtime.cache_manager = manager.config, manager
    runtime.finish_step(SparseStepContext(_mixed_prefill_seqs(), True, None))
    for layer in range(2):
        assert manager.row_seq_lens[layer].tolist() == [3, 4]
        for row, (seq_id, budget) in enumerate([(7, 3), (8, 4)]):
            slots = manager.buffer_req_to_token_slots[layer][row, :budget].long()
            for group in range(2):
                positions, payload = expected[layer, seq_id, group]
                assert manager._h2o_positions[layer, seq_id][group].tolist() == positions
                torch.testing.assert_close(manager.attention_cache_storage.cache[:, layer, slots, group], payload)
    assert manager._num_free_slots == [17, 17]
    assert manager._h2o_counters['final_prefill_evictions'] == 2
    assert manager._h2o_counters['intermediate_prefill_evictions'] == 2
    assert manager._h2o_counters['dropped_tokens'] == 8


def test_later_layer_capacity_failure_does_not_commit_earlier_layers():
    from sparsevllm.engine.sparse_methods.base import SparseStepContext
    from sparsevllm.engine.sparse_methods.h2o import H2ORuntime

    manager = _multi_request_manager()
    manager._num_free_slots[1] = 23
    old_cache = manager.attention_cache_storage.cache.clone()
    old_tables = [table.clone() for table in manager.buffer_req_to_token_slots]
    old_scores = {key: score.clone() for key, score in manager._h2o_scores.items()}
    runtime = object.__new__(H2ORuntime)
    runtime.config, runtime.cache_manager = manager.config, manager
    with pytest.raises(RuntimeError, match='overflow.*layer=1'):
        runtime.finish_step(SparseStepContext(_mixed_prefill_seqs(), True, None))
    torch.testing.assert_close(manager.attention_cache_storage.cache, old_cache)
    for actual, before in zip(manager.buffer_req_to_token_slots, old_tables):
        torch.testing.assert_close(actual, before)
    for key, before in old_scores.items():
        torch.testing.assert_close(manager._h2o_scores[key], before)
    assert manager._num_free_slots == [13, 23]
    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [[6, 5], [6, 5]]


@pytest.mark.parametrize('mla,heads,groups,tp', [(False, 8, 8, 1), (False, 32, 2, 1), (False, 16, 2, 2), (True, 32, 1, 1)])
def test_capacity_reserves_native_payload_and_live_head_metadata(monkeypatch, mla, heads, groups, tp):
    from sparsevllm.engine.cache_manager.snapkv import SnapKVCacheManager

    manager = object.__new__(H2OCacheManager)
    manager.tp_size, manager.num_kv_heads = tp, groups
    manager.hf_config = SimpleNamespace(num_attention_heads=heads * tp)
    manager.max_buffer_rows, manager.max_model_len = 3, 2048
    manager.config = SimpleNamespace(max_num_seqs_in_batch=2, engine_prefill_chunk_size=257,
                                     h2o_prefill_budget=129, h2o_prefill_score_window=0,
                                     enable_prefix_caching=False)
    storage = (MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16) if mla
               else ExplicitKVStorage(num_kv_heads=groups, head_dim=128, dtype=torch.bfloat16))
    manager.attention_cache_storage = storage
    native_bytes = storage.bytes_per_slot_per_layer()
    available = 64 * 1024 * 1024
    monkeypatch.setattr(SnapKVCacheManager, '_get_available_slots_info', lambda self: (available, native_bytes))
    remaining, slot_cost = manager._get_available_slots_info()
    # Scores and positions must survive while the newly gathered copies exist.
    assert slot_cost >= native_bytes + 2 * (4 * heads + 8 * groups)
    assert slot_cost - manager._h2o_metadata_bytes_per_slot == native_bytes
    score_buffers = 2 * 4 * 2 * heads * (129 + 257)
    copy_buffers = 129 * native_bytes
    assert available - remaining >= score_buffers + copy_buffers
    assert 0 < remaining < available
    monkeypatch.setattr(SnapKVCacheManager, '_get_available_slots_info', lambda self: (1, native_bytes))
    with pytest.raises(RuntimeError, match='Not enough memory'):
        manager._get_available_slots_info()


def test_mla_rejects_tp_without_cross_rank_head_reduction(monkeypatch):
    from sparsevllm.engine.cache_manager.snapkv import SnapKVCacheManager

    manager = manager_with_storage(mla=True)
    manager.tp_size = 2
    monkeypatch.setattr(SnapKVCacheManager, '_get_available_slots_info', lambda self: (2**30, 1152))
    with pytest.raises(ValueError, match='requires TP1'):
        manager._get_available_slots_info()


def test_rejected_first_chunk_preserves_resident_head_history():
    from sparsevllm.engine.sequence import Sequence

    manager = manager_with_storage()
    manager.num_layers = manager.num_kv_layers = 1
    manager.runtime_layout.kv_idx_to_layer_idx = (0,)
    seq = Sequence(list(range(6)))
    seq.seq_id, seq.current_chunk_size = 7, 3
    scores, positions = manager._h2o_scores[0, 7], manager._h2o_positions[0, 7]
    with pytest.raises(RuntimeError, match='non-empty physical row'):
        manager._prepare_prefill([seq])
    assert manager._h2o_scores[0, 7] is scores
    assert manager._h2o_positions[0, 7] is positions
    assert manager.row_seq_lens[0][0] == 6
    assert manager._num_free_slots == [6]
