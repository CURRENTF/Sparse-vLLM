import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.quest import QuestCacheManager, QuestPrefixBlockPayload
from sparsevllm.engine.cache_manager.standard import StandardCacheManager, StandardPrefixBlockPayload
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.prefix_cache import (
    PrefixCacheBlock,
    RadixPrefixIndex,
    RadixTreeBackend,
    build_prefix_cache_fingerprint,
    resolve_prefix_cache_block_size,
    usable_prefix_cache_tokens,
)
from sparsevllm.method_registry import PREFIX_CACHE_SUPPORTED_METHODS


def _cfg(method="", salt="", block_size=4):
    return SimpleNamespace(
        model="/models/qwen",
        hf_config=SimpleNamespace(model_type="qwen2", torch_dtype=torch.float16),
        tensor_parallel_size=1,
        vllm_sparse_method=method,
        prefix_cache_salt=salt,
        prefix_cache_block_size=block_size,
        chunk_prefill_accel_omnikv=False,
        num_top_tokens=64,
        num_top_tokens_in_prefill=64,
        num_sink_tokens=4,
        num_recent_tokens=8,
        full_attn_layers=[0],
        obs_layer_ids=None,
        quest_chunk_size=4,
        quest_skip_layers=2,
    )


def _insert_tokens(index: RadixPrefixIndex, token_ids: list[int]) -> bytes:
    parent_block_id = None
    last_block_id = None
    for logical_idx, start in enumerate(range(0, len(token_ids), index.block_size)):
        block_tokens = token_ids[start: start + index.block_size]
        stable_block_id = index.stable_block_id(block_tokens, parent_block_id)
        block = PrefixCacheBlock(
            stable_block_id=stable_block_id,
            parent_block_id=parent_block_id,
            block_size=index.block_size,
            logical_block_idx=logical_idx,
            payload=SimpleNamespace(name="dummy"),
            token_ids=tuple(block_tokens),
        )
        index.insert_block(block)
        parent_block_id = stable_block_id
        last_block_id = stable_block_id
    assert last_block_id is not None
    return last_block_id


def _hf_config():
    return SimpleNamespace(
        model_type="qwen2",
        torch_dtype=torch.float16,
        max_position_embeddings=32768,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def _make_config(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
            return Config(model=str(model_dir), **kwargs)


def _make_standard_manager_for_prefix(block_size=2):
    cfg = _cfg(block_size=block_size)
    cfg.num_kvcache_slots = 90
    fingerprint = build_prefix_cache_fingerprint(cfg, block_size)
    manager = object.__new__(StandardCacheManager)
    manager.config = cfg
    manager.device = torch.device("cpu")
    manager.enable_prefix_caching = True
    manager.prefix_cache_block_size = block_size
    manager.prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=fingerprint)
    manager.layer_batch_state = SimpleNamespace()
    manager.buffer_req_to_token_slots = torch.zeros((2, 16), dtype=torch.int32)
    manager.free_slots_stack = torch.arange(100, dtype=torch.int32)
    manager._num_free_slots = 90
    manager.seq_id_to_row = {}
    manager.free_rows = deque([0, 1])
    manager.row_seq_lens = np.zeros((2,), dtype=np.int32)
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_cached_ranges = {}
    manager._init_prefix_cache_runtime()
    return manager


def _make_quest_manager_for_prefix(page_size=2):
    cfg = _cfg(method="quest", block_size=page_size)
    fingerprint = build_prefix_cache_fingerprint(cfg, page_size)
    manager = object.__new__(QuestCacheManager)
    manager.config = cfg
    manager.device = torch.device("cpu")
    manager.enable_prefix_caching = True
    manager.page_size = page_size
    manager.num_pages = 10
    manager.prefix_cache_block_size = page_size
    manager.prefix_cache = RadixPrefixIndex(block_size=page_size, fingerprint=fingerprint)
    manager.layer_batch_state = SimpleNamespace()
    manager.page_offsets_i32 = torch.arange(page_size, dtype=torch.int32)
    manager.buffer_req_to_token_slots = torch.zeros((2, 16), dtype=torch.int32)
    manager.buffer_req_to_page_slots = torch.full((2, 8), -1, dtype=torch.int32)
    manager.free_pages_stack = torch.arange(10, dtype=torch.int32)
    manager._num_free_pages = 10
    manager.seq_id_to_row = {}
    manager.free_rows = deque([0, 1])
    manager.row_seq_lens = np.zeros((2,), dtype=np.int32)
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_cached_pages = {}
    manager._init_prefix_cache_runtime()
    return manager


def _remove_free_page(manager, page_slot: int):
    pages = [
        int(page)
        for page in manager.free_pages_stack[: manager._num_free_pages].tolist()
        if int(page) != int(page_slot)
    ]
    manager.free_pages_stack[: len(pages)] = torch.tensor(pages, dtype=torch.int32)
    manager._num_free_pages = len(pages)


def _remove_free_slots(manager, slots: list[int]):
    remove = {int(slot) for slot in slots}
    free_slots = [
        int(slot)
        for slot in manager.free_slots_stack[: manager._num_free_slots].tolist()
        if int(slot) not in remove
    ]
    manager.free_slots_stack[: len(free_slots)] = torch.tensor(free_slots, dtype=torch.int32)
    manager._num_free_slots = len(free_slots)


def test_usable_prefix_cache_tokens_leaves_logits_work():
    assert usable_prefix_cache_tokens(128, 16) == 112
    assert usable_prefix_cache_tokens(129, 16) == 128
    assert usable_prefix_cache_tokens(15, 16) == 0
    assert usable_prefix_cache_tokens(1, 16) == 0


def test_radix_prefix_index_block_id_is_stable_and_parent_sensitive():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)

    first = index.stable_block_id([1, 2, 3, 4], None)
    assert first == index.stable_block_id([1, 2, 3, 4], None)
    assert first != index.stable_block_id([1, 2, 3, 5], None)
    assert index.stable_block_id([5, 6, 7, 8], first) != index.stable_block_id([5, 6, 7, 8], None)


def test_prefix_cache_fingerprint_isolates_salt_and_method():
    vanilla = build_prefix_cache_fingerprint(_cfg(method="", salt="a"), 4)
    salted = build_prefix_cache_fingerprint(_cfg(method="", salt="b"), 4)
    omnikv = build_prefix_cache_fingerprint(_cfg(method="omnikv", salt="a"), 4)
    quest = build_prefix_cache_fingerprint(_cfg(method="quest", salt="a"), 4)

    assert vanilla != salted
    assert vanilla != omnikv
    assert omnikv != quest


def test_standard_prompt_admission_accounts_for_free_rows():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.free_rows = deque([0])
    seq = Sequence([1, 2, 3])

    budgets = manager.prompt_admission_budgets(deque(), chunk_prefill_size=4)
    costs = manager.prompt_admission_costs(seq)

    assert budgets["rows"] == 1
    assert costs["rows"] == 1
    assert budgets["slots"] == manager.prompt_admission_free_slots()
    assert costs["slots"] == manager.prompt_admission_cost(seq)


def test_lookup_returns_longest_full_block_prefix():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))

    hit_len, hit_last_block_id, hit_blocks = index.lookup_longest_prefix(
        list(range(12)),
        max_usable_tokens=usable_prefix_cache_tokens(12, 4),
    )

    assert hit_len == 8
    assert hit_last_block_id == last_block_id
    assert hit_blocks == 2
    chain = index.get_chain(hit_last_block_id, hit_blocks)
    assert [block.logical_block_idx for block in chain] == [0, 1]


def test_lookup_never_returns_half_block_match():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, [1, 2, 3, 4])

    hit_len, hit_last_block_id, hit_blocks = index.lookup_longest_prefix(
        [1, 2, 3, 4, 5, 6],
        max_usable_tokens=6,
    )

    assert hit_len == 4
    assert hit_last_block_id is not None
    assert hit_blocks == 1


def test_radix_backend_splits_edges_only_between_block_ids():
    backend = RadixTreeBackend()
    backend.insert((b"a", b"b", b"c"))
    backend.insert((b"a", b"b", b"d"))

    assert backend.lookup((b"a", b"b", b"c"), max_blocks=3).hit_block_count == 3
    assert backend.lookup((b"a", b"b", b"x"), max_blocks=3).hit_block_count == 2
    assert backend.child_count(b"a") == 1
    assert backend.child_count(b"b") == 2
    assert set(backend.leaf_block_ids()) == {b"c", b"d"}


def test_radix_backend_removes_leaf_from_compressed_segment_and_preserves_siblings():
    backend = RadixTreeBackend()
    backend.insert((b"a", b"b", b"c", b"d"))
    backend.insert((b"a", b"b", b"x", b"y"))
    backend.insert((b"a", b"q"))

    assert backend.path_to_block(b"d") == (b"a", b"b", b"c", b"d")
    assert backend.path_to_block(b"y") == (b"a", b"b", b"x", b"y")
    assert backend.child_count(b"b") == 2
    assert backend.child_count(b"c") == 1
    assert set(backend.subtree_block_ids(b"b")) == {b"b", b"c", b"d", b"x", b"y"}

    backend.remove_block(b"d")

    assert backend.path_to_block(b"c") == (b"a", b"b", b"c")
    assert backend.path_to_block(b"y") == (b"a", b"b", b"x", b"y")
    assert set(backend.leaf_block_ids()) == {b"c", b"y", b"q"}


def test_lookup_does_not_touch_lru_state():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_chain(last_block_id, 1)[0]
    last_access = block.last_access

    index.lookup_longest_prefix([1, 2, 3, 4, 5], max_usable_tokens=4)

    assert block.last_access == last_access


def test_leaf_only_eviction_preserves_parent_until_child_is_removed():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, list(range(8)))

    evicted = index.evict_until_freeable(1)
    assert [block.logical_block_idx for block in evicted] == [1]
    assert index.evictable_blocks() == 1

    evicted = index.evict_until_freeable(1)
    assert [block.logical_block_idx for block in evicted] == [0]
    assert len(index) == 0


def test_referenced_blocks_are_not_evictable():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_chain(last_block_id, 1)[0]
    block.ref_count = 1

    assert index.evict_until_freeable(1) == []
    block.ref_count = 0
    assert index.evict_until_freeable(1) == [block]


def test_duplicate_commit_returns_existing_block_and_counts_duplicate():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    stable_block_id = index.stable_block_id([1, 2, 3, 4], None)
    first = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=SimpleNamespace(name="first"),
        token_ids=(1, 2, 3, 4),
    )
    second = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=SimpleNamespace(name="second"),
        token_ids=(1, 2, 3, 4),
    )

    assert index.insert_block(first) is first
    assert index.insert_block(second) is first
    assert index.stats()["prefix_cache_duplicate_commits"] == 1


def test_eviction_priority_prefers_larger_positive_priority():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    first_id = _insert_tokens(index, [1, 2, 3, 4])
    second_id = _insert_tokens(index, [5, 6, 7, 8])
    first = index.get_block(first_id)
    second = index.get_block(second_id)
    assert first is not None and second is not None
    first.eviction_priority = 1
    second.eviction_priority = 10

    assert index.evict_until_freeable(1) == [second]
    assert first_id in index.blocks


def test_negative_priority_blocks_eviction_and_safe_delete():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_block(block_id)
    assert block is not None
    block.eviction_priority = -1

    assert index.evict_until_freeable(1) == []
    result = index.safe_delete_subtree([1, 2, 3, 4])
    assert result.deleted_blocks == []
    assert [blocked.reason for blocked in result.blocked_blocks] == ["negative_priority"]


def test_subtree_delete_reports_referenced_child_and_preserves_parent():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))
    parent, child = index.get_chain(last_block_id, 2)
    child.ref_count = 1

    result = index.safe_delete_subtree(list(range(4)))

    assert result.deleted_blocks == []
    assert [blocked.reason for blocked in result.blocked_blocks] == ["referenced", "has_children"]
    assert parent.stable_block_id in index.blocks
    assert child.stable_block_id in index.blocks


def test_subtree_delete_deletes_safe_child_and_blocks_protected_branch():
    fp = build_prefix_cache_fingerprint(_cfg(), 2)
    index = RadixPrefixIndex(block_size=2, fingerprint=fp)
    root_id = _insert_tokens(index, [1, 2])
    referenced_child_id = _insert_tokens(index, [1, 2, 3, 4])
    free_child_id = _insert_tokens(index, [1, 2, 5, 6])
    referenced_child = index.get_block(referenced_child_id)
    assert referenced_child is not None
    referenced_child.ref_count = 1

    result = index.safe_delete_subtree([1, 2])

    assert [block.stable_block_id for block in result.deleted_blocks] == [free_child_id]
    assert {blocked.reason for blocked in result.blocked_blocks} == {"referenced", "has_children"}
    assert root_id in index.blocks
    assert referenced_child_id in index.blocks
    assert free_child_id not in index.blocks


def test_max_blocks_requires_explicit_capacity_before_insert():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp, max_blocks=1)
    _insert_tokens(index, [1, 2, 3, 4])

    stable_block_id = index.stable_block_id([5, 6, 7, 8], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=SimpleNamespace(name="dummy"),
        token_ids=(5, 6, 7, 8),
    )
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        index.insert_block(block)

    evicted = index.ensure_insert_capacity(1)
    assert len(evicted) == 1
    assert index.insert_block(block) is block


def test_get_chain_fails_fast_on_incomplete_chain():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))
    parent = index.get_chain(last_block_id, 2)[0]
    del index.blocks[parent.stable_block_id]

    with pytest.raises(RuntimeError, match="incomplete"):
        index.get_chain(last_block_id, 2)


def test_resolve_prefix_cache_block_size_uses_quest_page_size():
    assert resolve_prefix_cache_block_size(_cfg(method="quest", block_size=None)) == 4
    with pytest.raises(ValueError, match="quest_chunk_size"):
        resolve_prefix_cache_block_size(_cfg(method="quest", block_size=8))
    with pytest.raises(ValueError, match="positive integer"):
        resolve_prefix_cache_block_size(_cfg(block_size=16.9))


def test_prefix_cache_supported_method_allowlist():
    assert PREFIX_CACHE_SUPPORTED_METHODS == {"", "omnikv", "quest"}


def test_config_resolves_prefix_cache_defaults():
    cfg = _make_config(enable_prefix_caching=True)
    assert cfg.vllm_sparse_method == ""
    assert cfg.prefix_cache_block_size == 16

    cfg = _make_config(
        vllm_sparse_method="quest",
        enable_prefix_caching=True,
        quest_chunk_size=8,
        prefix_cache_block_size=None,
    )
    assert cfg.prefix_cache_block_size == 8

    cfg = _make_config(enable_prefix_caching="false", prefix_cache_block_size="32")
    assert cfg.enable_prefix_caching is False
    assert cfg.prefix_cache_block_size == 32


def test_config_rejects_unsupported_prefix_cache_methods():
    with pytest.raises(ValueError, match="only supports vanilla"):
        _make_config(vllm_sparse_method="snapkv", enable_prefix_caching=True)


def test_config_rejects_unvalidated_prefix_cache_options():
    with pytest.raises(ValueError, match="capture_sampling"):
        _make_config(
            enable_prefix_caching=True,
            decode_cuda_graph=True,
            decode_cuda_graph_capture_sampling=True,
        )
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _make_config(
            enable_prefix_caching=True,
            decode_cuda_graph=True,
            tensor_parallel_size=2,
        )
    with pytest.raises(ValueError, match="quest_chunk_size"):
        _make_config(
            vllm_sparse_method="quest",
            enable_prefix_caching=True,
            quest_chunk_size=8,
            prefix_cache_block_size=16,
        )
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        _make_config(enable_prefix_caching="maybe")
    with pytest.raises(ValueError, match="prefix_cache_block_size"):
        _make_config(prefix_cache_block_size=16.9)
    with pytest.raises(ValueError, match="prefix_cache_max_blocks"):
        _make_config(prefix_cache_max_blocks="16.9")


def test_config_allows_prefix_cache_decode_cuda_graph_tp1_methods():
    vanilla = _make_config(enable_prefix_caching=True, decode_cuda_graph=True)
    assert vanilla.enable_prefix_caching is True
    assert vanilla.decode_cuda_graph is True
    assert vanilla.decode_cuda_graph_capture_sampling is False

    omnikv = _make_config(
        vllm_sparse_method="omnikv",
        enable_prefix_caching=True,
        decode_cuda_graph=True,
    )
    assert omnikv.enable_prefix_caching is True
    assert omnikv.decode_cuda_graph is True

    quest = _make_config(
        vllm_sparse_method="quest",
        enable_prefix_caching=True,
        decode_cuda_graph=True,
        quest_chunk_size=8,
        prefix_cache_block_size=None,
    )
    assert quest.enable_prefix_caching is True
    assert quest.decode_cuda_graph is True
    assert quest.prefix_cache_block_size == 8


def test_standard_attach_pins_prefix_slots_and_free_seq_keeps_cached_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id
    seq.prefix_cache_block_size = 2
    seq.prefix_cache_method = ""

    manager._attach_prefix_cache_if_needed(seq)
    assert manager.row_seq_lens[0] == 2
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == [10, 11]
    assert block.ref_count == 1

    manager._allocate(seq.seq_id, 1)
    assert manager.row_seq_lens[0] == 3
    assert manager._num_free_slots == 87

    manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == 88
    assert block.ref_count == 0
    assert manager.seq_id_to_row == {}


def test_standard_materializes_blocks_only_after_forward_end():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2])
    slots = torch.tensor([20, 21], dtype=torch.int32)

    manager._record_prefix_materialization(seq, [1, 2], slots)
    assert len(manager.prefix_cache) == 0

    manager.on_forward_end([seq], is_prefill=True)
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.ref_count == 1
    assert isinstance(block.payload, StandardPrefixBlockPayload)
    assert block.payload.token_slots.tolist() == [20, 21]
    assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 2)]


def test_standard_safe_delete_releases_payload_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    assert manager._num_free_slots == 88

    result = manager.prefix_cache_delete_subtree([1, 2])

    assert result["deleted_block_ids"] == [stable_block_id.hex()]
    assert manager._num_free_slots == 90
    assert stable_block_id not in manager.prefix_cache.blocks


def test_standard_safe_delete_partial_subtree_releases_only_deleted_child_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    root_id = manager.prefix_cache.stable_block_id([1, 2], None)
    referenced_child_id = manager.prefix_cache.stable_block_id([3, 4], root_id)
    free_child_id = manager.prefix_cache.stable_block_id([5, 6], root_id)
    root = PrefixCacheBlock(
        stable_block_id=root_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    referenced_child = PrefixCacheBlock(
        stable_block_id=referenced_child_id,
        parent_block_id=root_id,
        block_size=2,
        logical_block_idx=1,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([12, 13], dtype=torch.int32)),
        token_ids=(3, 4),
        ref_count=1,
    )
    free_child = PrefixCacheBlock(
        stable_block_id=free_child_id,
        parent_block_id=root_id,
        block_size=2,
        logical_block_idx=1,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([14, 15], dtype=torch.int32)),
        token_ids=(5, 6),
    )
    _remove_free_slots(manager, [10, 11, 12, 13, 14, 15])
    manager.prefix_cache.insert_block(root)
    manager.prefix_cache.insert_block(referenced_child)
    manager.prefix_cache.insert_block(free_child)
    assert manager._num_free_slots == 84

    result = manager.prefix_cache_delete_subtree([1, 2])

    assert result["deleted_block_ids"] == [free_child_id.hex()]
    assert {item["reason"] for item in result["blocked_blocks"]} == {"referenced", "has_children"}
    assert manager._num_free_slots == 86
    assert root_id in manager.prefix_cache.blocks
    assert referenced_child_id in manager.prefix_cache.blocks
    assert free_child_id not in manager.prefix_cache.blocks


def test_standard_pending_slots_do_not_alias_free_stack_storage():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2])
    first_slot_view = manager.free_slots_stack[89:90]

    manager._record_prefix_materialization(seq, [1], first_slot_view)
    manager.free_slots_stack[89] = 777
    manager._record_prefix_materialization(seq, [2], torch.tensor([88], dtype=torch.int32))

    pending = manager.pending_prefix_blocks[seq.seq_id][0]
    assert pending.slots.tolist() == [89, 88]


def test_standard_admission_reserves_evictable_hit_blocks():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager._num_free_slots = 1
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id

    assert manager.prompt_admission_free_slots() == 3
    assert manager.prompt_admission_cost(seq) == 3

    block.ref_count = 1
    assert manager.prompt_admission_cost(seq) == 1


def test_standard_materializes_child_after_prefix_hit_with_parent_sensitive_id():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3, 4])
    root_id = manager.prefix_cache.stable_block_id([1, 2], None)
    root = PrefixCacheBlock(
        stable_block_id=root_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(root)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = root_id
    seq.prefix_cache_block_size = 2

    manager._record_prefix_materialization(seq, [3, 4], torch.tensor([20, 21], dtype=torch.int32))
    manager.on_forward_end([seq], is_prefill=True)

    child_id = manager.prefix_cache.stable_block_id([3, 4], root_id)
    child = manager.prefix_cache.get_block(child_id)
    assert child is not None
    assert child.parent_block_id == root_id
    assert child.logical_block_idx == 1
    assert child.ref_count == 1
    assert isinstance(child.payload, StandardPrefixBlockPayload)
    assert child.payload.token_slots.tolist() == [20, 21]
    assert [block.stable_block_id for block in manager.prefix_cache.get_chain(child_id, 2)] == [root_id, child_id]


def test_standard_decode_token_completes_pending_prefix_block_by_default():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    assert len(manager.prefix_cache) == 0
    assert manager.pending_prefix_blocks[seq.seq_id] == []

    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)
    manager._prepare_decode([seq])
    manager.on_forward_end([seq], is_prefill=False)

    block_id = manager.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = manager.prefix_cache.get_block(block_id)
    assert block is not None
    assert block.token_ids == (1, 2, 3, 4)
    assert block.logical_block_idx == 0
    assert block.ref_count == 1
    assert isinstance(block.payload, StandardPrefixBlockPayload)
    assert block.payload.token_slots.tolist() == [87, 88, 89, 86]
    assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 4)]


def test_standard_static_decode_padding_does_not_materialize_padded_rows():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)

    input_ids = torch.empty((4,), dtype=torch.int64)
    positions = torch.empty((4,), dtype=torch.int64)
    slot_mapping = torch.empty((4,), dtype=torch.int32)
    context_lens = torch.empty((4,), dtype=torch.int32)
    req_indices = torch.empty((4,), dtype=torch.int32)

    manager.prepare_decode_static([seq], input_ids, positions, slot_mapping, context_lens, req_indices)
    manager.on_forward_end([seq], is_prefill=False)

    assert input_ids.tolist() == [4, 4, 4, 4]
    assert positions.tolist() == [3, 3, 3, 3]
    assert slot_mapping.tolist()[1:] == [-1, -1, -1]
    assert context_lens.tolist() == [4, 4, 4, 4]
    assert req_indices.tolist() == [0, 0, 0, 0]
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.token_ids == (1, 2, 3, 4)
    assert block.payload.token_slots.numel() == 4
    assert manager._num_free_slots == 86


def test_standard_decode_materialized_block_can_seed_later_prefix_hit():
    manager = _make_standard_manager_for_prefix(block_size=4)
    first = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(first.seq_id, 3)
    manager._record_prefix_materialization(first, [1, 2, 3], prompt_slots)
    first.num_prefilled_tokens = first.num_prompt_tokens
    first.append_token(4)
    manager._prepare_decode([first])
    manager.on_forward_end([first], is_prefill=False)

    second = Sequence([1, 2, 3, 4, 5])
    manager.refresh_prefix_cache_hit(second)

    assert second.prefix_cache_hit_len == 4
    assert second.prefix_cache_hit_block_count == 1
    manager._attach_prefix_cache_if_needed(second)
    row_idx = manager.seq_id_to_row[second.seq_id]
    assert manager.row_seq_lens[row_idx] == 4
    assert manager.buffer_req_to_token_slots[row_idx, :4].tolist() == [87, 88, 89, 86]
    block = manager.prefix_cache.get_block(second.prefix_cache_hit_last_block_id)
    assert block is not None
    assert block.ref_count == 2


def test_standard_reset_prefix_cache_clears_warmup_blocks_and_restores_allocator():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)
    assert len(manager.prefix_cache) == 2
    assert manager._num_free_slots == 86

    manager.reset_prefix_cache()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90
    assert manager.free_slots_stack[:90].tolist() == list(range(90))


def test_standard_reset_after_warmup_restores_allocator_without_prefix_cache():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.enable_prefix_caching = False
    manager.prefix_cache = None
    manager.free_slots_stack[:90] = torch.tensor(list(range(86)) + [89, 88, 87, 86], dtype=torch.int32)

    manager.reset_after_warmup()

    assert manager._num_free_slots == 90
    assert manager.free_slots_stack[:90].tolist() == list(range(90))


def test_standard_reset_after_warmup_clears_prefix_cache_and_allocator():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)

    manager.reset_after_warmup()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90
    assert manager.free_slots_stack[:90].tolist() == list(range(90))


def test_quest_attach_pins_pages_and_free_seq_keeps_cached_page():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id
    seq.prefix_cache_block_size = 2
    seq.prefix_cache_method = "quest"

    manager._attach_prefix_cache_if_needed(seq)
    assert manager.row_seq_lens[0] == 2
    assert manager.buffer_req_to_page_slots[0, 0].item() == 5
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == [10, 11]
    assert block.ref_count == 1

    manager._allocate(seq.seq_id, 1)
    assert manager.row_seq_lens[0] == 3
    assert manager._num_free_pages == 8

    manager.free_seq(seq.seq_id)
    assert manager._num_free_pages == 9
    assert block.ref_count == 0

    evicted = manager.prefix_cache.evict_until_freeable(1)
    manager._free_prefix_cache_blocks(evicted)
    assert manager._num_free_pages == 10
    reused = manager._allocate(Sequence([4]).seq_id, 1)
    assert reused.tolist() == [10]


def test_quest_allocate_can_fill_partial_page_without_free_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    manager._num_free_pages = 1

    first = manager._allocate(seq.seq_id, 1)
    assert first.tolist() == [0]
    assert manager._num_free_pages == 0

    second = manager._allocate(seq.seq_id, 1)
    assert second.tolist() == [1]
    assert manager._num_free_pages == 0


def test_quest_prefill_step_capacity_counts_partial_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    manager._num_free_pages = 1
    manager._allocate(seq.seq_id, 1)
    manager._num_free_pages = 0

    assert manager.prefill_step_free_slots() == 1
    assert manager.prefill_step_free_slots_for(seq) == 1
    assert manager.prefill_step_reservation_cost(seq, 1) == 1
    assert manager.prefill_step_reservation_cost(Sequence([3]), 1) == 2
    assert manager.prefill_step_free_slots_for(Sequence([3])) == 0


def test_quest_decode_capacity_counts_requests_not_tokens():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    manager._num_free_pages = 1

    assert manager.decode_step_free_slots() == 2
    assert manager.decode_step_free_slots_for(seq) == 2
    assert manager.decode_step_reservation_cost(seq) == 2

    manager._allocate(seq.seq_id, 1)
    manager._num_free_pages = 0
    assert manager.decode_step_free_slots() == 1
    assert manager.decode_step_free_slots_for(seq) == 1
    assert manager.decode_step_reservation_cost(seq) == 1
    assert manager.decode_step_free_slots_for(Sequence([3])) == 0
    assert manager.decode_step_reservation_cost(Sequence([3])) == 2


def test_quest_materializes_pages_only_after_forward_end():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    slots = torch.tensor([4, 5], dtype=torch.int32)

    manager._record_prefix_materialization(seq, [1, 2], slots)
    assert len(manager.prefix_cache) == 0

    manager.on_forward_end([seq], is_prefill=True)
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.ref_count == 1
    assert not hasattr(block, "page_slot")
    assert not hasattr(block, "slots")
    assert isinstance(block.payload, QuestPrefixBlockPayload)
    assert block.payload.block_slot == 2
    assert block.payload.token_slots.tolist() == [4, 5]
    assert manager.seq_id_to_cached_pages[seq.seq_id] == {0}


def test_quest_decode_token_completes_pending_prefix_page_by_default():
    manager = _make_quest_manager_for_prefix(page_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    assert len(manager.prefix_cache) == 0

    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)
    manager._prepare_decode([seq])
    manager.on_forward_end([seq], is_prefill=False)

    block_id = manager.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = manager.prefix_cache.get_block(block_id)
    assert block is not None
    assert block.token_ids == (1, 2, 3, 4)
    assert block.logical_block_idx == 0
    assert block.ref_count == 1
    assert isinstance(block.payload, QuestPrefixBlockPayload)
    assert block.payload.block_slot == 9
    assert block.payload.token_slots.tolist() == [36, 37, 38, 39]
    assert manager.seq_id_to_cached_pages[seq.seq_id] == {0}


def test_quest_static_decode_padding_does_not_materialize_padded_rows():
    manager = _make_quest_manager_for_prefix(page_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)

    input_ids = torch.empty((4,), dtype=torch.int64)
    positions = torch.empty((4,), dtype=torch.int64)
    slot_mapping = torch.empty((4,), dtype=torch.int32)
    context_lens = torch.empty((4,), dtype=torch.int32)
    req_indices = torch.empty((4,), dtype=torch.int32)

    manager.prepare_decode_static([seq], input_ids, positions, slot_mapping, context_lens, req_indices)
    manager.on_forward_end([seq], is_prefill=False)

    assert input_ids.tolist() == [4, 4, 4, 4]
    assert positions.tolist() == [3, 3, 3, 3]
    assert slot_mapping.tolist()[1:] == [-1, -1, -1]
    assert context_lens.tolist() == [4, 4, 4, 4]
    assert req_indices.tolist() == [0, 0, 0, 0]
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.token_ids == (1, 2, 3, 4)
    assert isinstance(block.payload, QuestPrefixBlockPayload)
    assert block.payload.block_slot == 9
    assert block.payload.token_slots.tolist() == [36, 37, 38, 39]
    assert manager._num_free_pages == 9


def test_quest_decode_materialized_page_can_seed_later_prefix_hit():
    manager = _make_quest_manager_for_prefix(page_size=4)
    first = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(first.seq_id, 3)
    manager._record_prefix_materialization(first, [1, 2, 3], prompt_slots)
    first.num_prefilled_tokens = first.num_prompt_tokens
    first.append_token(4)
    manager._prepare_decode([first])
    manager.on_forward_end([first], is_prefill=False)

    second = Sequence([1, 2, 3, 4, 5])
    manager.refresh_prefix_cache_hit(second)

    assert second.prefix_cache_hit_len == 4
    assert second.prefix_cache_hit_block_count == 1
    manager._attach_prefix_cache_if_needed(second)
    row_idx = manager.seq_id_to_row[second.seq_id]
    assert manager.row_seq_lens[row_idx] == 4
    assert manager.buffer_req_to_page_slots[row_idx, 0].item() == 9
    assert manager.buffer_req_to_token_slots[row_idx, :4].tolist() == [36, 37, 38, 39]
    block = manager.prefix_cache.get_block(second.prefix_cache_hit_last_block_id)
    assert block is not None
    assert block.ref_count == 2


def test_quest_reset_prefix_cache_clears_warmup_pages_and_restores_allocator():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)
    assert len(manager.prefix_cache) == 2
    assert manager._num_free_pages == 8

    manager.reset_prefix_cache()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_pages == 10
    assert manager.free_pages_stack[:10].tolist() == list(range(10))


def test_quest_reset_after_warmup_restores_allocator_without_prefix_cache():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager.enable_prefix_caching = False
    manager.prefix_cache = None
    manager.free_pages_stack[:10] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=torch.int32)

    manager.reset_after_warmup()

    assert manager._num_free_pages == 10
    assert manager.free_pages_stack[:10].tolist() == list(range(10))


def test_quest_reset_after_warmup_clears_prefix_cache_and_allocator():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)

    manager.reset_after_warmup()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_pages == 10
    assert manager.free_pages_stack[:10].tolist() == list(range(10))


def test_quest_safe_delete_releases_payload_block_slot():
    manager = _make_quest_manager_for_prefix(page_size=2)
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    assert manager._num_free_pages == 9

    result = manager.prefix_cache_delete_subtree([1, 2])

    assert result["deleted_block_ids"] == [stable_block_id.hex()]
    assert manager._num_free_pages == 10
    assert stable_block_id not in manager.prefix_cache.blocks


def test_quest_admission_is_page_aligned_and_reserves_hit_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager._num_free_pages = 1
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id

    assert manager.prompt_admission_free_slots() == 4
    assert manager.prompt_admission_cost(seq) == 4

    block.ref_count = 1
    assert manager.prompt_admission_cost(seq) == 2
