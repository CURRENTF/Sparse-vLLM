import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.prefix_cache import (
    PrefixCacheBlock,
    PrefixCacheIndex,
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


def _insert_tokens(index: PrefixCacheIndex, token_ids: list[int]) -> bytes:
    parent_key = None
    last_key = None
    for logical_idx, start in enumerate(range(0, len(token_ids), index.block_size)):
        block_tokens = token_ids[start: start + index.block_size]
        key = index.hash_block(block_tokens, parent_key)
        block = PrefixCacheBlock(
            key=key,
            parent_key=parent_key,
            block_size=index.block_size,
            logical_block_idx=logical_idx,
            token_ids=tuple(block_tokens),
        )
        index.insert_block(block)
        parent_key = key
        last_key = key
    assert last_key is not None
    return last_key


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
    fingerprint = build_prefix_cache_fingerprint(cfg, block_size)
    manager = object.__new__(StandardCacheManager)
    manager.config = cfg
    manager.enable_prefix_caching = True
    manager.prefix_cache_block_size = block_size
    manager.prefix_cache = PrefixCacheIndex(block_size=block_size, fingerprint=fingerprint)
    manager.buffer_req_to_token_slots = torch.zeros((2, 16), dtype=torch.int32)
    manager.free_slots_stack = torch.arange(100, dtype=torch.int32)
    manager._num_free_slots = 90
    manager.seq_id_to_row = {}
    manager.free_rows = deque([0, 1])
    manager.row_seq_lens = np.zeros((2,), dtype=np.int32)
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_materialized_blocks = {}
    manager.seq_id_to_cached_ranges = {}
    manager.prefix_runtime_states = {}
    manager.pending_prefix_blocks = {}
    return manager


def test_usable_prefix_cache_tokens_leaves_logits_work():
    assert usable_prefix_cache_tokens(128, 16) == 112
    assert usable_prefix_cache_tokens(129, 16) == 128
    assert usable_prefix_cache_tokens(15, 16) == 0
    assert usable_prefix_cache_tokens(1, 16) == 0


def test_prefix_cache_hash_is_stable_and_parent_sensitive():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp)

    first = index.hash_block([1, 2, 3, 4], None)
    assert first == index.hash_block([1, 2, 3, 4], None)
    assert first != index.hash_block([1, 2, 3, 5], None)
    assert index.hash_block([5, 6, 7, 8], first) != index.hash_block([5, 6, 7, 8], None)


def test_prefix_cache_fingerprint_isolates_salt_and_method():
    vanilla = build_prefix_cache_fingerprint(_cfg(method="", salt="a"), 4)
    salted = build_prefix_cache_fingerprint(_cfg(method="", salt="b"), 4)
    omnikv = build_prefix_cache_fingerprint(_cfg(method="omnikv", salt="a"), 4)
    quest = build_prefix_cache_fingerprint(_cfg(method="quest", salt="a"), 4)

    assert vanilla != salted
    assert vanilla != omnikv
    assert omnikv != quest


def test_lookup_returns_longest_full_block_prefix():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp)
    last_key = _insert_tokens(index, list(range(8)))

    hit_len, hit_last_key, hit_blocks = index.lookup_longest_prefix(
        list(range(12)),
        max_usable_tokens=usable_prefix_cache_tokens(12, 4),
    )

    assert hit_len == 8
    assert hit_last_key == last_key
    assert hit_blocks == 2
    chain = index.get_chain(hit_last_key, hit_blocks)
    assert [block.logical_block_idx for block in chain] == [0, 1]


def test_lookup_does_not_touch_lru_state():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp)
    last_key = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_chain(last_key, 1)[0]
    last_access = block.last_access

    index.lookup_longest_prefix([1, 2, 3, 4, 5], max_usable_tokens=4)

    assert block.last_access == last_access


def test_leaf_only_eviction_preserves_parent_until_child_is_removed():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, list(range(8)))

    evicted = index.evict_until_freeable(1)
    assert [block.logical_block_idx for block in evicted] == [1]
    assert index.evictable_blocks() == 1

    evicted = index.evict_until_freeable(1)
    assert [block.logical_block_idx for block in evicted] == [0]
    assert len(index) == 0


def test_referenced_blocks_are_not_evictable():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp)
    last_key = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_chain(last_key, 1)[0]
    block.ref_count = 1

    assert index.evict_until_freeable(1) == []
    block.ref_count = 0
    assert index.evict_until_freeable(1) == [block]


def test_max_blocks_requires_explicit_capacity_before_insert():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp, max_blocks=1)
    _insert_tokens(index, [1, 2, 3, 4])

    key = index.hash_block([5, 6, 7, 8], None)
    block = PrefixCacheBlock(key=key, parent_key=None, block_size=4, logical_block_idx=0)
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        index.insert_block(block)

    evicted = index.ensure_insert_capacity(1)
    assert len(evicted) == 1
    assert index.insert_block(block) is block


def test_get_chain_fails_fast_on_incomplete_chain():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = PrefixCacheIndex(block_size=4, fingerprint=fp)
    last_key = _insert_tokens(index, list(range(8)))
    parent = index.get_chain(last_key, 2)[0]
    del index._blocks[parent.key]

    with pytest.raises(RuntimeError, match="incomplete"):
        index.get_chain(last_key, 2)


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
    with pytest.raises(ValueError, match="decode_blocks"):
        _make_config(enable_prefix_caching=True, prefix_cache_cache_decode_blocks=True)
    with pytest.raises(ValueError, match="decode_cuda_graph"):
        _make_config(enable_prefix_caching=True, decode_cuda_graph=True)
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


def test_standard_attach_pins_prefix_slots_and_free_seq_keeps_cached_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3])
    key = manager.prefix_cache.hash_block([1, 2], None)
    block = PrefixCacheBlock(
        key=key,
        parent_key=None,
        block_size=2,
        logical_block_idx=0,
        slots=torch.tensor([10, 11], dtype=torch.int32),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_blocks = 1
    seq.prefix_cache_hit_last_key = key
    seq.prefix_cache_block_size = 2
    seq.prefix_cache_method = ""

    manager._attach_prefix_cache_if_needed(seq)
    assert manager.row_seq_lens[0] == 2
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == [10, 11]
    assert block.ref_count == 1

    manager._allocate(seq.seq_id, 1)
    assert manager.row_seq_lens[0] == 3
    assert manager._num_free_slots == 89

    manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == 90
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
    block = next(iter(manager.prefix_cache._blocks.values()))
    assert block.ref_count == 1
    assert block.slots.tolist() == [20, 21]
    assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 2)]


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
    key = manager.prefix_cache.hash_block([1, 2], None)
    block = PrefixCacheBlock(
        key=key,
        parent_key=None,
        block_size=2,
        logical_block_idx=0,
        slots=torch.tensor([10, 11], dtype=torch.int32),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_blocks = 1
    seq.prefix_cache_hit_last_key = key

    assert manager.prompt_admission_free_slots() == 3
    assert manager.prompt_admission_cost(seq) == 3

    block.ref_count = 1
    assert manager.prompt_admission_cost(seq) == 1
