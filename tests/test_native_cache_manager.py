"""CPU-owned native CacheManager/MemoryOracle/radix lifecycle integration."""

from types import SimpleNamespace

import pytest
import torch

from sparsevllm.engine.cache_manager import base
from sparsevllm.engine.cache_manager.methods.deepseek_v4_manager import DeepseekV4CacheManager
from sparsevllm.engine.runtime_state import RuntimeState
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import PREFILL_POLICY_ALL_CHUNKED


def make_manager(monkeypatch):
    monkeypatch.setattr(base.platforms, "_current_platform", SimpleNamespace(get_device=lambda _: torch.device("cpu")))
    parallel = SimpleNamespace(world_rank=0, world_size=1, attn_tp_rank=0, attn_tp_size=1,
                               moe_ep_rank=0, moe_ep_size=1, attn_dp_rank=0, attn_dp_size=1)
    hf = SimpleNamespace(num_hidden_layers=3, num_key_value_heads=1, head_dim=8, index_head_dim=4,
                         sliding_window=8, compress_ratios=(0, 4, 128), dtype=torch.bfloat16, model_type="deepseek_v4")
    config = SimpleNamespace(hf_config=hf, max_model_len=512, max_num_seqs_in_gpu=3,
                             runtime_layout=SimpleNamespace(num_kv_layers=3, kv_num_heads=(1, 1, 1),
                                                             kv_head_dims=(8, 8, 8), linear_attention_layer_indices=()),
                             enable_prefix_caching=True, prefix_cache_block_size=16, prefix_cache_max_blocks=4,
                             max_num_batched_tokens=32, max_decoding_seqs=3, max_num_seqs_in_batch=3,
                             engine_prefill_chunk_size=16, prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
                             sparse_method="", eos=-1, sink_keep_tokens=0, recent_keep_tokens=0, decode_keep_tokens=0)
    return config, DeepseekV4CacheManager(config, parallel, allocation_budget_bytes=128 * 1024)


def test_scheduler_attaches_committed_prefix_and_abort_releases_pending_lookup(monkeypatch):
    config, manager = make_manager(monkeypatch)
    runtime = RuntimeState(config, manager)
    parent = Sequence(list(range(48)))
    for start in range(0, 48, 16):
        parent.num_prefilled_tokens, parent.current_chunk_size = start, 16
        runtime.prepare_step([parent], True)
        runtime.on_forward_end([parent], True)
    runtime.free_seq(parent.seq_id)
    scheduler = Scheduler(config, runtime)
    child = Sequence(list(range(48)) + [999])
    scheduler.add(child)
    scheduled, prefill, _ = scheduler.schedule()
    assert scheduled == [child] and prefill
    assert child.num_prefilled_tokens == 48 and child.current_chunk_size == 1
    ids, positions, cu = runtime.prepare_step(scheduled, True)
    assert ids.tolist() == [999] and positions.tolist() == [48] and cu.tolist() == [0, 1]
    runtime.on_forward_end(scheduled, True)
    assert manager.pool.live[child.seq_id].length == 49
    runtime.free_seq(child.seq_id)
    waiting = Sequence(list(range(48)) + [1000])
    scheduler = Scheduler(config, runtime)
    scheduler.add(waiting)
    manager.refresh_prefix_cache_hit(waiting)
    assert waiting.seq_id in manager.prefix.hits
    assert not scheduler.abort(waiting.seq_id)
    assert waiting.seq_id not in manager.prefix.hits
    assert waiting.seq_id not in manager.prefix.paths
    assert manager.prefix.evict_for_capacity({}, snapshot_rows=4)
    assert not manager.pool.live and not manager.pool.snapshots
    assert all(allocator.free_count == len(allocator.refs) for allocator in manager.pool.slots.values())


def test_unfinished_forward_cannot_overlap_another_reservation(monkeypatch):
    _, manager = make_manager(monkeypatch)
    seq = Sequence(list(range(32)))
    seq.current_chunk_size = 16
    manager.prepare_step([seq], True)
    with pytest.raises(RuntimeError, match="finish or abort"):
        manager.prepare_step([seq], True)
    manager.abort_step()
    manager.free_seq(seq.seq_id)
    assert not manager.pool.live and not manager.pool.snapshots
    assert all(allocator.free_count == len(allocator.refs) for allocator in manager.pool.slots.values())


def test_native_prefix_controls_preserve_live_child_storage(monkeypatch):
    _, manager = make_manager(monkeypatch)
    parent = Sequence(list(range(48)))
    for start in range(0, 48, 16):
        parent.num_prefilled_tokens, parent.current_chunk_size = start, 16
        manager.prepare_step([parent], True)
        manager.on_forward_end([parent], True)
    manager.free_seq(parent.seq_id)
    child = Sequence(list(range(48)) + [999])
    assert manager.prefix_cache_match(child.token_ids)["matched_tokens"] == 48
    manager.refresh_prefix_cache_hit(child)
    assert manager.prefix_cache_delete_subtree(list(range(16)))["deleted_block_count"] == 0
    child.num_prefilled_tokens, child.current_chunk_size = 48, 1
    manager.prepare_step([child], True)
    manager.on_forward_end([child], True)
    manager.prefix_cache_set_eviction_priority(list(range(16)), priority=-1)
    assert manager.prefix_cache_delete_subtree(list(range(16)))["deleted_block_count"] == 0
    manager.prefix_cache_set_eviction_priority(list(range(16)), priority=0)
    assert manager.prefix_cache_delete_subtree(list(range(16)))["deleted_block_count"] > 0
    assert not manager.pool.snapshots
    assert manager.prefix_cache_match(child.token_ids)["matched_tokens"] == 0
    live = manager.pool.live[child.seq_id]
    assert live.length == 49
    for ratio, slots in manager.pool.slots.items():
        assert (slots.refs[manager.pool.host_tables[ratio][live.row, :49 // ratio]] > 0).all()
    manager.free_seq(child.seq_id)
    assert all(slots.free_count == len(slots.refs) for slots in manager.pool.slots.values())


def test_native_startup_history_reserves_and_releases_all_compressed_pools(monkeypatch):
    from sparsevllm.engine.startup.native_history import NativePrefillHistoryCacheManager

    config, original = make_manager(monkeypatch)
    config.enable_prefix_caching = False
    config.num_kvcache_slots = 160
    manager = NativePrefillHistoryCacheManager(config, original.parallel_context)
    seq = Sequence(list(range(132)))
    seq.num_prefilled_tokens, seq.current_chunk_size = 129, 3
    manager.seed_history(seq)
    row = manager.pool.live[seq.seq_id].row
    for ratio, table in manager.pool.host_tables.items():
        assert (table[row, :129 // ratio] >= 0).all()
        assert (table[row, 129 // ratio:] == -1).all()
    manager.prepare_step([seq], True)
    manager.on_forward_end([seq], True)
    assert manager.pool.live[seq.seq_id].length == 132
    manager.reset_after_warmup()
    assert not manager.pool.live
    assert all(allocator.free_count == len(allocator.refs) for allocator in manager.pool.slots.values())


def test_native_startup_budget_covers_physical_rows_and_prefix_state(monkeypatch):
    from sparsevllm.engine.startup.capacity import profiling_kv_budget_bytes

    config, original = make_manager(monkeypatch)
    config.attention_cache_layout = "shared_kv"
    config.parallel_topology = SimpleNamespace(attn_tp_size=1)
    config.startup_cache_phase = "profiling"
    budget = profiling_kv_budget_bytes(config, 64)
    manager = DeepseekV4CacheManager(config, original.parallel_context, allocation_budget_bytes=budget)
    assert manager.pool.allocated_bytes() <= budget
    assert manager.pool.free_snapshot_rows
    assert all(allocator.free_count >= 64 // ratio for ratio, allocator in manager.pool.slots.items())
