from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sparsevllm.engine.cache_manager.storage import CacheLayout
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.runtime_state import RuntimeState
from sparsevllm.engine.startup import (
    KVCapacityPlan,
    StartupMemoryProfile,
    feasible_startup_graph_plan,
    profiling_kv_budget_bytes,
    profiling_kv_slots,
    profiling_prefill_prompt_lengths,
)
from sparsevllm.models.layout import RuntimeLayout


def _config(*, sparse_method: str = ""):
    layout = RuntimeLayout.dense(4)
    layout = RuntimeLayout(
        **{
            **layout.__dict__,
            "kv_num_heads": (8, 8, 8, 8),
            "kv_head_dims": (128, 128, 128, 128),
        }
    )
    return SimpleNamespace(
        sparse_method=sparse_method,
        attention_cache_layout=CacheLayout.EXPLICIT_KV,
        runtime_layout=layout,
        parallel_topology=SimpleNamespace(attention_tp_size=2),
        hf_config=SimpleNamespace(
            torch_dtype=torch.bfloat16,
            num_key_value_heads=8,
            num_attention_heads=32,
            hidden_size=4096,
            head_dim=128,
        ),
        quest_chunk_size=16,
        max_num_batched_tokens=16,
        max_num_seqs_in_batch=4,
        max_num_seqs_in_gpu=8,
        max_decoding_seqs=4,
        engine_prefill_chunk_size=8,
        max_model_len=32,
        decode_graph_startup_capture=False,
        sink_keep_tokens=2,
        decode_keep_tokens=8,
        recent_keep_tokens=6,
    )


def test_capacity_plan_uses_larger_runtime_peak_and_external_headroom():
    profile = StartupMemoryProfile(
        total_bytes=1000,
        persistent_bytes=300,
        runtime_persistent_bytes=0,
        profile_persistent_growth_bytes=0,
        prefill_transient_bytes=120,
        decode_transient_bytes=80,
        cuda_graph_bytes=50,
    )

    plan = KVCapacityPlan.from_profile(profile, 0.9)

    assert plan.target_bytes == 900
    assert plan.safety_headroom_bytes == 100
    assert plan.runtime_transient_bytes == 120
    assert plan.local_kv_budget_bytes == 430


def test_explicit_profiling_budget_uses_tp_local_kv_shape():
    config = _config()

    budget = profiling_kv_budget_bytes(config, 10)

    expected_bytes_per_slot = 4 * 2 * 4 * 128 * 2
    expected_row_mapping = 8 * 32 * 4
    assert budget == 10 * (expected_bytes_per_slot + 4) + expected_row_mapping


def test_explicit_profiling_budget_uses_declared_head_dim_without_layout_shapes():
    config = _config()
    config.runtime_layout = RuntimeLayout.dense(4)

    budget = profiling_kv_budget_bytes(config, 10)

    expected_bytes_per_slot = 4 * 2 * 4 * 128 * 2
    expected_row_mapping = 8 * 32 * 4
    assert budget == 10 * (expected_bytes_per_slot + 4) + expected_row_mapping


def test_quest_profiling_budget_includes_page_metadata():
    config = _config(sparse_method="quest")

    budget = profiling_kv_budget_bytes(config, 17)

    bytes_per_slot = 4 * 2 * 4 * 128 * 2
    fixed_metadata = 8 * 32 * 4 + 8 * 2 * 4 + 16 * (4 + 8)
    assert budget == 32 * bytes_per_slot + 2 * (bytes_per_slot + 4) + fixed_metadata


def test_prefill_profile_fills_token_chunk_and_batch_limits_together():
    lengths = profiling_prefill_prompt_lengths(_config())

    assert lengths == (8, 6, 1, 1)
    assert sum(lengths) == 16


def test_profiling_kv_slots_cover_runtime_steps_not_maximum_context():
    config = _config()
    config.max_model_len = 1_000_000

    assert profiling_kv_slots(config) == 24


def test_quest_profiling_slots_round_each_request_to_a_page():
    config = _config(sparse_method="quest")

    assert profiling_kv_slots(config) == 64


def test_production_graph_plan_skips_families_larger_than_final_kv():
    config = _config(sparse_method="quest")
    config.sink_keep_tokens = 2
    config.decode_keep_tokens = 8
    config.recent_keep_tokens = 6
    plan = [(4, 32, True), (2, 32, True), (4, 16, False)]

    class AdmissionOracle:
        def startup_batch_fits(self, prompt_lengths, *, max_tokens):
            full_layers = sum(int(length) + int(max_tokens) for length in prompt_lengths)
            centers = sum((int(length) + 7) // 8 for length in prompt_lengths)
            return full_layers <= 32 and centers <= 4

    feasible, skipped = feasible_startup_graph_plan(
        config,
        plan,
        AdmissionOracle(),
    )

    assert feasible == [(4, 16, False)]
    assert skipped == [(4, 32, True), (2, 32, True)]


def test_explicit_budget_still_resolves_mixed_kv_and_recurrent_prefix_capacity():
    manager = object.__new__(StandardCacheManager)
    manager.config = SimpleNamespace(
        resolved_prefix_cache_mode="radix",
        prefix_recurrent_bytes_per_block=40,
        prefix_cache_max_blocks=None,
        prefix_cache_block_size=4,
        hf_config=SimpleNamespace(),
    )
    manager.allocation_budget_bytes = 1_000
    manager.attention_cache_bytes_per_slot_per_layer = lambda: 10
    manager._kv_allocation_bytes_per_prefix_block = lambda _slot_bytes: 60

    available, slot_bytes = manager._get_available_slots_info()

    assert available == 600
    assert slot_bytes == 10
    assert manager.config.prefix_cache_max_blocks == 10
    assert manager.config.prefix_recurrent_capacity_bytes == 400
    assert manager.config.prefix_kv_block_capacity == 10


def test_startup_batch_feasibility_uses_all_memory_oracle_budgets():
    class MultiBudgetManager:
        def scheduler_capacity_snapshot(self):
            return nullcontext()

        def prompt_admission_budgets(self, _waiting, _chunk_size):
            return {"full_layers": 100, "centers": 2}

        def prompt_admission_costs(self, seq):
            return {
                "full_layers": int(seq.num_prompt_tokens + seq.max_tokens),
                "centers": 1,
            }

        def prompt_logical_reservation_cost(self, seq):
            return int(seq.num_prompt_tokens)

        def prompt_admission_free_slots(self):
            return 100

    runtime = RuntimeState(
        SimpleNamespace(engine_prefill_chunk_size=8, max_num_seqs_in_gpu=8),
        MultiBudgetManager(),
    )

    assert runtime.startup_batch_fits((16, 16), max_tokens=2)
    assert not runtime.startup_batch_fits((16, 16, 16), max_tokens=2)
