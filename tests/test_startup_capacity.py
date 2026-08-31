from __future__ import annotations

from types import SimpleNamespace

import torch

from sparsevllm.engine.cache_manager.storage import CacheLayout
from sparsevllm.engine.startup import (
    KVCapacityPlan,
    StartupMemoryProfile,
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
    )


def test_capacity_plan_uses_larger_runtime_peak_and_external_headroom():
    profile = StartupMemoryProfile(
        total_bytes=1000,
        persistent_bytes=300,
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
