from sparsevllm.engine.startup import (
    DeviceMemorySnapshot,
    MemoryProfileMeasurement,
    build_startup_capacity_decision,
    validate_production_kv_records,
)


def _measurement(*, consumed: int = 0, transient: int = 0):
    return MemoryProfileMeasurement(
        consumed_bytes=consumed,
        transient_peak_bytes=transient,
    )


def test_capacity_decision_uses_each_rank_profile_and_global_minimum_budget():
    persistent = [
        {
            "world_rank": 0,
            "snapshot": DeviceMemorySnapshot(700, 1000, 300, 300),
            "post_graph_release_snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
        },
        {
            "world_rank": 1,
            "snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
            "post_graph_release_snapshot": DeviceMemorySnapshot(640, 1000, 360, 360),
        },
    ]
    prefill = [
        {"world_rank": 0, "measurement": _measurement(transient=100)},
        {"world_rank": 1, "measurement": _measurement(transient=80)},
    ]
    graph = [
        {
            "world_rank": 0,
            "after": DeviceMemorySnapshot(600, 1000, 400, 400),
            "measurement": _measurement(consumed=50),
        },
        {
            "world_rank": 1,
            "after": DeviceMemorySnapshot(580, 1000, 420, 420),
            "measurement": _measurement(consumed=60),
        },
    ]
    decode = [
        {"world_rank": 0, "measurement": _measurement(transient=70)},
        {"world_rank": 1, "measurement": _measurement(transient=120)},
    ]

    decision = build_startup_capacity_decision(
        prefill_records=prefill,
        graph_records=graph,
        decode_records=decode,
        persistent_records=persistent,
        gpu_memory_utilization=0.9,
    )

    assert [plan.capacity.local_kv_budget_bytes for plan in decision.rank_plans] == [450, 370]
    assert decision.selected_kv_budget_bytes == 370
    assert decision.limiting_rank == 1


def test_production_kv_records_must_match_across_ranks():
    records = [
        {"world_rank": 0, "num_kvcache_slots": 10},
        {"world_rank": 1, "num_kvcache_slots": 9},
    ]

    try:
        validate_production_kv_records(records)
    except RuntimeError as exc:
        assert "differs across ranks" in str(exc)
    else:  # pragma: no cover - contract failure path.
        raise AssertionError("mismatched production capacities were accepted")
