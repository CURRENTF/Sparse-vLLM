from sparsevllm.engine.startup import (
    DeviceMemorySnapshot,
    MemoryProfileMeasurement,
    build_startup_capacity_decision,
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
        },
        {
            "world_rank": 1,
            "snapshot": DeviceMemorySnapshot(650, 1000, 350, 350),
        },
    ]
    prefill = [
        {"world_rank": 0, "measurement": _measurement(transient=100)},
        {"world_rank": 1, "measurement": _measurement(transient=80)},
    ]
    graph = [
        {"world_rank": 0, "measurement": _measurement(consumed=50)},
        {"world_rank": 1, "measurement": _measurement(consumed=60)},
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
