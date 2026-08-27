import pytest

from sparsevllm.utils.profiler import Profiler


def test_profiler_snapshot_is_serializable_and_reports_average():
    instance = Profiler()
    instance.times["moe_router"] = 0.25
    instance.counts["moe_router"] = 2

    assert instance.snapshot() == {
        "moe_router": {
            "calls": 2,
            "total_s": 0.25,
            "avg_ms": 125.0,
            "p50_ms": 125.0,
            "p95_ms": 125.0,
            "p99_ms": 125.0,
        }
    }


def test_profiler_snapshot_reports_interpolated_percentiles(monkeypatch):
    timestamps = iter((1.0, 1.001, 2.0, 2.003, 3.0, 3.009))
    monkeypatch.setenv("SPARSEVLLM_PLATFORM", "cpu")
    monkeypatch.setattr("sparsevllm.utils.profiler.time.perf_counter", lambda: next(timestamps))
    instance = Profiler()
    instance.enabled = True

    for _ in range(3):
        with instance.record("decode_prepare"):
            pass

    entry = instance.snapshot()["decode_prepare"]
    assert entry["calls"] == 3
    assert entry["avg_ms"] == pytest.approx(13.0 / 3.0)
    assert entry["p50_ms"] == pytest.approx(3.0)
    assert entry["p95_ms"] == pytest.approx(8.4)
    assert entry["p99_ms"] == pytest.approx(8.88)
