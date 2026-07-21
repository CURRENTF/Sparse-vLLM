from unittest.mock import patch

import pytest

from scripts.validation import benchmark_minimax_m2_fp8 as benchmark


def test_minimax_benchmark_parses_and_validates_csv_values():
    assert benchmark._parse_int_csv("1, 8,32") == [1, 8, 32]
    assert benchmark._parse_nonnegative_int_csv("0, 1,3") == [0, 1, 3]
    assert benchmark._parse_str_csv("Reference, routed") == [
        "reference",
        "routed",
    ]
    with pytest.raises(ValueError, match="positive"):
        benchmark._parse_int_csv("1,0")
    with pytest.raises(ValueError, match="non-negative"):
        benchmark._parse_nonnegative_int_csv("0,-1")
    with pytest.raises(ValueError, match="non-empty"):
        benchmark._parse_str_csv(" , ")


def test_minimax_benchmark_refuses_busy_devices():
    devices = [
        {
            "index": 0,
            "name": "busy",
            "memory_used_mib": 1024,
            "utilization_percent": 90,
        },
        {
            "index": 1,
            "name": "idle",
            "memory_used_mib": 1,
            "utilization_percent": 0,
        },
    ]
    with patch.object(benchmark, "_query_gpus", return_value=devices):
        selected, observed = benchmark._select_idle_gpu(
            None,
            max_memory_used_mib=512,
            max_utilization_percent=5,
        )
        assert selected["index"] == 1
        assert observed == devices
        with pytest.raises(RuntimeError, match="busy"):
            benchmark._select_idle_gpu(
                0,
                max_memory_used_mib=512,
                max_utilization_percent=5,
            )


def test_minimax_benchmark_report_records_cold_and_steady_metrics(tmp_path):
    path = tmp_path / "report.md"
    benchmark._write_report(
        path,
        aggregate_status="success",
        run_config={
            "model_revision": benchmark.MODEL_REVISION,
            "git": {"commit": "abc123"},
            "seed": 27,
        },
        records=[
            {
                "case_id": "tokens_32_local_experts_32",
                "backend": "routed",
                "status": "success",
                "cold_start_ms": 12.0,
                "median_ms": 1.5,
                "p95_ms": 1.7,
                "peak_memory_bytes": 2 * 1024**3,
                "kernel_launches": 5,
                "correctness_vs_reference": {"relative_l2_error": 0.01},
            }
        ],
    )

    report = path.read_text(encoding="utf-8")
    assert "tokens_32_local_experts_32" in report
    assert "12.000" in report
    assert "1.500" in report
    assert "1.000000e-02" in report


def test_minimax_benchmark_aggregate_preserves_earlier_failure():
    assert benchmark._aggregate_status(
        [
            {"status": "metric_failed"},
            {"status": "success"},
        ]
    ) == "metric_failed"
