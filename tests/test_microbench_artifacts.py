import json
from types import SimpleNamespace

import pytest

from benchmark.microbench import (
    _benchmark_sparse_method,
    _decode_cuda_graph_status,
    _record_child_exit_failure,
    _write_output_dir,
)


def test_decode_cuda_graph_status_records_execution_counters():
    graph = object()
    runner = SimpleNamespace(
        _graphs={"bs4": SimpleNamespace(graph=graph)},
        last_state_key="bs4",
        capture_count=2,
        replay_count=17,
        eager_static_count=3,
        force_eager_count=1,
    )
    llm = SimpleNamespace(
        model_runner=SimpleNamespace(decode_cuda_graph_runner=runner),
        config=SimpleNamespace(decode_cuda_graph=True),
    )

    assert _decode_cuda_graph_status(llm) == {
        "decode_cuda_graph_configured": True,
        "decode_cuda_graph_runner_initialized": True,
        "decode_cuda_graph_state_count": 1,
        "decode_cuda_graph_graph_count": 1,
        "decode_cuda_graph_capture_count": 2,
        "decode_cuda_graph_replay_count": 17,
        "decode_cuda_graph_eager_static_count": 3,
        "decode_cuda_graph_force_eager_count": 1,
        "decode_cuda_graph_last_state_key": "bs4",
        "decode_cuda_graph_active": True,
    }


def test_benchmark_sparse_method_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported benchmark sparse method"):
        _benchmark_sparse_method("typo")


def test_nonzero_child_exit_overrides_partial_success_row():
    results = {
        ("h2o", 16, 1): {
            "method": "h2o",
            "length": 16,
            "batch_size": 1,
            "status": "SUCCESS",
            "prefill_tp": 123.0,
        }
    }

    _record_child_exit_failure(
        results,
        method="h2o",
        length=16,
        batch_size=1,
        exitcode=7,
        synchronize_step_timing=True,
    )

    row = results[("h2o", 16, 1)]
    assert row["status"] == "FAILED"
    assert row["child_exitcode"] == 7
    assert row["child_partial_status"] == "SUCCESS"
    assert row["prefill_tp"] == 123.0


def test_output_metadata_records_step_timing_mode(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_len=8,
        temperature=0.0,
        top_p=1.0,
        synchronize_step_timing=True,
        model_path="test-model",
        methods="h2o",
        lengths="16",
        batch_sizes="1",
        hyper_params_dict={},
    )
    monkeypatch.setattr(
        "benchmark.microbench._git_metadata",
        lambda: {"git_commit": "test", "git_branch": "test", "git_dirty": False},
    )

    _write_output_dir(args, [{"status": "SUCCESS", "length": 16}])

    run_info = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    aggregate = json.loads(
        (tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    performance = json.loads(
        (tmp_path / "performance.jsonl").read_text(encoding="utf-8")
    )
    assert run_info["synchronize_step_timing"] is True
    assert aggregate["synchronize_step_timing"] is True
    assert aggregate["records"][0]["synchronize_step_timing"] is True
    assert performance["synchronize_step_timing"] is True
