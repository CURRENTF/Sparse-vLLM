import json
from types import SimpleNamespace

import pytest

from benchmark.microbench import (
    _apply_prefill_policy_defaults,
    _artifact_records,
    _benchmark_sparse_method,
    _decode_cuda_graph_status,
    _record_child_exit_failure,
    _resolved_engine_config,
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


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("vanilla", "vanilla"),
        ("h2o", "h2o"),
        ("attention-sink", "streamingllm"),
        ("r-kv", "rkv"),
        ("deltakv-less-memory", "deltakv"),
    ],
)
def test_benchmark_sparse_method_preserves_runtime_method(method, expected):
    assert _benchmark_sparse_method(method) == expected


def test_benchmark_sparse_method_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported benchmark sparse method"):
        _benchmark_sparse_method("typo")


@pytest.mark.parametrize("method", ["vanilla", "snapkv", "quest"])
def test_microbench_defaults_all_chunked_methods_to_96k(method):
    hyper_params = {}

    _apply_prefill_policy_defaults(hyper_params, method)

    assert hyper_params == {"engine_prefill_chunk_size": 96 * 1024}


@pytest.mark.parametrize(
    "method",
    ["pyramidkv", "deltakv", "deltakv-less-memory-cudagraph"],
)
def test_microbench_defaults_long_prefill_methods_to_64k_boundary(method):
    hyper_params = {}

    _apply_prefill_policy_defaults(hyper_params, method)

    assert hyper_params == {"long_prefill_offload_threshold": 64 * 1024}


def test_microbench_preserves_explicit_policy_specific_prefill_controls():
    all_chunked = {"engine_prefill_chunk_size": 8192}
    long_prefill = {
        "engine_prefill_chunk_size": 8192,
        "long_prefill_offload_threshold": 65536,
    }

    _apply_prefill_policy_defaults(all_chunked, "vanilla")
    _apply_prefill_policy_defaults(long_prefill, "pyramidkv")

    assert all_chunked == {"engine_prefill_chunk_size": 8192}
    assert long_prefill == {
        "engine_prefill_chunk_size": 8192,
        "long_prefill_offload_threshold": 65536,
    }


def test_microbench_all_chunked_ignores_long_prefill_boundary():
    hyper_params = {"long_prefill_offload_threshold": 65536}

    _apply_prefill_policy_defaults(hyper_params, "vanilla")

    assert hyper_params == {"engine_prefill_chunk_size": 96 * 1024}


def test_microbench_long_prefill_accepts_chunk_without_explicit_boundary():
    hyper_params = {"engine_prefill_chunk_size": 8192}

    _apply_prefill_policy_defaults(hyper_params, "pyramidkv")

    assert hyper_params == {
        "engine_prefill_chunk_size": 8192,
        "long_prefill_offload_threshold": 64 * 1024,
    }


def test_microbench_rejects_chunk_larger_than_long_prefill_boundary():
    hyper_params = {
        "engine_prefill_chunk_size": 65537,
        "long_prefill_offload_threshold": 65536,
    }

    with pytest.raises(ValueError, match="engine_prefill_chunk_size <="):
        _apply_prefill_policy_defaults(hyper_params, "pyramidkv")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("vanilla", {"engine_prefill_chunk_size": 8192}),
        (
            "pyramidkv",
            {
                "engine_prefill_chunk_size": 8192,
                "long_prefill_offload_threshold": 65536,
            },
        ),
    ],
)
def test_microbench_selects_policy_control_when_both_are_declared(
    method,
    expected,
):
    hyper_params = {
        "engine_prefill_chunk_size": 8192,
        "long_prefill_offload_threshold": 65536,
    }

    _apply_prefill_policy_defaults(hyper_params, method)

    assert hyper_params == expected


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


@pytest.mark.parametrize(
    "method",
    [
        "deltakv-less-memory-cudagraph",
        "deltakv_less_memory_cudagraph",
    ],
)
def test_benchmark_sparse_method_preserves_graph_enabling_legacy_alias(method):
    assert _benchmark_sparse_method(method) == method


@pytest.mark.parametrize("enabled", [False, True])
def test_artifact_records_include_step_timing_mode(enabled):
    args = SimpleNamespace(
        output_len=8,
        temperature=0.0,
        top_p=1.0,
        synchronize_step_timing=enabled,
    )

    records = _artifact_records(args, [{"status": "SUCCESS", "length": 16}])

    assert records[0]["synchronize_step_timing"] is enabled


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


def test_resolved_engine_config_records_backend_and_jsonable_values():
    llm = SimpleNamespace(
        config=SimpleNamespace(
            vllm_sparse_method="deltakv",
            prefill_schedule_policy="long_bs1full_short_batch",
            chunk_prefill_size=4096,
            long_prefill_offload_threshold=4096,
            decode_cuda_graph=True,
            decode_cuda_graph_capture_sampling=False,
            deltakv_sparse_decode_backend="fa2",
            deltakv_triton_materialize_block_tokens=16,
            deltakv_triton_gather_heads_per_program=4,
            deltakv_triton_reconstruct_heads_per_program=2,
            full_layer_kv_quant_bits=4,
            kv_quant_bits=0,
            kv_quant_group_size=64,
            full_attn_layers=(0, 1, 2, 8),
            obs_layer_ids=[2, 8],
            h2o_decode_budget=4096,
            h2o_decode_eviction_interval=128,
            h2o_prefill_budget=8192,
            h2o_recent_ratio=0.5,
            h2o_prefill_score_window=128,
        )
    )

    resolved = _resolved_engine_config(llm)

    assert resolved["deltakv_sparse_decode_backend"] == "fa2"
    assert resolved["long_prefill_offload_threshold"] == 4096
    assert resolved["full_attn_layers"] == [0, 1, 2, 8]
    assert resolved["obs_layer_ids"] == [2, 8]
    assert resolved["h2o_decode_budget"] == 4096
    assert resolved["h2o_decode_eviction_interval"] == 128
    assert resolved["h2o_prefill_budget"] == 8192
    assert resolved["h2o_recent_ratio"] == 0.5
    assert resolved["h2o_prefill_score_window"] == 128
