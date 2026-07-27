import json
from types import SimpleNamespace

import pytest

from benchmark.microbench import (
    _artifact_records,
    _benchmark_sparse_method,
    _resolved_engine_config,
    _write_output_dir,
)


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
        )
    )

    resolved = _resolved_engine_config(llm)

    assert resolved["deltakv_sparse_decode_backend"] == "fa2"
    assert resolved["full_attn_layers"] == [0, 1, 2, 8]
    assert resolved["obs_layer_ids"] == [2, 8]
