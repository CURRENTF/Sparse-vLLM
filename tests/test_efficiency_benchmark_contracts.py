import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.efficiency import bench_probe
from benchmark.efficiency.bench_probe import (
    HardwareMetricError,
    _actual_hardware_metrics,
    _attach_churn_comparisons,
    _attach_saturation_metrics,
    _physical_gpu_metadata,
    _record_batch_first_tokens,
    _resolve_sparse_probe_protocol,
    _vllm_phase_metrics,
    _vllm_request_phase_seconds,
)
from benchmark.efficiency.hardware_monitor import GPUHardwareMonitor
from benchmark.efficiency.metrics_calculator import (
    ModelArchitectureSpecs,
    detect_gpu_hardware,
)
from benchmark.efficiency.validate_unified_suite import validate_suite
from benchmark.efficiency.workload import build_request_trace, derive_trace_seed
from benchmark.long_bench.pred_vllm import _effective_num_samples, build_chat_prompt
from benchmark.long_bench.pred import _merge_worker_outputs
from benchmark.long_bench.prompt_budget import encode_prompt_with_generation_budget


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_probe_cli_parser_builds_with_new_workload_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_probe.py",
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--scenario",
            "all",
            "--batch-sizes",
            "1,4",
        ],
    )

    args = bench_probe.parse_args()

    assert args.scenario == "all"
    assert args.batch_sizes == [1, 4]
    assert args.churn_request_multiplier == 4


def test_physical_gpu_metadata_uses_nvidia_smi_without_cuda_init(monkeypatch):
    monkeypatch.setattr(
        bench_probe.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="1, NVIDIA H100 80GB HBM3, 9.0, 81559, 590.44\n"
        ),
    )

    assert _physical_gpu_metadata([1]) == [
        {
            "physical_device_index": 1,
            "name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "total_memory_mib": 81559,
            "driver_version": "590.44",
        }
    ]


def test_unknown_hardware_does_not_fall_back_to_h100():
    with pytest.raises(ValueError, match="Unknown GPU hardware"):
        detect_gpu_hardware("Mystery Accelerator")


def test_ambiguous_h100_requires_explicit_profile():
    with pytest.raises(ValueError, match="Ambiguous H100"):
        detect_gpu_hardware("NVIDIA H100 80GB HBM3")
    assert detect_gpu_hardware("h100_sxm").peak_bandwidth_tbs == 3.35


def test_model_specs_require_real_architecture_fields():
    with pytest.raises(ValueError, match="missing required architecture fields"):
        ModelArchitectureSpecs.from_config_dict({"hidden_size": 2048})


def test_model_specs_accept_nested_explicit_non_factorized_head_dim():
    specs = ModelArchitectureSpecs.from_config_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "vocab_size": 248320,
                "intermediate_size": 17408,
            },
        }
    )

    assert specs.hidden_size == 5120
    assert specs.num_attention_heads == 24
    assert specs.num_key_value_heads == 4
    assert specs.head_dim == 256


def test_model_specs_require_factorized_head_dim_when_not_explicit():
    with pytest.raises(ValueError, match="without an explicit head_dim"):
        ModelArchitectureSpecs.from_config_dict(
            {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "vocab_size": 248320,
                "intermediate_size": 17408,
            }
        )


def test_probe_writes_metric_failed_when_model_discovery_fails(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 2048}))
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_probe.py",
            "--model-path",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--hardware",
            "h100_sxm",
        ],
    )

    with pytest.raises(ValueError, match="missing required architecture fields"):
        bench_probe.main()

    assert json.loads((output_dir / "run_status.json").read_text())["status"] == "metric_failed"
    assert json.loads((output_dir / "summary.json").read_text())["status"] == "metric_failed"


def test_probe_refuses_to_reuse_artifact_directory(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "raw_samples.jsonl").write_text("old\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_probe.py",
            "--model-path",
            "unused",
            "--output-dir",
            str(output_dir),
            "--hardware",
            "h100_sxm",
        ],
    )

    with pytest.raises(FileExistsError, match="Refusing to mix benchmark runs"):
        bench_probe.main()
    assert (output_dir / "raw_samples.jsonl").read_text() == "old\n"


def test_hardware_monitor_zero_samples_is_metric_failed(tmp_path):
    output = tmp_path / "timeline.json"
    monitor = GPUHardwareMonitor([0], output_file=output)
    monitor.start_time = 1.0
    monitor.end_time = 2.0

    summary = monitor.analyze_and_save()

    assert summary["status"] == "metric_failed"
    assert summary["total_samples"] == 0
    assert "host_launch_bubble" not in json.dumps(summary)
    assert json.loads(output.read_text())["summary"]["status"] == "metric_failed"


def test_hardware_monitor_names_sampled_idle_without_host_attribution():
    monitor = GPUHardwareMonitor([0])
    monitor.start_time = 1.0
    monitor.end_time = 2.0
    monitor.samples = [
        {
            "time_s": 0.1,
            "gpu0_util": 0.0,
            "gpu0_mem_util": 0.0,
            "gpu0_mem_mb": 1.0,
            "gpu0_power_w": 10.0,
            "gpu0_temp_c": 30.0,
        }
    ]

    summary = monitor.analyze_and_save()

    assert summary["status"] == "success"
    assert summary["gpus"]["gpu_0"]["coarse_gpu_idle_duty_pct"] == 100.0
    assert summary["gpus"]["gpu_0"]["avg_memory_io_activity_pct"] == 0.0
    assert summary["aggregate"]["mean_memory_io_activity_pct"] == 0.0
    assert "host_launch_bubble_pct" not in summary["gpus"]["gpu_0"]


def test_vllm_phase_metrics_use_one_request_timeline():
    output = SimpleNamespace(
        metrics=SimpleNamespace(arrival_time=10.0, first_token_time=10.2, finished_time=10.5),
        outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4])],
    )

    ttft_ms, tpot_ms = _vllm_phase_metrics([output], 4)

    assert ttft_ms == pytest.approx(200.0)
    assert tpot_ms == pytest.approx(100.0)


def test_vllm_single_token_request_has_no_tpot():
    output = SimpleNamespace(
        metrics=SimpleNamespace(arrival_time=10.0, first_token_time=10.2, finished_time=10.2),
        outputs=[SimpleNamespace(token_ids=[1])],
    )

    _ttft_ms, tpot_ms = _vllm_phase_metrics([output], 1)

    assert tpot_ms is None


def test_vllm_v1_phase_metrics_use_latency_and_monotonic_timestamps():
    ttft_s, decode_s, source = _vllm_request_phase_seconds(
        SimpleNamespace(
            arrival_time=1_700_000_000.0,
            first_token_latency=0.25,
            first_token_ts=10.0,
            last_token_ts=10.6,
        )
    )

    assert ttft_s == pytest.approx(0.25)
    assert decode_s == pytest.approx(0.6)
    assert source == "vllm_v1_first_token_latency_and_monotonic_decode"


def test_sparse_batch_ttft_waits_for_every_request_first_token():
    expected = {11, 12}
    observed: set[int] = set()

    assert not _record_batch_first_tokens(expected, observed, [(11, [1])], [])
    assert _record_batch_first_tokens(
        expected,
        observed,
        [],
        [(12, [2], None, None)],
    )
    assert observed == expected


def test_final_prompt_budget_includes_special_tokens_and_generation():
    class Tokenizer:
        bos_token = "<bos>"

        def __init__(self):
            self.add_special_tokens = None

        def encode(self, prompt, *, add_special_tokens):
            self.add_special_tokens = add_special_tokens
            token_ids = list(range(10, 20))
            return [1, *token_ids] if add_special_tokens else token_ids

    tokenizer = Tokenizer()
    token_ids = encode_prompt_with_generation_budget(
        tokenizer,
        "rendered chat prompt",
        max_model_len=8,
        max_gen=3,
    )

    assert tokenizer.add_special_tokens is True
    assert token_ids == [1, 10, 17, 18, 19]
    assert len(token_ids) + 3 == 8


def test_final_prompt_budget_rejects_generation_without_prompt_space():
    tokenizer = SimpleNamespace(
        bos_token=None,
        encode=lambda prompt, add_special_tokens: [1],
    )
    with pytest.raises(ValueError, match="leaves no prompt budget"):
        encode_prompt_with_generation_budget(
            tokenizer,
            "prompt",
            max_model_len=4,
            max_gen=4,
        )


def test_vllm_runner_requires_consistent_sample_limit():
    args = SimpleNamespace(num_samples=10, samples_per_task=8)
    with pytest.raises(ValueError, match="disagree"):
        _effective_num_samples(args)


def test_vllm_chat_template_failure_is_not_silently_hidden():
    tokenizer = SimpleNamespace(
        chat_template="template",
        apply_chat_template=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad template")),
    )
    with pytest.raises(RuntimeError, match="bad template"):
        build_chat_prompt(tokenizer, "prompt")


def test_snapkv_probe_requires_and_records_explicit_score_mode():
    args = SimpleNamespace(
        hyper_params="{}",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        sparse_method="snapkv",
        sparse_prefill_score_mode=None,
    )
    with pytest.raises(ValueError, match="explicit --sparse-prefill-score-mode"):
        _resolve_sparse_probe_protocol(args)

    args.sparse_prefill_score_mode = "probability"
    _hyper, budget, protocol, label = _resolve_sparse_probe_protocol(args)
    assert budget == 2176
    assert protocol == {
        "score_mode": "probability",
        "score_window": 64,
        "sparse_budget": 2176,
        "max_num_batched_tokens": 8192,
    }
    assert "probability-budget2176-window64" in label


def test_h2o_probe_records_explicit_budget_protocol():
    args = SimpleNamespace(
        hyper_params="{}",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        sparse_method="h2o",
        sparse_prefill_score_mode="probability",
    )

    hyper, budget, protocol, label = _resolve_sparse_probe_protocol(args)

    assert budget is None
    assert hyper["h2o_decode_budget"] == 4096
    assert protocol["decode_budget"] == 4096
    assert protocol["prefill_budget"] == 8192
    assert protocol["score_mode"] == "probability"
    assert protocol["max_num_batched_tokens"] == 8192
    assert "h2o-probability-decode4096-prefill8192-window0" in label


def test_omnikv_probe_requires_explicit_calibrated_layers():
    args = SimpleNamespace(
        hyper_params="{}",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        sparse_method="omnikv",
        sparse_prefill_score_mode=None,
        allow_single_omnikv_full_layer=False,
    )

    with pytest.raises(ValueError, match="explicit calibrated method config"):
        _resolve_sparse_probe_protocol(args)

    args.hyper_params = json.dumps(
        {
            "engine_prefill_chunk_size": 16384,
            "full_attention_layers": "0,2,4,11,16,22",
            "sink_keep_tokens": 0,
            "recent_keep_tokens": 32,
            "decode_keep_tokens": 2048,
            "pool_kernel_size": 1,
        }
    )
    hyper, budget, _protocol, label = _resolve_sparse_probe_protocol(args)

    assert hyper["full_attention_layers"] == "0,2,4,11,16,22"
    assert budget is None
    assert label == "sparsevllm-omnikv"


def test_random_trace_is_matched_reproducible_and_variable_length():
    seed = derive_trace_seed(
        42,
        scenario="fixed_batch",
        phase="measure",
        nominal_prompt_len=100,
        nominal_output_len=16,
        concurrency=4,
        iteration=0,
    )
    first = build_request_trace(
        seed=seed,
        request_count=4,
        nominal_prompt_len=100,
        nominal_output_len=16,
        vocab_size=1000,
        prompt_jitter_fraction=0.10,
        output_jitter_fraction=0.25,
        vary_output_lengths=False,
    )
    second = build_request_trace(
        seed=seed,
        request_count=4,
        nominal_prompt_len=100,
        nominal_output_len=16,
        vocab_size=1000,
        prompt_jitter_fraction=0.10,
        output_jitter_fraction=0.25,
        vary_output_lengths=False,
    )

    assert [trace.prompt_digest for trace in first] == [
        trace.prompt_digest for trace in second
    ]
    assert len({trace.prompt_digest for trace in first}) == 4
    assert len({trace.prompt_len for trace in first}) > 1
    assert all(90 <= trace.prompt_len <= 100 for trace in first)
    assert {trace.output_len for trace in first} == {16}


def test_random_trace_changes_every_measurement_iteration():
    seeds = [
        derive_trace_seed(
            42,
            scenario="fixed_batch",
            phase="measure",
            nominal_prompt_len=64,
            nominal_output_len=8,
            concurrency=2,
            iteration=iteration,
        )
        for iteration in range(3)
    ]
    digest_sets = []
    for seed in seeds:
        trace = build_request_trace(
            seed=seed,
            request_count=2,
            nominal_prompt_len=64,
            nominal_output_len=8,
            vocab_size=1000,
            prompt_jitter_fraction=0.10,
            output_jitter_fraction=0.25,
            vary_output_lengths=False,
        )
        digest_sets.append({request.prompt_digest for request in trace})

    assert len(set(seeds)) == 3
    assert not (digest_sets[0] & digest_sets[1] | digest_sets[0] & digest_sets[2] | digest_sets[1] & digest_sets[2])


def test_churn_trace_varies_prompt_and_output_lengths():
    trace = build_request_trace(
        seed=123,
        request_count=16,
        nominal_prompt_len=128,
        nominal_output_len=32,
        vocab_size=1000,
        prompt_jitter_fraction=0.10,
        output_jitter_fraction=0.25,
        vary_output_lengths=True,
    )

    assert len({request.prompt_len for request in trace}) > 1
    assert len({request.output_len for request in trace}) > 1


def test_actual_hardware_metrics_use_sampled_values_and_cross_gpu_peak():
    metrics = _actual_hardware_metrics(
        {
            "status": "success",
            "total_samples": 5,
            "sampling_interval_ms": 100,
            "aggregate": {
                "mean_compute_util_pct": 60.0,
                "mean_memory_io_activity_pct": 30.0,
                "mean_coarse_gpu_active_duty_pct": 80.0,
                "avg_total_power_w": 500.0,
            },
            "gpus": {
                "gpu_0": {"peak_vram_gb": 70.0},
                "gpu_1": {"peak_vram_gb": 71.5},
            },
        }
    )

    assert metrics["metric_source"] == "nvidia-smi sampled activity"
    assert metrics["gpu_compute_activity_pct_mean"] == 60.0
    assert metrics["gpu_memory_io_activity_pct_mean"] == 30.0
    assert metrics["peak_vram_gb_max"] == 71.5


def test_actual_hardware_metrics_do_not_fall_back_to_estimates():
    with pytest.raises(HardwareMetricError, match="collection failed"):
        _actual_hardware_metrics({"status": "metric_failed", "error": "no samples"})


def test_churn_summary_is_compared_to_matched_fixed_batch():
    rows = [
        {
            "engine": "sparsevllm",
            "sparse_method": "vanilla",
            "protocol_label": "sparsevllm-vanilla",
            "scenario": "fixed_batch",
            "prompt_len": 100,
            "output_len": 16,
            "concurrency": 4,
            "output_token_throughput_tps": 100.0,
            "request_throughput_rps": 10.0,
            "ttft_ms_p99": 20.0,
        },
        {
            "engine": "sparsevllm",
            "sparse_method": "vanilla",
            "protocol_label": "sparsevllm-vanilla",
            "scenario": "oversubscribed_churn",
            "prompt_len": 100,
            "output_len": 16,
            "concurrency": 4,
            "output_token_throughput_tps": 80.0,
            "request_throughput_rps": 8.0,
            "ttft_ms_p99": 50.0,
        },
    ]

    _attach_churn_comparisons(rows)

    assert rows[1]["fixed_batch_comparison_status"] == "success"
    assert rows[1]["churn_output_tps_ratio_vs_fixed_batch"] == pytest.approx(0.8)
    assert rows[1]["churn_ttft_p99_delta_ms_vs_fixed_batch"] == pytest.approx(30.0)


def test_saturation_metrics_use_observed_concurrency_ladder():
    rows = [
        {
            "engine": "sparsevllm",
            "sparse_method": "vanilla",
            "protocol_label": "sparsevllm-vanilla",
            "scenario": "fixed_batch",
            "prompt_len": 100,
            "output_len": 16,
            "concurrency": concurrency,
            "output_token_throughput_tps": rate,
        }
        for concurrency, rate in ((1, 100.0), (4, 300.0), (8, 310.0))
    ]

    _attach_saturation_metrics(rows)

    assert rows[0]["output_tps_pct_of_observed_sweep_peak"] == pytest.approx(
        100.0 / 310.0 * 100.0
    )
    assert rows[1]["output_tps_scaling_efficiency_pct_vs_min_concurrency"] == pytest.approx(75.0)
    assert rows[2]["marginal_output_tps_gain_pct_vs_previous_concurrency"] == pytest.approx(
        (310.0 / 300.0 - 1.0) * 100.0
    )
    assert rows[2]["observed_output_saturation_concurrency"] == 4
    assert all(row["saturation_analysis_status"] == "success" for row in rows)


def test_longbench_scorer_rejects_missing_status(tmp_path):
    prediction = {
        "pred": "label",
        "answers": ["label"],
        "all_classes": ["label"],
        "length": 1,
    }
    (tmp_path / "trec.jsonl").write_text(json.dumps(prediction) + "\n")

    result = subprocess.run(
        [sys.executable, "benchmark/long_bench/eval.py", "--path", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert metrics["task_statuses"]["trec"]["invalid_statuses"] == {"missing": 1}


def _write_valid_suite_fixture(root: Path, systems: list[str]) -> None:
    protocols = {
        "svllm-vanilla": ("sparsevllm", "vanilla"),
        "svllm-snapkv": ("sparsevllm", "snapkv"),
        "svllm-h2o": ("sparsevllm", "h2o"),
        "svllm-omnikv": ("sparsevllm", "omnikv"),
        "svllm-deltakv": ("sparsevllm", "deltakv"),
        "vllm-vanilla": ("vllm", "vanilla"),
        "vllm": ("vllm", "vanilla"),
    }
    for scenario in ("scenario_a_synthetic", "scenario_b_longbench"):
        for system in systems:
            engine, method = protocols[system]
            system_dir = root / scenario / system
            system_dir.mkdir(parents=True)
            (system_dir / "stage_status.json").write_text(
                json.dumps({"status": "success"}) + "\n"
            )
            (system_dir / "gpu_timeline_summary.json").write_text(
                json.dumps({"status": "success", "aggregate": {}}) + "\n"
            )
            if scenario == "scenario_a_synthetic":
                hardware = {
                    "metric_source": "nvidia-smi sampled activity",
                    "sample_count": 2,
                }
                fixed = {
                    "status": "success",
                    "scenario": "fixed_batch",
                    "prompt_len": 100,
                    "output_len": 16,
                    "prompt_len_min": 90,
                    "prompt_len_max": 100,
                    "output_len_min": 16,
                    "output_len_max": 16,
                    "concurrency": 2,
                    "request_count": 2,
                    "request_throughput_rps": 2.0,
                    "input_token_throughput_tps": 200.0,
                    "output_token_throughput_tps": 32.0,
                    "total_token_throughput_tps": 232.0,
                    "output_tps_pct_of_observed_sweep_peak": 100.0,
                    "output_tps_scaling_efficiency_pct_vs_min_concurrency": 100.0,
                    "saturation_analysis_status": "skipped_by_policy",
                    "actual_hardware_metrics": hardware,
                }
                churn = {
                    **fixed,
                    "scenario": "oversubscribed_churn",
                    "output_len_min": 12,
                    "request_count": 8,
                    "fixed_batch_comparison_status": "success",
                }
                (system_dir / "summary.json").write_text(
                    json.dumps({"status": "success", "summary": [fixed, churn]})
                    + "\n"
                )
                raw_rows = []
                for iteration, scenario_name in enumerate(
                    ("fixed_batch", "oversubscribed_churn")
                ):
                    raw_rows.append(
                        {
                            "status": "success",
                            "scenario": scenario_name,
                            "prompt_len": 100,
                            "output_len": 16,
                            "concurrency": 2,
                            "iteration": iteration,
                            "trace": {
                                "prompt_lengths": [90, 100],
                                "output_lengths": [16, 16],
                                "prompt_digests": [
                                    f"digest-{scenario_name}-a",
                                    f"digest-{scenario_name}-b",
                                ],
                            },
                        }
                    )
                (system_dir / "raw_samples.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in raw_rows)
                )
                (system_dir / "request_samples.jsonl").write_text(
                    json.dumps({"status": "success", "request_index": 0}) + "\n"
                )
                (system_dir / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "args": {"engine": engine, "sparse_method": method},
                            "workload": {
                                "prefix_caching_enabled": False,
                                "iteration_prompt_reuse_allowed": False,
                            },
                        }
                    )
                    + "\n"
                )
                if engine == "sparsevllm":
                    (system_dir / "operator_runtime_stats.json").write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "world_ranks": [
                                    {
                                        "world_rank": 0,
                                        "bindings": [
                                            {
                                                "operator_type": "test",
                                                "selected_provider": "test_provider",
                                            }
                                        ],
                                        "operators": {},
                                    }
                                ],
                            }
                        )
                        + "\n"
                    )
            else:
                (system_dir / "result.json").write_text(
                    json.dumps({"status": "success"}) + "\n"
                )
                sample = {
                    "dataset": "task",
                    "sample_idx": 0,
                    "source_idx": 0,
                    "status": "success",
                }
                task_sample = {"status": "success", "source_idx": 0}
                (system_dir / "task.jsonl").write_text(
                    json.dumps(task_sample) + "\n"
                )
                (system_dir / "run_status.json").write_text(
                    json.dumps({"status": "success"}) + "\n"
                )
                for artifact in (
                    "raw_outputs.jsonl",
                    "parsed_outputs.jsonl",
                    "sample_results.jsonl",
                ):
                    (system_dir / artifact).write_text(json.dumps(sample) + "\n")
                resolved = {
                    "backend": engine,
                    "args": {
                        "seed": 42,
                        "enable_prefix_caching": False,
                    },
                }
                if engine == "sparsevllm":
                    resolved["sparse_method"] = method
                    resolved["effective_runtime"] = {
                        "prefix_cache_enabled": False,
                    }
                if method == "omnikv":
                    resolved["requested_runtime"] = {
                        "config": {
                            "full_attention_layers": "0,2,4,11,16,22",
                        },
                    }
                    resolved["effective_runtime"]["benchmark_config"] = {
                        "full_attention_layers": [0, 2, 4, 11, 16, 22],
                        "obs_layer_ids": [0, 2, 4, 11, 16, 22],
                    }
                (system_dir / "resolved_config.json").write_text(
                    json.dumps(resolved) + "\n"
                )
                if engine == "sparsevllm":
                    (system_dir / "operator_runtime_stats.json").write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "world_ranks": [
                                    {
                                        "world_rank": 0,
                                        "bindings": [
                                            {
                                                "operator_type": "test",
                                                "selected_provider": "test_provider",
                                            }
                                        ],
                                        "operators": {},
                                    }
                                ],
                            }
                        )
                        + "\n"
                    )


def test_unified_suite_validator_rejects_any_failed_system(tmp_path):
    systems = ["svllm-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    failed = tmp_path / "scenario_a_synthetic/vllm-vanilla/stage_status.json"
    failed.write_text(json.dumps({"status": "failed", "task_exit_code": 3}) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("failed stage scenario_a_synthetic/vllm-vanilla" in error for error in report["errors"])


def test_longbench_worker_outputs_merge_without_shared_append(tmp_path):
    for rank, source_idx in ((0, 2), (1, 1)):
        worker_dir = tmp_path / f".worker_rank{rank}"
        worker_dir.mkdir()
        task_row = {"status": "success", "source_idx": source_idx}
        structured = {
            "dataset": "task",
            "sample_idx": source_idx,
            "source_idx": source_idx,
            "status": "success",
        }
        (worker_dir / "task.jsonl").write_text(json.dumps(task_row) + "\n")
        for artifact in (
            "raw_outputs.jsonl",
            "parsed_outputs.jsonl",
            "sample_results.jsonl",
        ):
            (worker_dir / artifact).write_text(json.dumps(structured) + "\n")

    _merge_worker_outputs(
        str(tmp_path),
        datasets=["task"],
        world_size=2,
    )

    for artifact in (
        "task.jsonl",
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "sample_results.jsonl",
    ):
        rows = [json.loads(line) for line in (tmp_path / artifact).read_text().splitlines()]
        assert [row["source_idx"] for row in rows] == [1, 2]


def test_unified_suite_validator_requires_longbench_structured_artifacts(tmp_path):
    systems = ["svllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    missing = (
        tmp_path
        / "scenario_b_longbench/svllm-vanilla/raw_outputs.jsonl"
    )
    missing.unlink()

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("raw_outputs.jsonl" in error for error in report["errors"])


def test_unified_suite_validator_requires_matched_source_ids(tmp_path):
    systems = ["svllm-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    mismatched = tmp_path / "scenario_b_longbench/vllm-vanilla/task.jsonl"
    mismatched.write_text(json.dumps({"status": "success", "source_idx": 7}) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("source IDs differ" in error for error in report["errors"])


def test_unified_suite_validator_rejects_method_label_mismatch(tmp_path):
    systems = ["svllm-omnikv"]
    _write_valid_suite_fixture(tmp_path, systems)
    resolved = tmp_path / "scenario_b_longbench/svllm-omnikv/resolved_config.json"
    resolved.write_text(
        json.dumps({"backend": "sparsevllm", "sparse_method": "vanilla"}) + "\n"
    )

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("protocol mismatch" in error for error in report["errors"])


def test_unified_suite_validator_rejects_single_layer_omnikv_runtime(tmp_path):
    systems = ["svllm-omnikv"]
    _write_valid_suite_fixture(tmp_path, systems)
    resolved_path = tmp_path / "scenario_b_longbench/svllm-omnikv/resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["effective_runtime"]["benchmark_config"]["full_attention_layers"] = [0]
    resolved_path.write_text(json.dumps(resolved) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("invalid OmniKV full_attention_layers" in error for error in report["errors"])


def test_unified_suite_validator_rejects_omnikv_requested_effective_mismatch(
    tmp_path,
):
    systems = ["svllm-omnikv"]
    _write_valid_suite_fixture(tmp_path, systems)
    resolved_path = tmp_path / "scenario_b_longbench/svllm-omnikv/resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["effective_runtime"]["benchmark_config"]["full_attention_layers"] = [
        0,
        3,
        9,
    ]
    resolved_path.write_text(json.dumps(resolved) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("layer mismatch" in error for error in report["errors"])


def test_unified_suite_validator_requires_sparse_operator_bindings(tmp_path):
    systems = ["svllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    stats_path = (
        tmp_path
        / "scenario_a_synthetic/svllm-vanilla/operator_runtime_stats.json"
    )
    stats_path.write_text(
        json.dumps({"status": "success", "world_ranks": []}) + "\n"
    )

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("empty operator runtime stats" in error for error in report["errors"])


def test_unified_suite_validator_requires_identical_random_traces(tmp_path):
    systems = ["svllm-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    path = tmp_path / "scenario_a_synthetic/vllm-vanilla/raw_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["trace"]["prompt_digests"][0] = "different-digest"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("synthetic random traces differ" in error for error in report["errors"])


def test_unified_suite_validator_rejects_repeated_prompts_across_iterations(tmp_path):
    systems = ["svllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    path = tmp_path / "scenario_a_synthetic/svllm-vanilla/raw_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["trace"]["prompt_digests"] = rows[0]["trace"]["prompt_digests"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("prompts repeat across synthetic iterations" in error for error in report["errors"])


def test_unified_suite_validator_requires_identical_output_length_trace(tmp_path):
    systems = ["svllm-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    path = tmp_path / "scenario_a_synthetic/vllm-vanilla/raw_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["trace"]["output_lengths"][0] = 15
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("synthetic random traces differ" in error for error in report["errors"])
