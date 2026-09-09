"""Reject misleading capacity curves even when a runner reports success."""
import json

import pytest

from scripts.official_experiments.decode_capacity_128k2k.plot_decode_capacity import validate_measurement


def make_case(tmp_path, engine="sparsevllm"):
    config = {"input_len": 128, "output_len": 4}
    row = dict(engine=engine, method="vanilla", length=128, output_len=4,
               status="success", stage_metrics_status="success",
               measurement_scope="full_batch_pure_decode_steps", actual_decode_peak=2,
               completed_requests=2, scheduler_preemptions=0, synchronize_step_timing=True,
               decode_stage_tokens=4, decode_stage_elapsed_s=0.5, decode_stage_throughput_tps=8.0)
    steps = [{"tokens": -2 if engine == "sparsevllm" else 2, "elapsed_s": dt,
              "measured": True, "pure_decode": True} for dt in (0.2, 0.3)]
    case = tmp_path / "vanilla-128-2"
    case.mkdir()
    outputs = [{"request_id": i, "token_ids": [1, 2, 3, 4], "status": "success"} for i in range(2)]
    return config, row, steps, outputs, case


def persist(tmp_path, row, steps, outputs, case):
    artifact = tmp_path / "performance.jsonl"
    artifact.write_text(json.dumps(row) + "\n")
    (case / "steps.jsonl").write_text("".join(json.dumps(item) + "\n" for item in steps))
    (case / "raw_outputs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in outputs))
    return artifact


@pytest.mark.parametrize("engine", ["sparsevllm", "vllm", "hisparse"])
def test_reconstruct_stage_rate_from_independent_token_work(tmp_path, engine):
    config, row, steps, outputs, case = make_case(tmp_path, engine)
    artifact = persist(tmp_path, row, steps, outputs, case)
    assert validate_measurement(artifact, 2, config)["decode_throughput_tps"] == 8.0


@pytest.mark.parametrize("corruption", ["queued", "short_output", "duplicate", "prefill", "wrong_rate", "nan_time"])
def test_reject_false_success_before_plotting(tmp_path, corruption):
    config, row, steps, outputs, case = make_case(tmp_path)
    if corruption == "queued":
        row["actual_decode_peak"] = 1
    elif corruption == "short_output":
        outputs[1]["token_ids"].pop()
    elif corruption == "duplicate":
        outputs[1]["request_id"] = 0
    elif corruption == "prefill":
        steps[0]["tokens"] = 2
    elif corruption == "wrong_rate":
        row["decode_stage_throughput_tps"] = 10.0
    elif corruption == "nan_time":
        steps[0]["elapsed_s"] = float("nan")
    artifact = persist(tmp_path, row, steps, outputs, case)
    with pytest.raises(ValueError):
        validate_measurement(artifact, 2, config)


def test_reject_mixed_vllm_step_even_with_matching_aggregate(tmp_path):
    config, row, steps, outputs, case = make_case(tmp_path, "vllm")
    steps[0]["pure_decode"] = False
    artifact = persist(tmp_path, row, steps, outputs, case)
    with pytest.raises(ValueError, match="Mixed"):
        validate_measurement(artifact, 2, config)
def test_missing_graph_requires_explicit_capacity_evidence(tmp_path):
    """Do not misclassify graph wiring regressions as a concurrency limit."""
    from scripts.official_experiments.decode_capacity_128k2k.sweep_decode_capacity import capacity_failure

    log = tmp_path / "run.log"
    error = "decode CUDA Graph has no startup-captured graph for batch_size=48, path='long'."
    log.write_text("graphs=2 short=1 long=1 skipped_for_kv_capacity=0")
    assert not capacity_failure(error, log)
    log.write_text("graphs=1 short=1 long=0 skipped_for_kv_capacity=1")
    assert capacity_failure(error, log)
    assert not capacity_failure("kernel launch failed", log)
    log.write_text("Full decode batch capacity exceeded: scheduler preemption")
    assert capacity_failure("benchmark child exited with code -9", log)
    log.write_text("illegal memory access")
    assert not capacity_failure("benchmark child exited with code -9", log)


@pytest.mark.parametrize("corruption", ["missing_failure", "wrong_boundary", "unobserved_point"])
def test_replot_rejects_unproven_capacity_before_rendering(tmp_path, monkeypatch, corruption):
    """A valid throughput alone must not turn an unbounded sweep into a maximum."""
    from scripts.official_experiments.decode_capacity_128k2k import plot_decode_capacity as plot

    curves = []
    for lane in plot.LANES:
        curves.append({
            "model": "fixture", "lane": lane, "max_concurrency": 1,
            "first_failed_concurrency": 2,
            "attempts": [{"concurrency": 1, "status": "success"},
                         {"concurrency": 2, "status": "capacity_exceeded"}],
            "points": [{"concurrency": 1, "decode_tokens": 2,
                        "decode_elapsed_s": 0.5, "decode_throughput_tps": 4.0}],
        })
    if corruption == "missing_failure":
        curves[0]["attempts"].pop()
    elif corruption == "wrong_boundary":
        curves[0]["first_failed_concurrency"] = 3
    else:
        curves[0]["attempts"].pop(0)
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"schema_version": 1, "config": {"models": {"fixture": {}}},
                               "curves": curves}))
    monkeypatch.setattr("sys.argv", ["plot", "--plot-data", str(path)])
    def unexpected_render(*args, **kwargs):
        pytest.fail("Invalid capacity export reached rendering")
    monkeypatch.setattr(plot, "render", unexpected_render)
    with pytest.raises(ValueError, match="capacity boundary|successful attempts"):
        plot.main()


@pytest.mark.parametrize("corruption", ["missing_evidence", "fabricated_zero"])
def test_unsupported_export_cannot_hide_missing_evidence_or_become_zero(tmp_path, monkeypatch, corruption):
    """A non-runnable combination must not silently become a measured data point."""
    from scripts.official_experiments.decode_capacity_128k2k import plot_decode_capacity as plot

    reason = "fixture rejects this attention contract"
    unsupported = {lane: reason for lane in plot.LANES}
    curves = [{"model": "fixture", "lane": lane, "status": "unsupported",
               "reason": reason, "points": [], "attempts": []} for lane in plot.LANES]
    if corruption == "missing_evidence":
        unsupported.pop(curves[0]["lane"])
    else:
        curves[0]["points"] = [{"concurrency": 1, "decode_throughput_tps": 0}]
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"schema_version": 1,
        "config": {"models": {"fixture": {}}, "unsupported": {"fixture": unsupported}},
        "curves": curves}))
    monkeypatch.setattr("sys.argv", ["plot", "--plot-data", str(path)])
    monkeypatch.setattr(plot, "render", lambda *a, **kw: pytest.fail("Invalid N/A reached rendering"))
    with pytest.raises(ValueError, match="Unsupported curve"):
        plot.main()
