"""Protect measured token accounting and reject prefill-contaminated windows."""
import pytest

from benchmark.efficiency.metrics import DecodeOnlyWindow


@pytest.mark.parametrize("measured_steps", [1, 2, 3])
def test_probe_closes_window_when_last_decode_finishes_workload(monkeypatch, tmp_path, measured_steps):
    # Exercise the real probe loop: collector-only tests cannot detect a missing
    # final boundary when is_finished() prevents another scheduler iteration.
    from types import SimpleNamespace

    import sparsevllm
    import torch
    from benchmark.efficiency import bench_probe

    class FakeLLM:
        def __init__(self):
            self.steps = 0
            self.scheduler = SimpleNamespace(decoding=[], waiting=[])
            self.last_step_token_outputs = []

        def add_request(self, *args):
            return 1

        def is_finished(self):
            return self.steps == 4

        def step(self):
            self.steps += 1
            self.last_step_token_outputs = [(1, [7])]
            self.scheduler.decoding = [SimpleNamespace(seq_id=1)]
            if self.is_finished():
                self.scheduler.decoding = []
                return [(1, [7] * 4, None, None)], -1
            return [], 1 if self.steps == 1 else -1

        def debug_sparse_state_summaries(self):
            return [{"decode_graph": {"capture_count": 0,
                     "replay_count": max(0, self.steps - 1),
                     "eager_static_count": 0, "force_eager_count": 0}}]

    llm = FakeLLM()
    syncs, windows = [], []
    monkeypatch.setattr(sparsevllm, "LLM", lambda *args, **kwargs: llm)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append(llm.steps))
    monkeypatch.setattr(bench_probe, "_resolve_sparse_probe_protocol",
                        lambda args: ({}, None, {}, "test"))
    monkeypatch.setattr(bench_probe, "_trace_for_iteration", lambda *args, **kwargs: [
        SimpleNamespace(prompt_token_ids=[7], output_len=4)])
    monkeypatch.setattr(bench_probe, "_case_monitor", lambda *args:
                        SimpleNamespace(stop=lambda: {}))

    def collect_window(*args, **kwargs):
        window = DecodeOnlyWindow(*args, **kwargs)
        windows.append(window)
        return window

    monkeypatch.setattr(bench_probe, "DecodeOnlyWindow", collect_window)

    class ReachedTimeline(Exception):
        pass

    def stop_at_timeline(**kwargs):
        raise ReachedTimeline

    monkeypatch.setattr(bench_probe, "_request_phase_metrics_from_timestamps", stop_at_timeline)
    args = SimpleNamespace(scenario="fixed", output_dir=str(tmp_path),
                           sparse_method="vanilla", model_path="unused", prompt_lens=[1],
                           output_lens=[4], batch_sizes=[1], num_warmups=0, num_iters=1,
                           decode_only_steps=measured_steps, decode_only_warmup_steps=1)
    if measured_steps == 3:
        with pytest.raises(RuntimeError, match="without the requested"):
            bench_probe.run_sparsevllm_probe(args, SimpleNamespace())
        assert windows[0].result is None
        assert syncs == [0, 2]
    else:
        with pytest.raises(ReachedTimeline):
            bench_probe.run_sparsevllm_probe(args, SimpleNamespace())
        result = windows[0].require_result()
        assert result["decode_stage_tokens"] == measured_steps
        assert result["graph_counter_delta"]["replay_count"] == measured_steps
        assert syncs == [0, 2, 2 + measured_steps]


def test_window_counts_only_full_decode_and_synchronizes_at_edges():
    now = [0.0]
    syncs = []
    graph = {"capture_count": 0, "replay_count": 0, "eager_decode_count": 0}
    window = DecodeOnlyWindow(2, 3, 1, synchronize=lambda: syncs.append(now[0]),
                              clock=lambda: now[0], graph_stats=lambda: dict(graph))
    window.observe(is_decode=False, request_ids=[1], tokens=100, admission_complete=False)
    now[0] = 100
    window.observe(is_decode=True, request_ids=[1, 2], tokens=2, admission_complete=True)
    for index in range(3):
        window.boundary()
        now[0] += 2
        graph["replay_count"] += 1
        window.observe(is_decode=True, request_ids=[2, 1], tokens=2, admission_complete=True)
    window.boundary()
    result = window.require_result()
    assert syncs == [100, 106]
    assert result["decode_stage_tokens"] == 6
    assert result["decode_stage_elapsed_s"] == 6
    assert result["decode_stage_throughput_tps"] == 1


@pytest.mark.parametrize("is_decode,ids,admitted", [(False, [1, 2], True),
                          (True, [1], True), (True, [1, 3], True), (True, [1, 2], False)])
def test_window_rejects_prefill_turnover_or_admission(is_decode, ids, admitted):
    window = DecodeOnlyWindow(2, 3, 1, synchronize=lambda: None, clock=lambda: 1,
                              graph_stats=lambda: {})
    window.observe(is_decode=True, request_ids=[1, 2], tokens=2, admission_complete=True)
    window.boundary()
    with pytest.raises(RuntimeError, match="inside decode-only"):
        window.observe(is_decode=is_decode, request_ids=ids, tokens=len(ids), admission_complete=admitted)
    with pytest.raises(RuntimeError, match="without the requested"):
        window.require_result()


@pytest.mark.parametrize("field", ["capture_count", "eager_decode_count", "replay_count"])
def test_window_rejects_non_replay_execution(field):
    graph = {"capture_count": 0, "replay_count": 0, "eager_decode_count": 0}
    window = DecodeOnlyWindow(1, 1, 1, synchronize=lambda: None, clock=lambda: 1,
                              graph_stats=lambda: dict(graph))
    window.observe(is_decode=True, request_ids=[1], tokens=1, admission_complete=True)
    window.boundary()
    graph["replay_count"] = 1
    graph[field] += 1
    window.observe(is_decode=True, request_ids=[1], tokens=1, admission_complete=True)
    with pytest.raises(RuntimeError, match="graph contract"):
        window.boundary()
