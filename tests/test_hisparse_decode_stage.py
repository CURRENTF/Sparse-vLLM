"""Independent CPU event-clock oracle for the scheduler timing boundary."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_scheduler_stage_includes_selection_and_cleanup_not_admission(tmp_path, monkeypatch):
    """Catch lazy-input counting and prefill/partial-batch timing contamination."""
    from benchmark import hisparse_microbench as adapter

    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: advance(.001), max_memory_allocated=lambda: 1024**3))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(adapter, "perf_counter", lambda: clock[0])

    def batch(active, decode):
        return SimpleNamespace(reqs=[SimpleNamespace(retraction_count=0) for _ in range(active)],
            input_ids=None, extend_num_tokens=25,
            forward_mode=SimpleNamespace(is_decode=lambda: decode,
                is_extend=lambda: not decode, is_mixed=lambda: False))

    batches = [batch(2, False), batch(1, True), batch(2, True), batch(2, True)]

    class Scheduler:
        enable_overlap = False

        def get_next_batch_to_run(self):
            advance(.002)
            return batches.pop(0)

        def run_batch(self, current):
            assert current.input_ids is None  # Materialization belongs to upstream.
            current.input_ids = object()
            advance(.007)
            return SimpleNamespace(can_run_cuda_graph=current.forward_mode.is_decode())

        def process_batch_result(self, current, result):
            advance(.003)

    def process():
        scheduler = Scheduler()
        while batches:
            current = scheduler.get_next_batch_to_run()
            result = scheduler.run_batch(current)
            scheduler.process_batch_result(current, result)

    fake_module = ModuleType("sglang.srt.managers.scheduler")
    fake_module.Scheduler = Scheduler
    fake_module.run_scheduler_process = process
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", fake_module)
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("HISPARSE_STAGE_STEPS", str(path))
    monkeypatch.setenv("HISPARSE_STAGE_BATCH", "2")
    monkeypatch.setenv("HISPARSE_STAGE_WARMUP", "1")
    adapter.measured_scheduler_process()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["tokens"] for row in rows] == [25, 1, 2, 2]
    assert [row["measured"] for row in rows] == [False, False, False, True]
    assert all(row["elapsed_s"] == pytest.approx(.013) for row in rows)
    measured = [row for row in rows if row["measured"]]
    assert sum(row["tokens"] for row in measured) == 2
    assert sum(row["elapsed_s"] for row in measured) == pytest.approx(.013)
