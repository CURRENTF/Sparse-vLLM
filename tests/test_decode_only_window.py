"""Protect measured token accounting and reject prefill-contaminated windows."""
import pytest

from benchmark.efficiency.metrics import DecodeOnlyWindow


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
