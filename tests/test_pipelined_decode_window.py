"""Protect boundary alignment: submitting work is not proof of completion."""
from collections import deque
from types import SimpleNamespace as NS

import pytest

from benchmark.efficiency.metrics import PipelinedDecodeWindow, decode_graph_delta


def test_pipeline_edges_wait_for_completed_work_and_all_ranks():
    now = [0.0]
    syncs = []
    graph = [dict(capture_count=0, replay_count=0, eager_decode_count=0) for _ in range(2)]
    window = PipelinedDecodeWindow(2, 3, 1, synchronize=lambda: syncs.append(now[0]),
                                  clock=lambda: now[0], graph_stats=lambda: [dict(g) for g in graph])
    def submit(context):
        return window.submit(is_decode=True, request_ids=[2, 1], tokens=2,
                             admission_complete=True, context_lengths=[context + 5, context])
    first = submit(10)
    with pytest.raises(RuntimeError, match="undrained"):
        window.boundary()
    window.complete(first, decode_tokens=2)
    window.boundary()
    pending = [submit(n) for n in (11, 12, 13)]
    with pytest.raises(RuntimeError, match="undrained"):
        submit(14)
    for ticket in reversed(pending):
        window.complete(ticket, decode_tokens=2)
        now[0] += 2
        for rank in graph:
            rank["replay_count"] += 1
    window.boundary()
    result = window.require_result()
    assert syncs == [0, 6]
    assert result["decode_stage_tokens"] == 6
    assert result["decode_stage_throughput_tps"] == 1
    assert result["context_lengths_start"] == [11, 16]
    assert result["context_lengths_end"] == [14, 19]
    with pytest.raises(RuntimeError, match="duplicate"):
        window.complete(pending[0], decode_tokens=2)
    graph[1]["eager_decode_count"] += 1
    with pytest.raises(RuntimeError, match="graph contract"):
        decode_graph_delta([dict(capture_count=0, replay_count=0, eager_decode_count=0)] * 2, graph, 3)


def test_completed_token_mismatch_is_not_a_measured_step():
    window = PipelinedDecodeWindow(2, 1, 1, synchronize=lambda: None,
                                  clock=lambda: 0, graph_stats=lambda: [])
    ticket = window.submit(is_decode=True, request_ids=[0, 1], tokens=2,
                           admission_complete=True, context_lengths=[10, 10])
    with pytest.raises(RuntimeError, match="Completed decode tokens"):
        window.complete(ticket, decode_tokens=1)
    assert not window.records[0]["completed"]


@pytest.mark.parametrize("measured_steps", [3, 6])
def test_vllm_native_batch_queue_is_drained_at_edges(monkeypatch, tmp_path, measured_steps):
    # Six measured steps end with the workload; no next loop iteration closes the edge.
    from benchmark import vllm_microbench as adapter
    now, syncs = [0.0], []
    graph = dict(capture_count=0, replay_count=0, eager_decode_count=0)
    requests = {k: NS(num_computed_tokens=0, num_prompt_tokens=4) for k in ("a", "b")}
    submitted, completed = [0], [0]

    def schedule():
        counts = {k: 4 if r.num_computed_tokens == 0 else 1 for k, r in requests.items()}
        for k, n in counts.items():
            requests[k].num_computed_tokens += n
        submitted[0] += 1
        return NS(num_scheduled_tokens=counts, preempted_req_ids=[])

    scheduler = NS(requests=requests, schedule=schedule,
                   update_from_output=lambda *a: None, has_requests=lambda: submitted[0] < 9)
    def rpc(fn):
        if fn is adapter._window_worker_boundary:
            assert not core.batch_queue
            syncs.append((submitted[0], completed[0]))
            return [dict(graph)]
        return [0.0]
    core = NS(scheduler=scheduler, batch_queue=deque(), async_scheduling=True,
              model_executor=NS(collective_rpc=rpc))
    def step():
        if scheduler.has_requests():
            out = scheduler.schedule()
            core.batch_queue.appendleft(out)
            if all(v == 1 for v in out.num_scheduled_tokens.values()):
                graph["replay_count"] += 1
            if len(core.batch_queue) < 2:
                return []
        if not core.batch_queue:
            return []
        out = core.batch_queue.pop()
        now[0] += 1
        scheduler.update_from_output(out, NS(sampled_token_ids=[[7], [7]], req_id_to_index={"a": 0, "b": 1}))
        completed[0] += 1
        if completed[0] == 9:
            return [NS(finished=True, request_id=k + "-external", outputs=[NS(token_ids=[7] * 9)]) for k in requests]
        return []
    engine = NS(step=step, has_unfinished_requests=lambda: completed[0] < 9,
                output_processor=NS(request_states={k: NS(external_req_id=k + "-external") for k in requests}))
    monkeypatch.setattr(adapter, "perf_counter", lambda: now[0])
    row, outputs, records = dict(resolved_parallel_topology={"tensor_parallel_size": 1}), [], []
    adapter._run_decode_window(engine, core, 2, 4,
        NS(decode_window_steps=measured_steps, decode_warmup_steps_after_full=2, output_len=9),
        row, outputs, records, tmp_path)
    assert syncs == [(3, 3), (3 + measured_steps, 3 + measured_steps)]
    assert row["decode_stage_tokens"] == 2 * measured_steps
    assert row["decode_stage_elapsed_s"] == measured_steps
    assert row["actual_async_scheduling"] is True
    assert len(outputs) == 2 and completed[0] == 9
    assert {r["request_id"] for r in outputs} == set(requests)
    assert all(r["completed"] for r in records)


def test_hisparse_overlap_queue_processes_each_result_once(monkeypatch, tmp_path):
    import torch
    from benchmark.hisparse_microbench import measured_window_process
    import benchmark.hisparse_microbench as adapter
    now, syncs, processed = [0.0], [], []
    monkeypatch.setenv("HISPARSE_STAGE_STEPS", str(tmp_path / "steps.jsonl"))
    monkeypatch.setenv("HISPARSE_STAGE_BATCH", "2")
    monkeypatch.setenv("HISPARSE_WINDOW_STEPS", "3")
    monkeypatch.setenv("HISPARSE_STAGE_WARMUP", "2")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append(len(processed)))
    monkeypatch.setattr(adapter, "perf_counter", lambda: now[0])

    class Scheduler:
        def __init__(self):
            self.enable_overlap, self.last_batch = True, None
            self.result_queue = deque()
            self.reqs = [NS(rid=k, output_ids=[], retraction_count=0) for k in ("a", "b")]
            self.iteration = 0

        def get_next_batch_to_run(self, **kwargs):
            i = self.iteration
            self.iteration += 1
            return NS(reqs=self.reqs, seq_lens_cpu=torch.tensor([4 + i] * 2),
                      forward_mode=NS(is_decode=lambda: i > 0), extend_num_tokens=8)

        def run_batch(self, batch):
            return NS(can_run_cuda_graph=True, index=self.iteration)

        def process_batch_result(self, batch, result):
            assert result.index not in processed
            processed.append(result.index)
            now[0] += 1
            for req in batch.reqs:
                req.output_ids.append(7)

    def run():
        scheduler = Scheduler()
        for _ in range(9):
            batch = scheduler.get_next_batch_to_run(last_batch=scheduler.last_batch)
            result = scheduler.run_batch(batch)
            scheduler.result_queue.append((batch, result))
            if scheduler.last_batch is not None:
                scheduler.process_batch_result(*scheduler.result_queue.popleft())
            scheduler.last_batch = batch
        while scheduler.result_queue:
            scheduler.process_batch_result(*scheduler.result_queue.popleft())
    measured_window_process(Scheduler, run)
    assert processed == list(range(1, 10))
    assert syncs == [3, 6]
