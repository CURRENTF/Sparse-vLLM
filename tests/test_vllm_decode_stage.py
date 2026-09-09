"""Independent step-work oracle for the optional vLLM stage adapter, without CUDA."""
import json
import sys
from types import SimpleNamespace

import pytest

from benchmark import vllm_microbench as bench


def stage_args(tmp_path):
    return SimpleNamespace(
        output_dir=str(tmp_path), output_len=4, model_path="fixture-model",
        hyper_params_dict={}, max_model_len_override=None,
        temperature=0.0, top_p=1.0, decode_warmup_steps_after_full=1,
        max_decode_steps_after_full=0, synchronize_step_timing=True,
        require_full_decode_batch=True, admission_wave_size=0,
        wave_decode_gap_steps=0, require_prefix_cache_hit=False,
    )


def install_engine(monkeypatch, *, preempt=False, short_output=False):
    # First-token prefill, mixed admission, warmup, full decode, falling tail.
    work = [{"0": 4}, {"0": 1, "1": 4}, {"0": 1, "1": 1},
            {"0": 1, "1": 1}, {"1": 1}]
    state = SimpleNamespace(clock=0.0, closed=False, constructed=False)

    class Scheduler:
        def __init__(self):
            self.requests = {}
            self.index = 0

        def schedule(self):
            counts = work[self.index]
            self.index += 1
            for key, tokens in counts.items():
                self.requests[key].num_computed_tokens += tokens
            return SimpleNamespace(num_scheduled_tokens=counts,
                                   preempted_req_ids=["0"] if preempt and self.index == 4 else [])

    class Executor:
        def collective_rpc(self, callback):
            if callback is bench._worker_peak_memory:
                return [1.0, 2.0]
            state.clock += 0.25  # All-worker completion barrier is timed too.
            return [None, None]

    class Engine:
        def __init__(self):
            self.scheduler = Scheduler()
            self.generated = {}
            core = SimpleNamespace(batch_queue=None, async_scheduling=False,
                                   scheduler=self.scheduler, model_executor=Executor())
            self.engine_core = SimpleNamespace(engine_core=core, shutdown=self.shutdown)

        def shutdown(self):
            state.closed = True

        def add_request(self, key, prompt, params):
            self.scheduler.requests[key] = SimpleNamespace(
                num_computed_tokens=0, num_prompt_tokens=len(prompt["prompt_token_ids"]))
            self.generated[key] = 0

        def has_unfinished_requests(self):
            return bool(self.scheduler.requests)

        def step(self):
            scheduled = self.scheduler.schedule()
            state.clock += 1.0
            outputs = []
            for key in scheduled.num_scheduled_tokens:
                self.generated[key] += 1
                finished = self.generated[key] == 4
                tokens = [7] * self.generated[key]
                if finished:
                    del self.scheduler.requests[key]
                    if short_output and key == "0":
                        tokens.pop()
                outputs.append(SimpleNamespace(
                    request_id=key, finished=finished,
                    outputs=[SimpleNamespace(token_ids=tokens)]))
            return outputs

    def llm(**kwargs):
        state.constructed = True
        state.config = kwargs
        return SimpleNamespace(llm_engine=Engine())

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        __version__="fixture", __file__=__file__, LLM=llm, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(bench, "perf_counter", lambda: state.clock)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    return state


@pytest.mark.parametrize("write_artifacts", [True, False])
def test_stage_counts_only_full_decode_after_warmup(tmp_path, monkeypatch, write_artifacts):
    """Protect scheduler-before-mutation classification and optional output_dir."""
    args = stage_args(tmp_path)
    if not write_artifacts:
        args.output_dir = None
    state = install_engine(monkeypatch)
    rows = {}
    bench.benchmark_decode_stage("vanilla", 4, 2, args, rows)
    row = rows[("vanilla", 4, 2)]
    assert row["status"] == "SUCCESS", row
    assert row["decode_stage_tokens"] == 2
    assert row["decode_stage_elapsed_s"] == 1.25
    assert row["decode_stage_throughput_tps"] == 2 / 1.25
    assert row["completed_requests"] == row["actual_decode_peak"] == 2
    assert state.closed
    if write_artifacts:
        steps = [json.loads(line) for line in (tmp_path / "vanilla-4-2/steps.jsonl").read_text().splitlines()]
        assert [step["measured"] for step in steps] == [False, False, False, True, False]
    else:
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ["preempt", "short_output"])
def test_failed_stage_keeps_evidence_and_closes_engine(tmp_path, monkeypatch, fault):
    args = stage_args(tmp_path)
    state = install_engine(monkeypatch, **{fault: True})
    rows = {}
    bench.benchmark_decode_stage("vanilla", 4, 2, args, rows)
    assert rows[("vanilla", 4, 2)]["status"] == "FAILED"
    assert (tmp_path / "vanilla-4-2/steps.jsonl").is_file()
    assert (tmp_path / "vanilla-4-2/raw_outputs.jsonl").is_file()
    assert state.closed


@pytest.mark.parametrize("override,hp", [
    ({"admission_wave_size": 1}, {}),
    ({"require_prefix_cache_hit": True}, {}),
    ({"decode_warmup_steps_after_full": -1}, {}),
    ({"synchronize_step_timing": False}, {}),
    ({}, {"tensor_parallel_size": 4, "expert_parallel_size": 2}),
    ({}, {"max_num_batched_tokens": 8, "engine_prefill_chunk_size": 4}),
])
def test_unrepresentable_protocol_fails_before_model_loading(tmp_path, monkeypatch, override, hp):
    """Do not publish metrics for silently remapped topology or prefill semantics."""
    args = stage_args(tmp_path)
    vars(args).update(override)
    args.hyper_params_dict = hp
    state = install_engine(monkeypatch)
    rows = {}
    bench.benchmark_decode_stage("vanilla", 4, 2, args, rows)
    assert rows[("vanilla", 4, 2)]["status"] == "FAILED"
    assert not state.constructed


@pytest.mark.parametrize("extra", [{"model": "different"}, {"seed": 17}, {"async_scheduling": True}])
def test_fork_options_cannot_override_measured_workload(tmp_path, monkeypatch, extra):
    args = stage_args(tmp_path)
    args.engine_kwargs_dict = extra
    state = install_engine(monkeypatch)
    rows = {}
    bench.benchmark_decode_stage("vanilla", 4, 2, args, rows)
    assert rows[("vanilla", 4, 2)]["status"] == "FAILED"
    assert not state.constructed


def test_labelled_fork_retains_stage_accounting(tmp_path, monkeypatch):
    args = stage_args(tmp_path)
    args.backend_label = "tangram-snapkv"
    args.engine_kwargs_dict = {"compression_scorer": "snapkv", "compression_budget_tokens": 4096}
    state = install_engine(monkeypatch)
    rows = {}
    bench.benchmark_decode_stage("snapkv", 4, 2, args, rows)
    row = rows[("snapkv", 4, 2)]
    assert row["status"] == "SUCCESS"
    assert row["decode_stage_tokens"] == 2
    assert row["decode_stage_elapsed_s"] == 1.25
    assert state.closed
