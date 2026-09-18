"""Scheduling and ownership contracts; CUDA equivalence is tested separately."""
from types import SimpleNamespace

import pytest
import torch

from sparsevllm.engine.async_execution import AsyncExecution
from sparsevllm.engine.async_scheduler import AsyncScheduler, execution_snapshot
from sparsevllm.engine.sequence import Sequence
from sparsevllm.sampling_params import SamplingParams
from test_prefill_schedule_policy import make_scheduler


class Executor:
    def __init__(self):
        self.pending = {}
        self.calls = []
        self.retired = set()

    def call(self, method, *args):
        self.calls.append((method, args))
        if method == 'submit_async':
            ticket, seqs, is_prefill = args
            self.pending[ticket] = (seqs, is_prefill)
        elif method == 'collect_async':
            seqs, prefill = self.pending.pop(args[0])
            return [1000 + (s.num_prompt_tokens if prefill else s.num_tokens) for s in seqs], None
        elif method == 'retire_async':
            referenced = {s.seq_id for seqs, _ in self.pending.values() for s in seqs}
            assert not referenced.intersection(args[0]), 'freed in-flight storage'
            self.retired.update(args[0])
        else:
            raise AssertionError(method)


def engine(depth=2, chunk=3):
    scheduler = make_scheduler('all_chunked', chunk=chunk, max_tokens=12)
    executor = Executor()
    def no_sync():
        raise AssertionError('unexpected synchronous execution')
    result = SimpleNamespace(
        config=SimpleNamespace(async_max_inflight=depth), scheduler=scheduler,
        model_runner=executor, _release_preempted_sequences=lambda _: None,
        _step_sync=no_sync,
        _throughput_logger=SimpleNamespace(record_step=lambda _: None),
    )
    return result, AsyncScheduler(result)


def request(length, outputs, **kwargs):
    return Sequence(list(range(1, length+1)), SamplingParams(max_tokens=outputs, **kwargs))


@pytest.mark.parametrize('depth', [2, 3, 4])
@pytest.mark.parametrize('outputs', [1, 2, 7])
def test_chunked_prefill_and_output_limits_without_placeholder_tokens(depth, outputs):
    e, driver = engine(depth)
    seqs = [request(5, outputs, ignore_eos=True), request(8, outputs, ignore_eos=True)]
    for seq in seqs:
        e.scheduler.add(seq)
    finished = []
    for _ in range(40):
        done, _ = driver.step()
        finished.extend(done)
        if not driver.pending and e.scheduler.is_finished():
            break
    else:
        pytest.fail('scheduler did not drain')
    assert len(finished) == len(seqs)
    for seq in seqs:
        assert seq.completion_token_ids == list(range(1000+seq.num_prompt_tokens, 1000+seq.num_prompt_tokens+outputs))
        assert seq.num_pending_outputs == 0
    assert e.model_runner.retired == {s.seq_id for s in seqs}
    methods = [m for m, _ in e.model_runner.calls]
    assert methods[:2] == ['submit_async', 'submit_async']


def test_eos_discards_lookahead_and_retires_only_after_completion():
    e, driver = engine(depth=3, chunk=16)
    seq = request(5, 20, eos_token_ids=[1005])
    e.scheduler.add(seq)
    done, _ = driver.step()
    assert done[0][1] == [1005]
    assert seq.num_pending_outputs == 0
    assert not driver.pending
    assert seq.seq_id in e.model_runner.retired


def test_abort_last_request_drains_gpu_owners_without_publishing_more_tokens():
    e, driver = engine(chunk=16)
    seq = request(5, 10, ignore_eos=True)
    e.scheduler.add(seq)
    driver.step()
    assert driver.pending
    before = list(seq.completion_token_ids)
    driver.abort(seq.seq_id)
    assert seq.completion_token_ids == before
    assert not driver.pending
    assert e.scheduler.is_finished()
    assert e.model_runner.retired == {seq.seq_id}


def test_new_request_can_join_while_prior_output_is_in_flight():
    e, driver = engine(chunk=16)
    first = request(5, 8, ignore_eos=True)
    e.scheduler.add(first)
    driver.step()
    later = request(7, 3, ignore_eos=True)
    e.scheduler.add(later)
    done = []
    for _ in range(30):
        rows, _ = driver.step()
        done.extend(rows)
        if e.scheduler.is_finished() and not driver.pending:
            break
    assert {x[0] for x in done} == {first.seq_id, later.seq_id}
    assert len(first.completion_token_ids) == 8
    assert len(later.completion_token_ids) == 3


def test_execution_snapshot_advances_position_without_fabricating_history():
    seq = request(5, 8, ignore_eos=True)
    seq.num_pending_outputs = 2
    snapshot = execution_snapshot(seq)
    assert snapshot.decode_input_position == seq.decode_input_position + 2
    assert snapshot.token_ids == seq.token_ids
    seq.append_token(17)
    assert snapshot.token_ids == [1, 2, 3, 4, 5]


def test_device_penalty_state_matches_independent_value_formula():
    state = object.__new__(AsyncExecution)
    state.penalties = {}
    seq = Sequence([1, 3], SamplingParams(repetition_penalty=1.3, presence_penalty=.7))
    seq.append_token(2)
    logits = torch.tensor([[.1, -2., 3., 4., -1.]])
    result = state.apply_penalties(logits, [seq])
    expected = logits.clone()
    for token in [1, 2, 3]:
        expected[0, token] = expected[0, token] / 1.3 if expected[0, token] > 0 else expected[0, token] * 1.3
    expected[0, 2] -= .7
    torch.testing.assert_close(result, expected)
    assert torch.equal(logits, torch.tensor([[.1, -2., 3., 4., -1.]]))
