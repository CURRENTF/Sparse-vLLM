"""Independent scheduler budgets cannot be spent twice within a mixed batch."""

import pytest

from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import PREFILL_POLICY_ALL_CHUNKED, PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
from test_prefill_schedule_policy import FakeMemoryOracle, make_scheduler


class Resources(FakeMemoryOracle):
    def __init__(self, budgets, costs, **kwargs):
        super().__init__(**kwargs)
        self.budgets, self.costs = budgets, costs

    def step_resource_budgets(self, *, is_prefill):
        return dict(self.budgets)

    def step_resource_costs(self, seq, scheduled_tokens, *, is_prefill):
        return self.costs(seq, scheduled_tokens)


def test_decode_reserves_each_pool_and_can_fill_with_zero_cost_requests():
    # The third request needs the already consumed coarse slot. A later
    # request crosses no allocation boundary and must still be runnable.
    seqs = [Sequence([index + 1]) for index in range(4)]
    costs = {seqs[0].seq_id: {"fine": 1, "coarse": 0},
             seqs[1].seq_id: {"fine": 1, "coarse": 1},
             seqs[2].seq_id: {"fine": 1, "coarse": 1},
             seqs[3].seq_id: {"fine": 0, "coarse": 0}}
    oracle = Resources({"fine": 3, "coarse": 1}, lambda seq, _: costs[seq.seq_id])
    scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, oracle=oracle)
    scheduler.decoding.extend(seqs)
    scheduled, prefill, preempted = scheduler.schedule()
    assert scheduled == [seqs[0], seqs[1], seqs[3]]
    assert not prefill and not preempted
    assert oracle.budgets == {"fine": 3, "coarse": 1}


def test_prefill_shortens_later_chunks_to_remaining_physical_budgets():
    # Token capacity remains available after the first request consumes the
    # last page; later chunks may stop before their next page boundary.
    oracle = Resources({"pairs": 4, "pages": 1},
                       lambda seq, tokens: {"pairs": (tokens + 1) // 2, "pages": tokens // 3})
    scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, chunk=4, max_tokens=12, oracle=oracle)
    seqs = [Sequence([index + 1] * 4) for index in range(3)]
    scheduler.waiting.extend(seqs)
    scheduled, prefill, preempted = scheduler.schedule()
    assert scheduled == seqs and prefill and not preempted
    assert [seq.current_chunk_size for seq in scheduled] == [4, 2, 2]


def test_whole_prefill_is_not_split_to_fit_an_extra_pool():
    oracle = Resources({"pages": 1}, lambda seq, tokens: {"pages": tokens // 3}, force_full_prefill=True)
    scheduler = make_scheduler(PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH, chunk=4, max_tokens=12, oracle=oracle)
    seqs = [Sequence([index + 1] * 4) for index in range(2)]
    scheduler.waiting.extend(seqs)
    scheduled, prefill, preempted = scheduler.schedule()
    assert scheduled == seqs[:1] and prefill and not preempted
    assert scheduled[0].current_chunk_size == 4


def test_blocked_partial_prefill_reports_resource_exhaustion_instead_of_spinning():
    oracle = Resources({"pages": 0}, lambda seq, tokens: {"pages": tokens})
    scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, oracle=oracle)
    seq = Sequence([1, 2])
    seq.num_prefilled_tokens = 1
    scheduler.waiting.append(seq)
    with pytest.raises(RuntimeError, match="physical resource budgets"):
        scheduler.schedule()
