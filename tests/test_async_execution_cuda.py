"""Real graph replay and DMA ownership against independent integer arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from sparsevllm.engine.async_execution import AsyncExecution, DeviceLogprobs
from sparsevllm.engine.sequence import Sequence
from sparsevllm.sampling_params import SamplingParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_device_feedback_survives_reorder_graph_output_reuse_and_async_copy():
    device = torch.device('cuda:0')
    source = torch.zeros(2, dtype=torch.long, device=device)
    capture = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            target = (source * 17 + 3) % 101
    stream.synchronize()
    with torch.cuda.graph(capture):
        target = (source * 17 + 3) % 101
    runner = SimpleNamespace(
        device=device, cache_manager=SimpleNamespace(),
        decode_graph_runner=SimpleNamespace(),
        parallel_context=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
        _post_sparse_forward=lambda *_: None,
    )
    execution = AsyncExecution(runner)
    def run(seqs, is_prefill):
        assert not is_prefill
        execution.prepare_inputs(source, seqs)
        capture.replay()
        return target, DeviceLogprobs(target.float())
    runner.run = run
    seqs = [Sequence([7], SamplingParams()), Sequence([11], SamplingParams())]
    expected = {s.seq_id: s.last_token for s in seqs}
    answers = []
    for step in range(12):
        order = seqs if step % 2 else list(reversed(seqs))
        execution.submit(step, order, False)
        expected = {sid: (token*17+3) % 101 for sid, token in expected.items()}
        answers.append([expected[s.seq_id] for s in order])
        if step:
            tokens, (logs, _) = execution.collect(step-1)
            assert tokens == answers[step-1]
            assert logs == [float(x) for x in tokens]
    assert execution.collect(11)[0] == answers[-1]
    assert execution.submitted == execution.completed == 12
    assert not execution.results
