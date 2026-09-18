"""Compare asynchronous TP with the synchronous execution reference.

This is a correctness runner, not a performance timing definition. All private
model and output paths are supplied by the caller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
import subprocess

import torch

from sparsevllm import LLM, SamplingParams
from sparsevllm.engine.async_scheduler import AsyncScheduler


def run_case(llm, prompts, params, *, asynchronous, cancel_first=False, append_prompt=None):
    assert llm.is_finished()
    if hasattr(llm, '_async_scheduler'):
        del llm._async_scheduler
    llm.model_runner.call('reset_after_warmup')
    if asynchronous:
        llm._async_scheduler = AsyncScheduler(llm)
    torch.manual_seed(123)
    ids = [llm.add_request(p, sp) for p, sp in zip(prompts, params)]
    results = {}
    steps = 0
    while not llm.is_finished():
        finished, _ = llm.step()
        for seq_id, tokens, logprobs, tops in finished:
            results[seq_id] = {'tokens': list(tokens), 'logprobs': list(logprobs), 'top_logprobs': tops}
        steps += 1
        if steps == 1:
            if cancel_first:
                llm.abort_request(ids[0])
            if append_prompt is not None:
                ids.append(llm.add_request(append_prompt, params[-1]))
        if steps > 1024:
            raise RuntimeError('Validation made no bounded progress')
    expected_ids = ids[1:] if cancel_first else ids
    assert set(results) == set(expected_ids)
    return [results[i] for i in expected_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--ep', type=int, default=2)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.7)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'model': args.model, 'cases': [],
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'torch': str(torch.__version__), 'tp': args.tp, 'ep': args.ep}
    destination.write_text(json.dumps(report, indent=2))
    llm = None
    try:
        llm = LLM(args.model, tensor_parallel_size=args.tp, expert_parallel_size=args.ep,
                  gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048,
                  max_num_seqs_in_batch=4, max_decoding_seqs=4, max_num_seqs_in_gpu=4,
                  max_num_batched_tokens=1024, engine_prefill_chunk_size=32,
                  enable_prefix_caching=True, prefix_cache_mode='radix',
                  async_scheduling=True, decode_graph_capture_sizes=[1, 2, 4])
        prompts = [[200+i % 31 for i in range(n)] for n in [19, 79, 113]]
        cases = [
            ('greedy', dict(temperature=0., ignore_eos=True), {}),
            ('penalties_logprobs', dict(temperature=0., ignore_eos=True, repetition_penalty=1.1,
                                       presence_penalty=.2, logprobs=3), {}),
            ('random_sampling', dict(temperature=.8, top_p=.9, top_k=20, ignore_eos=True), {}),
            ('cancellation', dict(temperature=0., ignore_eos=True), {'cancel_first': True}),
            ('new_arrival', dict(temperature=0., ignore_eos=True), {'append_prompt': prompts[0]}),
        ]
        for name, options, extra in cases:
            params = [SamplingParams(max_tokens=n, **options) for n in [17, 11, 7]]
            reference = run_case(llm, prompts, params, asynchronous=False, **extra)
            actual = run_case(llm, prompts, params, asynchronous=True, **extra)
            assert [len(r['tokens']) for r in reference] == [len(r['tokens']) for r in actual]
            equal = [r['tokens'] for r in reference] == [r['tokens'] for r in actual]
            if name != 'random_sampling':
                assert equal, f'{name}: greedy output mismatch'
            if name == 'penalties_logprobs':
                for ref, got in zip(reference, actual):
                    torch.testing.assert_close(torch.tensor(ref['logprobs']), torch.tensor(got['logprobs']), atol=.02, rtol=.02)
            report['cases'].append({'name': name, 'status': 'success', 'tokens_equal': equal,
                                    'reference': reference, 'asynchronous': actual})
            destination.write_text(json.dumps(report, indent=2))
        first = run_case(llm, [prompts[0]], [SamplingParams(max_tokens=1, temperature=0., ignore_eos=True)], asynchronous=False)[0]['tokens'][0]
        params = [SamplingParams(max_tokens=16, temperature=0., eos_token_ids=[first])]
        ref = run_case(llm, [prompts[0]], params, asynchronous=False)
        got = run_case(llm, [prompts[0]], params, asynchronous=True)
        assert ref[0]['tokens'] == got[0]['tokens'] == [first]
        report['cases'].append({'name': 'eos', 'status': 'success', 'reference': ref, 'asynchronous': got})
        report['status'] = 'success'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2))
        if llm is not None:
            llm.exit()


if __name__ == '__main__':
    main()
