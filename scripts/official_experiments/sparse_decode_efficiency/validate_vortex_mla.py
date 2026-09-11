"""Numerically validate rank-local Vortex QuEST MLA indexer + decode on one GPU.

Exercises the actual compiled indexer and backend forward_decode, without model
weights or distributed communication. Real TP/EP model smoke remains a separate
gate. Cache writes/prefill are outside this test's unchanged decode-only boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from vortex_torch.engine.sgl.attention_backend.cuda_mla_sm90 import VortexCudaMLASM90Backend
from vortex_torch.engine.sgl.attention_backend.cuda_mla_sm90_kernel import allocate_mla_buffers
from vortex_torch.flow.algorithms_mla import QuestMLA
from vortex_torch.indexer import Context
from vortex_torch.indexer.utils_sglang import get_decode_planner_trtllm


def check(backend, q, latent, cache, req_pages, lengths, output):
    """Independent FP32 attention and analytical QuEST upper-bound oracles."""
    md = backend.ctx.metadata
    errors, gaps = [], []
    for b, length in enumerate(lengths):
        sparse_len = int(md.sparse_seqlens[b].item())
        count = (sparse_len + 15) // 16
        pages = md.sparse_block_tables[b, :count].long()
        dense_count = (length + 15) // 16
        dense_pages = req_pages[b, :dense_count].long()
        if not torch.isin(pages, dense_pages).all() or pages.unique().numel() != pages.numel():
            raise AssertionError("Indexer selected duplicate or another request's pages")
        if count != min(dense_count, 128):
            raise AssertionError("Selected budget changed")
        if dense_count > 128:
            expected_reserved = torch.cat((dense_pages[:4], dense_pages[-32:]))
            if not torch.isin(expected_reserved, pages).all():
                raise AssertionError("Missing sink/recent pages")
            middle = dense_pages[4:-32]
            mean = q[b].float().mean(0)
            scores = (cache['cmax'][middle, 0].float() @ mean.clamp_min(0)
                      + cache['cmin'][middle, 0].float() @ mean.clamp_max(0))
            chosen = pages[~torch.isin(pages, expected_reserved)]
            chosen_scores = (cache['cmax'][chosen, 0].float() @ mean.clamp_min(0)
                             + cache['cmin'][chosen, 0].float() @ mean.clamp_max(0))
            cutoff = scores.topk(92).values[-1]
            # BF16 intermediate mean/GEMMs can exchange near-tied top-k pages.
            # Bound the score deficit, not equality of a discontinuous top-k set.
            gap = ((cutoff - chosen_scores.min()).clamp_min(0) / cutoff.abs().clamp_min(1)).item()
            if gap > 0.02:
                torch.save(dict(q=q.cpu(), pages=pages.cpu(), dense_pages=dense_pages.cpu(),
                                chosen=chosen.cpu(), scores=scores.cpu(), chosen_scores=chosen_scores.cpu(),
                                compiled_buffers={k: v.cpu() for k, v in vars(backend.compiled_indexer).items()
                                                  if isinstance(v, torch.Tensor)}),
                           backend.diagnostic_dir / f'selection-failure-b{b}.pt')
                raise AssertionError(f"QuEST score deficit exceeds 2%: {gap}")
            gaps.append(gap)
        slots = (pages[:, None] * 16 + torch.arange(16, device=q.device)).flatten()[:sparse_len]
        keys = latent[slots].float()
        reference = torch.softmax(q[b].float() @ keys.T / (576 ** 0.5), dim=-1) @ keys[:, :512]
        actual = output.view(q.shape[0], q.shape[1], 512)[b].float()
        torch.testing.assert_close(actual, reference, atol=0.015, rtol=0.02)
        errors.append((actual - reference).abs().max().item())
    return dict(max_abs_error=max(errors), max_selection_score_deficit=max(gaps, default=0))


def run_case(output_dir, batch, heads, context):
    device, dtype = torch.device('cuda'), torch.bfloat16
    torch.manual_seed(4200 + batch + heads + context)
    npage = (context + 15) // 16
    total_pages = batch * npage
    latent = torch.randn(total_pages * 16, 576, device=device, dtype=dtype)
    blocks = latent.view(total_pages, 16, 576)
    cache = dict(latent=blocks, cmin=blocks.amin(1, keepdim=True), cmax=blocks.amax(1, keepdim=True))
    pages = torch.randperm(total_pages, device=device, dtype=torch.int32).view(batch, npage)
    req_to_token = (pages[:, :, None] * 16 + torch.arange(16, device=device)).reshape(batch, -1).int()
    req_indices = torch.arange(batch, device=device, dtype=torch.int64)
    lengths = [context - (b % 3) * 17 for b in range(batch)]
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    q = torch.randn(batch, heads, 576, device=device, dtype=dtype)
    flow = QuestMLA()
    flow.initialize(16, 512, 64, dtype, dtype)
    config = NS(page_size=16, vortex_block_size=16, vortex_max_seq_lens=-1,
                vortex_workload_chunk_size=64, vortex_topk_val=92, vortex_max_topk_val=128,
                vortex_topk_ratio=0, vortex_dtype='bfloat16', vortex_block_reserved_bos=4,
                vortex_block_reserved_eos=32, vortex_compilation_cache_dir=str(output_dir / 'indexer'),
                vortex_impl_backend='triton', vortex_attention_backend='trtllm', vortex_use_tensor_core=True)
    runner = NS(device=device, server_args=config, model_config=NS(context_len=context),
                req_to_token_pool=NS(size=batch))
    # Use real backend methods after supplying the rank-local serving contract;
    # no mocked kernels, indexer, metadata planner, or attention implementation.
    backend = object.__new__(VortexCudaMLASM90Backend)
    backend.diagnostic_dir = output_dir
    backend.num_qo_heads = backend.group_size = heads
    backend.num_kv_heads = 1
    backend.head_dim = backend.kv_cache_dim = 576
    backend.kv_lora_rank = 512
    backend.q_data_type = dtype
    backend.sparse_attention = flow
    backend.ctx = Context()
    backend.layers_skip = []
    backend.block_size = 16
    backend._max_bs = batch
    backend._decoders = {}
    backend._compile(runner)
    backend._mla_buffers = allocate_mla_buffers(batch, heads, 16,
                                               backend.ctx.metadata.sparse_block_tables.size(1), device)
    planner = get_decode_planner_trtllm()
    pool = NS(get_cache=lambda layer: cache, get_key_buffer=lambda layer: latent)
    forward = NS(token_to_kv_pool=pool)
    layer = NS(layer_id=2, tp_q_head_num=heads, scaling=1 / (576 ** 0.5))

    def prepare():
        planner(seq_lens, req_to_token, req_indices, backend.ctx)
        backend._plan(seq_lens)

    def decode():
        return backend.forward_decode(q, None, None, layer, forward, save_kv_cache=False)

    prepare()
    for _ in range(3):
        result = decode()
    torch.cuda.synchronize()
    rows = [dict(mode='eager', **check(backend, q, latent, cache, pages, lengths, result))]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = decode()
    for replay in range(3):
        q.normal_()
        lengths = [context - replay - (b % 3) * 17 for b in range(batch)]
        seq_lens.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        for key in ('mid_o', 'mid_m', 'mid_l'):
            backend._mla_buffers[key].fill_(float('nan'))
        prepare()
        graph.replay()
        torch.cuda.synchronize()
        rows.append(dict(mode='graph', replay=replay,
                         **check(backend, q, latent, cache, pages, lengths, captured_result)))
    return dict(status='success', batch=batch, local_heads=heads, context=context, checks=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    rows = []
    with (args.output_dir / 'results.jsonl').open('x', buffering=1) as stream:
        for batch, heads, context in [(1,10,2049),(2,10,4097),(3,10,32768),
                                     (6,10,131072),(23,10,32768),(1,20,32768),(2,20,4097)]:
            case_dir = args.output_dir / f'b{batch}-h{heads}-c{context}'
            case_dir.mkdir()
            try:
                row = run_case(case_dir, batch, heads, context)
            except Exception as error:
                stream.write(json.dumps(dict(status='failed', batch=batch, local_heads=heads,
                                             context=context, error=repr(error))) + '\n')
                raise
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(dict(status='success', cases=rows,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
        scope='rank-local GPU indexer + decode; distributed model smoke required separately'), indent=2) + '\n')


if __name__ == '__main__':
    main()
