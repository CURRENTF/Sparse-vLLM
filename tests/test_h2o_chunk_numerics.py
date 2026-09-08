"""Real CUDA score/compaction/attention paths against logical-token oracles."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparsevllm.engine.cache_manager.base import (
    AttentionViewMeta, DecodeComputeView, ExplicitKVPayload, LayerBatchStates, PrefillComputeView,
)
from sparsevllm.engine.cache_manager.h2o import H2OCacheManager
from sparsevllm.engine.cache_manager.storage import ExplicitKVStorage
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_methods.base import PrefillScoreEvent, SparseStepContext
from sparsevllm.engine.sparse_methods.h2o import H2ORuntime
from sparsevllm.kernels.triton.context_flashattention_nopad import context_attention_fwd
from sparsevllm.kernels.triton.prefill_score import PrefillScoreWorkspace
from sparsevllm.operators.decode_attention import DecodeAttentionOpSpec, prepare_decode_attention_op
from sparsevllm.utils.context import reset_context, set_context

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def _manager(heads, kv_heads, length, budget):
    manager = object.__new__(H2OCacheManager)
    manager.device = torch.device('cuda:0')
    manager.num_kv_heads = kv_heads
    manager.num_layers = manager.num_kv_layers = 1
    manager.runtime_layout = SimpleNamespace(kv_layer_index=lambda layer: layer, kv_idx_to_layer_idx=(0,))
    manager.max_model_len = length + 1
    manager.attention_cache_storage = ExplicitKVStorage(num_kv_heads=kv_heads, head_dim=32, dtype=torch.bfloat16)
    manager.attention_cache_storage.allocate(num_layers=1, num_slots=length + 1, device=manager.device)
    manager.seq_id_to_row = [{7: 0}]
    manager.row_seq_lens = [np.zeros(1, dtype=np.int32)]
    manager.layer_batch_states = [LayerBatchStates()]
    manager.buffer_req_to_token_slots = [torch.zeros(1, length + 1, dtype=torch.int32, device=manager.device)]
    manager.free_slots_stack = [torch.randperm(length + 1, device=manager.device).int()]
    manager._num_free_slots = [length + 1]
    manager._h2o_scores, manager._h2o_positions = {}, {}
    manager._h2o_counters = dict(final_prefill_evictions=0, intermediate_prefill_evictions=0, dropped_tokens=0)
    manager.config = SimpleNamespace(sparse_method='h2o', prefill_sparse_method='h2o_prefill', h2o_prefill_budget=budget, h2o_decode_budget=budget,
                                     h2o_prefill_score_window=0, h2o_recent_ratio=.25)
    runtime = object.__new__(H2ORuntime)
    runtime.config, runtime.cache_manager = manager.config, manager
    runtime._prefill_score_workspace = PrefillScoreWorkspace()
    runtime._prefill_head_score_buffer = runtime._prefill_head_score_total = None
    return manager, runtime


def _run_chunks(q, k, v, chunks, budget):
    length, heads, dim = q.shape
    kv_heads = k.shape[1]
    group_size = heads // kv_heads
    manager, runtime = _manager(heads, kv_heads, length, budget)
    seq = Sequence(list(range(length)))
    seq.seq_id = 7
    histories = [[] for _ in range(kv_heads)]
    score_reference = [dict() for _ in range(heads)]
    output_chunks = []
    start = 0
    decode = None
    try:
        for chunk in chunks:
            seq.num_prefilled_tokens, seq.current_chunk_size = start, chunk
            manager._prepare_prefill([seq])
            state = manager.layer_batch_states[0]
            new_slots = state.slot_mapping.long()
            storage = manager.attention_cache_storage
            storage.cache[0, 0, new_slots] = k[start:start + chunk]
            storage.cache[1, 0, new_slots] = v[start:start + chunk]
            for history in histories:
                history.extend(range(start, start + chunk))
            resident = len(histories[0])
            zero = torch.zeros(1, device=q.device, dtype=torch.int32)
            chunk_tensor = torch.tensor([chunk], device=q.device, dtype=torch.int32)
            prefix = state.context_lens - chunk_tensor
            view = PrefillComputeView(
                meta=AttentionViewMeta(active_slots=manager.buffer_req_to_token_slots[0],
                                       req_indices=zero, context_lens=state.context_lens,
                                       max_context_len=resident),
                payload=ExplicitKVPayload(k_cache=storage.cache[0, 0], v_cache=storage.cache[1, 0]),
            )
            query = q[start:start + chunk]
            output = torch.empty_like(query)
            context_attention_fwd(query, view.payload.k_cache, view.payload.v_cache, output,
                                  zero, zero, state.context_lens, prefix, chunk, view.meta.active_slots)
            expected = torch.empty_like(query, dtype=torch.float32)
            # The oracle reads original tokens, never the packed production KV.
            for head in range(heads):
                group = head // group_size
                history = histories[group]
                keys = k[history, group].float()
                logits = query[:, head].float() @ keys.T * dim ** -.5
                causal = torch.tensor(history, device=q.device)[None] > torch.arange(start, start + chunk, device=q.device)[:, None]
                probability = logits.masked_fill(causal, -torch.inf).softmax(-1)
                expected[:, head] = probability @ v[history, group].float()
                for token, mass in zip(history, probability.sum(0).tolist()):
                    score_reference[head][token] = score_reference[head].get(token, 0.) + mass
            torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
            set_context(True, cache_manager=manager, seqs=[seq])
            runtime.collect_prefill_attention_score(PrefillScoreEvent(0, query, view, zero, chunk_tensor, dim ** -.5))
            for head in range(heads):
                expected_scores = torch.tensor([score_reference[head][token] for token in histories[head // group_size]], device=q.device)
                torch.testing.assert_close(manager._h2o_scores[0, 7][head], expected_scores, rtol=5e-3, atol=5e-3)
            runtime.finish_step(SparseStepContext([seq], True, None))
            if resident > budget:
                recent = max(1, int(budget * .25))
                for group, history in enumerate(histories):
                    def importance(token):
                        scores = [score_reference[h][token] for h in range(group * group_size, (group + 1) * group_size)]
                        return max(scores)
                    heavy = sorted(history[:-recent], key=lambda token: (-importance(token), token))[:budget - recent]
                    histories[group] = sorted(heavy + history[-recent:])
            assert manager._h2o_positions[0, 7].tolist() == histories
            for head in range(heads):
                score_reference[head] = {token: score_reference[head][token] for token in histories[head // group_size]}
                torch.testing.assert_close(manager._h2o_scores[0, 7][head], torch.tensor(list(score_reference[head].values()), device=q.device), rtol=5e-3, atol=5e-3)
            assert manager._num_free_slots[0] + len(histories[0]) == length + 1
            output_chunks.append(output)
            start += chunk
        # Exercise the registered native decode provider on the compressed KV.
        resident = len(histories[0])
        context = torch.tensor([resident], device=q.device, dtype=torch.int32)
        set_context(False, cache_manager=manager, seqs=[seq])
        decode = prepare_decode_attention_op(DecodeAttentionOpSpec(
            num_query_heads=heads, num_kv_heads=kv_heads, head_dim=dim,
            activation_dtype=q.dtype, softmax_scale=dim ** -.5,
            max_batch_size=1, cuda_graph=False, layer_varying_page_table=True,
        ), device_index=0)
        output = decode.run(q[:1], DecodeComputeView(
            meta=AttentionViewMeta(active_slots=view.meta.active_slots, req_indices=zero,
                                   context_lens=context, max_context_len=resident),
            payload=view.payload,
        ))
        for head in range(heads):
            group = head // group_size
            keys, values = k[histories[group], group].float(), v[histories[group], group].float()
            expected = (q[0, head].float() @ keys.T * dim ** -.5).softmax(-1) @ values
            torch.testing.assert_close(output[0, head].float(), expected, rtol=2e-2, atol=2e-2)
        return torch.cat(output_chunks), manager._h2o_scores[0, 7].clone()
    finally:
        if decode is not None:
            decode.close()
        reset_context()


@pytest.mark.parametrize('kv_heads', [4, 2])
def test_chunk_attention_and_retention_match_logical_token_oracle(kv_heads):
    torch.manual_seed(57)
    q = torch.randn(35, 4, 32, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(35, kv_heads, 32, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    _run_chunks(q, k, v, [11, 7, 17], budget=9)


@pytest.mark.parametrize('kv_heads', [4, 2])
def test_no_eviction_full_and_chunked_prefill_are_equivalent(kv_heads):
    torch.manual_seed(61)
    q = torch.randn(149, 4, 32, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(149, kv_heads, 32, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    full_output, full_scores = _run_chunks(q, k, v, [149], budget=149)
    for chunks in ([13, 129, 7], [71, 78]):
        output, scores = _run_chunks(q, k, v, chunks, budget=149)
        torch.testing.assert_close(output, full_output, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(scores, full_scores, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize('budget', [9, 37])
def test_mla_chunk_scores_and_outputs_follow_shared_latent_retention(budget):
    from sparsevllm.engine.cache_manager.base import MlaLatentPayload
    from sparsevllm.engine.cache_manager.storage import MlaLatentStorage
    from test_mla_attention_layer import _attention

    torch.manual_seed(73)
    attention = _attention(device='cuda', tp_size=1)
    heads, length = attention.spec.local_q_heads, 37
    latent = torch.randn(length, 512, device='cuda', dtype=torch.bfloat16)
    rope = torch.randn(length, 64, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(512, heads * 448, device='cuda', dtype=torch.bfloat16) * .025
    original_projection = (latent @ weights).reshape(length, heads, 448)
    q = torch.randn(length, heads, 256, device='cuda', dtype=torch.bfloat16)

    def run(chunks):
        manager, runtime = _manager(heads, 1, length, budget)
        storage = MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
        storage.allocate(num_layers=1, num_slots=length + 1, device=manager.device)
        manager.attention_cache_storage = storage
        seq = Sequence(list(range(length)))
        seq.seq_id = 7
        history, outputs = [], []
        cumulative = torch.zeros(heads, length, device='cuda')
        start = 0
        try:
            for chunk in chunks:
                seq.num_prefilled_tokens, seq.current_chunk_size = start, chunk
                manager._prepare_prefill([seq])
                state = manager.layer_batch_states[0]
                slots = state.slot_mapping.long()
                storage.latent_cache[0, slots, 0] = latent[start:start + chunk]
                storage.rope_cache[0, slots, 0] = rope[start:start + chunk]
                history.extend(range(start, start + chunk))
                zero = torch.zeros(1, device='cuda', dtype=torch.int32)
                chunk_tensor = torch.tensor([chunk], device='cuda', dtype=torch.int32)
                set_context(True, cache_manager=manager, seqs=[seq])
                view = PrefillComputeView(
                    meta=AttentionViewMeta(active_slots=manager.buffer_req_to_token_slots[0],
                                           req_indices=zero, context_lens=state.context_lens,
                                           max_context_len=len(history)),
                    payload=MlaLatentPayload(latent_cache=storage.latent_cache[0], rope_cache=storage.rope_cache[0]),
                )
                gathered = attention.prepare_prefill_history(view, query_tokens=chunk)
                projected = (gathered.gathered_latent @ weights).reshape(len(history), heads, 448)
                workset = attention.bind_prefill_kv(
                    gathered,
                    expanded_k=torch.cat((projected[..., :192], gathered.gathered_rope[:, None].expand(-1, heads, -1)), -1),
                    expanded_v=projected[..., 192:].contiguous(),
                )
                query = q[start:start + chunk]
                output = attention.run_prefill(query, workset, b_start_loc=zero, chunk_lens=chunk_tensor)
                expected = torch.empty_like(query, dtype=torch.float32)
                mask = torch.tensor(history, device='cuda')[None] > torch.arange(start, start + chunk, device='cuda')[:, None]
                for head in range(heads):
                    # Original latent projection and separate RoPE term form
                    # an oracle independent of the physical gather/compaction.
                    logits = (query[:, head, :192].float() @ original_projection[history, head, :192].float().T
                              + query[:, head, 192:].float() @ rope[history].float().T) * attention.spec.softmax_scale
                    probability = logits.masked_fill(mask, -torch.inf).softmax(-1)
                    expected[:, head] = probability @ original_projection[history, head, 192:].float()
                    cumulative[head, history] += probability.sum(0)
                torch.testing.assert_close(output.float(), expected, rtol=3e-2, atol=3e-2)
                runtime.collect_prefill_attention_score(PrefillScoreEvent(
                    0, query, attention.build_prefill_explicit_view(workset), zero, chunk_tensor, attention.spec.softmax_scale,
                ))
                torch.testing.assert_close(manager._h2o_scores[0, 7], cumulative[:, history], rtol=5e-3, atol=5e-3)
                runtime.finish_step(SparseStepContext([seq], True, None))
                if len(history) > budget:
                    recent = max(1, int(budget * .25))
                    heavy = sorted(history[:-recent], key=lambda token: (-float(cumulative[:, token].max()), token))[:budget - recent]
                    history = sorted(heavy + history[-recent:])
                assert manager._h2o_positions[0, 7].tolist() == [history]
                torch.testing.assert_close(manager._h2o_scores[0, 7], cumulative[:, history], rtol=5e-3, atol=5e-3)
                retained = manager.buffer_req_to_token_slots[0][0, :len(history)].long()
                torch.testing.assert_close(storage.latent_cache[0, retained, 0], latent[history])
                torch.testing.assert_close(storage.rope_cache[0, retained, 0], rope[history])
                assert manager._num_free_slots[0] + len(history) == length + 1
                outputs.append(output)
                start += chunk
            return torch.cat(outputs), manager._h2o_scores[0, 7].clone()
        finally:
            reset_context()

    output, scores = run([13, 17, 7])
    if budget >= length:
        full_output, full_scores = run([length])
        torch.testing.assert_close(output, full_output, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(scores, full_scores, rtol=5e-3, atol=5e-3)
