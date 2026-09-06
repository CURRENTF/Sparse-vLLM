"""Opt-in real-model H2O lifecycle check on an exclusively available GPU.

Set SPARSEVLLM_H2O_TEST_MODEL to a local checkpoint. Artifacts can be retained
with SPARSEVLLM_H2O_TEST_OUTPUT; this is a correctness smoke test, not a quality
or performance benchmark.
"""

import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(not os.environ.get('SPARSEVLLM_H2O_TEST_MODEL'), reason='requires an explicit local model')
def test_model_chunk_prefill_and_native_decode_handoff(tmp_path):
    import torch
    from sparsevllm import LLM, SamplingParams
    from sparsevllm.engine.cache_manager.storage import MlaLatentStorage

    model = os.environ['SPARSEVLLM_H2O_TEST_MODEL']
    output_dir = Path(os.environ.get('SPARSEVLLM_H2O_TEST_OUTPUT', str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(sparse_method='h2o', h2o_head_reduction='max',
                  h2o_prefill_budget=64, h2o_decode_budget=32,
                  h2o_prefill_score_window=0, engine_prefill_chunk_size=64,
                  max_num_batched_tokens=128, max_num_seqs_in_batch=2,
                  max_num_seqs_in_gpu=2, max_model_len=512,
                  decode_graph=False, enable_prefix_caching=False,
                  validate_runtime_invariants=True)
    artifact = dict(model=model, config=config, seed=19,
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    samples=[], steps=[])
    engine = None
    try:
        torch.manual_seed(19)
        engine = LLM(model, **config)
        manager = engine.model_runner.cache_manager
        free_before = list(manager._num_free_slots)
        groups = 1 if isinstance(manager.attention_cache_storage, MlaLatentStorage) else manager.num_kv_heads
        heads = int(engine.config.hf_config.num_attention_heads)
        sample_text = 'The archive contains notes about rivers, mountains, books, and gardens. '
        repeated = engine.tokenizer.encode(sample_text * 50, add_special_tokens=False)
        prompts = [repeated[:161], repeated[:99]]
        assert [len(prompt) for prompt in prompts] == [161, 99]
        for prompt in prompts:
            engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
            artifact['samples'].append(dict(status='model_failed', prompt_token_ids=prompt,
                                             token_ids=[], text=''))
        outputs = {}
        for _ in range(32):
            if engine.is_finished():
                break
            completed, scheduled_tokens = engine.step()
            for seq_id, tokens, *_ in completed:
                outputs[seq_id] = tokens
            step = dict(scheduled_tokens=scheduled_tokens, scores=[], counters=dict(manager._h2o_counters))
            for (layer, seq_id), scores in manager._h2o_scores.items():
                positions = manager._h2o_positions[layer, seq_id]
                row = manager.seq_id_to_row[layer][seq_id]
                resident = int(manager.row_seq_lens[layer][row])
                assert scores.ndim == 2 and scores.shape[0] == heads
                assert positions.shape == (groups, scores.shape[-1])
                assert resident >= scores.shape[-1]
                assert scores.shape[-1] <= config['h2o_prefill_budget']
                assert bool(torch.isfinite(scores).all())
                step['scores'].append(dict(layer=layer, seq_id=seq_id,
                                            score_length=scores.shape[-1], resident=resident))
            artifact['steps'].append(step)
        assert engine.is_finished(), 'bounded generation did not finish'
        assert len(outputs) == len(prompts)
        for sample, tokens in zip(artifact['samples'], [outputs[key] for key in sorted(outputs)]):
            sample.update(status='success', token_ids=tokens,
                          text=engine.tokenizer.decode(tokens, skip_special_tokens=True))
            assert len(tokens) == 8
        assert manager._h2o_counters['intermediate_prefill_evictions'] > 0
        assert manager._h2o_counters['final_prefill_evictions'] > 0
        assert manager._h2o_counters['decode_evictions'] == 0
        assert not manager._h2o_scores and not manager._h2o_positions
        assert manager._num_free_slots == free_before
        artifact['status'] = 'success'
    except Exception as error:
        artifact['status'] = 'model_failed'
        artifact['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (output_dir / 'model_validation.json').write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
        if engine is not None:
            engine.exit()
