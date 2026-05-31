# Prefill Schedule Policy And DeltaKV Full Prefill - 2026-05-16

## Scope

This change adds an explicit Sparse-vLLM prefill scheduling policy layer and
connects DeltaKV long-prompt prefill to a cache-manager-owned full-prefill
staging path.

Implemented files:

- `src/sparsevllm/method_registry.py`
- `src/sparsevllm/config.py`
- `src/sparsevllm/engine/scheduler.py`
- `src/sparsevllm/engine/cache_manager/base.py`
- `src/sparsevllm/engine/cache_manager/deltakv.py`
- `src/sparsevllm/engine/cache_manager/deltakv_delta_quant.py`
- `src/sparsevllm/engine/cache_manager/deltakv_snapkv.py`
- `src/sparsevllm/engine/cache_manager/deltakv_standalone.py`
- `src/sparsevllm/engine/sparse_controller.py`
- `src/sparsevllm/layers/attention.py`
- `src/sparsevllm/models/qwen2.py`
- `src/sparsevllm/models/qwen3.py`
- `tests/test_prefill_schedule_policy.py`
- `tests/test_mlp_chunking.py`
- `tests/test_runtime_param_normalization.py`

## Policies

Resolved policy constants live in `src/sparsevllm/method_registry.py`:

- `all_chunked`: all prefill requests are chunked by `chunk_prefill_size`.
  Long and short requests still use separate buckets.
- `long_bs1full_short_batch`: long prompts are scheduled one sequence at a
  time and prefilled to the full remaining prompt length. Short prompts still
  batch and chunk.
- `auto`: resolved from the method registry during `Config.__post_init__`.

Default mapping:

- vanilla, StreamingLLM/attention-sink, SnapKV, PyramidKV, Quest, OmniKV:
  `all_chunked`
- DeltaKV family (`deltakv`, triton variants, delta-quant, standalone,
  snapkv): `long_bs1full_short_batch`

Explicit policies must match the registry default. Mismatches fail fast.

## DeltaKV Full Prefill

For DeltaKV long full-prefill, sparse layers now use a shared per-layer staging
KV cache:

- current layer writes full raw KV to `deltakv_prefill_staging_kv_cache`
- attention reads the staging view
- `cache_manager.on_layer_attention_end(layer_idx)` performs layer-local
  compression after attention
- persistent sparse full KV stores only sink, recent, centers, and decode
  reconstruct buffers
- latent slots store non-center compressed tokens

The prefill staging temporary lifetime is separated from decode reconstruct
temporary accounting:

- `deltakv_prefill_staging_num_slots`
- `_deltakv_decode_reconstruct_full_reserve`
- `prefill_step_free_slots()` excludes temporary pools from persistent
  capacity scheduling

## Activation And Kernel Peak Fixes

Long full-prefill exposed two additional peak issues:

- Qwen2/Qwen3 MLP activation peak is bounded with `mlp_chunk_size` token
  chunks. This is correctness-preserving because MLP is token-local.
- DeltaKV V4 `batch_gather_mean` is now chunked by
  `deltakv_cluster_gather_chunk_size` to avoid oversized Triton launch grids
  during 128k layer-local compression.

## Validation

Local environment:

- Repo: `<PROJECT_ROOT>`
- Conda env: `svllm`
- Main local GPU used: GPU 6
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
- Local DeltaKV compressor:
  `<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor`
- Local output root:
  `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314`

Remote LongBench environment:

- Host: `ssh -p 26037 root@connect.westb.seetacloud.com`
- Repo copy:
  `<PROJECT_ROOT>-prefill-policy-20260515-2341`
- Conda env: `<REVIEW_CONDA_ENV>`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97GB
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
- Remote DeltaKV compressor:
  `<CHECKPOINT_ROOT>/cluster_e2e_cs256_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_cdownmlp_swiglud3072_cuplinear_0125_222950`
- Remote output root:
  `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314`
- Remote log root:
  `<AUTODL_FS>/logs/sparsevllm_prefill_policy_20260515_2314`

### Compile And Unit Tests

Commands:

```bash
conda run -n svllm python -m compileall -q src tests
conda run -n svllm python -m unittest tests.test_mlp_chunking tests.test_prefill_schedule_policy tests.test_runtime_param_normalization
conda run -n svllm python -m unittest discover -s tests -p 'test*.py'
```

Result:

- targeted tests: 23 tests, OK
- full unittest discover: 63 tests, OK

### Small Smoke

Command shape:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --lengths 2048 \
  --batch_sizes 2 \
  --methods vanilla,deltakv-triton-v4 \
  --output_len 2
```

Log:

- `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314/smoke_2048_vanilla_deltakv_triton_v4.log`

Result:

- vanilla: `prefill_schedule_policy='all_chunked'`, success, peak 70.73GB
- DeltaKV V4: `prefill_schedule_policy='long_bs1full_short_batch'`,
  success, peak 59.69GB

### 128k Smoke

Vanilla command log:

- `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314/smoke_128k_vanilla_local_gpu6_after_mlpchunk.log`

Vanilla result:

- `prefill_schedule_policy='all_chunked'`
- 131072 context, batch 1, success
- TTFT 14.92s, prefill 8783.8 tok/s, peak 47.78GB

DeltaKV command log:

- `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314/smoke_128k_deltakv_triton_v4_local_gpu6_chunk_gather.log`

DeltaKV result:

- `prefill_schedule_policy='long_bs1full_short_batch'`
- 131072 context, batch 1, success
- full-prefill staging allocation:
  `deltakv_prefill_staging_slots=131174`
- TTFT 18.23s, prefill 7190.5 tok/s, peak 74.34GB

Earlier failed 128k attempts before the final fixes:

- MLP activation OOM in `gate_up_proj`, fixed by `mlp_chunk_size`
- Triton `batch_gather_mean` invalid launch during DeltaKV layer-local
  compression, fixed by `deltakv_cluster_gather_chunk_size`

### LongBench

Sparse-vLLM backend one-sample smoke:

- Output:
  `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314/longbench_smoke_qasper_1_sparsevllm_deltakv`
- Log:
  `<AUTODL_FS>/logs/sparsevllm_prefill_policy_20260515_2314/longbench_smoke_qasper_1_sparsevllm_deltakv.log`
- Result: qasper 1-sample smoke passed and auto-eval produced `qasper: 25.0`

Full LongBench stress run:

```bash
tmux new-session -d -s sparsevllm_lb_bs64 \
  'cd <PROJECT_ROOT>-prefill-policy-20260515-2341 && \
   bash scripts/tmp/run_longbench_full_bs64.sh > \
   <AUTODL_FS>/logs/sparsevllm_prefill_policy_20260515_2314/longbench_full_sparsevllm_deltakv_triton_v4_bs64.log 2>&1'
```

Config:

- `--backend sparsevllm`
- `--batch_size 64`
- `sparse_method=deltakv-triton-v4`
- `engine_prefill_chunk_size=4096`
- `max_num_seqs_in_batch=64`
- `max_decoding_seqs=64`
- `decode_keep_tokens=4096`
- `prefill_keep_tokens=4096`
- `sink_keep_tokens=8`
- `recent_keep_tokens=128`
- `full_attention_layers=0,1,2,4,7,14`
- `deltakv_center_ratio=0.1`
- `deltakv_latent_dim=256`
- `deltakv_full_pool_reserve_ratio=0.2`
- `deltakv_neighbor_count=4`
- `deltakv_cluster_gather_chunk_size=16384`

Output:

- `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314/longbench_full_sparsevllm_deltakv_triton_v4_bs64`

Log:

- `<AUTODL_FS>/logs/sparsevllm_prefill_policy_20260515_2314/longbench_full_sparsevllm_deltakv_triton_v4_bs64.log`

Expected row counts from the remote LongBench data files:

- 200 rows: `narrativeqa`, `qasper`, `hotpotqa`, `2wikimqa`,
  `musique`, `gov_report`, `qmsum`, `multi_news`, `trec`, `triviaqa`,
  `samsum`, `passage_count`, `passage_retrieval_en`
- 150 rows: `multifieldqa_en`
- 500 rows: `lcc`, `repobench-p`

Final status:

- completed in tmux session `sparsevllm_lb_bs64`
- all output row counts matched the expected remote LongBench dataset counts
- `grep -Ein "Traceback|RuntimeError|OutOfMemory|CUDA out of memory|invalid argument"`
  found no errors in the run log
- engine logs confirmed high-pressure scheduling with 64 active sequences and
  separate long/short prefill buckets, for example `last_batch=pf-L` and
  `last_batch=pf-S`
- GPU memory reached about 96.7GB used on the 97GB card during the run
- after completion, the SSH remote was shut down with
  `ssh -p 26037 root@connect.westb.seetacloud.com shutdown`

Result file:

- `<OUTPUT_ROOT>/sparsevllm_prefill_policy_20260515_2314/longbench_full_sparsevllm_deltakv_triton_v4_bs64/result.json`

Scores:

- `narrativeqa`: 28.14
- `qasper`: 47.20
- `multifieldqa_en`: 48.81
- `hotpotqa`: 59.69
- `2wikimqa`: 48.32
- `musique`: 31.68
- `gov_report`: 33.64
- `qmsum`: 24.46
- `multi_news`: 25.51
- `trec`: 77.50
- `triviaqa`: 83.87
- `samsum`: 45.26
- `passage_count`: 7.50
- `passage_retrieval_en`: 95.00
- `lcc`: 49.19
- `repobench-p`: 39.85

Category scores:

- `SDQA`: 41.38
- `MDQA`: 46.56
- `SUM`: 27.87
- `FewShot`: 68.88
- `Syn`: 51.25
- `Code`: 44.52
- `overall_category_avg`: 46.74
