# SparseVLLM Regression Tests

## Purpose

This document describes how to run the fixed SparseVLLM regression harness under
`benchmark/sparsevllm_regression/`.

The harness is intended for reproducible method/model checks across:

- `quality`: LongBench-mini generation quality.
- `logits`: HF-reference vs SparseVLLM logits alignment.
- `perf`: prefill/decode throughput and memory accounting.
- `stress`: high-concurrency SparseVLLM admission/decode stress.
- `validate`: manifest and output-artifact validation.

The test plan is controlled by
`benchmark/sparsevllm_regression/manifest.json`.

## Prerequisites

Observed working environment on 2026-06-13:

- Working directory: `/root/autodl-tmp/DeltaKV`
- Conda env: `kv`
- Output root: `/root/autodl-tmp/outputs/deltakv`
- LongBench data: `/root/autodl-fs/datasets/LongBench`
- Models:
  - `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
  - `/root/autodl-fs/models/Qwen3-4B-Instruct-2507`
  - `/root/autodl-fs/models/Llama-3.1-8B-Instruct`
- Compressor checkpoints:
  - `/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
  - `/root/autodl-fs/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor`
  - `/root/autodl-fs/checkpoints/compressor/Llama-3.1-8B-Instruct-Compressor`

Set the environment before running the suite:

```bash
cd /root/autodl-tmp/DeltaKV

export DELTAKV_OUTPUT_DIR=/root/autodl-tmp/outputs/deltakv
export DELTAKV_LONGBENCH_DATA_DIR=/root/autodl-fs/datasets/LongBench

export DELTAKV_MODEL_QWEN25_7B=/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M
export DELTAKV_MODEL_QWEN3_4B=/root/autodl-fs/models/Qwen3-4B-Instruct-2507
export DELTAKV_MODEL_LLAMA31_8B=/root/autodl-fs/models/Llama-3.1-8B-Instruct

export DELTAKV_COMPRESSOR_QWEN25_7B=/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor
export DELTAKV_COMPRESSOR_QWEN3_4B=/root/autodl-fs/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor
export DELTAKV_COMPRESSOR_LLAMA31_8B=/root/autodl-fs/checkpoints/compressor/Llama-3.1-8B-Instruct-Compressor

export PYTHONPATH=/root/autodl-tmp/DeltaKV:/root/autodl-tmp/DeltaKV/src:${PYTHONPATH:-}
```

The manifest also contains `qwen25_32b`, but the current local regression
commands should omit it unless there is enough GPU memory and the corresponding
model/checkpoint environment variables are set.

## Quick Unit Tests

Run the unit tests that protect the regression harness, grading, manifest
policy, and OmniKV full-layer selector:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python -m unittest \
  tests.test_sparsevllm_regression_grading \
  tests.test_omnikv_full_layer_selector \
  -v
```

Expected result for the current harness: all tests pass.

## Manifest Validation

Use `validate` before long GPU runs. It resolves runtime paths, writes the
resolved manifest, and creates empty required artifact files.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer validate \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id validate_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

Use `--no-allow_skipped_policy` when missing model/checkpoint paths should fail
the run instead of being recorded as skipped.

## Common Run Commands

All commands write to:

```text
<output_root>/sparsevllm_regression/<run_id>/
```

### Quality

Quality is LongBench-mini with:

- tasks: `qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`
- LongBench batch size: `100`
- SparseVLLM `max_num_seqs_in_batch`: `16`
- SparseVLLM `max_decoding_seqs`: `16`
- samples per task: `50`

Run OmniKV against vanilla baselines:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id omnikv_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

For a full non-32B quality run:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv,deltakv-less-memory \
  --run_id quality_3models_all_methods_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Correctness / Logits

`logits` compares HF sparse reference outputs with SparseVLLM for methods that
declare `hf_logits_reference=true`. Methods without an HF reference are graded
`N/A` by policy.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer logits \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_logits_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Performance

Performance uses:

- prompt lengths: `16000,64000`
- batch sizes: `1,4`
- output tokens: `256`
- decode CUDA graph requested where the method supports it

For sparse methods, the benchmark also runs vanilla for the same shape so the
suite can compute decode speedup.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_perf_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Stress

Stress currently uses:

- prompt length: `16000`
- request count / batch size: `80`
- output tokens: `64`
- `max_num_seqs_in_batch=80`
- `max_decoding_seqs=80`
- max decode steps after full admission: `32`

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_stress80_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

### Combined Layers

`nightly` runs quality, logits, and performance. It does not run stress.

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer nightly \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id nightly_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root /root/autodl-tmp/outputs/deltakv
```

`pre-refactor` runs quality, logits, performance, and stress.

## Recent Result Records

This section records the recent runs from the current AutoDL workspace. It is a
result index, not a new pass/fail policy. Use the linked artifacts for exact
commands, raw rows, parsed rows, per-sample status, and aggregate metrics.

### 2026-06-18 Qwen3 DeltaKV Quality Fix

- run root:
  `/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality`
- layer/model/methods:
  `quality`, `qwen3_4b`, `vanilla,deltakv`
- data:
  LongBench-mini tasks
  `qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`,
  `50` samples per task per method.
- result:
  `grade_summary.json` reports quality grade `B`; vanilla
  `overall_category_avg=59.99`, DeltaKV `overall_category_avg=59.54`,
  `score_loss=0.45`.
- status:
  all task status counts are `success`; no skipped or failed samples were
  included in the aggregate.
- source artifacts:
  [`grade_summary.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/grade_summary.json),
  [`metrics.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/metrics.json),
  [`raw_outputs.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/raw_outputs.jsonl),
  [`parsed_outputs.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/parsed_outputs.jsonl),
  [`sample_results.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_quality_after_tempclone_20260618_045315_quality/sample_results.jsonl).

The corresponding full 16-task LongBench run is recorded in
`docs/longbench_results_summary.md` as `DeltaKV Sparse-VLLM temp-slot fix` with
overall `48.18` and 3750 `success` sample rows.

### 2026-06-18 Qwen3 DeltaKV Perf/Stress

- perf run root:
  `/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_perf`
- stress run root:
  `/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_stress`
- backend:
  Sparse-VLLM with decode CUDA graph active for all recorded rows.
- important interpretation:
  the quality bug was fixed, but the perf gate did not pass. Perf grades are
  `D` because DeltaKV decode throughput was slower than vanilla at the tested
  shapes. Stress grade is `B`; it completed with full admission and no
  preemptions, but utilization was below the A-grade threshold.

| Layer | Method | Length | BS | Status/grade | Decode tok/s | Mem GB | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| perf | vanilla | 32000 | 4 | SUCCESS | 164.117 | 83.457 | baseline |
| perf | deltakv | 32000 | 4 | D | 95.363 | 17.646 | speedup `0.581`, graph active |
| perf | vanilla | 32000 | 8 | SUCCESS | 209.585 | 83.584 | baseline |
| perf | deltakv | 32000 | 8 | D | 117.542 | 23.590 | speedup `0.561`, graph active |
| perf | vanilla | 64000 | 4 | SUCCESS | 106.581 | 83.459 | baseline |
| perf | deltakv | 64000 | 4 | D | 71.724 | 25.340 | speedup `0.673`, graph active |
| perf | vanilla | 64000 | 8 | SUCCESS | 124.318 | 83.586 | baseline |
| perf | deltakv | 64000 | 8 | D | 83.581 | 35.930 | speedup `0.672`, graph active |
| stress | deltakv | 16000 | 80 | B | 66.736 | 87.269 | full admission reached, preemptions `0`, avg BS `66.0` |

Source artifacts:
[`perf.jsonl`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_perf/perf.jsonl),
[`perf grade_summary.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_perf/grade_summary.json),
[`stress.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_stress/stress.json),
[`stress grade_summary.json`](/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/qwen3_deltakv_perf_stress_after_tempclone_20260618_072826_stress/grade_summary.json).

### 2026-06-16 Qwen2.5 128k Concurrent Decode Sanity

This was an ad hoc Sparse-VLLM throughput sanity run for whether the
full-layer-KIVI DeltaKV less-memory path can keep all requests admitted at
`128k` context and concurrent decode.

- run root:
  `/root/autodl-tmp/outputs/deltakv/perf/qwen25_7b_1m_lessmem_cudagraph_no_compressor_2settings_20260616_151628`
- model:
  `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
- method:
  `deltakv-less-memory-cudagraph`, `use_compression=false`,
  full-layer KIVI4 group32/residual32, full layers `0,2,4,11,16,22`,
  decode CUDA graph active.

| Length | BS | Output tokens | Status | Decode tok/s | Avg BS | Mem GB | Full admission |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 131072 | 8 | 256 | SUCCESS | 115.441 | 7.969 | 50.299 | yes |
| 131072 | 16 | 512 | SUCCESS | 135.191 | 15.969 | 77.228 | yes |

Source artifact:
[`summary.json`](/root/autodl-tmp/outputs/deltakv/perf/qwen25_7b_1m_lessmem_cudagraph_no_compressor_2settings_20260616_151628/summary.json).

### 2026-06-20 Qwen2.5 7B Max-BS Decode Sweep

This was an ad hoc max-batch sweep using
`scripts/benchmarks/bench_sparse_vllm.py`, not the fixed regression harness.
The selection rule in the run manifest was:
`status=SUCCESS and full_admission_reached=true`; max-success batch size and
best decode throughput are reported only from usable rows.

- run root:
  `/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry`
- model:
  `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
- compressor for attempted DeltaKV rows:
  `/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- common settings:
  output length `128`, `max_decode_steps_after_full=64`,
  `engine_prefill_chunk_size=8192`, `max_num_batched_tokens=65536`,
  `gpu_memory_utilization=0.9`, graph mode requested.

| Method | Length | Max success BS | Decode tok/s at max BS | Best decode BS | Best decode tok/s | Mem GB at best | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla | 65536 | 32 | 272.710 | 20 | 272.882 | 87.037 | success |
| vanilla | 131072 | 16 | 136.030 | 12 | 136.476 | 87.022 | success |
| vanilla | 262144 | 8 | 67.788 | 5 | 67.989 | 86.990 | success |
| vanilla | 524288 | 4 | 30.590 | 2 | 30.946 | 86.527 | success |
| vanilla | 921600 | 2 | 14.648 | 1 | 15.614 | 85.667 | success |
| deltakv | 65536 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 131072 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 262144 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 524288 | - | - | - | - | - | bs=1 failed; larger BS skipped |
| deltakv | 921600 | - | - | - | - | - | bs=1 failed; larger BS skipped |

DeltaKV rows are not throughput results. They failed during engine construction
because the launcher passed `prefill_keep_tokens`, which current Sparse-VLLM
rejects as an unknown config key:
`ValueError: Unknown Sparse-vLLM config keys: ['prefill_keep_tokens']`. The
sweep used `stop_after_first_failure`, so all larger DeltaKV batch sizes were
recorded as `SKIPPED_BY_POLICY`.

Source artifacts:
[`summary.json`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/summary.json),
[`summary.md`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/summary.md),
[`status.tsv`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/status.tsv),
[`deltakv_len65536_bs1.log`](/root/autodl-tmp/outputs/deltakv/qwen25_7b_svllm_vanilla_deltakv_maxbs_decode_20260620_after_scbench_d0p1_retry/logs/deltakv_len65536_bs1.log).

### 2026-06-24 Qwen2.5 7B Vanilla 128k No-Graph Full-Output Sweep

This was an ad hoc Sparse-VLLM vanilla throughput check on
`guest-KR6288-X2-A0-R0-00`, GPU4. The user target was no decode CUDA graph,
`128k` prompt length, max batch size `4`, and `512` output tokens per request.
The max-batch sweep tested batch sizes `1`, `2`, and `4`; because `bs4`
succeeded, the result is a lower bound rather than the true maximum batch size.

- status: completed, `EXIT_CODE=0`
- run root:
  `/data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput/qwen25_7b_vanilla_nograph_128k_fullout512_maxbs4_gpu4_20260624`
- launcher (local file ignored by `.gitignore`):
  `scripts/tmp/run_qwen25_7b_vanilla_nograph_128k_out512_maxbs4_gpu4.sh`
- model:
  `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- code:
  `codex/svllm-throughput` / `f9b712a4b096015abbc4b046596f157443bc4531`;
  worktree clean at launch
- environment:
  Python `3.10.20`, Torch `2.8.0+cu128`, H100 80GB GPU4
- common settings:
  `--methods vanilla`, `--lengths 128000`, `--max_batch_size 4`,
  `--output_len 512`, `--max_decode_steps_after_full 0`,
  `--disable_decode_cuda_graph`, `engine_prefill_chunk_size=8192`,
  `max_num_batched_tokens=65536`, `gpu_memory_utilization=0.9`

Command:

```bash
/home/haojitai/miniconda3/envs/svllm/bin/python -u scripts/benchmarks/run_sparsevllm_max_batch_throughput.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --compressor_path /data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor \
  --methods vanilla \
  --lengths 128000 \
  --gpus 4 \
  --output_root /data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput \
  --run_id qwen25_7b_vanilla_nograph_128k_fullout512_maxbs4_gpu4_20260624 \
  --max_batch_size 4 \
  --output_len 512 \
  --max_decode_steps_after_full 0 \
  --probe_timeout_s 3600 \
  --gpu_memory_utilization 0.9 \
  --engine_prefill_chunk_size 8192 \
  --mlp_chunk_size 16384 \
  --max_num_batched_tokens 65536 \
  --disable_decode_cuda_graph \
  --master_port_base 29800
```

| BS | Status | Decode tok/s | Prefill tok/s | TTFT s | ITL ms | Avg BS | Mem GB | Full admission | Preemptions | Decode graph active |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1 | SUCCESS | 28.267 | 9066.9 | 14.118 | 35.38 | 0.998 | 71.522 | yes | 0 | no |
| 2 | SUCCESS | 56.716 | 9182.5 | 27.880 | 35.26 | 1.996 | 72.375 | yes | 0 | no |
| 4 | SUCCESS | 100.076 | 9246.0 | 36.360 | 39.97 | 3.992 | 72.690 | yes | 0 | no |

Source artifacts:
[`summary.json`](/data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput/qwen25_7b_vanilla_nograph_128k_fullout512_maxbs4_gpu4_20260624/summary.json),
[`summary.md`](/data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput/qwen25_7b_vanilla_nograph_128k_fullout512_maxbs4_gpu4_20260624/summary.md),
[`status.tsv`](/data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput/qwen25_7b_vanilla_nograph_128k_fullout512_maxbs4_gpu4_20260624/status.tsv),
[`bs4 result.jsonl`](/data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput/qwen25_7b_vanilla_nograph_128k_fullout512_maxbs4_gpu4_20260624/probes/vanilla/len128000/bs4/result.jsonl).

The first launcher attempt at
`/data2/haojitai/outputs/deltakv/sparsevllm_max_batch_throughput/qwen25_7b_vanilla_nograph_128k_out512_maxbs4_gpu4_20260624`
was aborted manually and is not a comparable result: it used
`max_decode_steps_after_full=64`, so it only measured a 64-step decode window
despite passing `--output_len 512`.

### 2026-06-24 Qwen2.5 7B Vanilla Commit Throughput Scan

This was a commit-dimension throughput scan after the user clarified that
"bisect" meant moving backward through commits, not searching for max batch
size. Each successful row used the same benchmark shape: `GPU4`,
`Qwen2.5-7B-Instruct-1M`, Sparse-VLLM `vanilla`, no decode CUDA graph,
prompt length `128000`, batch size `4`, `output_len=512`, and
`max_decode_steps_after_full=0`.

- status: completed, `EXIT_CODE=0`
- run root:
  `/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624`
- launcher (local file ignored by `.gitignore`):
  `scripts/tmp/run_svllm_vanilla_commit_scan_gpu4.sh`
- model:
  `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- environment:
  Python `3.10.20`, Torch `2.8.0+cu128`, H100 80GB GPU4
- common runtime params:
  `enforce_eager=false`, `gpu_memory_utilization=0.9`,
  `engine_prefill_chunk_size=8192`, `max_num_batched_tokens=65536`,
  `max_num_seqs_in_batch=4`, `max_decoding_seqs=4`,
  `mlp_chunk_size=16384`

| Commit | Subject | Status | Decode tok/s | Prefill tok/s | TTFT s | ITL ms | Avg BS | Mem GB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `f9b712a` | Warn about experimental DeltaKV support | SUCCESS | 100.225 | 9245.916 | 36.344 | 39.910 | 3.992 | 72.690 |
| `1113e37` | chore: import sparsevllm snapshot with pruned docs | SUCCESS | 100.222 | 9237.617 | 36.352 | 39.911 | 3.992 | 72.690 |
| `30e62e4` | Remove tracked .DS_Store files | SUCCESS | 134.720 | 9232.550 | 36.370 | 29.690 | 4.000 | 72.690 |
| `50131ee` | Support PyramidKV full-prefill staging | SUCCESS | 147.390 | 9229.740 | 36.400 | 27.140 | 4.000 | 72.690 |
| `b0aa540` | Enable decode CUDA graphs for sparse methods | SUCCESS | 155.130 | 9277.120 | 36.200 | 25.780 | 4.000 | 72.690 |
| `774a71e` | Add decode CUDA graph batch padding | SUCCESS | 154.360 | 9273.750 | 36.210 | 25.910 | 4.000 | 72.690 |
| `d5aacd1` | Optimize OmniKV decode with CUDA graph prewarm | SUCCESS | 154.540 | 9257.120 | 36.290 | 25.880 | 4.000 | 72.690 |
| `e329ce6` | Optimize OmniKV decode bookkeeping | SUCCESS | 153.530 | 9165.610 | 36.870 | 26.050 | 4.000 | 72.690 |
| `44e26e8` | Implement prefill scheduling policy | SUCCESS | 144.860 | 8974.680 | 37.970 | 27.610 | 4.000 | 72.690 |

Interpretation:

- `f9b712a` and `1113e37` are effectively identical for this vanilla
  throughput case, so the latest warning-only commit did not cause the drop.
- The major observed drop is between `30e62e4` (`134.720` tok/s) and
  `1113e37` (`100.222` tok/s). That interval is not fine-bisectable on this
  branch because `1113e37` is a large imported snapshot touching
  `scripts/benchmarks/bench_sparse_vllm.py`, scheduler/model runner/cache
  manager paths, attention kernels, and Sparse-VLLM model files.
- `50131ee..30e62e4` changes only `src/sparsevllm/.DS_Store`, so the
  `147.390 -> 134.720` difference should be treated as a measurement point that
  may need repeat sampling, not as a source-code causal claim.
- `d5aacd1` was corrected with a second run at
  `cases/6b_d5aacd1_corrected`; it lands in the same throughput band as
  `774a71e` and `e329ce6`. The first invalid probe remains at `cases/6_d5aacd1`
  and only failed because that older commit did not yet have the generic
  `decode_cuda_graph` config key.

Source artifacts:
[`summary.md`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/summary.md),
[`summary.json`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/summary.json),
[`status.tsv`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/status.tsv).

### 2026-06-24 Qwen2.5 7B Vanilla Decode Regression Diagnosis

This diagnosis investigated whether the no-graph vanilla throughput drop was
caused by CUDA graph support. It used temporary detached worktrees under
`/data2/haojitai/outputs/deltakv/svllm_commit_throughput/diagnostics_*` and
kept the formal benchmark shape unchanged unless noted.

Diagnostic results:

| Probe | Variant | Decode tok/s | Finding |
| --- | --- | ---: | --- |
| baseline | `f9b712a`, no graph | 100.225 | Reproduces the slow current path. |
| path A/B | `f9b712a` with no-graph decode forced back to the ordinary eager branch | 99.840 | Static runner dispatch is not the main cause. |
| layernorm A/B | `f9b712a` with `RMSNorm` reverted to the `30e62e4` implementation | 102.300 | The graph-capture-friendly RMSNorm change is not the main cause. |
| scalar-check A/B | `f9b712a` with two decode-time `.max().item()` bounds checks removed | 145.680 | The CPU scalar reads explain most of the regression. |

Profiler evidence:

- Built-in profiler on 64 decode steps with `CUDA_SYNC_SVLLM=1` showed
  `model_run_model_decode` increased from `25.810 ms/step` at `30e62e4` to
  `40.521 ms/step` at `f9b712a`; `cache_prepare_decode` decreased from
  `4.198 ms/step` to `0.301 ms/step`, so the regression is inside model
  forward, not scheduling or cache preparation.
- `torch.profiler` over 16 decode steps showed the main CUDA kernels were
  effectively unchanged: GQA stage1 was `~165 ms/16 steps`, stage2 was
  `~29.5 ms/16 steps`, and GEMM was `~79 ms/16 steps` in both commits.
- The new path added `aten::item` / `_local_scalar_dense` calls:
  `896` calls over `16` decode steps, which equals `16 steps * 28 layers * 2`
  scalar reads. They cost `~157 ms` CPU time over those `16` steps, or about
  `9.8 ms/step`, matching the observed decode-time regression.

Concrete source locations in `f9b712a`:

- `src/sparsevllm/layers/attention.py`: the decode branch reads
  `decode_view.context_lens.max().item()` for a bounds check on every layer and
  decode step.
- `src/sparsevllm/triton_kernel/flash_decoding_stage2.py`: `flash_decode_stage2`
  reads `B_Seqlen.max().item()` for another bounds check on every layer and
  decode step.

Conclusion:

The regression is not caused by enabling CUDA graphs. Earlier commits that
already include decode CUDA graph support (`b0aa540`, `774a71e`, `d5aacd1`) are
still in the `154-155 tok/s` band when `decode_cuda_graph=false`. The measured
drop in `1113e37`/`f9b712a` is mainly caused by graph/static-decode safety
checks that still execute on the no-graph vanilla path and force repeated
GPU-to-CPU scalar synchronization.

Diagnostic artifacts:
[`f9_restore_ordinary_eager_decode`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/diagnostics/f9_restore_ordinary_eager_decode/run.log),
[`f9_old_layernorm_only`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/diagnostics/f9_old_layernorm_only/run.log),
[`profiler_64steps`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/diagnostics/profiler_64steps),
[`torch_profiler_decode16`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/diagnostics/torch_profiler_decode16),
[`f9_removed_decode_item_checks`](/data2/haojitai/outputs/deltakv/svllm_commit_throughput/qwen25_7b_vanilla_nograph_128k_bs4_out512_commit_scan_gpu4_20260624/diagnostics/f9_removed_decode_item_checks/run.log).

### 2026-06-24 Decode Bounds Check Fix Validation

The fix gates the decode-time GPU-to-CPU scalar bounds checks behind
`SVLLM_DEBUG_DECODE_BOUNDS=1`. Default decode no longer calls
`context_lens.max().item()` in `src/sparsevllm/layers/attention.py` or
`B_Seqlen.max().item()` in `src/sparsevllm/triton_kernel/flash_decoding_stage2.py`;
the debug path still performs the checks when explicitly requested.

- status: completed, all cases exited successfully
- run root:
  `/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624`
- model:
  `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- environment:
  `guest-KR6288-X2-A0-R0-00`, GPU4, conda env `svllm`
- common runtime params:
  Sparse-VLLM `vanilla`, `enforce_eager=false`,
  `engine_prefill_chunk_size=8192`, `gpu_memory_utilization=0.9`,
  `max_num_batched_tokens=65536`, `max_num_seqs_in_batch=4`,
  `max_decoding_seqs=4`, `mlp_chunk_size=16384`

| Case | Decode graph | Debug bounds | Context | BS | Output tokens | Decode tok/s | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `smoke_len4096_bs1_out16` | false | off | 4096 | 1 | 16 | 16.26 | SUCCESS |
| `smoke_len8192_bs2_out16` | false | off | 8192 | 2 | 16 | 64.19 | SUCCESS |
| `smoke_len16384_bs4_out16` | false | off | 16384 | 4 | 16 | 123.21 | SUCCESS |
| `debug_bounds_len4096_bs2_out8` | false | on | 4096 | 2 | 8 | 36.01 | SUCCESS |
| `perf_len128000_bs4_out512` | false | off | 128000 | 4 | 512 | 147.07 | SUCCESS |
| `graph_smoke_len4096_bs2_out8` | true | off | 4096 | 2 | 8 | 55.01 | SUCCESS |

Source artifacts:
[`smoke_len4096_bs1_out16`](/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624/smoke_len4096_bs1_out16/run.log),
[`smoke_len8192_bs2_out16`](/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624/smoke_len8192_bs2_out16/run.log),
[`smoke_len16384_bs4_out16`](/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624/smoke_len16384_bs4_out16/run.log),
[`debug_bounds_len4096_bs2_out8`](/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624/debug_bounds_len4096_bs2_out8/run.log),
[`perf_len128000_bs4_out512`](/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624/perf_len128000_bs4_out512/run.log),
[`graph_smoke_len4096_bs2_out8`](/data2/haojitai/outputs/deltakv/svllm_decode_bounds_fix_validation_20260624/graph_smoke_len4096_bs2_out8/run.log).

## OmniKV Full-Layer Selection

OmniKV full layers are model-specific. Use
`scripts/analysis/select_omnikv_full_layers.py` before publishing a new model's
OmniKV or OmniKV-aligned DeltaKV regression numbers.

The selector runs an offline decode-attention coverage calibration on a
LongBench task, chooses `--num-full-layers` layers, and writes the selected
layer string to `selected_full_layers.json`. This is not an online runtime mode:
the selected string must be passed back as `full_attention_layers`.

Example for Qwen2.5-7B with six full layers:

```bash
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python scripts/analysis/select_omnikv_full_layers.py \
  --model-path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --longbench-root /root/autodl-fs/datasets/LongBench \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir /root/autodl-tmp/outputs/deltakv/omnikv_full_layer_calibration_$(date -u +%Y%m%d)/qwen25_7b_full6 \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

Key outputs:

- `selected_full_layers.json`: selected layer ids and
  `full_attention_layers` string for runtime configs.
- `per_sample_points.jsonl`: sampled decode points used for calibration.
- `pair_scores.npy` and `segment_scores.npy`: raw coverage matrices for audit.
- `run_info.json`: command, git state, model/data paths, and calibration
  settings.
- `top128_kl_metrics.json`: optional validation output when running with
  `--top128-kl-only`.

To use the selected layers in an ad hoc Sparse-VLLM run, copy the
`full_attention_layers` value into `--hyper_params`:

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_DIR> \
  --methods omnikv \
  --lengths 131072 \
  --batch_sizes 4 \
  --output_len 128 \
  --hyper_params '{"sparse_method":"omnikv","full_attention_layers":"0,2,4,11,16,22","decode_keep_tokens":4096,"recent_keep_tokens":32,"sink_keep_tokens":0,"engine_prefill_chunk_size":512}'
```

For regression runs, update `methods.omnikv.model_configs` in
`benchmark/sparsevllm_regression/manifest.json`. If a DeltaKV regression config
is intentionally aligned to OmniKV observation/full layers, update the matching
DeltaKV model config in the same manifest and record that alignment in the run
summary. The current manifest uses:

```text
qwen25_7b:  0,2,4,11,16,22
qwen3_4b:   0,1,3,9,13,16,21,28
llama31_8b: 0,2,7,13,16,26
```

Run `validate` and rerun OmniKV quality/logits/perf/stress after changing these
layers.

## Outputs

Each run writes:

- `resolved_manifest.json`: manifest after environment-variable resolution.
- `grade_summary.json`: command records, grades, and final status.
- `metrics.json`: quality aggregate records.
- `logits_alignment.json`: logits comparison summaries.
- `perf.jsonl`: flattened performance rows.
- `memory.json`: memory grades derived from performance rows.
- `stress.json`: stress rows and stress grades.
- `raw_outputs.jsonl`, `parsed_outputs.jsonl`, `sample_results.jsonl`: quality
  generation artifacts, when quality is run.
- Layer-specific logs:
  - `quality/<model>/<method>/run.log`
  - `logits/<model>/<method>/run.log`
  - `perf/<model>/<method>.log`
  - `stress/<model>/<method>.log`

Quick summary command:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/root/autodl-tmp/outputs/deltakv/sparsevllm_regression/<run_id>")
data = json.loads((root / "grade_summary.json").read_text())
print("status:", data["status"])
print("worst_required_grade:", data.get("worst_required_grade"))
for grade in data.get("grades", []):
    print(grade.get("model"), grade.get("method"), grade["name"], grade["grade"], grade["status"], grade["metrics"])
PY
```

## Grade Meanings

The gate rules live in `benchmark/sparsevllm_regression/grading.py`.

- Quality:
  - A: sparse score loss `< 0.1` vs vanilla.
  - B: loss `<= 0.5`.
  - C: loss `<= 1.0`.
  - D: loss `> 1.0`, missing score, or failed quality run.
- Logits:
  - A: all decode top-1 match, mean top-5 overlap `>= 0.8`, mean top-10 overlap
    `>= 0.9`, and p99 diff is within threshold.
  - B: top-1 match and mean top-5 overlap `>= 0.8`, but misses A.
  - C: top-1 match, but misses B/A.
  - D: top-1 mismatch, missing decode metrics, or run failure.
  - N/A: no HF logits reference exists for that method.
- Performance:
  - A: decode speedup `>= 2.0` and required decode CUDA graph active.
  - B: speedup `>= 1.5`.
  - C: speedup `> 1.0`.
  - D: speedup `<= 1.0`, run failure, or expected CUDA graph inactive.
- Memory:
  - A/B/C depend on positive observed saving and absolute error from expected
    saving within `0.05/0.10/0.20`.
  - D: missing accounting, non-positive saving, or error `> 0.20`.
- Stress:
  - A: completed, no crash, no preemptions, full admission window reached, and
    utilization OK.
  - B: completed with no preemptions, but not all A conditions met.
  - C: completed with preemptions.
  - D: crashed, stuck, failed rows, or did not finish.

## Updating The Bug Matrix

Use `benchmark/sparsevllm_regression/BUGS_20260613.md` for the current
ABCD gate matrix and open blockers. When rerunning a subset, update only the
affected model/method rows and record the new run IDs near the execution
summary.

For a new campaign date, create a new `BUGS_YYYYMMDD.md` instead of overwriting
the old report.

## Troubleshooting

- Missing model or compressor paths:
  - Run `validate`.
  - Check `resolved_manifest.json`.
  - If a run should fail on missing paths, pass `--no-allow_skipped_policy`.
- Import errors:
  - Ensure `PYTHONPATH=/root/autodl-tmp/DeltaKV:/root/autodl-tmp/DeltaKV/src:${PYTHONPATH:-}`.
  - Use the `kv` conda env on the observed AutoDL machine.
- Quality dataset errors:
  - Set `DELTAKV_LONGBENCH_DATA_DIR=/root/autodl-fs/datasets/LongBench`.
- GPU memory failures:
  - Do not add fallback behavior inside the harness.
  - Record the exact run ID, model, method, layer, log path, and error in the
    bug report.
- A command exits early:
  - Inspect `<run_id>/grade_summary.json`; failed commands are recorded with
    `returncode`, `cmd`, and `log_path`.
  - Inspect the layer-specific log path from the command record.
