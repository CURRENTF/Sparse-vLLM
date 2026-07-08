# SparseVLLM Regression Tests

## Purpose

This document describes how to run the fixed SparseVLLM regression harness under
`benchmark/sparsevllm_regression/`.

The harness is intended for reproducible method/model checks across:

- `quality`: LongBench-mini generation quality.
- `logits`: HF-reference vs SparseVLLM logits alignment.
- `perf`: prefill/decode throughput and memory accounting.
- `stress`: fixed-length high-concurrency SparseVLLM admission/decode stress.
- `stress_v2`: synthetic serving-trace stress with shared-prefix and multi-turn
  workloads, variable prompt lengths, and prefix-cache hit validation for
  supported methods.
- `validate`: manifest and output-artifact validation.

The test plan is controlled by
`benchmark/sparsevllm_regression/manifest.json`.

## Prerequisites

Configure these paths for the machine running the suite:

- Working directory: `<REPO_ROOT>`
- Conda env: `<CONDA_ENV>`
- Output root: `<OUTPUT_ROOT>`
- LongBench data: `<LONGBENCH_ROOT>`
- Models:
  - `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
  - `<MODEL_ROOT>/Qwen3-4B-Instruct-2507`
  - `<MODEL_ROOT>/Llama-3.1-8B-Instruct`
- Compressor checkpoints:
  - `<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor`
  - `<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor`
  - `<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor`

Set the environment before running the suite:

```bash
cd <REPO_ROOT>

export DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT>
export DELTAKV_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>

export DELTAKV_MODEL_QWEN25_7B=<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M
export DELTAKV_MODEL_QWEN3_4B=<MODEL_ROOT>/Qwen3-4B-Instruct-2507
export DELTAKV_MODEL_LLAMA31_8B=<MODEL_ROOT>/Llama-3.1-8B-Instruct

export DELTAKV_COMPRESSOR_QWEN25_7B=<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor
export DELTAKV_COMPRESSOR_QWEN3_4B=<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor
export DELTAKV_COMPRESSOR_LLAMA31_8B=<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor

export PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}
```

The manifest also contains `qwen25_32b`; omit it unless there is enough GPU
memory and the corresponding model/checkpoint environment variables are set.

## Quick Unit Tests

Run the unit tests that protect the regression harness, grading, manifest
policy, and OmniKV full-layer selector:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
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
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer validate \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id validate_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Use `--no-allow_skipped_policy` when missing model/checkpoint paths should fail
the run instead of being recorded as skipped.

## Common Run Commands

All commands write to:

```text
<OUTPUT_ROOT>/sparsevllm_regression/<run_id>/
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
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id omnikv_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For a full non-32B quality run:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv,deltakv-less-memory \
  --run_id quality_3models_all_methods_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For TP decode CUDA graph v1 quality validation, keep LongBench data-worker
parallelism at its default and pass engine TP through the regression-suite
override:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_quality_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

This compares sparse methods against TP vanilla in the same run. A/B/C grades
are recorded; crashes or D grades fail the TP graph quality gate.

For the fast TP prefix-cache + decode-graph regression gate, keep the same
method coverage but use explicit small-sample overrides and a child-command
timeout. This still exercises LongBench quality, SCBench quality, and stress,
but it is a smoke/regression gate rather than the full 50-sample-per-task
quality suite:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --quality_tasks qasper,hotpotqa \
  --quality_batch_size 2 \
  --quality_samples_per_task 2 \
  --quality_min_required_samples 2 \
  --quality_sparsevllm_max_num_seqs_in_batch 2 \
  --quality_sparsevllm_max_decoding_seqs 2 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_quality_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Then run the matching SCBench quality and prefix-hit stress layers:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer scbench \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --scbench_decode_cuda_graph \
  --scbench_tasks scbench_kv \
  --scbench_num_eval_examples 1 \
  --scbench_max_turns 2 \
  --scbench_max_seq_length 1024 \
  --scbench_batch_size 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_scbench_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>

conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --require_prefix_cache_hit \
  --stress_length 256 \
  --stress_request_counts 2 \
  --stress_output_len 2 \
  --stress_max_num_seqs_in_batch 2 \
  --stress_max_decoding_seqs 2 \
  --stress_max_decode_steps_after_full 1 \
  --stress_admission_wave_size 1 \
  --stress_wave_decode_gap_steps 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_stress_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Correctness / Logits

`logits` compares HF sparse reference outputs with SparseVLLM for methods that
declare `hf_logits_reference=true`. Methods without an HF reference are graded
`N/A` by policy.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer logits \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_logits_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
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
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_perf_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

For TP decode CUDA graph v1 performance validation:

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_perf_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

Inspect `perf.jsonl` for `decode_cuda_graph_expected=true` and
`decode_cuda_graph_active=true`.

### Stress

Stress currently uses:

- prompt length: `16000`
- request count / batch size: `80`
- output tokens: `64`
- `max_num_seqs_in_batch=80`
- `max_decoding_seqs=80`
- max decode steps after full admission: `32`

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_stress80_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Stress V2

`stress_v2` uses `scripts/benchmarks/bench_prefix_cache.py` as a regression
layer for serving-like traces. Unlike fixed `stress`, it runs seeded synthetic
requests with:

- workloads: `shared_prefix,multiturn`
- supported methods: `vanilla`, `omnikv`, `quest`
- `vanilla` cases: `baseline_full,prefix_full`
- `omnikv` case: `prefix_omnikv`
- `quest` case: `prefix_quest`
- sessions / turns: `8 / 4`
- shared-prefix requests: `8`
- output tokens: `64`
- max active requests: `8`
- variable multi-turn user lengths: `128..1024`
- variable session-prefix lengths: `1024..4096`
- variable shared suffix lengths: `512..4096`
- max prompt length: about `16.5k` tokens for multi-turn and `12.3k` for
  shared-prefix
- prefix-cache block size: `16`

The gate fails if a prefix-cache-enabled case does not observe cache hits or if
the realized prompt lengths do not vary. Unsupported methods are recorded as
`skipped_by_policy` because this layer specifically validates prefix-cache
serving behavior.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress_v2 \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --run_id stress_v2_qwen3_serving_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Combined Layers

`nightly` runs quality, logits, and performance. It does not run stress.

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer nightly \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id nightly_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

`pre-refactor` runs quality, logits, performance, and stress.

## Result Records

Keep this file as the stable regression runbook. Do not add chronological
experiment records or local result indexes here. If a repo-facing result claim
is needed, cite the original run artifact path directly.

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
conda run -n <CONDA_ENV> --no-capture-output \
  python scripts/analysis/select_omnikv_full_layers.py \
  --model-path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --longbench-root <LONGBENCH_ROOT> \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir <OUTPUT_ROOT>/omnikv_full_layer_calibration_$(date -u +%Y%m%d)/qwen25_7b_full6 \
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
- `stress_v2.json`: serving-trace stress rows and stress_v2 grades.
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

root = Path("<OUTPUT_ROOT>/sparsevllm_regression/<run_id>")
data = json.loads((root / "grade_summary.json").read_text())
print("status:", data["status"])
print("worst_required_grade:", data.get("worst_required_grade"))
for grade in data.get("grades", []):
    print(grade.get("model"), grade.get("method"), grade["name"], grade["grade"], grade["status"], grade["metrics"])
PY
```

## Regression Rubrics

The executable gate rules live in `benchmark/sparsevllm_regression/grading.py`.
The stable human-facing rubric lives in
`benchmark/sparsevllm_regression/rubrics.md`.

Update `benchmark/sparsevllm_regression/rubrics.md` only when stable ABCD
rubric definitions change. Do not add dated campaign results, open blockers,
run IDs, or remote log paths to the rubric file.

## Troubleshooting

- Missing model or compressor paths:
  - Run `validate`.
  - Check `resolved_manifest.json`.
  - If a run should fail on missing paths, pass `--no-allow_skipped_policy`.
- Import errors:
  - Ensure `PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}`.
  - Use an environment with the dependencies from [Getting Started](../getting_started/README.md).
- Quality dataset errors:
  - Set `DELTAKV_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>`.
- GPU memory failures:
  - Do not add fallback behavior inside the harness.
  - Record the exact run ID, model, method, layer, log path, and error in the
    issue note.
- A command exits early:
  - Inspect `<run_id>/grade_summary.json`; failed commands are recorded with
    `returncode`, `cmd`, and `log_path`.
  - Inspect the layer-specific log path from the command record.
