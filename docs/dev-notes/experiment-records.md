# Experiment Records

### 2026-06-23 21:33 Asia/Shanghai - skipkv-rkv-gpu7-debug

- Status: completed, with one intentional fail-fast run followed by a successful debug run
- Goal: validate first-class `rkv` and `skipkv` Sparse-VLLM methods on GPU7 after implementation.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: synthetic prompt token ids from `scripts/benchmarks/bench_sparse_vllm.py`; no external dataset.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; backend `sparsevllm`; bf16 model config.
- Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen3-4B-Instruct-2507 \
  --lengths 32 --batch_sizes 1 --methods rkv,skipkv --output_len 48 --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":32,"tensor_parallel_size":1,"decode_cuda_graph":false,"enforce_eager":true,"sink_keep_tokens":1,"recent_keep_tokens":4,"decode_keep_tokens":16,"rkv_compression_interval":8,"skipkv_compression_interval":8,"skipkv_segment_size":4,"rkv_max_redundancy_tokens":64,"skipkv_max_redundancy_tokens":64,"max_num_seqs_in_batch":1,"max_decoding_seqs":1}' \
  --output_jsonl /data2/haojitai/outputs/sparsevllm_sparse_method_support/skipkv_rkv_debug_gpu7_20260623/bench_debug_len32.jsonl
```

- Results: `rkv` completed with `decode_tokens=47`, `decode_tp=9.86 tok/s`, peak memory `41.56 GB`, cache manager `RKVCacheManager`; `skipkv` completed with `decode_tokens=47`, `decode_tp=13.31 tok/s`, peak memory `41.51 GB`, cache manager `SkipKVCacheManager`.
- Artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/skipkv_rkv_debug_gpu7_20260623/bench_debug_len32.jsonl`.
- Notes: the earlier `--lengths 128` run intentionally used `rkv_max_redundancy_tokens=64` / `skipkv_max_redundancy_tokens=64`; both methods failed fast because candidate tokens were `124`, confirming the explicit quadratic-work guard. Artifact: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/skipkv_rkv_debug_gpu7_20260623/bench_debug.jsonl`. The `output_len=48` numbers are debug-only and must not be used as final decode-speed evidence.

### 2026-06-23 22:51 Asia/Shanghai - rkv-skipkv-gpu7-performance-tuning

- Status: completed.
- Goal: retest R-KV and SkipKV decode speed with enough generated tokens after bounding redundancy scoring to a trailing window and delaying softmax/post-processing until a decode eviction actually triggers.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: synthetic prompt token ids from `scripts/benchmarks/bench_sparse_vllm.py`; prompt length `128`, batch size `1`; no external dataset.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; backend `sparsevllm`; bf16 model config.
- Command shape:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen3-4B-Instruct-2507 \
  --lengths 128 --batch_sizes 1 --methods vanilla,snapkv,rkv,skipkv --output_len <128-or-256> --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":64,"tensor_parallel_size":1,"decode_cuda_graph":true,"enforce_eager":false,"sink_keep_tokens":1,"recent_keep_tokens":8,"decode_keep_tokens":32,"snapkv_window_size":8,"rkv_compression_interval":64,"skipkv_compression_interval":64,"rkv_redundancy_window":32,"skipkv_redundancy_window":32,"rkv_max_redundancy_tokens":256,"skipkv_max_redundancy_tokens":256,"max_num_seqs_in_batch":1,"max_decoding_seqs":1}' \
  --output_jsonl <artifact.jsonl>
```

- Results:

| artifact | graph | decoded tokens | vanilla decode tp | snapkv decode tp | rkv decode tp | skipkv decode tp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `/data2/haojitai/outputs/sparsevllm_sparse_method_support/perf_tuning_20260623/matched_eager_len128_out128_after_window.jsonl` | false | 127 | 21.49 | 17.69 | 17.64 | 17.47 |
| `/data2/haojitai/outputs/sparsevllm_sparse_method_support/perf_tuning_20260623/matched_graph_len128_out128_after_softmax_gate.jsonl` | true | 127 | 128.29 | 80.87 | 72.96 | 72.74 |
| `/data2/haojitai/outputs/sparsevllm_sparse_method_support/perf_tuning_20260623/matched_graph_len128_out256_after_softmax_gate.jsonl` | true | 255 | 129.62 | 79.80 | 73.47 | 71.96 |

- Notes: the 255-token graph run is the primary performance check. R-KV reached `92.1%` of matched SnapKV decode throughput and SkipKV reached `90.2%`. The score-dependent graph path now captures attention-score tensors up front for R-KV/SkipKV so later interval-triggered decode evictions are not silently skipped after CUDA graph capture.

### 2026-06-24 00:45 Asia/Shanghai - rkv-skipkv-gpu6-multi-bs-benchmark

- Status: completed.
- Goal: satisfy the add-sparse-method validation rule that new methods must run an end-to-end benchmark with multiple batch sizes, including `bs=1` and `bs>1`, while comparing against `vanilla` and the nearest existing sparse method (`snapkv`).
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=6`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: synthetic prompt token ids from `scripts/benchmarks/bench_sparse_vllm.py`; prompt length `128`; batch sizes `1,2`; `output_len=256`; no external dataset.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; backend `sparsevllm`; bf16 model config.
- Command:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen3-4B-Instruct-2507 \
  --lengths 128 --batch_sizes 1,2 --methods vanilla,snapkv,rkv,skipkv --output_len 256 --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":64,"tensor_parallel_size":1,"decode_cuda_graph":true,"enforce_eager":false,"sink_keep_tokens":1,"recent_keep_tokens":8,"decode_keep_tokens":32,"snapkv_window_size":8,"rkv_compression_interval":64,"skipkv_compression_interval":64,"rkv_redundancy_window":32,"skipkv_redundancy_window":32,"rkv_max_redundancy_tokens":256,"skipkv_max_redundancy_tokens":256,"max_num_seqs_in_batch":2,"max_decoding_seqs":2}' \
  --output_jsonl /data2/haojitai/outputs/sparsevllm_sparse_method_support/multi_bs_gpu6_20260624/matched_graph_len128_bs1_2_out256.jsonl
```

- Results:

| method | batch size | status | graph active | decoded tokens | prefill tp | decode tp | speedup vs vanilla |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| vanilla | 1 | SUCCESS | true | 255 | 1419.3 | 92.16 | 1.00 |
| vanilla | 2 | SUCCESS | true | 510 | 3244.1 | 248.44 | 1.00 |
| snapkv | 1 | SUCCESS | true | 255 | 820.7 | 64.60 | 0.70 |
| snapkv | 2 | SUCCESS | true | 510 | 1382.8 | 155.61 | 0.63 |
| rkv | 1 | SUCCESS | true | 255 | 886.2 | 60.54 | 0.66 |
| rkv | 2 | SUCCESS | true | 510 | 1599.7 | 138.23 | 0.56 |
| skipkv | 1 | SUCCESS | true | 255 | 883.1 | 60.02 | 0.65 |
| skipkv | 2 | SUCCESS | true | 510 | 1688.5 | 131.94 | 0.53 |

- Artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/multi_bs_gpu6_20260624/matched_graph_len128_bs1_2_out256.jsonl`; log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/multi_bs_gpu6_20260624/matched_graph_len128_bs1_2_out256.log`.
- Notes: all R-KV and SkipKV rows completed with `SUCCESS`, full-admission decode scope, `decode_cuda_graph_active=true`, and enough generated tokens for a meaningful quick benchmark. Relative to matched SnapKV on GPU6, R-KV reached `93.7%` at `bs=1` and `88.8%` at `bs=2`; SkipKV reached `92.9%` at `bs=1` and `84.8%` at `bs=2`.

### 2026-06-24 01:34 Asia/Shanghai - rkv-skipkv-gpu6-cudagraph-explicit

- Status: completed.
- Goal: explicitly rerun the matched benchmark with CUDA Graph enabled and verify graph activation from the JSONL fields, not just command-line parameters.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=6`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: synthetic prompt token ids from `scripts/benchmarks/bench_sparse_vllm.py`; prompt length `128`; batch sizes `1,2`; `output_len=256`; no external dataset.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; backend `sparsevllm`; bf16 model config.
- Command:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen3-4B-Instruct-2507 \
  --lengths 128 --batch_sizes 1,2 --methods vanilla,snapkv,rkv,skipkv --output_len 256 --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":64,"tensor_parallel_size":1,"decode_cuda_graph":true,"enforce_eager":false,"sink_keep_tokens":1,"recent_keep_tokens":8,"decode_keep_tokens":32,"snapkv_window_size":8,"rkv_compression_interval":64,"skipkv_compression_interval":64,"rkv_redundancy_window":32,"skipkv_redundancy_window":32,"rkv_max_redundancy_tokens":256,"skipkv_max_redundancy_tokens":256,"max_num_seqs_in_batch":2,"max_decoding_seqs":2}' \
  --output_jsonl /data2/haojitai/outputs/sparsevllm_sparse_method_support/cudagraph_gpu6_20260624/matched_graph_len128_bs1_2_out256.jsonl
```

- Results:

| method | batch size | status | graph active | graph count | decoded tokens | prefill tp | decode tp |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| vanilla | 1 | SUCCESS | true | 2 | 255 | 1437.0 | 94.81 |
| vanilla | 2 | SUCCESS | true | 1 | 510 | 3130.8 | 248.87 |
| snapkv | 1 | SUCCESS | true | 2 | 255 | 838.0 | 60.73 |
| snapkv | 2 | SUCCESS | true | 1 | 510 | 1497.2 | 155.32 |
| rkv | 1 | SUCCESS | true | 2 | 255 | 691.8 | 61.13 |
| rkv | 2 | SUCCESS | true | 1 | 510 | 1285.8 | 135.43 |
| skipkv | 1 | SUCCESS | true | 2 | 255 | 920.4 | 59.37 |
| skipkv | 2 | SUCCESS | true | 1 | 510 | 1760.5 | 134.20 |

- Artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/cudagraph_gpu6_20260624/matched_graph_len128_bs1_2_out256.jsonl`; log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/cudagraph_gpu6_20260624/matched_graph_len128_bs1_2_out256.log`.
- Notes: every row reported `decode_cuda_graph_active=true` with a concrete `DecodeCudaGraphKey` for the requested method, batch size, and context capacity. Relative to matched SnapKV on this run, R-KV reached `100.7%` at `bs=1` and `87.2%` at `bs=2`; SkipKV reached `97.8%` at `bs=1` and `86.4%` at `bs=2`.

### 2026-06-23 21:37 Asia/Shanghai - rkv-skipkv-math500-smoke

- Status: completed smoke; inconclusive for quality.
- Goal: try the paper-style reasoning benchmark path without downloading large benchmarks.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=1`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B`; tokenizer same path; backend `sparsevllm`.
- Commands:

```bash
CUDA_VISIBLE_DEVICES=7 DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_smoke_20260623 PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python benchmark/math_bench/pred.py \
  --model rkv_math500_smoke --model_path /data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B --tokenizer_path /data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B \
  --backend sparsevllm --sparse_method rkv --task math500 --data_path_math500 /data2/haojitai/datasets/math500/test.jsonl --num_samples 1 --batch_size 1 \
  --max_new_tokens 64 --max_model_len 512 --temperature 0.6 --top_p 0.95 --top_k 0 \
  --hyper_param '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":64,"tensor_parallel_size":1,"decode_cuda_graph":false,"enforce_eager":true,"sink_keep_tokens":1,"recent_keep_tokens":8,"decode_keep_tokens":32,"rkv_compression_interval":16,"rkv_max_redundancy_tokens":256,"max_num_seqs_in_batch":1,"max_decoding_seqs":1}'

CUDA_VISIBLE_DEVICES=7 DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_smoke_20260623 PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python benchmark/math_bench/pred.py \
  --model skipkv_math500_smoke --model_path /data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B --tokenizer_path /data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B \
  --backend sparsevllm --sparse_method skipkv --task math500 --data_path_math500 /data2/haojitai/datasets/math500/test.jsonl --num_samples 1 --batch_size 1 \
  --max_new_tokens 64 --max_model_len 512 --temperature 0.6 --top_p 0.95 --top_k 0 \
  --hyper_param '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":64,"tensor_parallel_size":1,"decode_cuda_graph":false,"enforce_eager":true,"sink_keep_tokens":1,"recent_keep_tokens":8,"decode_keep_tokens":32,"skipkv_compression_interval":16,"skipkv_segment_size":8,"skipkv_max_redundancy_tokens":256,"max_num_seqs_in_batch":1,"max_decoding_seqs":1}'
```

- Results: both smoke runs completed; both produced `math500 pass@1=0.0`, `correct=0`, `total=1`, `missing_extracted=1`. This is not a quality result because `max_new_tokens=64` truncates reasoning and only one sample was evaluated.
- Artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_smoke_20260623/benchmark/math_bench/pred/rkv_math500_smoke/None_0623_2137`, `/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_smoke_20260623/benchmark/math_bench/pred/skipkv_math500_smoke/None_0623_2137`, `/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_smoke_20260623/mathbench_eval.log`.

### 2026-06-23 21:30 Asia/Shanghai - regression-manifest-validate

- Status: completed.
- Goal: validate the regression manifest after adding `rkv` and `skipkv`.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Command:

```bash
PYTHONPATH=$PWD:$PWD/src /home/haojitai/miniconda3/envs/svllm/bin/python benchmark/sparsevllm_regression/run_suite.py \
  --layer validate --models qwen25_7b --methods rkv,skipkv \
  --run_id skipkv_rkv_validate_20260623_r2 \
  --output_root /data2/haojitai/outputs/sparsevllm_sparse_method_support --dry_run
```

- Results: manifest validation completed.
- Artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/sparsevllm_regression/skipkv_rkv_validate_20260623_r2/grade_summary.json`.

### 2026-06-24 12:38 Asia/Shanghai - math500-bs4-cudagraph-sanity

- Status: completed sanity; not a quality score.
- Goal: verify the MATH-500 runner can expose generation throughput and confirm active decode CUDA Graph before launching the high-batch MATH-500 run.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method and benchmark instrumentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=6`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=4`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B`; tokenizer same path; backend `sparsevllm`.
- Command: see log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_bs_high_cudagraph_20260624_sanity/rkv_bs4_sanity.log`.
- Hyperparameters: `batch_size=4`, `max_new_tokens=128`, `max_model_len=1024`, `temperature=0.6`, `top_p=0.95`, `decode_cuda_graph=true`, `enforce_eager=false`, `sink_keep_tokens=1`, `recent_keep_tokens=16`, `decode_keep_tokens=128`, `rkv_compression_interval=128`, `max_num_seqs_in_batch=4`, `max_decoding_seqs=4`.
- Results: R-KV generated 4 samples and wrote `math500.jsonl`, `math500_per_sample_results.jsonl`, `result.json`, and `perf_rank0.json`. `decode_cuda_graph_active=true`, `decode_cuda_graph_graph_count=6`, generated text throughput `61.33 tok/s`. `pass@1=0.0` with `parse_failed=4`, expected because `max_new_tokens=128` truncates reasoning.
- Artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_bs_high_cudagraph_20260624_sanity/benchmark/math_bench/pred/rkv_math500_bs4_cg_sanity/None_0624_1238`.

### 2026-06-24 12:40 Asia/Shanghai - math500-rkv-skipkv-vanilla-bs16-cudagraph

- Status: running; R-KV completed, SkipKV currently running, vanilla pending.
- Goal: run full local MATH-500 for `rkv`, `skipkv`, and `vanilla` with a higher batch size while measuring generation throughput and verifying active decode CUDA Graph for each method.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Command:

```bash
tmux new-session -d -s math500_cg_bs16_gpu6_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_cudagraph_bs16_gpu6_20260624 GPU_ID=6 bash scripts/tmp/run_math500_cudagraph_bs16_gpu6_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method and benchmark instrumentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=6`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=500`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B`; tokenizer same path; backend `sparsevllm`.
- Hyperparameters: `batch_size=16`, `max_new_tokens=8192`, `max_model_len=12288`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; hparams file `scripts/tmp/math500_cudagraph_bs16_hparams_20260624.json` with `decode_cuda_graph=true`, `enforce_eager=false`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `rkv_compression_interval=128`, `skipkv_compression_interval=128`, `max_num_seqs_in_batch=16`, `max_decoding_seqs=16`.
- Expected outputs: per method, `math500.jsonl` with 500 rows, `math500_per_sample_results.jsonl` with 500 rows, `result.json`, `perf_rank0.json`; aggregate summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_cudagraph_bs16_gpu6_20260624/method_summary.jsonl`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_cudagraph_bs16_gpu6_20260624/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_cudagraph_bs16_gpu6_20260624/status.tsv`; tmux session `math500_cg_bs16_gpu6_20260624`.
- Current status: R-KV completed at `2026-06-24T13:48:46+08:00` with 500 prediction rows and active decode CUDA Graph; SkipKV started at `2026-06-24T13:48:47+08:00` and had 128 prediction rows at the latest check; vanilla has not started yet.
- Partial results:

| method | batch size | status | pass@1 | correct/total | missing extracted | generated text tok/s | graph count | artifact |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| rkv | 16 | completed | 63.8 | 319/500 | 58 | 298.55 | 25 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_cudagraph_bs16_gpu6_20260624/benchmark/math_bench/pred/math500_rkv_bs16_cg/None_0624_1240` |
| skipkv | 16 | running | TBD | 128/500 rows written | TBD | TBD | TBD | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_cudagraph_bs16_gpu6_20260624/benchmark/math_bench/pred/math500_skipkv_bs16_cg/None_0624_1348` |

### 2026-06-24 12:50 Asia/Shanghai - math500-skipkv-bs32-cudagraph-sanity

- Status: completed sanity; not a final quality score.
- Goal: test whether SkipKV can run MATH-500 with higher batch size (`bs=32`) and active decode CUDA Graph on GPU7 before launching the full 500-sample run.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method and benchmark instrumentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; `SPARSEVLLM_MASTER_PORT=2347`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=32`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B`; tokenizer same path; backend `sparsevllm`; method `skipkv`.
- Hyperparameters: `batch_size=32`, `max_new_tokens=512`, `max_model_len=12288`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; hparams file `scripts/tmp/math500_skipkv_cudagraph_bs32_hparams_20260624.json` with `decode_cuda_graph=true`, `enforce_eager=false`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `skipkv_compression_interval=128`, `max_num_seqs_in_batch=32`, `max_decoding_seqs=32`.
- Results: initial run without `SPARSEVLLM_MASTER_PORT` failed fast before model execution because default port `2333` was already used by the GPU6 run. Rerun with port `2347` completed; `decode_cuda_graph_active=true`, `decode_cuda_graph_graph_count=2`, generated text throughput `1393.81 tok/s`. The sanity score was `math500 pass@1=18.75` on 32 samples with `missing_extracted=24`, expected to be truncation-heavy because `max_new_tokens=512`.
- Artifacts: failed port-conflict log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624_sanity/skipkv_bs32_sanity.log`; successful run `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624_sanity_port2347/benchmark/math_bench/pred/skipkv_math500_bs32_cg_sanity/None_0624_1250`; log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624_sanity_port2347/skipkv_bs32_sanity.log`.

### 2026-06-24 12:51 Asia/Shanghai - math500-skipkv-bs32-cudagraph-gpu7

- Status: completed.
- Goal: run full local MATH-500 for `skipkv` only on GPU7 with larger batch size (`bs=32`), active decode CUDA Graph, and throughput artifacts.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Command:

```bash
tmux new-session -d -s math500_skipkv_cg_bs32_gpu7_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && bash scripts/tmp/run_math500_skipkv_cudagraph_bs32_gpu7_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method and benchmark instrumentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; `SPARSEVLLM_MASTER_PORT=2347`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=500`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B`; tokenizer same path; backend `sparsevllm`; method `skipkv`.
- Hyperparameters: `batch_size=32`, `max_new_tokens=8192`, `max_model_len=12288`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; hparams file `scripts/tmp/math500_skipkv_cudagraph_bs32_hparams_20260624.json` with `decode_cuda_graph=true`, `enforce_eager=false`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `skipkv_compression_interval=128`, `max_num_seqs_in_batch=32`, `max_decoding_seqs=32`.
- Expected outputs: `math500.jsonl` with 500 rows, `math500_per_sample_results.jsonl` with 500 rows, `result.json`, `perf_rank0.json`, and aggregate summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624/method_summary.jsonl`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624/status.tsv`; tmux session `math500_skipkv_cg_bs32_gpu7_20260624`.
- Results: completed at `2026-06-24T13:32:44+08:00`; 500 prediction rows and 500 per-sample rows verified; active decode CUDA Graph verified. `pass@1=65.4`, `correct=327/500`, `missing_extracted=61`, status counts `success=439`, `parse_failed=61`, generated text tokens `1,225,156`, generation elapsed `2456.93s`, throughput `498.65 text tok/s`, graph count `23`.
- Artifacts: predictions/results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624/benchmark/math_bench/pred/math500_skipkv_bs32_cg/None_0624_1251`; summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_cudagraph_bs32_gpu7_20260624/method_summary.jsonl`.

### 2026-06-24 14:34 Asia/Shanghai - math500-rkv-llama8b-bs500-cudagraph-gpu7

- Status: completed; first full run with `gpu_memory_utilization=0.90` failed with CUDA OOM, retry with the same 64-concurrency limit and `gpu_memory_utilization=0.83` completed.
- Goal: retest R-KV on the model used by the R-KV paper's Llama-family main table, using `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`, and submit all 500 MATH-500 prompts in one `Sparse-VLLM` generate call so the engine scheduler queues internally.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Command:

```bash
tmux new-session -d -s math500_rkv_llama8b_cg_bs500_gpu7_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gpu7_20260624 GPU_ID=7 bash scripts/tmp/run_math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624.sh; echo EXIT_CODE=\$?; sleep 300"

tmux new-session -d -s math500_rkv_llama8b_cg_bs500_gmem083_gpu7_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624 GPU_ID=7 HPARAMS_JSON=scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_hparams_20260624.json bash scripts/tmp/run_math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method and benchmark instrumentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; `SPARSEVLLM_MASTER_PORT=2348`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=500`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; config reports `model_type=llama`, `architectures=["LlamaForCausalLM"]`; backend `sparsevllm`; method `rkv`.
- Hyperparameters: `batch_size=500`, `max_new_tokens=8192`, `max_model_len=12288`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; current hparams file `scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_hparams_20260624.json` with `gpu_memory_utilization=0.83`, `decode_cuda_graph=true`, `enforce_eager=false`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `rkv_compression_interval=128`, `rkv_redundancy_window=64`, `max_num_seqs_in_batch=64`, `max_decoding_seqs=64`.
- Smoke notes: a `max_decoding_seqs=256` smoke failed during graph-sized warmup with CUDA OOM (`Tried to allocate 278.00 MiB`, only `170.38 MiB` free); log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624_smoke256/smoke.log`. A `max_decoding_seqs=128` smoke completed but was superseded after the user requested a 64 concurrency limit; artifact `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624_smoke128/benchmark/math_bench/pred/math500_rkv_llama8b_smoke128/None_0624_1431`. The final 64-concurrency smoke completed with 8 rows, `decode_cuda_graph_active=true`, graph count `2`, generated text throughput `503.32 tok/s`, and expected truncation-heavy `parse_failed=8` because `max_new_tokens=512`; artifact `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624_smoke64/benchmark/math_bench/pred/math500_rkv_llama8b_smoke64/None_0624_1433`.
- Expected outputs: `math500.jsonl` with 500 rows, `math500_per_sample_results.jsonl` with 500 rows, `result.json`, `perf_rank0.json`, and aggregate summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/method_summary.jsonl`.
- Logs/status: completed retry log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/run.log`; status `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/status.tsv`; tmux session `math500_rkv_llama8b_cg_bs500_gmem083_gpu7_20260624`.
- Failure notes: the first full run started at `2026-06-24T14:34:04+08:00` and failed at `2026-06-24T14:44:49+08:00`; it had submitted the single `batch_size=500` request, but no prediction rows were written because the batch did not return. The failing log reports `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB` with only `90.38 MiB` free while capturing/building a decode CUDA graph attention-score buffer; failed artifact root `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gpu7_20260624`.
- Results: retry started at `2026-06-24T14:48:01+08:00` and completed at `2026-06-24T15:07:08+08:00`; `LLM Config` confirms `max_num_seqs_in_batch=64`, `max_decoding_seqs=64`, and `decode_cuda_graph=True`. The run submitted one `batch_size=500` request and verified `rows=500 submitted_batch_size=500 graph_active=true`. MATH-500 result: `pass@1=63.0`, `correct=315/500`, `missing_extracted=62`, status counts `success=438`, `parse_failed=62`. Throughput artifact reports `generated_text_tokens=1,330,922`, `generation_elapsed_s=1122.847`, `generated_text_tokens_per_s=1185.31`, `decode_cuda_graph_active=true`, graph count `16`, last graph key `DecodeCudaGraphKey(method='rkv', batch_size=1, context_capacity=16384, is_long_text=True, capture_sampling=False)`.
- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_cg/None_0624_1448/math500.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_cg/None_0624_1448/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_cg/None_0624_1448/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_cg/None_0624_1448/perf_rank0.json`; summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxseq64_gmem083_gpu7_20260624/method_summary.jsonl`.
- Paper comparison notes: R-KV paper uses `Bbuffer=128`, observation `alpha=8`, `lambda=0.1`, maximum generation length `16384` for MATH-500, and reports pass@1 from `64` generated responses per question. Appendix Table 2 reports Llama3-8B MATH FullKV `82.38%` and R-KV from `51.08%` at budget `128` to `82.65%` at budget `2048`. This run is a single-response MATH-500 evaluation with `max_new_tokens=8192`, so it is not directly comparable to the paper's 64-response protocol; its `63.0%` is below the paper's FullKV and high-budget R-KV numbers, but within the paper's low-budget-to-mid-budget R-KV range.

### 2026-06-24 15:19 Asia/Shanghai - math500-rkv-llama8b-bs500-maxnew32k-cudagraph-gpu7

- Status: completed.
- Goal: test whether increasing MATH-500 generation budget from `8192` to `32768` improves R-KV alignment with the paper-style Llama-8B result while preserving one submitted batch of 500 prompts, scheduler concurrency cap `64`, and active decode CUDA Graph.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Command:

```bash
tmux new-session -d -s math500_rkv_llama8b_cg_bs500_maxnew32k_gpu7_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624 GPU_ID=7 HPARAMS=scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_hparams_20260624.json BATCH_SIZE=500 EXPECTED_ROWS=500 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=33792 MODEL_NAME=math500_rkv_llama8b_bs500_maxnew32k_cg SPARSEVLLM_MASTER_PORT=2349 bash scripts/tmp/run_math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Code: `codex/sparse-method-support` / `f9b712a`; worktree had relevant uncommitted Sparse-VLLM method, benchmark instrumentation, and experiment-launcher changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; `SPARSEVLLM_MASTER_PORT=2349`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=500`; previous `8192` run had 59/500 generations at or above the truncation boundary.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; backend `sparsevllm`; method `rkv`.
- Hyperparameters: `batch_size=500`, `max_new_tokens=32768`, `max_model_len=33792`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; hparams file `scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_hparams_20260624.json` with `gpu_memory_utilization=0.72`, `decode_cuda_graph=true`, `decode_cuda_graph_capture_sizes=1,2,4,8,16,32,64`, `decode_cuda_graph_context_sizes=1024,2048,4096,8192,16384,32768,33792`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `rkv_compression_interval=128`, `rkv_redundancy_window=64`, `max_num_seqs_in_batch=64`, `max_decoding_seqs=64`.
- Smoke: 4-sample smoke on GPU7 completed before full launch with `max_model_len=33792`, graph warmup at `num_seqs=64`, and `pass@1=75.0`; smoke log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_gpu7_20260624_smoke/smoke.log`.
- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_maxnew32k_cg/None_0624_1519/math500.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_maxnew32k_cg/None_0624_1519/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_maxnew32k_cg/None_0624_1519/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/benchmark/math_bench/pred/math500_rkv_llama8b_bs500_maxnew32k_cg/None_0624_1519/perf_rank0.json`; summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/method_summary.jsonl`.
- Logs/status: completed log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/run.log`; status `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_gpu7_20260624/status.tsv`; tmux session `math500_rkv_llama8b_cg_bs500_maxnew32k_gpu7_20260624` printed `EXIT_CODE=0` and was cleaned up after log capture.
- Results: completed at `2026-06-24T15:52:54+08:00`; `math500.jsonl` rows `500`, per-sample rows `500`, submitted batch size `500`; `pass@1=63.4` (`317/500` correct), `missing_extracted=17`, status counts `success=483`, `parse_failed=17`.
- Performance: `generated_text_tokens=1,915,302`, `generation_elapsed_s=1995.753`, `generated_text_tokens_per_s=959.69`; decode CUDA Graph active with graph count `16` and last key `DecodeCudaGraphKey(method='rkv', batch_size=1, context_capacity=33792, is_long_text=True, capture_sampling=False)`.
- Length audit: local tokenizer counted `avg=3832.6`, `median=1527`, `p90=10197.1`, `p95=16326.5`, `p99=32770`, `max=32771`; `63/500` generations were `>=8190` tokens, `27/500` were `>=16000`, and `6/500` were `>=32700`.
- Comparison to the prior `8192` R-KV run: score changed from `63.0` to `63.4` (`+2` correct), parse failures changed from `62` to `17` (`-45`), generated tokens increased from `1,330,922` to `1,915,302`, and throughput decreased from `1185.31` to `959.69 tok/s`.
- Interpretation: raising `max_new_tokens` to `32768` fixed many extraction failures but did not materially close the accuracy gap; the low score is therefore not primarily an `8192` output-token cap issue. The run is still not directly paper-comparable because this local run used one response per problem, while the R-KV paper reports MATH-500 with multiple sampled responses per question.

### 2026-06-24 16:19 Asia/Shanghai - skipkv-vector-math500-cudagraph-qwen7b

- Status: completed.
- Goal: build a SkipKV activation steering vector from MATH train, then test SkipKV on full local MATH-500 with CUDA Graph enabled, comparing no steering, the self-generated vector, the official SkipKV Qwen7B layer-20 vector, and a high-batch scheduler-throughput run.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`.
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree had relevant uncommitted Sparse-VLLM method, activation-controller, vector-script, benchmark, and documentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPUs `CUDA_VISIBLE_DEVICES=4,5,6,7`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `num_samples=500`. Vector training data was exported from `EleutherAI/hendrycks_math` train to `/data2/haojitai/datasets/hendrycks_math_train/eleutherai_hendrycks_math_train_7500.jsonl`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Qwen-7B`; tokenizer same path; backend `sparsevllm`.
- Vector script at the time: a now-removed heuristic script constructed `steering_vector = execution_mean - non_execution_mean`, preferring explicit `execution_spans` / `non_execution_spans` and otherwise using bounded MATH-style sentence heuristics. It was removed because the heuristic vector was not paper-equivalent and current SkipKV support is official-vector-only.
- Generated vector artifact: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/skipkv_vector_qwen7b_math_train_layer20_gpu7_20260624_retry_localdata/qwen7b_layer20_math_train_heuristic.pt`; layer `20`; shape `[3584]`; raw vector norm `26.0718`; usable samples `877/1000`; execution tokens `223028`; non-execution tokens `50038`; label source counts `explicit=0`, `heuristic=877`.
- Official-vector diagnostic artifact: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/official_skipkv_vectors/qwen7b_layer20_transition_reflection_steervec.pt`; norm `83.2809`; cosine with the self-generated heuristic vector `-0.23234`, so the heuristic vector is not paper-equivalent.
- Commands:

```bash
tmux new-session -d -s math500_skipkv_sentence_delimiterfix_gpu4_20260624 \
  "bash /home/haojitai/projects/Sparse-vLLM-sparse-method-support/scripts/tmp/run_math500_skipkv_sentence_delimiterfix_gpu4_20260624.sh"

tmux new-session -d -s math500_skipkv_steering_delimiterfix_gpu7_20260624 \
  "bash /home/haojitai/projects/Sparse-vLLM-sparse-method-support/scripts/tmp/run_math500_skipkv_steering_delimiterfix_gpu7_20260624.sh"

tmux new-session -d -s math500_skipkv_steering_officialvec_gpu6_20260624 \
  "bash /home/haojitai/projects/Sparse-vLLM-sparse-method-support/scripts/tmp/run_math500_skipkv_steering_officialvec_gpu6_20260624.sh"

tmux new-session -d -s math500_skipkv_steering_bs500_maxseq64_gpu5_20260624 \
  "bash /home/haojitai/projects/Sparse-vLLM-sparse-method-support/scripts/tmp/run_math500_skipkv_steering_bs500_maxseq64_gpu5_20260624.sh"
```

- Shared Qwen7B MATH-500 hyperparameters: `max_new_tokens=8192`, `max_model_len=12288`, `temperature=0.6`, `top_p=0.95`, `top_k=0`, `decode_cuda_graph=true`, `enforce_eager=false`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `skipkv_compression_interval=128`, `skipkv_redundancy_window=64`, `skipkv_max_redundancy_tokens=4096`, newline-oriented SkipKV delimiters, and CUDA Graph verification from `perf_rank0.json`.
- Results:

| run | batch size | scheduler max seqs | vector | pass@1 | correct | parse_failed | text tok/s | graph count | artifact root |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla baseline | 10 | 10 | none | 68.4 | 342/500 | 40 | 340.74 | 16 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_cudagraph_bs10_gpu6_20260624` |
| R-KV baseline | 10 | 10 | none | 65.4 | 327/500 | 60 | 220.38 | 16 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_cudagraph_bs10_gpu5_20260624` |
| SkipKV no steering | 10 | 10 | none | 67.2 | 336/500 | 50 | 147.70 | 16 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_sentence_cudagraph_bs10_gpu4_delimiterfix_20260624` |
| SkipKV self-generated vector | 10 | 10 | heuristic MATH train | 66.8 | 334/500 | 58 | 148.49 | 16 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_steering_cudagraph_bs10_gpu7_delimiterfix_20260624` |
| SkipKV official vector diagnostic | 10 | 10 | official Qwen7B layer20 | 67.0 | 335/500 | 59 | 145.90 | 16 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_steering_officialvec_cudagraph_bs10_gpu6_20260624` |
| SkipKV self-generated vector high batch | 500 | 64 | heuristic MATH train | 65.0 | 325/500 | 59 | 354.64 | 16 | `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_skipkv_steering_cudagraph_bs500_maxseq64_gpu5_20260624` |

- Validation: each completed row has `math500.jsonl` rows `500`, `math500_per_sample_results.jsonl` rows `500`, `result.json`, `perf_rank0.json`, `method_summary.jsonl`, and `decode_cuda_graph_graph_count > 0`. GPUs 4-7 were free after completion.
- Interpretation: the SkipKV runtime path is functional with CUDA Graph, but the bs10 SkipKV implementation is slow on long MATH-500 generations, around `146-148 text tok/s`, well below vanilla `340.74` and R-KV `220.38`. Submitting all 500 prompts in one scheduler batch raises the measured throughput to `354.64 text tok/s`, but the score drops to `65.0` and the last long-answer tail still dominates wall time.
- Paper-comparison caveats: SkipKV paper-style settings use MATH train activation vectors, layer `20`, negative steering strength, newline-oriented delimiters, and MATH-500 max generation length `8192`. This local benchmark is a same-runner comparison using the repo's chat prompt and sampled decoding (`temperature=0.6`, `top_p=0.95`), not the official SkipKV greedy prompt. The self-generated vector uses heuristic labels because plain MATH train lacks execution/non-execution spans; its negative cosine against the official vector confirms it is not a reproduction of the paper's vector construction.

### 2026-06-24 19:48 Asia/Shanghai - math500-vanilla-llama8b-bs500-maxnew32k-top-p-cudagraph-gpu7

- Status: completed; 4-sample smoke completed first.
- Goal: retest the FullKV/vanilla MATH-500 baseline with paper-style sampled decoding (`temperature=0.6`, `top_p=0.95`) and `max_new_tokens=32768`, after confirming the Sparse-VLLM adapter forwards `top_p` into `SamplingParams`.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree had relevant uncommitted Sparse-VLLM method, math benchmark, experiment-launcher, and `src/deltakv/get_chat_api.py` adapter changes.
- Adapter change: `src/deltakv/get_chat_api.py` now maps HF-style `top_p` to `sparsevllm.SamplingParams`. A monkeypatch test verified `{'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 32}`.
- Environment: host `guest-KR6288-X2-A0-R0-00`; full run GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; `SPARSEVLLM_MASTER_PORT=2353`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; full run `num_samples=500`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; backend `sparsevllm`; method `vanilla`, normalized runtime config `vllm_sparse_method=''`.
- Hyperparameters: `batch_size=500`, `max_new_tokens=32768`, `max_model_len=33792`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; hparams file `scripts/tmp/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_hparams_20260624.json` with `gpu_memory_utilization=0.82`, `decode_cuda_graph=true`, `decode_cuda_graph_capture_sizes=1,2,4,8`, `decode_cuda_graph_context_sizes=1024,2048,4096,8192,16384,32768,33792`, `max_num_seqs_in_batch=8`, `max_decoding_seqs=8`.
- FullKV concurrency note: the run still submits all 500 prompts in one outer batch, but vanilla keeps full KV for generated tokens, so the internal active decode cap is 8 rather than R-KV's 64 to avoid H100 80GB OOM. This affects throughput comparability, not sampling or scoring semantics.
- Smoke command:

```bash
RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_smoke_bs4_maxnew512_top_p095_gpu5_20260624 \
GPU_ID=5 EXPECTED_ROWS=4 BATCH_SIZE=4 MAX_NEW_TOKENS=512 MAX_MODEL_LEN=33792 \
MODEL_NAME=math500_vanilla_llama8b_smoke_bs4_top_p095_maxnew512_cg SPARSEVLLM_MASTER_PORT=2352 \
bash scripts/tmp/run_math500_vanilla_llama8b_cudagraph_bs500_top_p095_gpu5_20260624.sh
```

- Smoke result: completed at `2026-06-24T19:48:53+08:00`; verified `rows=4`, `submitted_batch_size=4`, `decode_cuda_graph_active=true`, `top_p=0.95`, graph count `2`, last graph key `DecodeCudaGraphKey(method='', batch_size=4, context_capacity=1024, is_long_text=False, capture_sampling=False)`, generated text throughput `311.97 tok/s`. The short smoke scored `pass@1=25.0` with `3/4` parse failures, expected because `max_new_tokens=512` truncates reasoning.
- Full command:

```bash
tmux new-session -d -s math500_vanilla32k_topp_gpu7_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624 GPU_ID=7 EXPECTED_ROWS=500 BATCH_SIZE=500 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=33792 MODEL_NAME=math500_vanilla_llama8b_bs500_top_p095_maxnew32768_cg SPARSEVLLM_MASTER_PORT=2353 bash scripts/tmp/run_math500_vanilla_llama8b_cudagraph_bs500_top_p095_gpu5_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Expected outputs: predictions `.../math500.jsonl`, per-sample results `.../math500_per_sample_results.jsonl`, aggregate result `.../result.json`, performance `.../perf_rank0.json`, and summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/method_summary.jsonl`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/status.tsv`; tmux session `math500_vanilla32k_topp_gpu7_20260624`.
- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_cg/None_0624_1949/math500.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_cg/None_0624_1949/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_cg/None_0624_1949/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_cg/None_0624_1949/perf_rank0.json`; summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_gpu7_20260624/method_summary.jsonl`.
- Results: completed at `2026-06-24T20:36:16+08:00`; tmux pane printed `EXIT_CODE=0`; `math500.jsonl` rows `500`, per-sample rows `500`, submitted batch size `500`; `pass@1=69.4` (`347/500` correct), `missing_extracted=14`, status counts `success=486`, `parse_failed=14`.
- Performance: `generated_text_tokens=1,616,855`, `generation_elapsed_s=2794.787`, `generated_text_tokens_per_s=578.53`; decode CUDA Graph active with graph count `16` and last key `DecodeCudaGraphKey(method='', batch_size=1, context_capacity=33792, is_long_text=True, capture_sampling=False)`. Full run log confirms `vllm_sparse_method=''`, `decode_cuda_graph=True`, `max_num_seqs_in_batch=8`, `max_decoding_seqs=8`; GPU7 reached about `67.2GB` during generation and returned to idle after completion.
- Length audit: local tokenizer counted `avg=3235.7`, `median=1507`, `p90=7164`, `p95=11713`, `p99=32770`, `max=32770`; `42/500` generations were `>=8190` tokens, `15/500` were `>=16000`, and `9/500` were `>=32700`.
- Comparison to the local R-KV `32768` run: vanilla scored `69.4` vs R-KV `63.4`, with fewer parse failures (`14` vs `17`) and fewer generated tokens (`1,616,855` vs `1,915,302`). Vanilla throughput was lower (`578.53 tok/s` vs R-KV `959.69 tok/s`) because FullKV used `max_decoding_seqs=8` to fit 80GB H100 memory while R-KV used `max_decoding_seqs=64`.
- Interpretation: forwarding `top_p=0.95` and raising `max_new_tokens` to `32768` improves the vanilla baseline relative to earlier local sampled baselines, but the local single-response result is still below the R-KV paper's reported Llama3-8B FullKV MATH-500 `82.38%`. This remains not directly paper-comparable because the paper reports MATH-500 with multiple sampled responses per question, while this run uses one response per problem.

### 2026-06-24 23:04 Asia/Shanghai - math500-vanilla-llama8b-prefill-think-bs500-maxnew32k-cudagraph-gpu7

- Status: completed; 2-sample smoke completed first on GPU7.
- Goal: retest the FullKV/vanilla MATH-500 baseline with the DeepSeek-R1 usage recommendation more closely aligned by appending `<think>\n` to the actual generation prompt, rather than only adding it back to saved outputs after generation.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree had relevant uncommitted Sparse-VLLM method, math benchmark, launcher, and documentation changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; full run GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; `SPARSEVLLM_MASTER_PORT=2357`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; full run `num_samples=500`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; backend `sparsevllm`; method `vanilla`, normalized runtime config `vllm_sparse_method=''`.
- Hyperparameters: `batch_size=500`, `max_new_tokens=32768`, `max_model_len=33792`, `temperature=0.6`, `top_p=0.95`, `top_k=0`; hparams file `scripts/tmp/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_hparams_20260624.json` with `gpu_memory_utilization=0.82`, `decode_cuda_graph=true`, `decode_cuda_graph_capture_sizes=1,2,4,8`, `decode_cuda_graph_context_sizes=1024,2048,4096,8192,16384,32768,33792`, `max_num_seqs_in_batch=8`, `max_decoding_seqs=8`.
- Prompt/decoding alignment: `benchmark/math_bench/pred.py` now supports `--prefill_think_prefix` and `--no_prompt_think_instruction`; this run uses both, so the user prompt keeps the math directive and the actual generation prompt ends with `<think>\n`.
- Smoke artifact: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_prefillthink_smoke_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_prefillthink_smoke_gpu7_cg/None_0624_2303`; smoke verified rows `2`, `prefill_think_prefix=true`, `prompt_think_instruction=false`, `decode_cuda_graph_active=true`, graph count `3`, and generated text throughput `108.27 tok/s`. The smoke score `50.0` is not a quality metric because `max_new_tokens=512`.
- Full command:

```bash
tmux new-session -d -s math500_vanilla_prefill_gpu7_20260624 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624 GPU_ID=7 SPARSEVLLM_MASTER_PORT=2357 bash scripts/tmp/run_math500_vanilla_llama8b_cudagraph_prefillthink_gpu4_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Expected outputs: predictions `.../math500.jsonl`, per-sample results `.../math500_per_sample_results.jsonl`, aggregate result `.../result.json`, performance `.../perf_rank0.json`, and summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/method_summary.jsonl`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/status.tsv`; tmux session `math500_vanilla_prefill_gpu7_20260624`.
- Running observation: first 30-second generation heartbeat reported `prefill_tp=916 tok/s`, `decode_tp=513 tok/s`, `seq(run/prf/dc)=494/484/10`; GPU7 reached about `66.8GB` and `95%` utilization.
- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_prefillthink_cg/None_0624_2304/math500.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_prefillthink_cg/None_0624_2304/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_prefillthink_cg/None_0624_2304/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/benchmark/math_bench/pred/math500_vanilla_llama8b_bs500_top_p095_maxnew32768_prefillthink_cg/None_0624_2304/perf_rank0.json`; summary `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_vanilla_llama8b_cudagraph_bs500_maxnew32k_maxseq8_top_p095_prefillthink_gpu7_20260624/method_summary.jsonl`.
- Results correction: completed at `2026-06-24T23:59:45+08:00`; tmux pane printed `EXIT_CODE=0`; `math500.jsonl` rows `500`, submitted batch size `500`. The original local regex/string-equality score `pass@1=71.4` (`357/500`, `missing_extracted=7`) was wrong and should not be cited. On `2026-06-25`, the standard `result.json` and `math500_per_sample_results.jsonl` were overwritten with the `math-verify==0.9.0` scorer: `pass@1=89.8` (`449/500` correct), `missing_extracted=0`, status counts `success=500`; `math500_parsed_outputs.jsonl` was added.
- Performance: `generated_text_tokens=1,975,185`, `generation_elapsed_s=3325.758`, `generated_text_tokens_per_s=593.91`; decode CUDA Graph active with graph count `15` and last key `DecodeCudaGraphKey(method='', batch_size=1, context_capacity=32768, is_long_text=True, capture_sampling=False)`.
- Comparison note: the earlier score comparison that used `69.4`/`71.4` came from the removed regex evaluator and is invalid. The corrected `89.8` score indicates the main discrepancy was evaluator undercount, not generation quality.

### 2026-06-25 02:14 Asia/Shanghai - math500-official-vllm-llama8b-bs500-maxnew32k-gpu0

- Status: completed; two official-vLLM smoke runs completed first.
- Goal: check whether an official vLLM environment can reproduce a higher DeepSeek-R1-Distill-Llama-8B MATH-500 vanilla score than the local Sparse-VLLM FullKV baseline, while also measuring high-throughput scheduling with all 500 prompts submitted at once.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree had relevant uncommitted math benchmark, official-vLLM launcher, Sparse-VLLM method, and experiment-record changes.
- Environment setup: the first latest-vLLM attempt at `/data2/haojitai/conda_envs/vllm-math500-py310` was aborted at `2026-06-25T01:11:59+08:00` because `pip install --upgrade vllm` began pulling a very large CUDA 13 / torch 2.11 dependency chain. The completed pinned environment is `/data2/haojitai/conda_envs/vllm-math500-torch28`, created by offline-cloning the local `svllm` torch2.8/cu128 environment and installing `vllm==0.10.2` plus `transformers==4.55.2`; setup log `/data2/haojitai/outputs/sparsevllm_sparse_method_support/vllm_math500_torch28_env_setup_20260625/setup.log`, status `/data2/haojitai/outputs/sparsevllm_sparse_method_support/vllm_math500_torch28_env_setup_20260625/status.tsv`.
- Environment versions: Python `/data2/haojitai/conda_envs/vllm-math500-torch28/bin/python`; `torch=2.8.0+cu128`, `transformers=4.55.2`, `vllm=0.10.2`, `datasets=4.1.0`, `sympy=1.14.0`; GPU `CUDA_VISIBLE_DEVICES=0`, NVIDIA H100 80GB HBM3.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; full run `num_samples=500`.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; backend `official_vllm`.
- Prompt/decoding alignment: math prompt asks `Please reason step by step, and put your final answer within \boxed{}.`; no system prompt; actual generation prompt appends `<think>\n`; sampled decoding uses `temperature=0.6`, `top_p=0.95`, `seed=42`, `max_tokens=32768`, `max_model_len=33792`, `max_num_seqs=64`, `gpu_memory_utilization=0.9`, `dtype=bfloat16`.
- CUDA graph/performance note: official vLLM V1 enabled CUDA graph capture with sizes up to `128` and captured the full-run graph set successfully. It also warned that FlashInfer was unavailable and top-p/top-k sampling fell back to PyTorch-native sampling, so the measured throughput is not the fastest possible official-vLLM top-p path.
- Smoke artifacts: 2-row API smoke `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_smoke_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_smoke_top_p095_maxnew512_maxseq4/official_vllm_0625_0212`; max-length/max-seqs smoke `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_smoke_maxlen33792_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_smoke_top_p095_maxnew512_maxlen33792_maxseq64/official_vllm_0625_0213`. Both wrote 2 predictions, per-sample results, raw outputs, result, perf, and run config; scores were not quality metrics because `max_tokens=512`.
- Full command:

```bash
tmux new-session -d -s math500_official_vllm_llama8b_gpu0_20260625 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && ENV_PREFIX=/data2/haojitai/conda_envs/vllm-math500-torch28 RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625 GPU_ID=0 EXPECTED_ROWS=500 MAX_TOKENS=32768 MAX_MODEL_LEN=33792 RUN_NAME=math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64 MAX_NUM_SEQS=64 bash scripts/tmp/run_math500_vllm_llama8b_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64/official_vllm_0625_0214/math500.jsonl`; raw outputs `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64/official_vllm_0625_0214/raw_outputs/math500_raw_outputs.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64/official_vllm_0625_0214/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64/official_vllm_0625_0214/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64/official_vllm_0625_0214/perf_rank0.json`; run config `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_top_p095_maxnew32768_maxseq64/official_vllm_0625_0214/run_config.json`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_bs500_maxnew32k_maxseq64_top_p095_gpu0_20260625/status.tsv`; tmux session printed `EXIT_CODE=0` and was cleaned up after log capture.
- Validation: `math500.jsonl`, `math500_per_sample_results.jsonl`, and `raw_outputs/math500_raw_outputs.jsonl` each have `500` rows; status completed at `2026-06-25T02:35:44+08:00`; no traceback, runtime error, CUDA OOM, or illegal-memory lines were found in `run.log`.
- Results correction: the original local regex/string-equality score `pass@1=71.8` (`359/500`, `missing_extracted=6`) was wrong and should not be cited. On `2026-06-25`, the standard `result.json` and `math500_per_sample_results.jsonl` were overwritten with the `math-verify==0.9.0` scorer: `pass@1=89.2` (`446/500` correct), `missing_extracted=1`, status counts `success=499`, `parse_failed=1`; `math500_parsed_outputs.jsonl` was added.
- Performance: `generated_text_tokens=2,349,466`, `generation_elapsed_s=1243.328`, `generated_text_tokens_per_s=1889.66`. This is about `3.18x` the Sparse-VLLM vanilla prefill-think run (`593.91 tok/s`). After scorer correction, this run is no longer far below the DeepSeek model-card MATH-500 number `89.1`; it is essentially aligned at `89.2`.

### 2026-06-25 11:07 Asia/Shanghai - math500-official-vllm-llama8b-openr1-prompt-gpu7

- Status: completed; 2-sample smoke completed first.
- Goal: mimic the Open-R1 MATH-500 reproduction path without installing or invoking LightEval, by using its MATH prompt template with the existing official-vLLM runner and local evaluator.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree has relevant uncommitted changes in `benchmark/math_bench/pred.py`, `scripts/tmp/run_math500_vllm_llama8b_20260624.py`, and `scripts/tmp/run_math500_vllm_llama8b_20260624.sh`.
- Environment: host `guest-KR6288-X2-A0-R0-00`; env `/data2/haojitai/conda_envs/vllm-math500-torch28`; GPU `CUDA_VISIBLE_DEVICES=7`, NVIDIA H100 80GB HBM3; backend `official_vllm`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; full run `500` samples.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; dtype `bfloat16`.
- Prompt/decoding alignment: `PROMPT_STYLE=openr1` uses the Open-R1/LightEval MATH query wording that requires the last line to be `Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct`; chat template is enabled and already renders `<｜Assistant｜><think>\n`; sampled decoding uses `temperature=0.6`, `top_p=0.95`, `seed=42`, `max_tokens=32768`, `max_model_len=32768`, `max_num_seqs=64`, `gpu_memory_utilization=0.9`.
- Smoke command:

```bash
ENV_PREFIX=/data2/haojitai/conda_envs/vllm-math500-torch28 \
RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_smoke_gpu7_20260625 \
GPU_ID=7 EXPECTED_ROWS=2 NUM_SAMPLES=2 MAX_TOKENS=512 MAX_MODEL_LEN=32768 \
RUN_NAME=math500_official_vllm_llama8b_openr1_smoke_top_p095_maxnew512_maxlen32768_maxseq64 \
MAX_NUM_SEQS=64 PROMPT_STYLE=openr1 bash scripts/tmp/run_math500_vllm_llama8b_20260624.sh
```

- Smoke result: completed at `2026-06-25T11:07:17+08:00`; rows `2`, per-sample rows `2`, generated tokens `1024`, throughput `247.62 tok/s`; score is not a quality metric because `max_tokens=512` truncated reasoning.
- Full command:

```bash
tmux new-session -d -s math500_official_vllm_openr1_llama8b_gpu7_20260625 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && ENV_PREFIX=/data2/haojitai/conda_envs/vllm-math500-torch28 RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625 GPU_ID=7 EXPECTED_ROWS=500 MAX_TOKENS=32768 MAX_MODEL_LEN=32768 RUN_NAME=math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64 MAX_NUM_SEQS=64 PROMPT_STYLE=openr1 bash scripts/tmp/run_math500_vllm_llama8b_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Expected outputs: predictions `.../math500.jsonl`, raw outputs `.../raw_outputs/math500_raw_outputs.jsonl`, per-sample results `.../math500_per_sample_results.jsonl`, aggregate result `.../result.json`, performance `.../perf_rank0.json`, and run config `.../run_config.json`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/status.tsv`; tmux session `math500_official_vllm_openr1_llama8b_gpu7_20260625`.
- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/math500.jsonl`; raw outputs `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/raw_outputs/math500_raw_outputs.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/perf_rank0.json`; run config `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/run_config.json`.
- Validation: `math500.jsonl`, `math500_per_sample_results.jsonl`, and `raw_outputs/math500_raw_outputs.jsonl` each have `500` rows; status completed at `2026-06-25T11:20:48+08:00`; tmux pane printed `EXIT_CODE=0`; no traceback, runtime error, CUDA OOM, or illegal-memory lines were found in `run.log`.
- Results correction: the original local regex/string-equality score `pass@1=75.4` (`377/500`, `missing_extracted=4`) was wrong and should not be cited. On `2026-06-25`, the standard `result.json` and `math500_per_sample_results.jsonl` were overwritten with the `math-verify==0.9.0` scorer: `pass@1=90.0` (`450/500` correct), `missing_extracted=0`, status counts `success=500`; `math500_parsed_outputs.jsonl` was added.
- Performance: `generated_text_tokens=1,750,272`, `generation_elapsed_s=750.132`, `generated_text_tokens_per_s=2333.28`; vLLM captured CUDA graphs with max capture size `128`, used `max_num_seqs=64`, and reported maximum full-length concurrency `13.59x` for 32,768-token requests.
- Open-R1-style rescore: installed `math-verify==0.9.0` into `/data2/haojitai/conda_envs/vllm-math500-torch28` with dependencies `latex2sympy2_extended==1.11.0` and `antlr4-python3-runtime==4.13.2`, then rescored the same `math500.jsonl` predictions without regenerating. Gold parsing uses `solution` with `answer` fallback and `LatexExtractionConfig()`; prediction parsing uses `ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)`, matching the important Open-R1/LightEval-style extraction behavior more closely than the removed repo regex evaluator.
- Open-R1-style rescore artifacts: aggregate `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/math500_math_verify_result.json`; parsed outputs `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/math500_math_verify_parsed_outputs.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/math500_math_verify_per_sample_results.jsonl`.
- Open-R1-style rescore result: `pass@1=90.0` (`450/500` correct), `missing_extracted=0`, status counts `success=500`; both new JSONL artifacts have `500` rows. Interpretation: the earlier `75.4` result was an evaluator undercount by `73` samples, not a generation-quality gap. With `PROMPT_STYLE=openr1`, sampled decoding, and `math-verify` scoring, this run is aligned with the DeepSeek/Open-R1 MATH-500 model-card range (`~88.6-89.1`) within expected single-run variance. The standard `result.json` now also contains this corrected `90.0` result.

### 2026-06-25 11:32 Asia/Shanghai - mathbench-eval-py-math-verify-switch

- Status: completed.
- Goal: replace the local `benchmark/math_bench/eval.py` scorer with `math-verify` so subsequent local MATH-500/GSM8K/AIME/HMMT evaluations use symbolic/equivalence checking rather than regex string equality.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`.
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree has relevant uncommitted changes in `benchmark/math_bench/eval.py`, `tests/test_math_bench_eval.py`, `benchmark/math_bench/README.md`, `pyproject.toml`, and this experiment record.
- Environment: host `guest-KR6288-X2-A0-R0-00`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; installed `math-verify==0.9.0`, `latex2sympy2_extended==1.11.0`, and `antlr4-python3-runtime==4.13.2` into the local `svllm` env; cache path `/data2/haojitai/pip_cache`.
- Implementation: `eval.py` now imports `math_verify.parse`/`verify`, writes `result.json`, `<dataset>_parsed_outputs.jsonl`, and `<dataset>_per_sample_results.jsonl`; prediction parsing uses `ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)`; MATH-500 gold prefers `solution` with `answer` fallback; parse/verify timeouts are bounded and every row gets an explicit status. The previous regex/string-equality scoring path and `--allow_unboxed` compatibility flag were removed.
- Validation commands:

```bash
/home/haojitai/miniconda3/envs/svllm/bin/python -m py_compile benchmark/math_bench/eval.py tests/test_math_bench_eval.py
/home/haojitai/miniconda3/envs/svllm/bin/python -m unittest tests.test_math_bench_eval
SRC=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_official_vllm_llama8b_openr1_bs500_maxnew32k_maxseq64_top_p095_gpu7_20260625/benchmark/math_bench/pred/math500_official_vllm_llama8b_openr1_top_p095_maxnew32768_maxlen32768_maxseq64/official_vllm_0625_1107/math500.jsonl
EVAL_DIR=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_eval_py_mathverify_openr1_verify_20260625
rm -rf "$EVAL_DIR"
mkdir -p "$EVAL_DIR"
ln -s "$SRC" "$EVAL_DIR/math500.jsonl"
/home/haojitai/miniconda3/envs/svllm/bin/python benchmark/math_bench/eval.py --path "$EVAL_DIR"
```

- Validation artifacts: aggregate `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_eval_py_mathverify_openr1_verify_20260625/result.json`; parsed outputs `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_eval_py_mathverify_openr1_verify_20260625/math500_parsed_outputs.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_eval_py_mathverify_openr1_verify_20260625/math500_per_sample_results.jsonl`.
- Results: unit tests `4/4` passed. Full 500-row offline validation reproduced the prior `math-verify` rescore: `pass@1=90.0`, `450/500` correct, `missing_extracted=0`, status counts `success=500`; parsed and per-sample JSONL files each have `500` rows.

### 2026-06-25 11:35 Asia/Shanghai - mathbench-existing-full-runs-rescored

- Status: completed.
- Goal: update existing full MATH-500 result artifacts in place so standard `result.json` and `math500_per_sample_results.jsonl` use the new `math-verify` scorer, instead of preserving separate old-regex and new-math-verify paths.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`.
- Environment: host `guest-KR6288-X2-A0-R0-00`; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; scorer `math-verify==0.9.0`.
- Command summary: found all `/data2/haojitai/outputs/sparsevllm_sparse_method_support/**/math500.jsonl`; skipped files with row counts other than `500`; ran `benchmark/math_bench/eval.py --path <prediction_dir>` for each full directory, overwriting standard eval artifacts and adding `math500_parsed_outputs.jsonl`.
- Status/summary artifacts: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_eval_py_mathverify_rescore_20260625/status.tsv`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/mathbench_eval_py_mathverify_rescore_20260625/summary.tsv`.
- Results: `16` full 500-row directories rescored; smoke, partial, and 0-row directories were skipped. Any earlier MATH-500 pass@1 numbers produced by the regex/string-equality evaluator are superseded and should be treated as wrong.

| Run directory suffix | pass@1 | correct/total | status |
| --- | ---: | ---: | --- |
| `official_vllm_0625_1107` Open-R1 prompt | `90.0` | `450/500` | `success=500` |
| `official_vllm_0625_0214` DeepSeek-style prompt | `89.2` | `446/500` | `success=499`, `parse_failed=1` |
| `math500_vanilla_llama8b_bs500_top_p095_maxnew32768_prefillthink_cg/None_0624_2304` | `89.8` | `449/500` | `success=500` |
| `math500_vanilla_llama8b_bs500_top_p095_maxnew32768_cg/None_0624_1949` | `86.6` | `433/500` | `success=500` |
| `math500_vanilla_bs10_cg/None_0624_1540` | `87.6` | `438/500` | `success=500` |
| `math500_vanilla_bs16_cg/None_0624_1456` | `87.4` | `437/500` | `success=500` |
| `math500_rkv_llama8b_bs500_maxnew32k_cg/None_0624_1519` | `79.8` | `399/500` | `success=500` |
| `math500_rkv_llama8b_bs500_cg/None_0624_1448` | `79.0` | `395/500` | `success=500` |
| `math500_rkv_bs10_cg/None_0624_1540` | `84.2` | `421/500` | `success=500` |
| `math500_rkv_bs16_cg/None_0624_1240` | `81.0` | `405/500` | `success=500` |
| `math500_skipkv_bs500_cg/None_0624_1724` | `84.4` | `422/500` | `success=500` |
| `math500_skipkv_bs32_cg/None_0624_1251` | `82.0` | `410/500` | `success=500` |
| `math500_skipkv_bs16_cg/None_0624_1348` | `83.6` | `418/500` | `success=500` |
| `math500_skipkv_bs10_cg/None_0624_1648` | `84.8` | `424/500` | `success=500` |
| `math500_skipkv_bs10_cg/None_0624_1619` sentence delimiter fix | `84.6` | `423/500` | `success=500` |
| `math500_skipkv_bs10_cg/None_0624_1619` steering delimiter fix | `84.2` | `421/500` | `success=500` |

### 2026-06-25 13:03 Asia/Shanghai - math500-rkv-formula-lambda-llama8b-bs500-gpu4

- Status: completed.
- Goal: retest R-KV on local MATH-500 after changing the retention formula to the paper-style joint score `alpha * importance - (1 - alpha) * redundancy`, using all 500 prompts submitted at once so Sparse-VLLM can queue internally up to the configured decode concurrency.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`.
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree had relevant uncommitted R-KV formula, benchmark, scorer, skill, and experiment-record changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=4`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`; `SPARSEVLLM_MASTER_PORT=2365`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `500` samples.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; backend `sparsevllm`; method `rkv`.
- Prompt/decoding: sampled decoding with `temperature=0.6`, `top_p=0.95`, `top_k=0`, `max_new_tokens=32768`, `max_model_len=33792`, `batch_size=500`, `max_num_seqs_in_batch=64`, `max_decoding_seqs=64`, `decode_cuda_graph=true`.
- R-KV hparams: `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `rkv_compression_interval=128`, `rkv_alpha=0.1`, `rkv_redundancy_window=64`, `rkv_max_redundancy_tokens=4096`, `gpu_memory_utilization=0.72`, `engine_prefill_chunk_size=1024`.
- Command:

```bash
tmux new-session -d -s math500_rkv_formula_gpu4_20260625 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625 GPU_ID=4 HPARAMS=scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_gmem072_hparams_20260624.json BATCH_SIZE=500 EXPECTED_ROWS=500 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=33792 MODEL_NAME=math500_rkv_formula_lambda_bs500_maxnew32k_cg SPARSEVLLM_MASTER_PORT=2365 bash scripts/tmp/run_math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Result artifacts: predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_formula_lambda_bs500_maxnew32k_cg/None_0625_1303/math500.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_formula_lambda_bs500_maxnew32k_cg/None_0625_1303/result.json`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_formula_lambda_bs500_maxnew32k_cg/None_0625_1303/math500_per_sample_results.jsonl`; parsed outputs `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_formula_lambda_bs500_maxnew32k_cg/None_0625_1303/math500_parsed_outputs.jsonl`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_formula_lambda_bs500_maxnew32k_cg/None_0625_1303/perf_rank0.json`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_formula_lambda_gpu4_20260625/status.tsv`; tmux pane printed `EXIT_CODE=0`.
- Validation: `math500.jsonl` has `500` rows; `math500_per_sample_results.jsonl` has `500` rows; status completed at `2026-06-25T13:38:07+08:00`; no missing extracted answers; all rows have status `success`.
- Results: `math-verify==0.9.0` score `pass@1=83.0`, `415/500` correct, `missing_extracted=0`, status counts `success=500`.
- Performance: `generated_text_tokens=2,013,714`, `generation_elapsed_s=2082.107`, `generated_text_tokens_per_s=967.15`; decode CUDA Graph active with graph count `16`, last key `DecodeCudaGraphKey(method='rkv', batch_size=1, context_capacity=33792, is_long_text=True, capture_sampling=False)`. The run showed a strong long tail: after high-concurrency phases, the final 7 samples dropped the 30s decode window to `358`, `311`, `187`, then `64 tok/s`.
- Interpretation: the formula change improves R-KV from the previous `79.8` full Llama-8B R-KV max-new-32k run to `83.0`, but it remains below the corrected Sparse-VLLM vanilla baseline `89.8` and official-vLLM/Open-R1-style baselines around `89-90`. This is still not strict paper-aligned R-KV because redundancy scoring remains limited to the trailing `rkv_redundancy_window=64` candidate tokens rather than the full budget-candidate redundancy selection range.

### 2026-06-25 13:59 Asia/Shanghai - math500-rkv-strict-full-candidate-i1024-gpu4

- Status: completed.
- Goal: retest R-KV after changing the default R-KV redundancy selection to full candidate-set scoring (`rkv_redundancy_window=0`) and increasing decode eviction interval to `1024`, so the MATH-500 run is strict with respect to the paper-style budget-candidate redundancy range.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-sparse-method-support`.
- Code: `codex/sparse-method-support` / `f9b712a4b096015abbc4b046596f157443bc4531`; worktree has relevant uncommitted strict R-KV, benchmark, scorer, skill, and experiment-record changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=4`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`; `SPARSEVLLM_MASTER_PORT=2367`.
- Data: local MATH-500 file `/data2/haojitai/datasets/math500/test.jsonl`; `500` samples.
- Model: `/data2/haojitai/models/DeepSeek-R1-Distill-Llama-8B`; tokenizer same path; backend `sparsevllm`; method `rkv`.
- Prompt/decoding: sampled decoding with `temperature=0.6`, `top_p=0.95`, `top_k=0`, `max_new_tokens=32768`, `max_model_len=33792`, `batch_size=500`, `max_num_seqs_in_batch=64`, `max_decoding_seqs=64`, `decode_cuda_graph=true`.
- R-KV hparams: config file `scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_strict_hparams_20260625.json`; `sink_keep_tokens=8`, `recent_keep_tokens=128`, `decode_keep_tokens=1024`, `rkv_compression_interval=1024`, `rkv_alpha=0.1`, `rkv_redundancy_window=0`, `rkv_max_redundancy_tokens=4096`, `gpu_memory_utilization=0.72`, `engine_prefill_chunk_size=1024`.
- Command:

```bash
tmux new-session -d -s math500_rkv_strict_i1024_gpu4_20260625 \
  "cd /home/haojitai/projects/Sparse-vLLM-sparse-method-support && RUN_ROOT=/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625 GPU_ID=4 HPARAMS=scripts/tmp/math500_rkv_llama8b_cudagraph_bs500_maxnew32k_maxseq64_strict_hparams_20260625.json BATCH_SIZE=500 EXPECTED_ROWS=500 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=33792 MODEL_NAME=math500_rkv_strict_i1024_bs500_maxnew32k_cg SPARSEVLLM_MASTER_PORT=2367 bash scripts/tmp/run_math500_rkv_llama8b_cudagraph_bs500_gpu7_20260624.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Result artifacts: prediction directory `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_strict_i1024_bs500_maxnew32k_cg/None_0625_1359`; predictions `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_strict_i1024_bs500_maxnew32k_cg/None_0625_1359/math500.jsonl`; parsed outputs `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_strict_i1024_bs500_maxnew32k_cg/None_0625_1359/math500_parsed_outputs.jsonl`; per-sample results `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_strict_i1024_bs500_maxnew32k_cg/None_0625_1359/math500_per_sample_results.jsonl`; aggregate result `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_strict_i1024_bs500_maxnew32k_cg/None_0625_1359/result.json`; performance `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/benchmark/math_bench/pred/math500_rkv_strict_i1024_bs500_maxnew32k_cg/None_0625_1359/perf_rank0.json`.
- Logs/status: `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/run.log`; `/data2/haojitai/outputs/sparsevllm_sparse_method_support/math500_rkv_strict_i1024_gpu4_20260625/status.tsv`; tmux pane printed `EXIT_CODE=0`.
- Launch validation: foreground strict synthetic smoke passed on GPU4 with `rkv_redundancy_window=0` for `bs=1,2`; an earlier smoke with `rkv_max_redundancy_tokens=256` failed fast at warmup because full-candidate scoring saw `candidate_tokens=314`, confirming the strict path and explicit quadratic-work guard. Full run loaded the model, finished CUDA Graph warmup, and completed without `rkv_max_redundancy_tokens` errors.
- Validation: `math500.jsonl`, `math500_parsed_outputs.jsonl`, and `math500_per_sample_results.jsonl` each have `500` rows; status completed at `2026-06-25T14:28:42+08:00`; no missing extracted answers; all rows have status `success`; grep found no traceback, runtime error, CUDA OOM, illegal-memory, or error lines in `run.log`.
- Results: `math-verify==0.9.0` score `pass@1=68.8`, `344/500` correct, `missing_extracted=0`, status counts `success=500`.
- Performance: `generated_text_tokens=1,800,813`, `generation_elapsed_s=1717.811`, `generated_text_tokens_per_s=1048.32`; decode CUDA Graph active with graph count `16`, last key `DecodeCudaGraphKey(method='rkv', batch_size=1, context_capacity=32768, is_long_text=True, capture_sampling=False)`.
- Runtime notes: the run submitted all 500 prompts in one outer batch and let Sparse-VLLM queue internally up to `64` decode sequences. The middle high-concurrency windows were around `1100-2200 tok/s`, while the final long-answer tail dropped from `461` to `80 tok/s` as only a few long sequences remained.
- Interpretation: this run is stricter on the R-KV redundancy range than the prior `rkv_redundancy_window=64` run, but the score drops from `83.0` to `68.8`. The likely cause is not scorer/parser failure (`success=500`, `missing_extracted=0`) but an algorithm/runtime-semantics mismatch introduced by combining full candidate redundancy with the current online eviction schedule and `rkv_compression_interval=1024`. It is therefore not aligned with the paper's reported high-budget Llama-family MATH result yet, despite using the strict full-candidate redundancy range.

### 2026-06-25 20:23 Asia/Shanghai - scbench-prefix-cache-regression-qwen3-4b-gpu4

- Status: completed.
- Goal: validate the new block-level radix prefix cache on a small SCBench multi-turn subset and add the same workload as an explicit Sparse-VLLM regression layer covering `vanilla`, `omnikv`, and `quest`.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-radix-prefix-cache`.
- Code: `codex/radix-prefix-cache` / `b3e58593ef8e9e6ac1d861bd301b174138ddfb30`; worktree had relevant uncommitted SCBench runner, regression-suite, manifest, test, and experiment-record changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU `CUDA_VISIBLE_DEVICES=4`, NVIDIA H100 80GB HBM3; Python `/home/haojitai/miniconda3/envs/svllm/bin/python`; `PYTHONPATH=$PWD:$PWD/src`; `SPARSEVLLM_MASTER_PORT=25376`.
- Data: local SCBench snapshot `/data2/haojitai/hf_cache/hub/datasets--microsoft--SCBench/snapshots/283310bb8c5ba6909dd9a6b1be087d2937f76f6d`; tasks `scbench_kv` and `scbench_qa_eng`; `4` examples per task; `2` turns per example; multi-turn mode, not SCDQ.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; tokenizer same path; backend `sparsevllm`; methods `vanilla`, `omnikv`, `quest`; prefix cache enabled for all methods.
- Runtime knobs: `max_seq_length=4096`, `batch_size=4`, `max_num_seqs_in_batch=4`, `max_decoding_seqs=4`, `max_num_batched_tokens=16384`, `prefix_cache_block_size=16`, `gpu_memory_utilization=0.5`, `decode_cuda_graph=false`, `enforce_eager=true`. QuEST used `quest_chunk_size=16`, so the prefix-cache block size matched the QuEST chunk/block size.
- Command:

```bash
tmux new-session -d -s svllm_scbench_reg_gpu4 \
  "cd /home/haojitai/projects/Sparse-vLLM-radix-prefix-cache && RUN_TAG=sparsevllm_scbench_regression_20260625_gpu4 CUDA_VISIBLE_DEVICES=4 SPARSEVLLM_MASTER_PORT=25376 bash scripts/tmp/run_scbench_regression_gpu4.sh; echo EXIT_CODE=\$?; sleep 300"
```

- Result artifacts: regression root `/data2/haojitai/outputs/sparsevllm_scbench_regression_20260625_gpu4/regression/sparsevllm_regression/sparsevllm_scbench_regression_20260625_gpu4`; aggregate SCBench result `/data2/haojitai/outputs/sparsevllm_scbench_regression_20260625_gpu4/regression/sparsevllm_regression/sparsevllm_scbench_regression_20260625_gpu4/scbench.json`; child SCBench log `/data2/haojitai/outputs/sparsevllm_scbench_regression_20260625_gpu4/regression/sparsevllm_regression/sparsevllm_scbench_regression_20260625_gpu4/scbench/qwen3_4b/run.log`; launcher log `/data2/haojitai/outputs/sparsevllm_scbench_regression_20260625_gpu4/run.log`; status `/data2/haojitai/outputs/sparsevllm_scbench_regression_20260625_gpu4/status.tsv`.
- Validation: tmux pane printed `EXIT_CODE=0`; launcher status completed at `2026-06-25T20:26:58+08:00`; artifact validator found one SCBench record with methods `vanilla`, `omnikv`, and `quest`; every method/task pair had `request_count=8`, `status=success`, and prefix summary `status=success`.
- Results:

| Method | Task | Score by turn | Requests | Hit requests | Cached tokens | Cached blocks | Eligible hit rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vanilla` | `scbench_kv` | `50.000,25.000` | `8` | `4` | `15168` | `948` | `1.000` |
| `vanilla` | `scbench_qa_eng` | `25.000,0.000` | `8` | `4` | `16064` | `1004` | `1.000` |
| `omnikv` | `scbench_kv` | `50.000,25.000` | `8` | `4` | `15168` | `948` | `1.000` |
| `omnikv` | `scbench_qa_eng` | `25.000,0.000` | `8` | `4` | `16064` | `1004` | `1.000` |
| `quest` | `scbench_kv` | `50.000,25.000` | `8` | `4` | `15168` | `948` | `1.000` |
| `quest` | `scbench_qa_eng` | `25.000,0.000` | `8` | `4` | `16064` | `1004` | `1.000` |

- Notes: a one-example smoke on GPU4 first confirmed all three methods could generate and reuse prefix cache in SCBench multi-turn mode. The initial multi-method runner attempted to load all methods sequentially in one Python process; OmniKV then failed cache-slot initialization because method teardown did not fully reset CUDA/NCCL allocator state. The runner was changed so the parent command launches one child Python process per method, while preserving a single aggregate `scbench_methods_summary.json`.
