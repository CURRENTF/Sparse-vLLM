# Decode CUDA Graph Forward Path - 2026-05-16

## Scope

Branch: `perf/omnikv-decode-128k-bs4`

Base commit during this run: `d5aacd1`

Goal: turn the OmniKV-only greedy CUDA Graph prototype into a cleaner
decode-forward graph path, add a vanilla/full-attention graph baseline, keep
sampling outside the graph by default so non-greedy sampling can run, and
measure fair 128k bs4 results on local GPU 1.

Retained code changes:

- Added `src/sparsevllm/engine/decode_cuda_graph.py` with a
  `DecodeCudaGraphRunner` that owns fixed-shape graph buckets keyed by method,
  batch size, max context length, and whether sampling is captured.
- Reduced `ModelRunner` to graph-runner orchestration instead of storing
  OmniKV-specific graph state inline.
- Added generic config flags:
  - `decode_cuda_graph`
  - `decode_cuda_graph_capture_sampling`
  - kept `omnikv_decode_cuda_graph` as an OmniKV-only alias to
    `decode_cuda_graph=True`
- Default graph behavior captures `run_model()` forward only and returns the
  graph-resident logits tensor; the existing sampler runs outside the graph.
- Optional `decode_cuda_graph_capture_sampling=True` captures greedy `argmax`
  in the graph for comparison; it fails fast for non-greedy decoding.
- Added `SamplingParams.top_p` and sampler top-p filtering. Greedy decode still
  bypasses the top-p path.
- Added `scripts/debug/compare_decode_graph_eager_logits.py` to compare
  Sparse-VLLM eager decode logits with graph decode logits.

## Environment

- Host: local `guest-KR6288-X2-A0-R0-00`
- Working dir: `<PROJECT_ROOT>`
- GPU: `CUDA_VISIBLE_DEVICES=1`, NVIDIA H100 80GB HBM3
- Conda env: `svllm`
- Base model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
- Compressor path for HF logits check:
  `<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor`
- Output root:
  `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124`

## Validation

CPU-side checks:

```bash
conda run -n svllm python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/decode_cuda_graph.py \
  src/sparsevllm/engine/model_runner.py \
  src/sparsevllm/engine/llm_engine.py \
  src/sparsevllm/engine/sequence.py \
  src/sparsevllm/layers/sampler.py \
  src/sparsevllm/sampling_params.py \
  tests/test_sampler.py \
  tests/test_prefill_schedule_policy.py

conda run -n svllm python -m unittest discover -s tests -p 'test*.py'
```

Result: `Ran 75 tests`, `OK`.

## Correctness Checks

Graph-vs-eager decode logits, small smoke:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_decode_graph_eager_logits.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --method <vanilla|omnikv> \
  --prompt_len 2048 \
  --batch_size 2 \
  --max_tokens 3 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":1024,"tensor_parallel_size":1,"decode_keep_tokens":256,"prefill_keep_tokens":256,"sink_keep_tokens":8,"recent_keep_tokens":64,"full_attention_layers":"0,1,2,4,7,14","chunk_prefill_accel_omnikv":true,"mlp_chunk_size":4096,"throughput_log_interval_s":0.0}' \
  --output <output-json>
```

Results:

| Method | Result file | max abs diff | mean abs diff | Argmax | Top-k |
| --- | --- | ---: | ---: | --- | --- |
| vanilla | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/vanilla_graph_vs_eager_logits.json` | 0.0 | 0.0 | match | top1/5/10/50 all 1.0 |
| omnikv | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/omnikv_graph_vs_eager_logits.json` | 0.0 | 0.0 | match | top1/5/10/50 all 1.0 |

HF-vs-Sparse-VLLM eager OmniKV logits:

- Output:
  `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/hf_vs_sparse_eager_logits/long_omnikv.json`
- Decode result: `max_abs_diff=0.28125`,
  `mean_abs_diff=0.030360523611307144`, `argmax_match=true`
- Decode top-k overlap: top1 `1.0`, top5 `1.0`, top10 `0.9`, top50 `1.0`

Top-p graph smoke:

- Command used `decode_cuda_graph=true`, `temperature=0.7`, `top_p=0.9`,
  `length=2048`, `bs=2`, `output_len=4`.
- Log:
  `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/top_p_graph_smoke.log`
- Result: vanilla and OmniKV completed successfully with sampling outside the
  graph.

Implementation note: the first version of the graph-vs-eager logits script
loaded eager and graph engines in the same process, which left too much 7B model
memory live before constructing the second engine and caused CUDA OOM. The
script now runs eager and graph collection in isolated spawned worker
processes.

## 128k BS4 Benchmark

Shared command shape:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods vanilla,omnikv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 4096 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"tensor_parallel_size":1,"max_num_seqs_in_batch":4,"max_decoding_seqs":4,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,4,7,14","chunk_prefill_accel_omnikv":true,"mlp_chunk_size":16384,"throughput_log_interval_s":0.0,<graph flags>}' \
  > <log> 2>&1
```

Resolved OmniKV parameters:

- `prefill_schedule_policy=all_chunked`
- `num_top_tokens=4096`
- `num_top_tokens_in_prefill=4096`
- `num_sink_tokens=8`
- `num_recent_tokens=128`
- `full_attn_layers=[0,1,2,4,7,14]`
- `chunk_prefill_accel_omnikv=True`

Results:

| Run | Method | Decode graph | Sampling captured | Decode tok/s | ITL ms | TTFT s | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| eager | vanilla | no | no | 150.98 | 26.49 | 57.49 | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/eager_128k_bs4_out4096.log` |
| eager | omnikv | no | no | 139.83 | 28.61 | 28.73 | same |
| forward graph | vanilla | yes | no | 219.11 | 18.26 | 57.43 | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/graph_forward_128k_bs4_out4096.log` |
| forward graph | omnikv | yes | no | 399.24 | 10.02 | 28.84 | same |
| graph+argmax | vanilla | yes | yes | 218.51 | 18.31 | 58.03 | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_forward_20260516_141124/graph_argmax_128k_bs4_out4096.log` |
| graph+argmax | omnikv | yes | yes | 397.80 | 10.06 | 28.99 | same |

Interpretation:

- Comparing OmniKV graph against vanilla eager gives `399.24 / 150.98 = 2.64x`,
  but that is not a fair graph-vs-graph baseline.
- The fair forward-graph comparison is `399.24 / 219.11 = 1.82x`.
- Moving greedy `argmax` outside the graph did not materially hurt throughput:
  OmniKV `399.24 tok/s` forward-only vs `397.80 tok/s` graph+argmax.
- This supports using forward-only graph as the default because it keeps
  non-greedy sampling compatible while preserving the useful graph speedup.

## VLLM-Style Batch Padding Update

Date/time: 2026-05-16 16:46 Asia/Shanghai

Status: completed.

Goal: replace exact-batch decode CUDA Graph buckets with vLLM-style capture
sizes, where runtime decode batch size is padded to the smallest captured graph
batch size that can hold it.

Code: branch `perf/omnikv-decode-128k-bs4`, base commit `d5aacd1`, with
relevant uncommitted changes in this working tree.

Implementation:

- Added `decode_cuda_graph_capture_sizes`; `auto` expands to powers of two up
  to the next power of two that covers `max_decoding_seqs`.
  Example: `max_decoding_seqs=6` resolves to `[1,2,4,8]`.
- `DecodeCudaGraphRunner` now selects `graph_batch_size >= real_batch_size`,
  keeps long/short decode graphs separate, and returns logits/token ids sliced
  back to the real batch size.
- `StandardCacheManager.prepare_decode_static()` only allocates KV slots for
  real requests. Padded rows mirror the first real request for read-only
  attention and use `slot_mapping=-1`, so they do not write KV or consume
  persistent cache slots.
- OmniKV CUDA Graph attn-score tensors use the graph batch size under static
  decode so padded rows cannot write outside the captured tensor shape.
- The debug logits script records the runner's last graph state, so it can
  compare real logits even when `real_bs != graph_bs`.

Validation:

```bash
conda run -n svllm python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/decode_cuda_graph.py \
  src/sparsevllm/engine/cache_manager/standard.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/engine/model_runner.py \
  scripts/debug/compare_decode_graph_eager_logits.py \
  tests/test_prefill_schedule_policy.py

conda run -n svllm python -m unittest tests.test_prefill_schedule_policy tests.test_sampler
conda run -n svllm python -m compileall -q src tests scripts/debug/compare_decode_graph_eager_logits.py scripts/benchmarks/bench_sparse_vllm.py
conda run -n svllm python -m unittest discover -s tests -p 'test*.py'
git diff --check
```

Results: targeted unit tests ran `23 tests OK`; full discovery ran `77 tests
OK`; `compileall` and `git diff --check` passed.

GPU smoke environment:

- Host: local `guest-KR6288-X2-A0-R0-00`
- GPU: `CUDA_VISIBLE_DEVICES=5`, NVIDIA H100 80GB HBM3
- Conda env: `svllm`
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
- Output root:
  `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_vllm_padding_20260516_1646`

GPU smoke commands:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_decode_graph_eager_logits.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --method vanilla \
  --prompt_len 1024 \
  --batch_size 6 \
  --max_tokens 3 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"tensor_parallel_size":1,"throughput_log_interval_s":0.0}' \
  --output <OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_vllm_padding_20260516_1646/vanilla_bs6_graph8_logits.json

CUDA_VISIBLE_DEVICES=5 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_decode_graph_eager_logits.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --method omnikv \
  --prompt_len 1024 \
  --batch_size 6 \
  --max_tokens 3 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"tensor_parallel_size":1,"decode_keep_tokens":256,"prefill_keep_tokens":256,"sink_keep_tokens":8,"recent_keep_tokens":64,"full_attention_layers":"0,1,2,4,7,14","chunk_prefill_accel_omnikv":true,"mlp_chunk_size":4096,"throughput_log_interval_s":0.0}' \
  --output <OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_vllm_padding_20260516_1646/omnikv_bs6_graph8_logits.json
```

GPU smoke results:

| Method | real bs | auto capture sizes | graph bs | max abs diff | mean abs diff | argmax | result |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| vanilla | 6 | `[1,2,4,8]` | 8 | 0.0 | 0.0 | match | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_vllm_padding_20260516_1646/vanilla_bs6_graph8_logits.json` |
| omnikv | 6 | `[1,2,4,8]` | 8 | 0.0 | 0.0 | match | `<OUTPUT_ROOT>/Sparse-vLLM/decode_cuda_graph_vllm_padding_20260516_1646/omnikv_bs6_graph8_logits.json` |

Notes: GPU 1 and GPU 6 were busy during this small smoke, so the run used GPU
5. This was a correctness smoke only, not a throughput benchmark.

## Remote 128k BS1-6 Benchmark

Date/time: 2026-05-16 17:28 Asia/Shanghai

Status: completed.

Goal: re-measure full-attention and OmniKV throughput on the remote Blackwell
GPU after adding vLLM-style CUDA Graph batch padding, for `bs=1..6` at 128k
context.

Remote environment:

- Host: `autodl-container-nqmeqbvtjn-072cfeb1`
- SSH target: `root@connect.westb.seetacloud.com:51823`
- Working dir: `<PROJECT_ROOT>`
- Output dir:
  `<OUTPUT_ROOT>/Sparse-vLLM/remote_128k_bs1_6_graph_20260516_172815`
- Local watcher dir:
  `<OUTPUT_ROOT>/Sparse-vLLM/remote_watch_128k_bs1_6_20260516_172815_fixedprobe`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB,
  driver `595.58.03`
- Conda env: `kv`, Python `3.12.3`
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
- Code: branch `perf/omnikv-decode-128k-bs4`, base commit `d5aacd1`;
  relevant working-tree changes were synced to the remote before running.

Pre-run validation:

- Remote `py_compile` passed for `src/sparsevllm/models/qwen2.py` and
  `src/sparsevllm/models/qwen3.py`.
- A short smoke run completed for `vanilla,omnikv`, `length=4096`, `bs=6`,
  `output_len=8`, with `decode_cuda_graph=true`. It verified the remote Qwen
  config path, graph padding to capture size 8, and sampling outside the graph.
- Local validation after the Qwen RoPE config compatibility patch:
  `py_compile` for Qwen2/Qwen3 and targeted unit tests
  `tests.test_prefill_schedule_policy tests.test_sampler` passed
  (`Ran 23 tests OK`).

Failure notes before the completed run:

- `remote_128k_bs1_6_graph_20260516_171448` failed before model execution
  because shell quoting corrupted the inline `--hyper_params` JSON.
- `remote_128k_bs1_6_graph_20260516_171720` and
  `remote_128k_bs1_6_graph_20260516_172008` exposed Transformers 5.3 Qwen
  config drift: `rope_theta` moved under `rope_parameters`, and default
  `rope_parameters` could appear through `rope_scaling` as a dict. The retained
  code change treats default/no-scaling RoPE dicts as `None` and still fails
  fast for unsupported non-default scaling.

Command:

```bash
cd <PROJECT_ROOT>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src <CONDA_BIN> run -n kv --no-capture-output \
  python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods vanilla,omnikv \
  --lengths 128000 \
  --batch_sizes 1,2,3,4,5,6 \
  --output_len 512 \
  --temperature 0 \
  --top_p 1 \
  --hyper_params @<OUTPUT_ROOT>/Sparse-vLLM/remote_128k_bs1_6_graph_20260516_172815/hyper_params.json
```

Hyperparameters:

```json
{
  "gpu_memory_utilization": 0.9,
  "engine_prefill_chunk_size": 4096,
  "tensor_parallel_size": 1,
  "decode_keep_tokens": 4096,
  "prefill_keep_tokens": 4096,
  "sink_keep_tokens": 8,
  "recent_keep_tokens": 128,
  "full_attention_layers": "0,1,2,4,7,14",
  "chunk_prefill_accel_omnikv": true,
  "mlp_chunk_size": 16384,
  "throughput_log_interval_s": 0.0,
  "decode_cuda_graph": true,
  "decode_cuda_graph_capture_sampling": false
}
```

CUDA Graph behavior:

- Forward graph enabled for both vanilla and OmniKV:
  `decode_cuda_graph=True`.
- Sampling remained outside the graph:
  `decode_cuda_graph_capture_sampling=False`.
- Auto capture sizes resolved by batch: bs1 -> `[1]`, bs2 -> `[1,2]`,
  bs3/4 -> `[1,2,4]`, bs5/6 -> `[1,2,4,8]`. Thus bs6 replay used a
  graph batch size of 8 with padded rows sliced away before sampling/results.

Results:

| Method | Context | BS | TTFT s | Prefill tok/s | Decode tok/s | ITL ms | Avg BS | Mem GB | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 128000 | 1 | 17.72 | 7222.2 | 60.5 | 16.52 | 1.0 | 85.20 | 1.00x |
| vanilla | 128000 | 2 | 34.42 | 7438.1 | 88.8 | 22.51 | 2.0 | 85.64 | 1.00x |
| vanilla | 128000 | 3 | 51.68 | 7429.9 | 107.6 | 27.89 | 3.0 | 86.08 | 1.00x |
| vanilla | 128000 | 4 | 68.70 | 7453.4 | 123.1 | 32.51 | 4.0 | 86.51 | 1.00x |
| vanilla | 128000 | 5 | 85.52 | 7484.0 | 129.4 | 38.64 | 5.0 | 86.78 | 1.00x |
| vanilla | 128000 | 6 | 102.77 | 7473.4 | 136.7 | 43.89 | 6.0 | 86.92 | 1.00x |
| omnikv | 128000 | 1 | 9.66 | 13249.6 | 78.4 | 12.76 | 1.0 | 85.26 | 1.30x |
| omnikv | 128000 | 2 | 18.41 | 13905.3 | 139.6 | 14.32 | 2.0 | 85.75 | 1.57x |
| omnikv | 128000 | 3 | 27.74 | 13845.9 | 188.6 | 15.91 | 3.0 | 86.25 | 1.75x |
| omnikv | 128000 | 4 | 36.63 | 13977.0 | 235.4 | 16.99 | 4.0 | 86.73 | 1.91x |
| omnikv | 128000 | 5 | 45.64 | 14023.8 | 264.4 | 18.91 | 5.0 | 87.06 | 2.04x |
| omnikv | 128000 | 6 | 54.72 | 14036.3 | 298.9 | 20.07 | 6.0 | 87.25 | 2.19x |

Interpretation:

- The remote Blackwell baseline is different from the local H100 baseline, as
  expected. Using the same run and same GPU, OmniKV reaches `2.19x` at bs6,
  but does not reach the `2.5x` target.
- Decode CUDA Graph is active for both the full-attention baseline and OmniKV,
  so these speedups are graph-vs-graph, not OmniKV graph compared against
  vanilla eager.
- OmniKV prefill is consistently faster than vanilla under this config because
  `chunk_prefill_accel_omnikv=true`, but the requested target was decode
  throughput; the highest measured decode speedup is the bs6 `2.19x`.
