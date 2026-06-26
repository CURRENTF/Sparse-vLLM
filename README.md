<div align="center">
  <img src="docs/assets/logo.png" alt="Sparse-vLLM" style="width:42%; height:auto;">

  <p>
    <a href="https://deepwiki.com/CURRENTF/Sparse-vLLM"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <a href="https://arxiv.org/abs/2602.08005"><img src="https://img.shields.io/badge/arXiv-2602.08005-b31b1b.svg" alt="arXiv"></a>
    <a href="https://arxiv.org/pdf/2602.08005.pdf"><img src="https://img.shields.io/badge/PDF-download-brightgreen.svg" alt="PDF"></a>
  </p>
</div>

A sparse-first inference engine for long-context LLM serving, which also includes DeltaKV compressor training and evaluation tooling.

<div align="center">
  <img src="docs/assets/sparse_vllm_throughput.png" alt="Sparse-vLLM throughput" style="width:86%; height:auto;">
</div>

## Project Overview

Sparse-vLLM is an inference framework built with sparsity as the first design principle. Instead of layering sparse methods on top of a conventional KV cache, it rethinks cache layout, controller flow, and kernels so that multiple sparse mechanisms can plug in cleanly.

DeltaKV-related compressor training, HF wrapper comparisons, and benchmark
adapters live under `src/deltakv/` and `benchmark/`.

## Key Runtime Principles

- Public commands and `LLM(...)` kwargs should use `sparse_method`; Sparse-vLLM
  normalizes it internally to `vllm_sparse_method`.
- Sparse method runtime state belongs in
  `src/sparsevllm/engine/cache_manager/`; `attention.py` should stay generic.
- Prefill scheduling is method-specific and registry-owned. The source of
  truth is `src/sparsevllm/method_registry.py`, not benchmark scripts.
- Sparse-vLLM currently uses two prefill policies: `all_chunked` and the
  special `long_bs1full_short_batch` policy.
- `long_bs1full_short_batch` is only for methods that are registered to need a
  complete long-prefill pass before their sparse/cache transformation. Long
  requests run as full prefill with batch size 1; short requests still use
  chunked batching.
- Benchmark reports should record the sparse method, prefill policy, prefill
  chunk size, prompt length, batch size, and any DeltaKV checkpoint.

## Core Sparse Methods

Sparse-vLLM supports physical eviction, logical masking, query-aware selection,
and hybrid KV compression. The main method families are `streamingllm`,
`snapkv`, `pyramidkv`, `omnikv`, `quest`, and `deltakv`.

| Method | Type | Short Description |
| --- | --- | --- |
| `vanilla` | Dense baseline | Runs full attention and keeps the standard KV cache behavior for correctness and performance baselines. |
| `streamingllm` / `attention-sink` | Physical eviction | Keeps fixed sink tokens plus a recent window, then physically evicts older tokens outside that policy. |
| `snapkv`, `pyramidkv` | Physical eviction | Selects important historical tokens during prefill/finalization and stores only the retained KV tokens. |
| `omnikv` | Logical masking | Keeps tokens in storage but masks the attention read view so sparse layers attend only selected context. |
| `quest` | Query-aware selection | Uses decode-time query-aware page selection while keeping prefill dense. |
| `deltakv` / `deltakv-*` | Hybrid compression | Keeps a small full-precision pool and stores older context through DeltaKV compression or related ablations. |

Read the method overview and integration rules in
[Core Sparse Methods](docs/features/sparse-methods.md).

## Documentation

| Topic | Link |
| --- | --- |
| Quick setup and minimal usage | [Getting Started](docs/getting_started/README.md) |
| Sparse method taxonomy and extension rules | [Core Sparse Methods](docs/features/README.md) |
| Runtime architecture | [Architecture](docs/design/README.md) |
| Runtime parameter semantics | [Runtime Parameter Semantics](docs/configuration/runtime-parameter-semantics.md) |
| Benchmark commands | [Benchmarks](docs/benchmarking/README.md) |
| DeltaKV inference and training | [DeltaKV](docs/features/deltakv.md) |
| Reproducibility checklist | [Reproducibility](docs/getting_started/reproducibility.md) |

The full documentation index is maintained in [docs/README.md](docs/README.md).

## Quick Start

Install the package from the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0
uv pip install \
  transformers==5.12.1 pillow accelerate deepspeed datasets matplotlib \
  seaborn wandb loguru ansible bitsandbytes datatrove tqdm fire fastapi \
  "uvicorn[standard]" pydantic
uv pip install -e . --no-deps

# Optional CUDA kernels. Install only after the base environment works.
# MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
# uv pip install "flash-linear-attention[cuda,conv1d]==0.5.1"
```

For the full dependency list and a minimal `LLM(...)` example, see
[Getting Started](docs/getting_started/README.md).

## Benchmarks

Use `benchmark/microbench.py` for synthetic throughput measurements and the
other `benchmark/` entrypoints for LongBench, MathBench, SCBench, NIAH, and
multimodal evaluations.

See [Benchmarks](docs/benchmarking/README.md) for command examples and backend notes.

## Contributing Sparse Methods

New sparse methods should keep method-specific runtime state in
`src/sparsevllm/engine/cache_manager/` and keep
`src/sparsevllm/layers/attention.py` generic.

For Codex-assisted method work, use the repo-local
[`$add-sparse-method`](.agents/skills/add-sparse-method/SKILL.md) skill.

**Common knobs**

- `tensor_parallel_size`: number of GPU ranks (processes) to spawn.
- `gpu_memory_utilization`: fraction of total GPU memory to allocate for the KV cache.
- `max_model_len`: max (prompt + generated) tokens allowed.
- `engine_prefill_chunk_size`: Sparse-vLLM prefill scheduling and memory-admission chunk size.
- `max_num_batched_tokens`, `max_num_seqs_in_batch`, `max_decoding_seqs`: scheduler throughput/latency constraints.

**Sparse knobs (method-dependent)**

- `sparse_method`: method selector.
- `deltakv_checkpoint_path`: DeltaKV compressor checkpoint directory or file.
- `sink_keep_tokens`: always-kept prefix/sink tokens.
- `recent_keep_tokens`: always-kept recent tail tokens.
- `decode_keep_tokens`: decode-time top/important token budget.
- `full_attention_layers`: comma-separated layer indices (or list) that run full attention.

Sparse-vLLM requires explicit integer keep budgets. Ratio-style values such as
`decode_keep_tokens=0.17` are accepted on HF paths that support ratios, but must
be converted to token counts before running Sparse-vLLM.

### Supported methods

Set `sparse_method` to one of:

- `"vanilla"` (full attention)
- `"streamingllm"` / `"attention-sink"` (fixed sink + recent-window physical eviction)
- `"snapkv"`, `"pyramidkv"` (physical eviction)
- `"omnikv"` (logical masking)
- `"quest"` (query-aware page selection on decode; prefill stays full attention)
- `"rkv"` / `"r-kv"` (decode-time R-KV physical eviction using attention importance and key-vector redundancy)
- `"skipkv"` (SkipKV sentence-aware KV storage skipping; supported only for the official DeepSeek-R1-Distill Llama-8B, Qwen-7B, and Qwen-14B model basenames with released steering vectors)
- `"deltakv"` / `"deltakv-*"` (hybrid compression; optional / experimental, see [DeltaKV](#deltakv))

Sparse-vLLM internally stores this as `vllm_sparse_method`, but public commands
and `LLM(...)` kwargs should use `sparse_method`.

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens (default `16`)
- `quest_token_budget`: decode-time token budget before page rounding (default `1024`)
- `quest_skip_layers`: keep the first N layers dense during decode (default `2`)

`rkv` runtime knobs:

- `rkv_compression_interval`: generated-token buffer size before decode eviction (default `128`; use larger values such as `1024` for fewer, larger paper-style decode evictions)
- `rkv_observation_tokens`: query-cache observation window used to score candidate KV tokens (default `8`; must be `<= 128` and `<= rkv_compression_interval`)
- `rkv_alpha`: paper-style joint score lambda, `alpha * importance - (1 - alpha) * redundancy` (default `0.1`)
- `rkv_similarity_threshold`: cosine threshold used to count key-token redundancy (default `0.8`)
- `rkv_recent_similar_keep`: preserve links to the most recent similar tokens when estimating redundancy (default `1`)
- `rkv_max_redundancy_tokens`: explicit safety bound for quadratic redundancy scoring (default `4096`)
- `rkv_redundancy_window`: candidate window scored for redundancy on each eviction; `0` scores the full candidate set and is the paper-aligned default, positive values opt into a trailing-window speed approximation

`skipkv` runtime knobs:

- `skipkv_compression_interval`: generated-token buffer size before decode eviction (default `128`)
- `skipkv_alpha`: token-level redundancy penalty weight (default `0.1`)
- `skipkv_similarity_threshold`: segment cosine threshold for sentence-inspired eviction (default `0.95`)
- `skipkv_segment_size`: token count per segment when sentence spans are unavailable in the engine (default `32`)
- `skipkv_max_redundancy_tokens`: explicit safety bound for scoring candidates (default `4096`)
- `skipkv_redundancy_window`: trailing candidate window scored for token/segment redundancy on each eviction (default `64`)
- `skipkv_enable_sentence_scoring`: use tokenizer-delimited sentence spans and hidden-state sentence embeddings when available (default `true`)
- `skipkv_sentence_score_weight`: weight for sentence-level redundancy penalty (default `1.0`)
- `skipkv_sentence_min_tokens` / `skipkv_sentence_max_tokens`: sentence tracking bounds (defaults `4` / `256`)
- `skipkv_sentence_embedding_layer`: transformer layer used for sentence embeddings, negative values count from the end (default `-1`)
- `skipkv_enable_activation_steering`: enable SkipKV activation steering only when `skipkv_steering_vector_path` is set (default `false`); enabling steering without a vector path raises `ValueError`.
- `skipkv_steering_vector_path`: `.pt` file containing an official released SkipKV steering vector. The repo does not support locally generated heuristic vectors as paper-equivalent vectors.
- `skipkv_steering_layer`, `skipkv_steering_alpha`, `skipkv_steering_alpha_increment`, `skipkv_steering_alpha_max`: steering injection layer and signed adaptive strength controls. Official SkipKV-style Qwen/Llama runs use layer `20` with negative alpha, e.g. `-1.25` and increment `-0.02`.
- SkipKV sentence tracking and steering use the paper/official newline-oriented delimiter set (`\n`, `.\n`, `)\n`, `\n\n`, `.\n\n`, `)\n\n`, `?\n\n`), not standalone punctuation such as `.`.
- SkipKV updates the dynamic steering strength only after completed sentences that contain non-execution markers such as `Alternatively`, `Wait`, or `again`; sentence-similarity redundancy remains an eviction signal.
- SkipKV compatibility is checked with `model_path.rstrip("/").split("/")[-1]`. Supported basenames are exactly `DeepSeek-R1-Distill-Llama-8B`, `DeepSeek-R1-Distill-Qwen-7B`, and `DeepSeek-R1-Distill-Qwen-14B`; other basenames raise `ValueError`.

## How to test

For the fixed SparseVLLM regression harness (`quality`, `logits`, `perf`,
`stress`, and ABCD gate reports), see
[`docs/sparsevllm-regression-tests.md`](docs/sparsevllm-regression-tests.md).

### Throughput benchmark

Use `scripts/benchmarks/bench_sparse_vllm.py` to measure TTFT, prefill throughput, decode throughput, and GPU memory.

Notes:

- Prefer `--hyper_params` to pass Sparse-vLLM settings as a JSON object.
- `--hyper_params` accepts canonical runtime names such as `sparse_method`, `engine_prefill_chunk_size`, and `decode_keep_tokens`; legacy runtime names are rejected.
- `--lengths` measures *prompt length*; the script sets `max_model_len = length + output_len + 100` internally.

Baseline (vanilla):

```bash
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization": 0.9}'
```

#### MathBench with `sparsevllm` backend

These examples are convenient for quick GSM8K / AIME-style comparisons while exercising the Sparse-vLLM engine directly. For dataset details, see `benchmark/math_bench/README.md`.

Full-attention baseline:

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-full \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "vanilla"}'
```

OmniKV:

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024}'
```

DeltaKV:

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 512, "max_num_batched_tokens": 8192, "max_num_seqs_in_batch": 30, "sparse_method": "deltakv", "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024, "deltakv_checkpoint_path": "/root/autodl-fs/checkpoints/compressor/<COMPRESSOR_DIR>", "deltakv_latent_dim": 256}'
```

When `--backend sparsevllm`, method selection happens through `sparse_method`
and checkpoints through `deltakv_checkpoint_path`.

#### LongBench with `sparsevllm` backend

Use this path when you want LongBench results from the actual Sparse-vLLM engine rather than the HF wrapper models.

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend sparsevllm \
  --task qasper,hotpotqa,multi_news \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "decode_keep_tokens": 2048, "full_attention_layers": "0,1,2,4,7,14", "recent_keep_tokens": 128, "sink_keep_tokens": 8}'
```

For a full LongBench run, omit `--task`. To switch to DeltaKV, keep
`--backend sparsevllm` and set `sparse_method="deltakv"` plus
`deltakv_checkpoint_path=...`.
For the no-checkpoint direct residual ablation, set
`sparse_method="deltakv-less-memory"` and omit `deltakv_checkpoint_path`.

#### LongBench with HF wrappers

Use the HF backend when you want to compare against the DeltaKV / SnapKV / PyramidKV wrapper models implemented under `src/deltakv/`.

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "/root/autodl-fs/checkpoints/compressor/<COMPRESSOR_DIR>" \
  --hyper_param '{"hf_prefill_chunk_size": 2048000, "prefill_keep_tokens": 4096, "chunk_prefill_accel_omnikv": true, "decode_keep_tokens": 0.11, "full_attention_layers": "0,1,2,4,7,14", "recent_keep_tokens": 128, "sink_keep_tokens": 8, "use_compression": true, "use_cluster": true, "deltakv_center_ratio": 0.1}'
```

To compare other baselines, keep `--backend hf` and switch `--sparse_method` /
`--hyper_param`, e.g. `omnikv` with
`{"hf_prefill_chunk_size":4096,"prefill_keep_tokens":4096,"decode_keep_tokens":2048,"full_attention_layers":"0,1,2,4,7,14","recent_keep_tokens":128,"sink_keep_tokens":8}`,
`snapkv` with `{"decode_keep_tokens":0.2,"pool_kernel_size":7}`, or `kvzip`
with `{"ratio":0.3,"level":"pair","kv_type":"evict","prefill_chunk_size":16000}`.

For `kvzip`, the vendored baseline lives in `baselines/kvzip/`. Build its CUDA extension first:

```bash
cd baselines/kvzip/csrc
make
```

## DeltaKV

DeltaKV is a method for **compressing the KV cache** to enable more efficient long-context inference for Transformer LLMs.
This repo includes DeltaKV compressor training code and some inference/benchmark integrations, but DeltaKV-specific
speed/quality/perf trade-offs are still under active iteration.

### DeltaKV inference

Use one public Sparse-vLLM method:

- `"deltakv"`

The older `"deltakv-less-memory"` and `"deltakv-less-memory-cudagraph"` method
names are accepted as compatibility aliases and are normalized to `"deltakv"`.
`"deltakv-less-memory-cudagraph"` also enables `decode_cuda_graph=true` for old
configs. New configs should keep the method name stable and control graph mode
with `decode_cuda_graph`.

Compressor-backed DeltaKV inference requires
`deltakv_checkpoint_path="/path/to/trained_compressor_dir_or_file"`. The maintained
Sparse-vLLM runtime stores sparse layers as learned-compressor latent residuals:

```text
latent_delta = compress_down(KV_before_rope) - compress_down(mean(selected_center_KV_before_rope))
```

The supported storage profiles are intentionally small:

- BF16/FP16 full layers + BF16/FP16 compressor sparse layers: `full_layer_kv_quant_bits=0`, `deltakv_latent_quant_bits=0`.
- KIVI int4 full layers + compressor int4 sparse layers: `full_layer_kv_quant_bits=4`, `deltakv_latent_quant_bits=4`, usually with group size 32.

DeltaKV knobs you may need:

- `deltakv_checkpoint_path`: path to trained compressor weights (directory containing `*.safetensors`/`*.pt`/`*.bin`, or a single file).
- `deltakv_latent_dim`: latent dimension of compressed KV.
- `deltakv_center_ratio`, `cluster_metric`: reference selection / clustering behavior.
- `deltakv_neighbor_count`: number of selected center/reference tokens used for reconstruction.
- `deltakv_latent_quant_bits`: `0` keeps sparse latent residuals in model dtype; `4` stores them as grouped int4.
- `decode_cuda_graph`: enables decode CUDA Graph capture/replay.
- `decode_cuda_graph_context_sizes`: context buckets, defaulting to `1024,2048,4096,8192,...` up to `max_model_len`.
- `decode_cuda_graph_context_policy`: `current` by default, so a short request uses its current-length bucket instead of a previously captured long-context graph; `requested` reproduces the older final-length capacity behavior for diagnostics.

CUDA Graph support is shared across Sparse-vLLM methods through the same decode
runner. The graph key is method × batch bucket × context bucket × long/short
batch state × sampling-capture mode, so `snapkv`, `pyramidkv`, `streamingllm`,
`quest`, `omnikv`, vanilla, and `deltakv` all use the same bucketed capture
policy.
With prefix cache enabled, decode CUDA Graph is supported for `vanilla`,
`omnikv`, and `quest` with `decode_cuda_graph_capture_sampling=false`.
When `tensor_parallel_size>1`, each rank keeps a rank-local mirrored prefix
cache with stable logical block ids and rank-local KV payloads.

For the HF backend, `deltakv_cache_impl="delta_compressed_quant_kivi_full_fp8_ref"`
keeps DeltaKV center selection, but stores sparse-layer learned-compressor latent
deltas as int4 token-level groups:

```text
latent_delta = compress_down(KV_before_rope) - compress_down(mean(selected_center_KV_before_rope))
```

It currently uses `deltakv_latent_quant_bits=4` and a trained compressor
checkpoint. Full-attention layers are KIVI-4bit round-tripped in place with
`group_size=32` and `residual_length=32`; sparse-layer reference tokens are
fp8 round-tripped in place before reconstruction. These two paths measure the
numerical loss while preserving the existing BF16/FP16 attention kernels, so
they are not yet physical int4/fp8 storage savings for those tokens.

Quick throughput smoke:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"deltakv_checkpoint_path":"/path/to/compressor","decode_cuda_graph":true,"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"deltakv_latent_quant_group_size":32,"deltakv_full_pool_reserve_ratio":0.2}'
```

### Train a compressor

The main entrypoint is:

- Python: `python src/deltakv/train_compressor.py ...`
- CLI script (after installation): `deltakv-train ...`

The training script expects a **tokenized + packed** dataset saved by Hugging Face `datasets` (`load_from_disk`).

```bash
python src/deltakv/train_compressor.py \
  --model_name_or_path <PATH_TO_BASE_MODEL> \
  --dataset_path <PATH_TO_DATASET_ON_DISK> \
  --output_dir <PATH_TO_OUTPUT_CHECKPOINT_DIR> \
  --deltakv_latent_dim 512 \
  --compressor_token_group_size 1 \
  --deltakv_neighbor_count 4 \
  --layer_chunk_size 1 \
  --batch_size 1 \
  --warmup_ratio 0.02 \
  --max_steps 20000 \
  --learning_rate 2e-4 \
  --use_nonlinear_compressor True \
  --ref_mode avg \
  --collect_kv_before_rope True \
  --model_type cluster_e2e \
  --cluster_soft_assignment False \
  --compressor_down_type mlp_swiglu \
  --compressor_down_intermediate_size 3072 \
  --compressor_up_type linear \
  --compressor_linear_bias False
```

Common knobs:

- `--deltakv_latent_dim`: compressed KV latent width (smaller = more compression)
- `--compressor_token_group_size`: token grouping for non-cluster compressor references.
- `--deltakv_neighbor_count`: number of selected ref/center tokens for cluster DeltaKV.
- `--model_type`: `e2e`, `cluster_e2e`, `cluster_e2e_big` (see `src/deltakv/train_compressor.py`)
- `--collect_kv_before_rope`: whether to collect KV before RoPE (model-dependent)

### Evaluate on LongBench

`benchmark/long_bench/pred.py` runs LongBench prediction and writes JSONL outputs under a local output directory.

```bash
python benchmark/long_bench/pred.py \
  --model all \
  --model_path <PATH_TO_BASE_MODEL> \
  --tokenizer_path <PATH_TO_TOKENIZER_OR_MODEL> \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "<PATH_TO_TRAINED_COMPRESSOR_DIR>" \
  --hyper_param '{"hf_prefill_chunk_size": 2048000, "prefill_keep_tokens": 4096,
  "chunk_prefill_accel_omnikv": true, "decode_keep_tokens": 0.17, "full_attention_layers": "0,1,2,8,18",
  "recent_keep_tokens": 128, "sink_keep_tokens": 8, "use_compression": true, "use_cluster": true, "deltakv_center_ratio": 0.1}'
```

Notes:

- `--backend` supports `hf` and `sparsevllm` (see `benchmark/long_bench/pred.py`).
- `--hyper_param` accepts either a JSON string or a path to a JSON file.
- `full_attention_layers` is passed as a comma-separated string of layer indices (example: `"0,1,2,8,18"`).

### DeltaKV checkpoints

- `deltakv_checkpoint_path` can point to either a directory (the loader scans `*.safetensors` first, then `*.bin`/`*.pt`) or a single checkpoint file.
- Split-KV checkpoints (`k_compress_*` / `v_compress_*`) are currently not supported by the Sparse-vLLM loader.

## Troubleshooting

### `SamplingParams` does not allow greedy decoding

`SamplingParams.temperature` must be `> 1e-10` (see `src/sparsevllm/sampling_params.py`). Use a tiny temperature (e.g. `1e-5`) for “almost greedy”.

### `Mixed long/short batch detected`

Sparse-vLLM enforces that each step runs either a “long-text” batch or a “short-text” batch, never mixed, to keep kernels simpler.
If you hit this error, it usually means you are bypassing the scheduler separation logic or mixing very different-length requests in a custom loop.

### `Insufficient KV cache slots to admit prompt`

This means the engine cannot allocate enough KV slots to place the prompt (or a chunk of it), given your method and current KV budgets.
Try one or more of:

- Increase `gpu_memory_utilization`.
- Reduce `max_model_len`, `batch_sizes`, or prompt length.
- Reduce `recent_keep_tokens` / `decode_keep_tokens` / `sink_keep_tokens` for long-context methods.

## Acknowledgements

This project is inspired by and/or references ideas and implementation techniques from:

- `LightLLM` (`ModelTC/LightLLM`)
- `ShadowKV` (`ByteDance-Seed/ShadowKV`)
- `nano-vllm` (`GeeeekExplorer/nano-vllm`)

## License

[Apache License 2.0](LICENSE)

## Citation
```text
@article{hao2026deltakv,
  title={DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity},
  author={Hao, Jitai and Huang, Qiang and Wang, Yaowei and Zhang, Min and Yu, Jun},
  journal={arXiv preprint arXiv:2602.08005},
  year={2026}
}

@inproceedings{hao2025omnikv,
  title={Omnikv: Dynamic context selection for efficient long-context llms},
  author={Hao, Jitai and Zhu, Yuke and Wang, Tian and Yu, Jun and Xin, Xin and Zheng, Bo and Ren, Zhaochun and Guo, Sheng},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
