# Sparse-vLLM & DeltaKV

**Model checkpoints, datasets, and papers are all about to be uploaded.**

## Sparse-vLLM (Sparse-first Inference Framework)

Sparse-vLLM (implemented in `src/sparsevllm/`) is an inference framework built with **sparsity as the first design principle**.
Instead of layering sparse methods onto a conventional KV cache, it rethinks the cache layout, controller flow, and kernels to make diverse sparse mechanisms plug in cleanly and run efficiently.

### CacheManager: extensible cache data layouts

At the core is a `CacheManager` whose internal memory layout can be swapped to match different algorithms’ access patterns. The repo currently supports representative backends for:

- **Physical eviction** (e.g., SnapKV, PyramidKV): tokens are truly removed / moved in physical storage.
- **Logical masking** (e.g., Full Attention, OmniKV): tokens are retained globally but masked at the attention level.
- **Hybrid compression** (DeltaKV): some tokens remain high-precision while older tokens are stored in compressed form.

Concretely, Sparse-vLLM uses different mapping structures depending on the method:

- **Per-layer independent mapping (physical eviction)**: maintains `L` independent page tables
  `buffer_req_to_token_slots[layer_idx]`, because each layer can have a different physically-discontinuous KV view.
- **Global shared mapping (logical masking)**: uses a unified `req_to_token_slots` shared by all layers to reduce metadata
  overhead and improve locality.
- **Heterogeneous DeltaKV storage (hybrid compression)**:
  - **Dual physical pools**: a `Full Pool` for high-precision tokens (Sink/Recent) and a `Latent Pool` for compressed
    vectors; slots are allocated based on token lifecycle.
  - **Intra-group slot sharing**: within Observation→Sparse layer groups, reconstructed temporary slots are shared in a
    copy-on-write style to avoid repeated decompression and reduce pre-forward bandwidth pressure.

### Sparse Controller: method-specific workflows

Sparse-vLLM separates *policy* (which tokens to keep / reconstruct) from *mechanism* (how memory is managed) via a
Sparse Controller that orchestrates interaction between the model and `CacheManager`.

For **DeltaKV**, the workflow is:

- **Pre-forward (DeltaKV view construction)**:
  1. **Index resolution**: resolve logical indices of tokens that require decompression based on reference tokens.
  2. **Batch reconstruction**: fetch compressed vectors from the `Latent Pool` plus their references.
  3. **Slot virtualization**: write reconstructed KV into a temporary physical buffer, then build a virtual `slot_mapping`
     that stitches static slots (Sink/Recent) with temporary slots to present a contiguous logical view to attention.

- **Post-forward (lifecycle management)**:
  - When the Recent buffer overflows, a specialized fused kernel:
    1. computes the residual w.r.t. assigned reference tokens,
    2. compresses the residual via the encoder (down-projection),
    3. writes into the `Latent Pool` and frees `Full Pool` slots immediately (keeping memory bounded vs. sequence length).

### Kernel optimizations

Sparse-vLLM includes kernel-side optimizations to keep the sparse/compress loop efficient:

- **Indirect addressing via slot mapping**: Flash-Decoding kernels are modified to accept token-level index arrays
  (e.g., `req_to_token_slots`), enabling direct reads from non-contiguous physical KV without extra copies or block-table
  lookups.
- **Fused DeltaKV kernels (Triton)**: custom kernels accelerate reference search and reconstruction, including:
  - batch distance computation (e.g., L2) for reference selection,
  - fused reconstruction that combines gathering references, mean computation, and residual addition to reduce bandwidth.

You can benchmark Sparse-vLLM-based methods via `scripts/bench_sparse_vllm.py` (see examples below).

## DeltaKV

DeltaKV is a method for **compressing the KV cache** to enable more efficient long-context inference for Transformer LLMs.
This repository contains:

- Training code for a KV compressor (Hugging Face + Accelerate/DeepSpeed).
- Inference / benchmarking utilities (LongBench, throughput micro-benchmarks).
- A lightweight inference framework (`sparsevllm`).


## Highlights

- **Trainable KV compressor**: freezes the base model and trains only compression-related modules.
- **Architecture support**: Llama-family and Qwen2-family.
- **Benchmarks included**: LongBench + SCBench + AIME + throughput scripts (see `benchmark/` and `scripts/`).
- **Sparse-vLLM Inference Framework**: a sparse-first inference engine with a modular cache system designed for physical eviction, logical masking, and hybrid compression.

## Repository layout

- `src/deltakv/`: core implementation (configs, modeling, training, data utilities)
- `src/deltakv/train_compressor.py`: compressor training entrypoint
- `benchmark/long_bench/`: LongBench evaluation wrapper
- `scripts/bench_sparse_vllm.py`: throughput / TTFT / prefill-decode benchmarking

## Installation

### Python environment

- Python `>= 3.8`
- A CUDA-capable GPU is recommended for training and most benchmarks.

Create an environment and install this repo in editable mode:

```bash
conda create -n deltakv python=3.10 -y
conda activate deltakv
pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==4.1.0
pip install transformers==4.53.3 fire matplotlib seaborn wandb loguru ansible
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
```

### Notes on dependencies

- `flash-attn` is listed in `pyproject.toml` and typically requires a CUDA build toolchain.
- Some benchmark folders provide their own `requirements.txt` (e.g. `benchmark/long_bench/requirements.txt`).

## Quick start

### 1) Train a compressor

The main entrypoint is:

- Python: `python src/deltakv/train_compressor.py ...`
- CLI script (after installation): `deltakv-train ...`

The training script expects a **tokenized + packed** dataset saved by Hugging Face `datasets` (`load_from_disk`).

#### Training example (command template)

Replace paths (`<...>`) with your local paths.

```bash
# linear / mlp example
git pull; python src/deltakv/train_compressor.py \
  --model_name_or_path <PATH_TO_BASE_MODEL> \
  --dataset_path <PATH_TO_DATASET_ON_DISK> \
  --output_dir <PATH_TO_OUTPUT_CHECKPOINT_DIR> \
  --kv_compressed_size 512 \
  --seq_chunk_size 4 \
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

**Common knobs**

- `--kv_compressed_size`: compressed KV length (smaller = more compression)
- `--model_type`: `e2e`, `cluster_e2e`, `cluster_e2e_big` (see `src/deltakv/train_compressor.py`)
- `--collect_kv_before_rope`: whether to collect KV before RoPE (model-dependent)

### 2) Evaluate on LongBench

`benchmark/long_bench/pred.py` runs LongBench prediction and writes JSONL outputs under a local output directory.

#### LongBench example (HF backend)

```bash
# test quality (LongBench)
git pull; python benchmark/long_bench/pred.py \
  --model all \
  --model_path <PATH_TO_BASE_MODEL> \
  --tokenizer_path <PATH_TO_TOKENIZER_OR_MODEL> \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --model_cls deltakv \
  --compressor_path "<PATH_TO_TRAINED_COMPRESSOR_DIR>" \
  --hyper_param '{"chunk_prefill_size": 2048000, "num_top_tokens_in_prefill": 4096,
  "chunk_prefill_accel_omnikv": true, "num_top_tokens": 0.17, "full_attn_layers": "0,1,2,8,18",
  "num_recent_tokens": 128, "num_sink_tokens": 8, "use_compression": true, "use_cluster": true, "cluster_ratio": 0.1}'
```

Notes:

- `--backend` supports `hf` and `sparsevllm` (see `benchmark/long_bench/pred.py`).
- `--hyper_param` accepts either a JSON string or a path to a JSON file.
- `full_attn_layers` is passed as a comma-separated string of layer indices (example: `"0,1,2,8,18"`).

### 3) Benchmark throughput (prefill/decode/TTFT)

Use `scripts/bench_sparse_vllm.py` to measure TTFT, prefill throughput, decode throughput, and GPU memory.

#### Throughput example: baseline (full / vanilla)

```bash
# throughput baseline
git pull; python scripts/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --gpu_util 0.9
```

#### Throughput example: DeltaKV (Triton variant)

```bash
# throughput with DeltaKV
git pull; python scripts/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 131000 \
  --batch_sizes 16 \
  --methods deltakv-triton-v4 \
  --output_len 768 \
  --hyper_params '{"gpu_memory_utilization": 0.85,
  "kv_compressed_size": 256,
  "num_top_tokens": 2048,
  "deltakv_path": "<PATH_TO_TRAINED_COMPRESSOR_DIR>",
  "full_attn_layers": "0,1,2,4,7,14"}'
```

## Data preparation (optional)

This repo includes utilities under `src/deltakv/data_prepare/` for tokenization/packing and building training datasets.
Typical training uses a dataset saved via `datasets.Dataset.save_to_disk(...)` and loaded by `load_from_disk(...)`.
