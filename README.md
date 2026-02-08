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

## Codebase Tour (File-by-file)

If you want to understand the repo by reading code, a good order is:

1. `src/sparsevllm/engine/llm_engine.py`: engine entrypoint (public API + main loop)
2. `src/sparsevllm/engine/model_runner.py`: per-GPU-rank runner (load weights, allocate KV, run forward)
3. `src/sparsevllm/engine/cache_manager.py`: KV cache allocation/layout/free (method-specific implementations live here)
4. `src/sparsevllm/engine/sparse_controller.py`: sparse policy/controller (what to read/reconstruct/evict and when)
5. `src/sparsevllm/layers/attention.py` + `src/sparsevllm/triton_kernel/`: attention integration + kernels

### Top-level

| Path | What it does |
| --- | --- |
| `README.md` | Project overview + install/train/eval/throughput benchmark entrypoints |
| `pyproject.toml` | Python packaging; installs `deltakv-train` CLI (points to `deltakv.train_compressor:main`) |
| `src/` | Two main packages: `sparsevllm` (inference engine) and `deltakv` (compressor training/analysis/eval tooling) |
| `scripts/` | Experiment scripts (throughput, correctness, bandwidth, ablations); not a stable library API |
| `benchmark/` | Evaluation harnesses (LongBench / SCBench / math_bench / NIAH) |
| `baselines/` | Placeholder (currently mostly empty) |

### Sparse-vLLM (`src/sparsevllm/`)

Sparse-vLLM is the inference-side core: scheduling + KV management + sparse control + Triton/CUDA kernels.

| Path | What it does |
| --- | --- |
| `src/sparsevllm/__init__.py` | Public exports: `from sparsevllm import LLM, SamplingParams` |
| `src/sparsevllm/llm.py` | `LLM` convenience entrypoint (currently a thin alias of `LLMEngine`) |
| `src/sparsevllm/config.py` | `Config`: inference-time knobs (chunked prefill, KV budgets, method selection, DeltaKV offload, etc.) |
| `src/sparsevllm/sampling_params.py` | `SamplingParams`: minimal sampling knobs (note: `temperature` must be > `1e-10`; greedy is not supported) |
| `src/sparsevllm/constant.py` | Global constants (e.g., batch redundancy factor) |
| `src/sparsevllm/engine/llm_engine.py` | Engine: multi-process tensor-parallel lifecycle, public API (`add_request/step/generate/exit`), throughput logger |
| `src/sparsevllm/engine/model_runner.py` | Per-TP-rank runner: NCCL init, load weight shards, create `CacheManager` + `SparseController`, run forward + sampling |
| `src/sparsevllm/engine/scheduler.py` | Scheduler: waiting/decoding queues, chunked prefill, long/short batch separation, preemption/rollback under memory pressure |
| `src/sparsevllm/engine/sequence.py` | `Sequence`: per-request state machine (prompt/prefill progress/generated tokens/finish criteria) |
| `src/sparsevllm/engine/cache_manager.py` | KV cache managers: Standard / SnapKV / OmniKV / DeltaKV (incl. Triton v2/v3/v4 and offload variants) |
| `src/sparsevllm/engine/sparse_controller.py` | Sparse controller: builds per-layer read-view (active slots/indices), aggregates attention scores, triggers eviction/reconstruction |
| `src/sparsevllm/models/qwen2.py` | Qwen2 inference model wiring (custom layers/kernels + sparse controller integration) |
| `src/sparsevllm/models/qwen3.py` | Qwen3 inference model wiring |
| `src/sparsevllm/layers/` | Layer building blocks (attention/linear/rmsnorm/rotary/sampler, etc.) |
| `src/sparsevllm/triton_kernel/` | Triton kernels (flash decoding, OmniKV fused, quant pack/unpack, embedding, rmsnorm, etc.) |
| `src/sparsevllm/cuda_kernel/` | Custom CUDA extension for efficient CPU(pinned)→GPU gather (used by DeltaKV CUDA-offload path; see its README) |
| `src/sparsevllm/utils/` | Logging, per-step context (`is_prefill`/length flags), profiler, compressor factory + weight loader, etc. |

### DeltaKV (`src/deltakv/`)

`deltakv` is mostly training/analysis/evaluation tooling: it trains the KV compressor and evaluates via HF or sparsevllm backends.

| Path | What it does |
| --- | --- |
| `src/deltakv/train_compressor.py` | Compressor training entrypoint (Fire CLI): load HF model, inject compressor modules, freeze base model, train, and save only trainable params |
| `src/deltakv/save_trainable_trainer.py` | Custom `Trainer`: when saving checkpoints, only persists `requires_grad=True` parameters |
| `src/deltakv/configs/model_config_cls.py` | HF Config extensions: attach DeltaKV/SnapKV/OmniKV/PyramidKV hyper-params onto the config |
| `src/deltakv/modeling/` | Model variants and utilities (KV cache, token selection, Qwen2/Llama variants: e2e/cluster/snapkv/pyramidkv, etc.) |
| `src/deltakv/data_prepare/` | Data preparation (tokenize + pack, collators, dataset generation) |
| `src/deltakv/analysis/` | Research analysis scripts (similarity analyses, ablation visualizations, stats tools) |

### Benchmarks / scripts

| Path | What it does |
| --- | --- |
| `scripts/bench_sparse_vllm.py` | Throughput/TTFT/ITL/memory benchmark (supports `vanilla/snapkv/omnikv/deltakv` + Triton/offload variants; pass `Config` via `--hyper_params`) |
| `benchmark/long_bench/pred.py` | LongBench prediction runner (backend: `hf` or `sparsevllm`; `--hyper_param` can be inline JSON or a JSON file path) |
| `benchmark/scbench/run_scbench.py` | SCBench runner + glue scripts (see `benchmark/scbench/readme.md` and `requirements.txt`) |
| `benchmark/math_bench/pred.py` | Math benchmark prediction runner |
| `benchmark/niah/gen_niah.py` | Generate NIAH data |
| `benchmark/niah/test_niah.py` | Run NIAH evaluation |

#### `scripts/` file guide

| File | What it does |
| --- | --- |
| `scripts/bench_sparse_vllm.py` | Sparse-vLLM throughput/TTFT/ITL benchmark (multiple methods/lengths/batch sizes) |
| `scripts/test_sparse_vllm_correctness.py` | End-to-end generation sanity checks (long/short separation, long-gen stability, etc.) |
| `scripts/test_compressor_chunk_loss.py` | Compressor/chunking-related loss or alignment sanity check (training-side) |
| `scripts/test_llama_snapkv_init.py` | Llama + SnapKV init/behavior checks |
| `scripts/test_omnikv_fused_compare.py` | OmniKV fused kernel comparison/validation |
| `scripts/test_gqa_flash_decoding_score.py` | GQA flash-decoding score/behavior validation |
| `scripts/test_gather_bandwidth.py` | Gather bandwidth/throughput micro-benchmark |
| `scripts/test_pcie_bandwidth.py` | PCIe bandwidth test (useful for offload path evaluation) |
| `scripts/test_tensor_op_overhead.py` | Tensor-op pipeline overhead test (can mimic LLM MLP scale) |
| `scripts/test_derope.py` | RoPE/De-RoPE experiments |
| `scripts/get_train_dataset_loss.py` | Compute average loss of a model on a tokenized dataset subset (sanity-check data/model) |
| `scripts/stat_longbench_len.py` | LongBench length statistics and related analysis |
| `scripts/compare_indices.py` | Compare token-selection indices across implementations (e.g., SnapKV overlap) |
| `scripts/visualize_profiling.py` | Visualize profiling breakdowns (stacked bar charts, etc.) |
| `scripts/tune_omnikv.py` | OmniKV experiment entrypoint (reads `scripts/_exp_lst.py`) |
| `scripts/_exp_lst.py` | OmniKV experiment list/config (currently placeholder) |
| `scripts/run_llama_ablation.sh` | Shell script to run Llama ablations |

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

## Sparse-vLLM Quickstart (Inference)

After `pip install -e .`, you can run inference via the lightweight `sparsevllm` engine.

### Minimal example

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    "/path/to/base_model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    chunk_prefill_size=8192,
    vllm_sparse_method="",
)

outputs = llm.generate(
    prompts=["Write a short story about sparse attention."],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

### Method selection

Set `vllm_sparse_method` to one of:

- `""` (vanilla / full attention)
- `"snapkv"`, `"pyramidkv"` (physical eviction)
- `"omnikv"` (logical masking)
- `"deltakv"` (hybrid compression)
- `"deltakv-triton"`, `"deltakv-triton-v2"`, `"deltakv-triton-v3"`, `"deltakv-triton-v4"` (DeltaKV with Triton kernels)
- `"deltakv-triton-v3-offload"` / `"deltakv-triton-v3-cuda-offload"` (DeltaKV with CPU latent offload; the CUDA-offload path uses the custom CUDA gather extension)

For DeltaKV inference, also pass `deltakv_path="/path/to/trained_compressor_dir_or_file"`.

### Optional: build the CUDA gather extension (only for `*-cuda-offload`)

The CUDA extension lives in `src/sparsevllm/cuda_kernel/` and is only required for `deltakv-triton-v3-cuda-offload`.

```bash
cd src/sparsevllm/cuda_kernel
pip install -e .
```

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
python src/deltakv/train_compressor.py \
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
python benchmark/long_bench/pred.py \
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

Notes:

- Prefer `--hyper_params` to pass Sparse-vLLM `Config` values (JSON object). Flags like `--gpu_util/--chunk_size/--tp` are kept only for backward compatibility.
- `--lengths` measures *prompt length*; the script sets `max_model_len = length + output_len + 100` internally.

#### Throughput example: baseline (full / vanilla)

```bash
# throughput baseline
python scripts/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization": 0.9}'
```

#### Throughput example: DeltaKV (Triton variant)

```bash
# throughput with DeltaKV
python scripts/bench_sparse_vllm.py \
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

## Configuration Cheatsheet (Sparse-vLLM)

Sparse-vLLM runtime knobs are defined in `src/sparsevllm/config.py` and can be passed as keyword args to `LLM(...)`.

### Common knobs

- `tensor_parallel_size`: number of GPU ranks (processes) to spawn.
- `gpu_memory_utilization`: fraction of total GPU memory to allocate for the KV cache.
- `max_model_len`: max (prompt + generated) tokens allowed.
- `chunk_prefill_size`: chunk size for long-prompt prefill (reduces peak memory and improves scheduling for long contexts).
- `max_num_batched_tokens`, `max_num_seqs_in_batch`, `max_decoding_seqs`: scheduler throughput/latency constraints.

### Sparse knobs (method-dependent)

- `vllm_sparse_method`: method string (see “Method selection” above).
- `num_sink_tokens`: always-kept “sink” tokens at the beginning.
- `num_recent_tokens`: always-kept “recent” tail tokens.
- `num_top_tokens`: keep top-K tokens (importance-based selection; used by OmniKV/DeltaKV sparse layers).
- `num_top_tokens_in_prefill`: top-K used during prefill (defaults to `num_top_tokens` if unset).
- `full_attn_layers`: comma-separated layer indices (or list) that run full attention; used by OmniKV/DeltaKV as “observation” anchors.

### DeltaKV knobs

- `deltakv_path`: path to trained compressor weights (a directory containing `*.safetensors`/`*.pt`/`*.bin`, or a single file).
- `kv_compressed_size`: latent dimension of compressed KV.
- `cluster_ratio`, `cluster_metric`: reference selection / clustering behavior (method variants may use these differently).
- `deltakv_offload_latent`: offload latent cache to CPU (enabled automatically by `*-offload` methods).
- `deltakv_offload_cpu_threads`: CPU gather thread count for offload mode.

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
- Reduce `num_recent_tokens` / `num_top_tokens` / `num_sink_tokens` for long-context methods.
- For DeltaKV, ensure `deltakv_path` points to a valid compressor checkpoint.

## Notes on DeltaKV checkpoints

- `deltakv_path` can point to either a directory (the loader scans `*.safetensors` first, then `*.bin`/`*.pt`) or a single checkpoint file.
- Split-KV checkpoints (`k_compress_*` / `v_compress_*`) are currently not supported by the Sparse-vLLM loader.
