# Sparse-vLLM

<p align="center">
  <a href="https://arxiv.org/abs/2602.08005">
    <img src="https://img.shields.io/badge/arXiv-2602.08005-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://arxiv.org/pdf/2602.08005.pdf">
    <img src="https://img.shields.io/badge/PDF-download-brightgreen.svg" alt="PDF">
  </a>
</p>

**Model checkpoints and datasets are all about to be uploaded.**

This repo is primarily a **sparse-first inference engine** (`sparsevllm`). It also contains DeltaKV compressor training + evaluation tooling (`deltakv`).

## Contents

- [Sparse-vLLM](#sparse-vllm)
  - [Contents](#contents)
  - [Sparse-vLLM](#sparse-vllm-1)
    - [Install](#install)
    - [Minimal usage](#minimal-usage)
    - [Key parameters](#key-parameters)
    - [Supported methods](#supported-methods)
  - [How to test](#how-to-test)
    - [Throughput benchmark](#throughput-benchmark)
  - [Codebase tour (file-by-file)](#codebase-tour-file-by-file)
    - [Top-level](#top-level)
    - [Sparse-vLLM (`src/sparsevllm/`)](#sparse-vllm-srcsparsevllm)
    - [Benchmarks / scripts](#benchmarks--scripts)
    - [`scripts/` file guide](#scripts-file-guide)
  - [DeltaKV](#deltakv)
    - [DeltaKV code layout](#deltakv-code-layout)
    - [DeltaKV inference](#deltakv-inference)
    - [Train a compressor](#train-a-compressor)
    - [Evaluate on LongBench](#evaluate-on-longbench)
    - [DeltaKV checkpoints](#deltakv-checkpoints)
    - [CUDA gather extension (only for `*-cuda-offload`)](#cuda-gather-extension-only-for--cuda-offload)
  - [Troubleshooting](#troubleshooting)
    - [`SamplingParams` does not allow greedy decoding](#samplingparams-does-not-allow-greedy-decoding)
    - [`Mixed long/short batch detected`](#mixed-longshort-batch-detected)
    - [`Insufficient KV cache slots to admit prompt`](#insufficient-kv-cache-slots-to-admit-prompt)
  - [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Sparse-vLLM

Sparse-vLLM (implemented in `src/sparsevllm/`) is an inference framework built with **sparsity as the first design principle**. Instead of layering sparse methods on top of a conventional KV cache, it rethinks cache layout, controller flow, and kernels so that multiple sparse mechanisms can plug in cleanly.

At a high level, Sparse-vLLM supports:

- **Physical eviction** (e.g., SnapKV, PyramidKV): tokens are truly removed/moved in physical storage.
- **Logical masking** (e.g., OmniKV): tokens remain in storage but are masked at the attention level.
- **Hybrid approaches** (DeltaKV): keep a small high-precision pool + store older tokens in a compressed pool (optional/experimental).

More sparse methods can be added over time. The modular `CacheManager` design keeps it straightforward to integrate new
methods efficiently without rewriting the whole engine.

### Install

```bash
conda create -n svllm python=3.10 -y
conda activate svllm
pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==4.1.0
pip install fire matplotlib seaborn wandb loguru ansible
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
```

### Minimal usage

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen2.5-7B-Instruct-1M",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    chunk_prefill_size=4096,
    vllm_sparse_method="omnikv",
    # OmniKV knobs (simple baseline; tune as needed)
    full_attn_layers="0,1,2,4,7,14",  # layers that run full attention (must include layer 0)
    num_top_tokens=2096,  # top-K tokens kept for sparse layers
    num_top_tokens_in_prefill=8192,  # top-K during prefill (defaults to num_top_tokens)
    chunk_prefill_accel_omnikv=False,  # disable OmniKV chunk-prefill acceleration for easier comparisons
)

outputs = llm.generate(
    prompts=["Write a short story about sparse attention."],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

### Key parameters

Sparse-vLLM runtime knobs are defined in `src/sparsevllm/config.py` and can be passed as keyword args to `LLM(...)`.

**Common knobs**

- `tensor_parallel_size`: number of GPU ranks (processes) to spawn.
- `gpu_memory_utilization`: fraction of total GPU memory to allocate for the KV cache.
- `max_model_len`: max (prompt + generated) tokens allowed.
- `chunk_prefill_size`: chunk size for long-prompt prefill (reduces peak memory and improves scheduling for long contexts).
- `max_num_batched_tokens`, `max_num_seqs_in_batch`, `max_decoding_seqs`: scheduler throughput/latency constraints.

**Sparse knobs (method-dependent)**

- `vllm_sparse_method`: method string (see below).
- `num_sink_tokens`: always-kept “sink” tokens at the beginning.
- `num_recent_tokens`: always-kept “recent” tail tokens.
- `num_top_tokens`: keep top-K tokens (importance-based selection; used by OmniKV and some hybrid modes).
- `num_top_tokens_in_prefill`: top-K used during prefill (defaults to `num_top_tokens` if unset).
- `full_attn_layers`: comma-separated layer indices (or list) that run full attention; used by OmniKV/DeltaKV as “observation” anchors.

### Supported methods

Set `vllm_sparse_method` to one of:

- `""` (vanilla / full attention)
- `"snapkv"`, `"pyramidkv"` (physical eviction)
- `"omnikv"` (logical masking)
- `"deltakv"` / `"deltakv-*"` (hybrid compression; optional / experimental, see [DeltaKV](#deltakv))

## How to test

### Throughput benchmark

Use `scripts/bench_sparse_vllm.py` to measure TTFT, prefill throughput, decode throughput, and GPU memory.

Notes:

- Prefer `--hyper_params` to pass Sparse-vLLM `Config` values (JSON object). Flags like `--gpu_util/--chunk_size/--tp` are kept only for backward compatibility.
- `--lengths` measures *prompt length*; the script sets `max_model_len = length + output_len + 100` internally.

Baseline (vanilla):

```bash
python scripts/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization": 0.9}'
```

## Codebase tour (file-by-file)

If you want to understand the repo by reading code, a good order is:

1. `src/sparsevllm/engine/llm_engine.py`: engine entrypoint (public API + main loop)
2. `src/sparsevllm/engine/model_runner.py`: per-GPU-rank runner (load weights, allocate KV, run forward)
3. `src/sparsevllm/engine/cache_manager.py`: KV cache allocation/layout/free (method-specific implementations live here)
4. `src/sparsevllm/engine/sparse_controller.py`: sparse policy/controller (what to read/reconstruct/evict and when)
5. `src/sparsevllm/layers/attention.py` + `src/sparsevllm/triton_kernel/`: attention integration + kernels

### Top-level

| Path | What it does |
| --- | --- |
| `README.md` | Project overview + quickstart + test entrypoints |
| `pyproject.toml` | Python packaging; installs `deltakv-train` CLI (points to `deltakv.train_compressor:main`) |
| `src/` | Two packages: `sparsevllm` (inference engine) and `deltakv` (optional compressor tooling) |
| `scripts/` | Experiment scripts (throughput, correctness, bandwidth, ablations); not a stable library API |
| `benchmark/` | Evaluation harnesses (LongBench / SCBench / math_bench / NIAH) |
| `baselines/` | Placeholder (currently mostly empty) |

### Sparse-vLLM (`src/sparsevllm/`)

| Path | What it does |
| --- | --- |
| `src/sparsevllm/__init__.py` | Public exports: `from sparsevllm import LLM, SamplingParams` |
| `src/sparsevllm/llm.py` | `LLM` convenience entrypoint (currently a thin alias of `LLMEngine`) |
| `src/sparsevllm/config.py` | `Config`: inference-time knobs (chunked prefill, KV budgets, method selection, etc.) |
| `src/sparsevllm/sampling_params.py` | `SamplingParams`: minimal sampling knobs (note: `temperature` must be > `1e-10`; greedy is not supported) |
| `src/sparsevllm/engine/llm_engine.py` | Engine: multi-process tensor-parallel lifecycle, public API (`add_request/step/generate/exit`) |
| `src/sparsevllm/engine/model_runner.py` | Per-TP-rank runner: NCCL init, load weight shards, create `CacheManager` + `SparseController`, run forward + sampling |
| `src/sparsevllm/engine/scheduler.py` | Scheduler: waiting/decoding queues, chunked prefill, long/short batch separation, preemption/rollback under memory pressure |
| `src/sparsevllm/engine/sequence.py` | `Sequence`: per-request state machine (prompt/prefill progress/generated tokens/finish criteria) |
| `src/sparsevllm/engine/cache_manager.py` | KV cache managers: Standard / SnapKV / OmniKV / PyramidKV / (optional DeltaKV variants) |
| `src/sparsevllm/engine/sparse_controller.py` | Sparse controller: builds per-layer read-view, aggregates attention scores, triggers eviction/reconstruction |
| `src/sparsevllm/models/qwen2.py` | Qwen2 inference model wiring (custom layers/kernels + sparse controller integration) |
| `src/sparsevllm/models/qwen3.py` | Qwen3 inference model wiring |
| `src/sparsevllm/layers/` | Layer building blocks (attention/linear/rmsnorm/rotary/sampler, etc.) |
| `src/sparsevllm/triton_kernel/` | Triton kernels (flash decoding, OmniKV fused, quant pack/unpack, embedding, rmsnorm, etc.) |
| `src/sparsevllm/cuda_kernel/` | Optional CUDA extension used by the DeltaKV CUDA-offload path (see DeltaKV section) |
| `src/sparsevllm/utils/` | Logging, per-step context (`is_prefill`/length flags), profiler, compressor factory + weight loader, etc. |

### Benchmarks / scripts

| Path | What it does |
| --- | --- |
| `scripts/bench_sparse_vllm.py` | Throughput/TTFT/ITL/memory benchmark (supports multiple methods; pass `Config` via `--hyper_params`) |
| `scripts/test_sparse_vllm_correctness.py` | End-to-end generation sanity checks |
| `benchmark/long_bench/pred.py` | LongBench prediction runner (backend: `hf` or `sparsevllm`) |
| `benchmark/scbench/run_scbench.py` | SCBench runner + glue scripts |
| `benchmark/math_bench/pred.py` | Math benchmark prediction runner |
| `benchmark/niah/gen_niah.py` | Generate NIAH data |
| `benchmark/niah/test_niah.py` | Run NIAH evaluation |

### `scripts/` file guide

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

## DeltaKV

DeltaKV is a method for **compressing the KV cache** to enable more efficient long-context inference for Transformer LLMs.
This repo includes DeltaKV compressor training code and some inference/benchmark integrations, but DeltaKV-specific
speed/quality/perf trade-offs are still under active iteration.

### DeltaKV code layout

| Path | What it does |
| --- | --- |
| `src/deltakv/train_compressor.py` | Compressor training entrypoint (Fire CLI) |
| `src/deltakv/save_trainable_trainer.py` | Custom `Trainer` that saves only trainable (`requires_grad=True`) params |
| `src/deltakv/configs/model_config_cls.py` | HF Config extensions for compressor + sparse-method hyper-params |
| `src/deltakv/modeling/` | Model variants/utilities (Qwen2/Llama variants, KV cache helpers, token selection, etc.) |
| `src/deltakv/data_prepare/` | Data preparation (tokenize + pack, collators, dataset generation) |
| `src/deltakv/analysis/` | Research analysis scripts (similarity, ablations, visualizations) |

### DeltaKV inference

Set `vllm_sparse_method` to one of:

- `"deltakv"`
- `"deltakv-triton"`, `"deltakv-triton-v2"`, `"deltakv-triton-v3"`, `"deltakv-triton-v4"`
- `"deltakv-triton-v3-offload"` / `"deltakv-triton-v3-cuda-offload"`

For DeltaKV inference, also pass `deltakv_path="/path/to/trained_compressor_dir_or_file"`.

DeltaKV knobs you may need:

- `deltakv_path`: path to trained compressor weights (directory containing `*.safetensors`/`*.pt`/`*.bin`, or a single file).
- `kv_compressed_size`: latent dimension of compressed KV.
- `cluster_ratio`, `cluster_metric`: reference selection / clustering behavior (method variants may use these differently).
- `deltakv_offload_latent`: offload latent cache to CPU (enabled automatically by `*-offload` methods).
- `deltakv_offload_cpu_threads`: CPU gather thread count for offload mode.

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

Common knobs:

- `--kv_compressed_size`: compressed KV length (smaller = more compression)
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

### DeltaKV checkpoints

- `deltakv_path` can point to either a directory (the loader scans `*.safetensors` first, then `*.bin`/`*.pt`) or a single checkpoint file.
- Split-KV checkpoints (`k_compress_*` / `v_compress_*`) are currently not supported by the Sparse-vLLM loader.

### CUDA gather extension (only for `*-cuda-offload`)

The CUDA extension lives in `src/sparsevllm/cuda_kernel/` and is only required for `deltakv-triton-v3-cuda-offload`.

```bash
cd src/sparsevllm/cuda_kernel
pip install -e .
```

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

## Acknowledgements

This project is inspired by and/or references ideas and implementation techniques from:

- `LightLLM` (`ModelTC/LightLLM`)
- `ShadowKV` (`ByteDance-Seed/ShadowKV`)
- `nano-vllm` (`GeeeekExplorer/nano-vllm`)


# Citation
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