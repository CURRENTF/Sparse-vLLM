# Sparse-vLLM

<p align="center">
  <a href="https://deepwiki.com/CURRENTF/Sparse-vLLM"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://arxiv.org/abs/2602.08005">
    <img src="https://img.shields.io/badge/arXiv-2602.08005-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://arxiv.org/pdf/2602.08005.pdf">
    <img src="https://img.shields.io/badge/PDF-download-brightgreen.svg" alt="PDF">
  </a>
</p>

This repo is primarily a **sparse-first inference engine** (`sparsevllm`). It also contains DeltaKV compressor training + evaluation tooling (`deltakv`).

*Model checkpoints and datasets are all about to be uploaded.*

## Sparse-vLLM

Sparse-vLLM (implemented in `src/sparsevllm/`) is an inference framework built with **sparsity as the first design principle**. Instead of layering sparse methods on top of a conventional KV cache, it rethinks cache layout, controller flow, and kernels so that multiple sparse mechanisms can plug in cleanly.

For codebase structure and file-level navigation, use the DeepWiki badge at the top of this page.

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

### Practical recipes

The following commands reflect the day-to-day experiment patterns we have been using. They are written in AutoDL / GPUHub style paths, but the same structure works on any Linux box.

#### AutoDL / GPUHub setup

```bash
ssh -p <PORT> root@connect.westc.gpuhub.com
cd /root/autodl-tmp/Sparse-vLLM
source /etc/network_turbo

export XDG_CACHE_HOME=/root/autodl-fs/.cache
export HF_HOME=/root/autodl-fs/.cache/huggingface
export TORCH_HOME=/root/autodl-fs/.cache/torch
export DELTAKV_DATA_DIR=/root/autodl-fs/datasets
export DELTAKV_OUTPUT_DIR=/root/autodl-fs/deltakv_outputs
export PYTHONPATH="$PYTHONPATH:/root/autodl-tmp/Sparse-vLLM/src"

conda activate kv
# or: /root/miniconda3/bin/conda run -n kv --no-capture-output <cmd>
```

Notes:

- Keep code under `/root/autodl-tmp`, and put models / datasets / caches / benchmark outputs under `/root/autodl-fs`.
- Run `source /etc/network_turbo` before `git pull`, `pip install`, or downloading checkpoints/datasets.
- Use `tmux` for long jobs if you expect the SSH session to disconnect.

#### Measured sparse-method comparisons

The following numbers were collected on `2026-03-06` on the AutoDL server `connect.westd.seetacloud.com:37226`.

Testbed:

- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`~96 GB`)
- Model: `/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M`
- Script: `scripts/bench_sparse_vllm.py`
- Raw logs: `/root/autodl-fs/deltakv_outputs/bench_bs1_*.log`, `/root/autodl-fs/deltakv_outputs/bench_fullmem_*.log`, and `/root/autodl-fs/deltakv_outputs/bench_throughput_*.log`
- DeltaKV checkpoint: `cluster_e2e_cs256_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_cdownmlp_swiglud3072_cuplinear_0125_222950`

Long single-request speed case (`prompt_len=262144`, `batch_size=1`, `output_len=512`, `gpu_memory_utilization=0.9`, `chunk_prefill_size=4096`):

| Method | Key sparse settings | TTFT | Prefill | Decode | End-to-end | GPU mem |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla | full attention | `64.64s` | `4055.67 tok/s` | `27.13 tok/s` | `83.47s` | `85.30 GB` |
| SnapKV | `num_top_tokens=2048`, `num_sink_tokens=8`, `num_recent_tokens=128`, `snapkv_window_size=32` | `65.24s` | `4018.19 tok/s` | `42.95 tok/s` | `77.14s` | `88.32 GB` |
| OmniKV | `full_attn_layers=0,1,2,4,7,14`, `num_top_tokens_in_prefill=4096`, `num_top_tokens=2048`, `num_recent_tokens=128`, `num_sink_tokens=8` | `27.74s` | `9451.44 tok/s` | `39.83 tok/s` | `40.56s` | `85.41 GB` |

Notes:

- This is the stronger `bs=1` proof point: with a longer prompt and a longer decode, both SnapKV and OmniKV are faster than full attention end-to-end.
- In this setting, SnapKV is about `1.08x` faster end-to-end than full attention and about `1.58x` faster in decode throughput.
- OmniKV is about `2.06x` faster end-to-end than full attention, mainly from a much faster prefill stage.

Full-memory throughput case (`prompt_len=262144`, `output_len=32`, `gpu_memory_utilization=0.9`, `chunk_prefill_size=4096`):

| Method | Batch | Admitted prompt tokens | TTFT | Prefill | Decode | End-to-end | GPU mem | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Vanilla | `5` | `1310720` | `308.70s` | `4246.03 tok/s` | `68.39 tok/s` | `310.96s` | `87.01 GB` | near the standard-cache admission limit |
| SnapKV | `5` | `1310720` | `310.04s` | `4227.59 tok/s` | `197.73 tok/s` | `310.82s` | `91.39 GB` | much faster decode, but similar admission limit |
| OmniKV | `5` | `1310720` | `130.68s` | `10030.42 tok/s` | `129.68 tok/s` | `131.87s` | `87.55 GB` | fastest same-capacity throughput on this setup |
| DeltaKV (`deltakv-triton-v4`) | `8` | `2097152` | `238.87s` | `8779.82 tok/s` | `113.74 tok/s` | `241.04s` | `90.61 GB` | admits `1.60x` more prompt tokens than the standard-cache methods above |

Additional notes:

- For vanilla / SnapKV / OmniKV, `batch_size=5` is already close to the admission boundary here because `5 x 262144 = 1310720` prompt tokens and the cache manager reports `1318342` available slots.
- Vanilla at `batch_size=8` fails immediately with `Insufficient KV cache slots to admit prompt`, while DeltaKV completes the same `262144 x 8` workload.
- On this codebase and hardware, SnapKV and OmniKV still use the standard cache manager for prompt admission, so their main gain is speed, while DeltaKV is the method that materially increases admitted workload under the same memory budget.
- The DeltaKV row above uses the asymmetric compressor variant (`down=mlp_swiglu(inter=3072, bias=False)`, `up=linear(bias=False)`).
- Quick `PyramidKV` reference on the same server: at `128K, bs=1, output_len=64`, it reaches `TTFT 18.78s`, `Prefill 6981.35 tok/s`, `Decode 33.69 tok/s`, and requires `pyramidkv_least_ratio=0.05` on this setup.

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
  --hyper_param '{"chunk_prefill_size": 4096, "vllm_sparse_method": ""}'
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
  --hyper_param '{"chunk_prefill_size": 4096, "vllm_sparse_method": "omnikv", "chunk_prefill_accel_omnikv": false, "full_attn_layers": "0,1,2,4,7,14", "num_top_tokens": 1024}'
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
  --hyper_param '{"chunk_prefill_size": 512, "num_top_tokens_in_prefill": 16384, "max_num_batched_tokens": 8192, "max_num_seqs_in_batch": 30, "vllm_sparse_method": "deltakv-triton-v4", "chunk_prefill_accel_omnikv": true, "full_attn_layers": "0,1,2,4,7,14", "num_top_tokens": 1024, "deltakv_path": "/root/autodl-fs/checkpoints/compressor/<COMPRESSOR_DIR>", "kv_compressed_size": 256}'
```

When `--backend sparsevllm`, method selection happens entirely through `--hyper_param` (`vllm_sparse_method`, `deltakv_path`, etc.). `--model_cls` and `--compressor_path` are not used by the Sparse-vLLM backend.

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
  --hyper_param '{"chunk_prefill_size": 4096, "vllm_sparse_method": "omnikv", "chunk_prefill_accel_omnikv": true, "num_top_tokens_in_prefill": 4096, "num_top_tokens": 2048, "full_attn_layers": "0,1,2,4,7,14", "num_recent_tokens": 128, "num_sink_tokens": 8}'
```

For a full LongBench run, simply omit `--task`. To switch to DeltaKV, keep `--backend sparsevllm` and replace the method-specific part of `--hyper_param` with `vllm_sparse_method="deltakv"` (or `deltakv-triton-v4`) plus `deltakv_path=...`.

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
  --model_cls deltakv \
  --compressor_path "/root/autodl-fs/checkpoints/compressor/<COMPRESSOR_DIR>" \
  --hyper_param '{"chunk_prefill_size": 2048000, "num_top_tokens_in_prefill": 4096, "chunk_prefill_accel_omnikv": true, "num_top_tokens": 0.11, "full_attn_layers": "0,1,2,4,7,14", "num_recent_tokens": 128, "num_sink_tokens": 8, "use_compression": true, "use_cluster": true, "cluster_ratio": 0.1}'
```

To compare other baselines, keep `--backend hf` and switch `--model_cls` / `--hyper_param`, e.g. `snapkv` with `{"num_top_tokens": 0.2, "pool_kernel_size": 7}` or `pyramidkv` with a similar token budget.

## DeltaKV

DeltaKV is a method for **compressing the KV cache** to enable more efficient long-context inference for Transformer LLMs.
This repo includes DeltaKV compressor training code and some inference/benchmark integrations, but DeltaKV-specific
speed/quality/perf trade-offs are still under active iteration.

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
