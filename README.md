![logo.png](assets/logo.png)

![sparse_vllm_throughput.png](assets/sparse_vllm_throughput.png)

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

Public DeltaKV compressor checkpoints are available on Hugging Face. Benchmark datasets are still configured by each benchmark entrypoint and its documented data paths.

## Sparse-vLLM

Sparse-vLLM (implemented in `src/sparsevllm/`) is an inference framework built with **sparsity as the first design principle**. Instead of layering sparse methods on top of a conventional KV cache, it rethinks cache layout, controller flow, and kernels so that multiple sparse mechanisms can plug in cleanly.

For codebase structure and file-level navigation, use the DeepWiki badge at the top of this page.

At a high level, Sparse-vLLM supports:

- **Physical eviction** (e.g., SnapKV, PyramidKV): tokens are truly removed/moved in physical storage.
- **Logical masking** (e.g., OmniKV): tokens remain in storage but are masked at the attention level.
- **Hybrid approaches** (DeltaKV): keep a small high-precision pool + store older tokens in a compressed pool (optional/experimental).

More sparse methods can be added over time. The modular `CacheManager` design keeps it straightforward to integrate new
methods efficiently without rewriting the whole engine.

> If you want Codex to add a new sparse method following this repo's architecture, use the repo skill
[`$add-sparse-method`](skills/add-sparse-method/SKILL.md). It encodes the expected structure for new methods
(`cache_manager`-first, generic `attention.py`, decode-time hooks through `build_decode_view(...)`, and method-specific
state kept out of `utils/`).

### Install

```bash
conda create -n svllm python=3.10 -y
conda activate svllm
pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==4.1.0 bitsandbytes
pip install fire matplotlib seaborn wandb loguru ansible
MAX_JOBS=8 pip install flash-attn --no-build-isolation
pip install -e .
```

### Download DeltaKV compressor checkpoints

Compressor-backed DeltaKV runs require a local checkpoint directory. Download
the compressor that matches the base model before passing
`deltakv_checkpoint_path`; the current loaders read `model.safetensors` from the
local filesystem.

| Base model | Compressor checkpoint |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct-1M` | [`JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor`](https://huggingface.co/JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor) |
| `Qwen/Qwen2.5-32B-Instruct` | [`JitaiHao/Qwen2.5-32B-Instruct-Compressor`](https://huggingface.co/JitaiHao/Qwen2.5-32B-Instruct-Compressor) |
| `meta-llama/Llama-3.1-8B-Instruct` | [`JitaiHao/Llama-3.1-8B-Instruct-Compressor`](https://huggingface.co/JitaiHao/Llama-3.1-8B-Instruct-Compressor) |

Example download location used by commands below:

```bash
export DELTAKV_CKPT_ROOT=/root/autodl-fs/checkpoints/compressor
mkdir -p "$DELTAKV_CKPT_ROOT"

huggingface-cli download JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Qwen2.5-7B-Instruct-1M-Compressor"

huggingface-cli download JitaiHao/Qwen2.5-32B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Qwen2.5-32B-Instruct-Compressor"

huggingface-cli download JitaiHao/Llama-3.1-8B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Llama-3.1-8B-Instruct-Compressor"
```

Use the downloaded local directory as `deltakv_checkpoint_path`. Do not reuse a
compressor checkpoint with a different base model unless it was trained for that
model and its layer/head dimensions match.

### Documentation map

- [`docs/README.md`](docs/README.md): stable docs index.
- [`docs/architecture.md`](docs/architecture.md): Sparse-vLLM and HF DeltaKV
  runtime flow.
- [`docs/reproducibility.md`](docs/reproducibility.md): environment,
  checkpoint, data path, smoke test, and artifact checklist.
- [`docs/runtime-parameter-semantics.md`](docs/runtime-parameter-semantics.md):
  canonical parameters and backend-specific semantics.
- Historical implementation notes and exact local run records live under
  [`docs/dev-notes/`](docs/dev-notes/).

### Minimal usage

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen2.5-7B-Instruct-1M",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    engine_prefill_chunk_size=4096,
    sparse_method="omnikv",
    # OmniKV knobs (simple baseline; tune as needed)
    full_attention_layers="0,1,2,4,7,14",  # layers that run full attention (must include layer 0)
    decode_keep_tokens=2096,  # top-K tokens kept for sparse layers
    prefill_keep_tokens=8192,  # top-K during prefill (defaults to decode_keep_tokens)
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
For backend-agnostic DeltaKV/HF and Sparse-vLLM configs, use the canonical names
and behavior notes in the repo-wide parameter audit:
[`docs/runtime-parameter-semantics.md`](docs/runtime-parameter-semantics.md).
Legacy runtime names such as `chunk_prefill_size`, `vllm_sparse_method`,
`num_top_tokens`, `model_cls`, and `compressor_path` are rejected at public
runtime/API boundaries. Use the canonical names below.

For LLaVA-OneVision visual-token experiments, see
[`docs/multimodal_models_adapation/llava-onevision-visual-cache-benchmarks.md`](docs/multimodal_models_adapation/llava-onevision-visual-cache-benchmarks.md).
The no-checkpoint keep-ratio path is a visual-token uniform-pruning baseline,
not DeltaKV cluster/compressor inference.

**Common knobs**

- `tensor_parallel_size`: number of GPU ranks (processes) to spawn.
- `gpu_memory_utilization`: fraction of total GPU memory to allocate for the KV cache.
- `max_model_len`: max (prompt + generated) tokens allowed.
- `engine_prefill_chunk_size`: Sparse-vLLM prefill scheduling and memory-admission chunk size.
- `max_num_batched_tokens`, `max_num_seqs_in_batch`, `max_decoding_seqs`: scheduler throughput/latency constraints.

**Sparse knobs (method-dependent)**

- `sparse_method`: method selector.
- `deltakv_checkpoint_path`: local DeltaKV compressor checkpoint directory or file.
- `sink_keep_tokens`: always-kept prefix/sink tokens.
- `recent_keep_tokens`: always-kept recent tail tokens.
- `decode_keep_tokens`: decode-time top/important token budget.
- `prefill_keep_tokens`: prefill/finalization top/important token budget.
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
- `"deltakv"` / `"deltakv-*"` (hybrid compression; optional / experimental, see [DeltaKV](#deltakv))

Sparse-vLLM internally stores this as `vllm_sparse_method`, but public commands
and `LLM(...)` kwargs should use `sparse_method`.

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens (default `16`)
- `quest_token_budget`: decode-time token budget before page rounding (default `1024`)
- `quest_skip_layers`: keep the first N layers dense during decode (default `2`)

## How to test

### Throughput benchmark

Use `scripts/benchmarks/bench_sparse_vllm.py` to measure TTFT, prefill throughput, decode throughput, and GPU memory.

Notes:

- Prefer `--hyper_params` to pass Sparse-vLLM settings as a JSON object.
- `--hyper_params` accepts canonical runtime names such as `sparse_method`, `engine_prefill_chunk_size`, `decode_keep_tokens`, and `prefill_keep_tokens`; legacy runtime names are rejected.
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
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "chunk_prefill_accel_omnikv": false, "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024}'
```

DeltaKV requires a compressor trained for the same base model. The public
checkpoints above are for Qwen2.5-7B-1M, Qwen2.5-32B, and Llama-3.1-8B; replace
the checkpoint path below with a matching compressor for the model you run.

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
  --hyper_param '{"engine_prefill_chunk_size": 512, "prefill_keep_tokens": 16384, "max_num_batched_tokens": 8192, "max_num_seqs_in_batch": 30, "sparse_method": "deltakv-triton-v4", "chunk_prefill_accel_omnikv": true, "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024, "deltakv_checkpoint_path": "/root/autodl-fs/checkpoints/compressor/<MATCHING_COMPRESSOR_DIR>", "deltakv_latent_dim": 256}'
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
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "chunk_prefill_accel_omnikv": true, "prefill_keep_tokens": 4096, "decode_keep_tokens": 2048, "full_attention_layers": "0,1,2,4,7,14", "recent_keep_tokens": 128, "sink_keep_tokens": 8}'
```

For a full LongBench run, omit `--task`. To switch to DeltaKV, keep
`--backend sparsevllm` and set `sparse_method="deltakv"` (or
`"deltakv-triton-v4"`) plus a matching `deltakv_checkpoint_path`, for example
`/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`.
For the no-checkpoint direct residual ablation, set
`sparse_method="deltakv-delta-quant"` and omit `deltakv_checkpoint_path`.

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
  --deltakv_checkpoint_path "/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor" \
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

Set `sparse_method` to one of:

- `"deltakv"`
- `"deltakv-triton"`, `"deltakv-triton-v2"`, `"deltakv-triton-v3"`, `"deltakv-triton-v4"`
- `"deltakv-delta-quant"` for the no-checkpoint direct residual quantization ablation

For compressor-backed DeltaKV inference, also pass
`deltakv_checkpoint_path="/path/to/local/trained_compressor_dir_or_file"`.
`deltakv-delta-quant` does not load or require a compressor checkpoint.

DeltaKV knobs you may need:

- `deltakv_checkpoint_path`: local path to trained compressor weights (directory containing `*.safetensors`/`*.pt`/`*.bin`, or a single file).
- `deltakv_latent_dim`: latent dimension of compressed KV.
- `deltakv_center_ratio`, `cluster_metric`: reference selection / clustering behavior.
- `deltakv_neighbor_count`: number of selected center/reference tokens used for reconstruction.
- `deltakv_latent_quant_bits`: `4` packs the DeltaKV-style cached state as int4 where supported.

`deltakv-delta-quant` is a Sparse-vLLM-only ablation that reuses DeltaKV center
selection and sparse decode views, but stores the token-space residual directly:

```text
residual = KV_before_rope - mean(selected_center_KV_before_rope)
```

With `deltakv_latent_quant_bits=4`, that residual is packed as int4 plus
per-token scale/min metadata. With `deltakv_latent_quant_bits=0`, the residual
is stored in the model dtype. This path deliberately does not use learned
`compress_down` or `compress_up` modules. The int4 reconstruction path uses a
fused Triton kernel that dequantizes the residual, adds the selected-center
mean, applies RoPE to K, and writes K/V back into the cache in one pass.

Quick throughput smoke:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv-delta-quant \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"prefill_keep_tokens":64,"deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_quant_bits":4,"deltakv_full_pool_reserve_ratio":0.2}'
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
  --deltakv_checkpoint_path "<LOCAL_PATH_TO_TRAINED_COMPRESSOR_DIR>" \
  --hyper_param '{"hf_prefill_chunk_size": 2048000, "prefill_keep_tokens": 4096,
  "chunk_prefill_accel_omnikv": true, "decode_keep_tokens": 0.17, "full_attention_layers": "0,1,2,8,18",
  "recent_keep_tokens": 128, "sink_keep_tokens": 8, "use_compression": true, "use_cluster": true, "deltakv_center_ratio": 0.1}'
```

Notes:

- `--backend` supports `hf` and `sparsevllm` (see `benchmark/long_bench/pred.py`).
- `--hyper_param` accepts either a JSON string or a path to a JSON file.
- `full_attention_layers` is passed as a comma-separated string of layer indices (example: `"0,1,2,8,18"`).

### DeltaKV checkpoints

- Public compressor checkpoints are listed in [Download DeltaKV compressor checkpoints](#download-deltakv-compressor-checkpoints).
- `deltakv_checkpoint_path` can point to either a local directory (the loader scans `*.safetensors` first, then `*.bin`/`*.pt`) or a single checkpoint file.
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
