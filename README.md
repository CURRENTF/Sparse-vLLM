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

*Model checkpoints and datasets for DeltaKV are all about to be uploaded.*

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
[`docs/llava-onevision-visual-cache-benchmarks.md`](docs/llava-onevision-visual-cache-benchmarks.md).
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
- `"deltakv"` / `"deltakv-*"` (hybrid compression; optional / experimental, see [DeltaKV](#deltakv))

Sparse-vLLM internally stores this as `vllm_sparse_method`, but public commands
and `LLM(...)` kwargs should use `sparse_method`.

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens (default `16`)
- `quest_token_budget`: decode-time token budget before page rounding (default `1024`)
- `quest_skip_layers`: keep the first N layers dense during decode (default `2`)

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

#### Claw-Eval with a local Sparse-vLLM OpenAI API

Claw-Eval drives models through an OpenAI-compatible chat API. This repo
provides a text-only Sparse-vLLM shim at
`benchmark/claw_eval/serve_sparsevllm_openai.py` and a reproducible runner at
`benchmark/claw_eval/run_sparsevllm_claw_eval.sh`.

On the `guest-KR6288` evaluation host, the runner defaults to GPUs `4,5,6`,
writes logs/traces under `/data2/haojitai/outputs/Sparse-vLLM/claw-eval`, and
uses the Clash download-only proxy on `127.0.0.1:7898` for Claw-Eval cloning,
fixtures, and package downloads.
The default `ENGINE_KWARGS` uses `tensor_parallel_size=1`; increase it only for
models whose tensor-parallel dimensions divide evenly across the requested GPUs.

```bash
OPENROUTER_API_KEY=... \
MODEL_PATH=/data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
ENGINE_KWARGS='{"tensor_parallel_size":1,"gpu_memory_utilization":0.88,"max_model_len":131072,"engine_prefill_chunk_size":4096,"sparse_method":"vanilla"}' \
bash benchmark/claw_eval/run_sparsevllm_claw_eval.sh
```

The shim supports text chat completions only. It explicitly rejects streaming,
server-side tool/function calling, and multimodal content so unsupported
Claw-Eval splits fail visibly instead of producing misleading scores.

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
