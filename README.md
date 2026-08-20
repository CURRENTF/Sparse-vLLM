<div align="center">
  <img src="docs/assets/logo.png" alt="Sparse-vLLM" style="width:42%; height:auto;">

  <p>
    <a href="https://deepwiki.com/CURRENTF/Sparse-vLLM"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <a href="https://arxiv.org/abs/2602.08005"><img src="https://img.shields.io/badge/arXiv-2602.08005-b31b1b.svg" alt="arXiv"></a>
    <a href="https://arxiv.org/pdf/2602.08005.pdf"><img src="https://img.shields.io/badge/PDF-download-brightgreen.svg" alt="PDF"></a>
  </p>
</div>

<p align="center">English | <a href="README_zh.md">简体中文</a></p>

A sparse-first inference engine for long-context LLM serving.

<div align="center">
  <img src="docs/assets/sparse_vllm_throughput.png" alt="Sparse-vLLM throughput" style="width:86%; height:auto;">
</div>

## Project Overview

Sparse-vLLM is an inference framework built with sparsity as the first design principle. Instead of layering sparse methods on top of a conventional KV cache, it rethinks cache layout, controller flow, and kernels so that multiple sparse mechanisms can plug in cleanly.

> **Note:** DeltaKV compressor training code is maintained separately in
> [CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV). This repository only
> keeps the native DeltaKV inference implementation under `src/sparsevllm/`;
> it does not include DeltaKV training code or an HF reference implementation.

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
`snapkv`, `h2o`, `pyramidkv`, `omnikv`, `quest`, and `deltakv`.

| Method | Type | Short Description |
| --- | --- | --- |
| `vanilla` | Dense baseline | Runs full attention and keeps the standard KV cache behavior for correctness and performance baselines. |
| `streamingllm` / `attention-sink` | Physical eviction | Keeps fixed sink tokens plus a recent window, then physically evicts older tokens outside that policy. |
| `snapkv`, `pyramidkv` | Physical eviction | Selects important historical tokens during prefill/finalization and stores only the retained KV tokens. |
| `h2o` | Physical eviction | Accumulates normalized token importance in one score vector per layer/sequence. Prefill scores and physically evicts after every chunk; decode scores every token and normally selects, compacts, and frees in a burst when a row reaches `h2o_decode_budget + h2o_decode_eviction_interval`, then returns it to `h2o_decode_budget` (interval default: 128). If a decode step consumes the final free KV slot, all active decode rows already over budget compact early, including temporarily unscheduled rows, so the next step can proceed. Idle chain rows are excluded. The budget-plus-interval sum must be divisible by 64 for the scored decode kernel. Each eviction retains heavy hitters plus a recent suffix. Prefill uses normalized attention mass; the optimized decode path max-reduces raw QK logits across query heads before token-wise normalization. `h2o_prefill_budget` controls intermediate prefill rows, `h2o_recent_ratio` splits each budget, and `h2o_prefill_score_window` controls chunk scoring. |
| `omnikv` | Logical masking | Keeps tokens in storage but masks the attention read view so sparse layers attend only selected context. |
| `quest` | Query-aware selection | Uses decode-time query-aware page selection while keeping prefill dense. |
| `deltakv` / `deltakv-*` | Hybrid compression | Keeps a small full-precision pool and stores older context through DeltaKV compression or related ablations. |

Read the method overview and integration rules in
[Core Sparse Methods](docs/en/features/sparse-methods.md).

## Supported Models

| Model | Supported |
| --- | :---: |
| Qwen2.5 | ✅ |
| Qwen3 | ✅ |
| Qwen3MoE | ✅ |
| Qwen3.5 / 3.6 / 3.8 | ✅ |
| Qwen3.5 / Qwen3.6 MoE | ✅ |
| GLM-4.7-Flash | ✅ |
| Gemma 4 Dense / MoE | ✅ |
| Llama 3 / 3.1 | ✅ |
| MiniMax M2.7 | ✅ |

See [Supported Models](docs/en/features/supported-models.md) for the precision,
parallelism, and sparse-method compatibility matrices.

Native image, video, and audio inputs are enabled per checkpoint with
`enable_multimodal=True`; see the supported-model matrix for media coverage.

## Documentation

| Topic | Link |
| --- | --- |
| Quick setup and minimal usage | [Getting Started](docs/en/getting_started/README.md) |
| Model, precision, and parallelism support | [Supported Models](docs/en/features/supported-models.md) |
| Sparse method taxonomy and extension rules | [Core Sparse Methods](docs/en/features/README.md) |
| Runtime architecture | [Architecture](docs/en/design/README.md) |
| Runtime parameter semantics | [Runtime Parameter Semantics](docs/en/configuration/runtime-parameter-semantics.md) |
| Benchmark commands | [Benchmarks](docs/en/benchmarking/README.md) |
| DeltaKV inference | [DeltaKV](docs/en/features/deltakv.md) |
| Reproducibility checklist | [Reproducibility](docs/en/getting_started/reproducibility.md) |

The full documentation index is maintained in [docs/en/README.md](docs/en/README.md).

## Quick Start

Sparse-vLLM requires Python 3.10 or newer. Default dependencies are declared in
`pyproject.toml`.

### Conda

```bash
conda create -n svllm python=3.10 -y
conda activate svllm

CUDA_VERSION=cu130
python -m pip config --site set global.extra-index-url \
  "https://download.pytorch.org/whl/${CUDA_VERSION} https://flashinfer.ai/whl/${CUDA_VERSION}"
python -m pip install -e ".[${CUDA_VERSION}]"

# Optional
MAX_JOBS=8 pip install flash-attn --no-build-isolation
pip install flashinfer-cubin --index-url https://flashinfer.ai/whl
```

PyTorch wheels include their CUDA runtime, while compiled extensions such as
`flash-attn` use the CUDA toolchain active in the environment.

### uv

```bash
uv venv --python 3.10
source .venv/bin/activate

uv pip install -e ".[cu130]"

# Optional
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
uv pip install flashinfer-cubin --index-url https://flashinfer.ai/whl
```

Use `cu129` instead of `cu130` for CUDA 12.9.

`einops`, `sglang-kernel`, and the training, benchmark, and test packages are all
part of the main installation; no workflow-specific extras are required.

Sparse-vLLM supports Qwen3.5/Qwen3.6/Qwen3.8 checkpoints in unquantized BF16
and block-scaled FP8 formats. These releases share the `qwen3_5` runtime
architecture and the same precision, parallelism, sparse-method, and
multimodal support.

Their prefill causal Conv1D and decode Conv1D/GDN packing paths use
repository-local Triton kernels; they do not call `sglang-kernel` themselves.

For the full dependency list and a minimal `LLM(...)` example, see
[Getting Started](docs/en/getting_started/README.md).

## Benchmarks

Use `scripts/benchmarks/bench_sparse_vllm.py` for throughput measurements and
the `benchmark/` entrypoints for LongBench, MathBench, SCBench, NIAH, and
multimodal evaluations.

See [Benchmarks](docs/en/benchmarking/README.md) for command examples and backend notes.

## Contributing Sparse Methods

New sparse methods should keep method-specific runtime state in
`src/sparsevllm/engine/cache_manager/` and keep
`src/sparsevllm/layers/attention.py` generic.


## Acknowledgements

This project is inspired by and/or references ideas and implementation techniques from:

- `LightLLM` (`ModelTC/LightLLM`)
- `SGLang` (`sgl-project/sglang`)
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
