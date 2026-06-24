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
