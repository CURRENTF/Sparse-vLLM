# Sparse-vLLM Documentation

English | [简体中文](../zh/README.md)

This directory contains stable user-facing documentation: setup guides,
feature descriptions, architecture notes, configuration references, and
benchmark runbooks.

Keep `docs/` focused on stable project guides, contracts, and runbooks. Do not
add local experiment ledgers here; cite concrete repo artifacts directly when a
repo-facing result claim needs evidence.

## Stable Docs

- [Getting Started](getting_started/README.md): installation, checkpoint
  download, and a minimal Sparse-vLLM usage example.
- [Features](features/README.md): sparse method taxonomy, DeltaKV notes, and
  Qwen3MoE expert parallelism.
- [Design](design/README.md): repository layout, runtime flow, and method
  ownership boundaries.
- [Configuration](configuration/README.md): canonical runtime parameters and
  native runtime semantics.
- [Benchmarking](benchmarking/README.md): throughput, LongBench, MathBench /
  AIME / MATH-500, SCBench, Claw-Eval, multimodal, RULER-VT, NIAH, and
  regression benchmark entrypoints.
- [Governance](governance/README.md): reliability rules for research code.

## Reference Docs

- [Supported models](features/supported-models.md)
- [Research code guidelines](governance/research-code-guidelines.md)
- [Runtime parameter semantics](configuration/runtime-parameter-semantics.md)
- [Sparse-vLLM control map](design/control-map.md)

## Benchmark Runbooks

- [Benchmark inventory](benchmarking/README.md)
- [Sparse-vLLM regression tests](benchmarking/sparsevllm-regression-tests.md)
- [Multimodal benchmarks](benchmarking/multimodal/README.md)
