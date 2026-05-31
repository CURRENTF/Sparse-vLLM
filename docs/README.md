# Sparse-vLLM Documentation

This directory contains stable user-facing documentation: setup guides,
feature descriptions, architecture notes, configuration references, and
benchmark runbooks.

Developer logs, dated implementation notes, and local experiment records live
outside this tree under [`../dev_docs/`](../dev_docs/). Keep new temporary
records there so `docs/` stays useful as the public project guide.

## Stable Docs

- [Getting Started](getting_started/README.md): installation, checkpoint
  download, and a minimal Sparse-vLLM usage example.
- [Features](features/README.md): sparse method taxonomy and DeltaKV notes.
- [Design](design/README.md): repository layout, runtime flow, and method
  ownership boundaries.
- [Configuration](configuration/README.md): canonical runtime parameters and
  backend-specific semantics.
- [Benchmarking](benchmarking/README.md): throughput, LongBench, MathBench,
  and multimodal benchmark entrypoints.
- [Governance](governance/README.md): reliability rules for research code.

## Reference Docs

- [Research code guidelines](governance/research-code-guidelines.md)
- [HF vs Sparse-vLLM backend parameter guide](configuration/hf-vs-sparsevllm-parameter-guide.md)

## Benchmark Runbooks

- [LLaVA-OneVision visual-cache benchmarks](benchmarking/multimodal/llava-onevision-visual-cache-benchmarks.md)
- [LLaVA-OneVision StreamingBench](benchmarking/multimodal/llava-onevision-streamingbench.md)
- [LLaVA-OneVision Video-MME](benchmarking/multimodal/llava-onevision-videomme.md)
- [LLaVA-OneVision vanilla batch benchmarks](benchmarking/multimodal/llava-onevision-vanilla-batch-benchmarks.md)
- [LLaVA-OneVision ReKV-style QA-Ego4D](benchmarking/multimodal/llava-onevision-rekv-qaego4d.md)

## Developer And Experiment Archive

- [Developer documentation archive](../dev_docs/README.md): dated code-change
  notes, local experiment records, repository review notes, and maintenance
  TODOs that are useful for audits but not stable user documentation.
