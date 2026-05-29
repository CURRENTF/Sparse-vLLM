# Sparse-vLLM Documentation

This directory separates stable user-facing docs from development notes and
historical experiment records.

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
- [Governance](governance/README.md): reliability rules, repo review, and
  maintenance notes.

## Reference Docs

- [Research code guidelines](governance/research-code-guidelines.md)
- [HF vs Sparse-vLLM backend parameter guide](configuration/hf-vs-sparsevllm-parameter-guide.md)

## Benchmark Runbooks

- [LLaVA-OneVision visual-cache benchmarks](benchmarking/multimodal/llava-onevision-visual-cache-benchmarks.md)
- [LLaVA-OneVision StreamingBench](benchmarking/multimodal/llava-onevision-streamingbench.md)
- [LLaVA-OneVision Video-MME](benchmarking/multimodal/llava-onevision-videomme.md)
- [LLaVA-OneVision vanilla batch benchmarks](benchmarking/multimodal/llava-onevision-vanilla-batch-benchmarks.md)
- [LLaVA-OneVision ReKV-style QA-Ego4D](benchmarking/multimodal/llava-onevision-rekv-qaego4d.md)

## Development Notes

These files preserve implementation history and exact local experiment records.
They are useful for debugging and audit trails, but they are not the primary
entrypoint for new users.

- [Experiment records](dev_notes/experiment-records.md)
- [Code change history](dev_notes/code-change-history/)
