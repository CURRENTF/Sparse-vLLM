# Sparse-vLLM Documentation

This directory separates stable user-facing docs from development notes and
historical experiment records.

## Stable Docs

- [Architecture](architecture.md): repository layout, runtime flow, and method
  ownership boundaries.
- [Reproducibility](reproducibility.md): environment, checkpoint, data, smoke
  test, and artifact conventions for reproducing runs.
- [Runtime Parameter Semantics And Audit](runtime-parameter-semantics.md):
  canonical runtime parameters, backend differences, method routing, and
  fail-fast rules.
- [Research Code Guidelines](research-code-guidelines.md): reliability rules
  for experiments and evaluation artifacts.
- [HF vs Sparse-vLLM Backend Parameter Guide](hf-vs-sparsevllm-parameter-guide.md):
  compatibility entrypoint that redirects to the runtime parameter audit.

## Benchmark Runbooks

- [LLaVA-OneVision visual-cache benchmarks](multimodal_models_adapation/llava-onevision-visual-cache-benchmarks.md)
- [LLaVA-OneVision StreamingBench](multimodal_models_adapation/llava-onevision-streamingbench.md)
- [LLaVA-OneVision Video-MME](multimodal_models_adapation/llava-onevision-videomme.md)
- [LLaVA-OneVision vanilla batch benchmarks](multimodal_models_adapation/llava-onevision-vanilla-batch-benchmarks.md)
- [LLaVA-OneVision ReKV-style QA-Ego4D](multimodal_models_adapation/llava-onevision-rekv-qaego4d.md)

## Development Notes

These files preserve implementation history and exact local experiment records.
They are useful for debugging and audit trails, but they are not the primary
entrypoint for new users.

- [Experiment records](dev-notes/experiment-records.md)
- [Code change history](dev-notes/code-change-history/)
- [Repository review, 2026-05-12](repo-review-2026-05-12.md)
- [TODO](todo.md)
