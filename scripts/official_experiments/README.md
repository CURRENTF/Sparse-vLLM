# Official experiment packages

Each experiment has a self-contained directory for its orchestration scripts,
parameter JSON, plotting code, and lightweight result exports. Drivers call the
canonical benchmark entrypoints; they do not implement a separate timing engine.
Supply machine-specific model, environment, checkout, and output paths at runtime.
Large logs, token outputs, checkpoints, and source archives remain outside Git.

- [128K input / 2K output decode capacity](sparse_decode_efficiency/README.md):
  two models, five methods, exact concurrency boundaries, linear/log-y figures.
- [Sparse-vLLM vs Vortex](sparsevllm_vs_vortex/README.md): guarded single-card
  QuEST and H2O-like comparisons, full-residency decode timing, configs,
  recorded JSON data and plots.

Here “official” identifies the maintained experiment recipe, not universal
performance guarantees or parity with an upstream method implementation.
Recorded results must retain source identity and caveats.
