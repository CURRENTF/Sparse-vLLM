# Official experiments

Reproducible experiments and figure assets maintained with Sparse-vLLM.
The directory name identifies project-maintained experiments; it does not
assert algorithmic equivalence to a baseline author's official implementation.

- [Sparse-vLLM vs Vortex](sparsevllm_vs_vortex/README.md): guarded single-card QuEST and H2O-like
  comparisons, full-residency decode timing, configs, recorded JSON data and plots.

Keep experiment-specific orchestration and adapters here. Reuse canonical
benchmark entrypoints and shared metric definitions. Pass machine-specific paths
as arguments; keep checkpoints, compiler caches, and large raw run artifacts
outside the repository. Recorded results must retain source identity and caveats.
