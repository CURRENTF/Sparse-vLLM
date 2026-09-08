# Official experiment packages

Each experiment has a self-contained directory for its orchestration scripts,
parameter JSON, plotting code, and lightweight result exports. Drivers call the
canonical benchmark entrypoints; they do not implement a separate timing engine.
Supply machine-specific model, environment, checkout, and output paths at runtime.
Large logs, token outputs, checkpoints, and source archives remain outside Git.

- [128K input / 2K output decode capacity](decode_capacity_128k2k/README.md):
  two models, five methods, exact concurrency boundaries, linear/log-y figures.

Here “official” identifies the maintained experiment recipe, not universal
performance guarantees or parity with an upstream method implementation.
