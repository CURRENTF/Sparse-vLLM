# Sparse-vLLM Agent Guide

This is a research codebase. Optimize for correct, reproducible results and
minimal, readable implementations.

## Engineering Principles

- Prefer the simplest implementation that remains clear and efficient. Use a
  one-liner when it fully and clearly expresses the logic; otherwise, do not
  compress code at the expense of readability, correctness, debuggability, or
  performance. Avoid unnecessary wrappers, helpers, layers, and boilerplate.
- Design for maintainability where responsibilities or extension paths are real.
  Give components clear ownership and stable boundaries; keep shared mechanisms
  separate from method-, backend-, or experiment-specific implementations. Split
  growing files by coherent responsibility instead of appending unrelated logic.
  Add abstractions only when they simplify current code or serve multiple concrete
  implementations.
- Preserve existing behavior outside the requested scope.
- Handle failures that can occur at real input, system, or experiment boundaries.
  Surface them clearly; avoid speculative guards, fallbacks, retries, recovery,
  or compatibility branches for scenarios the system does not support.
- Resolve routine ambiguity by inspecting the code and making a reasonable,
  explicit assumption. Ask only when the choice would materially change behavior
  or scope.
- Validate changed behavior in proportion to its risk. Keep experiment semantics
  stable unless the task explicitly changes them.
- Comment only non-obvious research logic, tensor shapes, algorithmic choices, or
  paper-specific behavior.

## Repository Skills

Use the relevant repo skill when its trigger matches the task:

- `$add-sparse-method` (`.agents/skills/add-sparse-method/SKILL.md`): add or
  refactor a first-class sparse method.
- `$code-review` (`.agents/skills/code-review/SKILL.md`): review diffs, PRs,
  runtime changes, benchmarks, or documentation.
- `$review-operator-organization`
  (`.agents/skills/review-operator-organization/SKILL.md`): review operator,
  provider, platform, or kernel boundaries. `$code-review` loads it when needed.
- `$optimize-sparsevllm-kernel`
  (`.agents/skills/optimize-sparsevllm-kernel/SKILL.md`): implement, tune,
  profile, or integrate GPU kernels.

## Architecture Invariants

- Cache managers own physical KV storage, slot allocation and accounting,
  method-specific persistent cache metadata, and physical prefill/decode view
  construction.
- `SparseController` owns logical sparse selection, cross-layer observation,
  attention-score coordination, and scheduler-facing sparse orchestration. Keep
  physical cache metadata and slot lifetimes in cache managers.
- Keep `src/sparsevllm/layers/attention.py` method-agnostic. It executes the
  shared prefill/decode sequence through generic controller, cache-manager, and
  operator interfaces; add a reusable hook instead of a method-specific branch.
- Treat the cache manager's `MemoryOracle` interfaces as the single source of
  truth for cache capacity, reservation, and admission. The scheduler consumes
  these interfaces instead of duplicating method-specific memory accounting.
- Keep canonical method names, aliases, runtime compatibility, attention
  contracts, and prefill policies in `src/sparsevllm/method_registry.py`.
  Normalize public inputs once; configs, serving code, schedulers, and benchmark
  scripts must not redefine method semantics.
- Models express stable operator semantics. Provider selection, optional
  dependencies, kernels, workspaces, and physical weight layouts belong in
  `src/sparsevllm/operators/` or provider-owned modules and are resolved before
  the forward hot path.

## Experiments and Benchmarks

- Before starting a GPU workload, check device idleness and use an idle device.
  If every device is busy, wait or report the conflict instead of sharing a busy
  device.
- For efficiency benchmarks, follow the canonical
  [English](docs/en/benchmarking/efficiency.md) runbook, including its matched
  traces, artifact validation, and metric interpretation. Sampled GPU activity is
  not theoretical MFU/MBU; use the documented Nsight diagnostic for attribution.
- Do not hardcode private local or remote paths in test scripts; pass them as
  arguments or variables. `scripts/tmp/` is exempt.

For evaluation pipelines that score individual samples:

- Record each sample as `success`, `invalid_input`, `model_failed`,
  `parse_failed`, `metric_failed`, or `skipped_by_policy`.
- Save raw outputs, parsed outputs, per-sample results, aggregate metrics, and the
  run configuration needed to reproduce the result.
- Keep retries and parsing attempts bounded, and do not silently change metrics or
  sample inclusion rules.

## Git

Use Conventional Commit titles in the form `<type>: <description>`, where `type`
is `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, or `chore`. Write
the English description in lowercase imperative mood and keep the title under
200 characters.
