# Repo Skills

This repository includes repo-local Codex skills.

## Available skills

- `add-sparse-method`: Add or refactor a first-class Sparse-vLLM sparse method following this repo's architecture. Use when Codex needs to introduce a new `vllm_sparse_method`, move method logic out of `attention.py` or `utils/`, add method-specific cache metadata or decode-time view building, and preserve the cache-manager-first design. File: `.agents/skills/add-sparse-method/SKILL.md`
- `code-review`: Review Sparse-vLLM diffs for correctness, sparse-runtime and operator architecture, scheduling semantics, reproducibility, performance, and tests. Use when reviewing PRs, git diffs, sparse method integrations, operator/provider or kernel changes, cache-manager or scheduler changes, benchmark/evaluation scripts, OpenAI serving changes, or when the user asks for a code review. File: `.agents/skills/code-review/SKILL.md`
- `review-operator-organization`: Review operator/provider boundaries, device capability selection, kernel ownership, dependency compatibility, weight layouts, fallback semantics, and validation. Use for changes under `operators/`, `platforms/`, Triton kernels, external kernel integrations, or model-to-operator call sites. File: `.agents/skills/review-operator-organization/SKILL.md`
- `optimize-sparsevllm-kernel`: Find, implement, tune, profile, and integrate Sparse-vLLM GPU kernels across Triton, TileLang, CUDA/CuTe, and external SGL providers. Use for kernel hotspots, fusion, correctness baselines, microbenchmarks, Nsight Compute analysis, provider integration, or matched end-to-end performance validation. File: `.agents/skills/optimize-sparsevllm-kernel/SKILL.md`

## How to use

- In this repo, invoke the sparse-method skill as `$add-sparse-method`.
- In this repo, invoke the review skill as `$code-review`.
- Invoke focused operator reviews as `$review-operator-organization`;
  `$code-review` loads it automatically for relevant diffs.
- Invoke the end-to-end kernel workflow as `$optimize-sparsevllm-kernel`; it
  loads only the selected DSL and profiling references.
- Keep method-specific runtime state in `src/sparsevllm/engine/cache_manager/`.
- Keep `src/sparsevllm/layers/attention.py` generic and hook new methods through shared cache-manager interfaces when possible.

# Task Running Rules

1. Before running a task, check whether each device is idle. Select an idle device when one is available. If all devices are busy, wait first; if the wait becomes too long, report the situation instead of starting the task on a busy device.
2. Do not hardcode private paths (including local machine paths and remote paths) in test scripts; pass them via variables or arguments instead. Scripts located under `scripts/tmp/` are exempt from this restriction.
3. When using a conda environment, activate it or use `conda run`; invoking only its absolute `python` path does not expose environment-provided executables such as `ninja` to child processes.

# Standardized Efficiency & Performance Benchmark Suite

The canonical runbooks are:

- [English efficiency benchmark runbook](docs/en/benchmarking/efficiency.md)
- [简体中文效率基准运行手册](docs/zh/benchmarking/efficiency.md)

Follow the runbook's matched-trace, idle-GPU, artifact-validation, and metric-
interpretation rules. Do not treat sampled GPU activity as theoretical MFU/MBU;
use the documented Nsight diagnostic for kernel-timeline attribution.


# Research Code Skill

You are writing research code, not production SaaS code.

Primary goals:
1. Make experiments reproducible.
2. Make results easy to verify.
3. Keep implementation minimal and readable.
4. Avoid hiding failures.

Rules:
- Prefer simple, explicit code over abstraction-heavy frameworks.
- Do not introduce new dependencies unless necessary. If necessary, explain why.
- Do not add broad fallback logic, silent exception handling, or auto-recovery paths unless explicitly requested.
- Do not mask errors with default values, random substitutes, empty outputs, or warning-only behavior.
- Fail fast with clear error messages when required files, configs, checkpoints, datasets, or API keys are missing.
- Keep changes scoped to the requested experiment or bug.
- Preserve existing experiment semantics unless the user explicitly asks to refactor.
- Add comments only for non-obvious research logic, tensor shapes, algorithmic choices, or paper-specific details.

# Research Code Reliability Rules

This is a research codebase. The priority is trustworthy experimental results.

1. Do not hide failures. Missing files, bad configs, failed API calls, parse errors, and metric errors must be explicit.
2. Do not add fallback behavior unless requested. Any fallback must be opt-in, logged, and reflected in final results.
3. Every evaluated sample must have an explicit status: success, invalid_input, model_failed, parse_failed, metric_failed, or skipped_by_policy.
4. Save raw outputs, parsed outputs, per-sample results, and aggregate metrics separately.
5. Do not change metric definitions or sample inclusion rules unless explicitly requested.
6. Bound all retries, loops, API calls, and parsing attempts.
7. Validate inputs at config, dataset, model-loading, parsing, and metric boundaries.
8. Save enough run information to reproduce the experiment: config, command, model, dataset split, prompt, decoding parameters, seed, and sample count.
9. Make the smallest correct change. Avoid unrelated refactors, new dependencies, and renamed interfaces.

# Git Rules

## Git Commit Messages Rules
1. **Specification**: Strictly follow the Conventional Commits specification.
2. **Format**: Use the format `<type>: <description>`.
3. **Allowed Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
4. **Style**:
   - Write the description in English, using the imperative mood (e.g., "add" not "added").
   - Start the description with a lowercase letter.
   - Keep the entire line under 200 characters.
