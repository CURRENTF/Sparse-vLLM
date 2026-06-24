# Repo Skills

This repository includes a repo-local Codex skill.

## Available skills

- `add-sparse-method`: Add or refactor a first-class Sparse-vLLM sparse method following this repo's architecture. Use when Codex needs to introduce a new `vllm_sparse_method`, move method logic out of `attention.py` or `utils/`, add method-specific cache metadata or decode-time view building, and preserve the cache-manager-first design. File: `.agents/skills/add-sparse-method/SKILL.md`

## How to use

- In this repo, invoke the skill as `$add-sparse-method`.
- Keep method-specific runtime state in `src/sparsevllm/engine/cache_manager/`.
- Keep `src/sparsevllm/layers/attention.py` generic and hook new methods through shared cache-manager interfaces when possible.

# Project Structure

- `src/sparsevllm/`: core Sparse-vLLM implementation, including engine, cache managers, layers, models, and Triton kernels.
- `src/deltakv/`: DeltaKV-specific modeling, training, analysis, and config code.
- `benchmark/`: benchmark drivers and evaluation scripts for text, multimodal, and SCBench-style workloads.
- `scripts/`: analysis, validation, profiling, debugging, and benchmark entry scripts.
- `tests/`: unit and contract tests for runtime behavior, checkpoints, and research invariants.
- `docs/`: design notes, configuration guides, benchmarking docs, and governance notes.
- `baselines/`: external baseline implementations and wrappers.

# Task Running Rules

1. Before running a task, check whether each device is idle. Select an idle device when one is available. If all devices are busy, wait first; if the wait becomes too long, report the situation instead of starting the task on a busy device.

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
2. **Format**: Use the format `<type>: <short description>` (e.g., `feat: add hybrid attention support`).
3. **Allowed Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
4. **Style**:
   - Write the description in English, using the imperative mood (e.g., "add" not "added").
   - Start the description with a lowercase letter.
   - Do NOT end the message with a period.
   - Keep the entire line under 50 characters.
5. **Constraint**: Return ONLY the single line of the commit message. Do NOT include any markdown code blocks, introductory text, or explanations.
