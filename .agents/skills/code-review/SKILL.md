---
name: code-review
description: Review Sparse-vLLM diffs for correctness, sparse-runtime architecture, scheduling semantics, reproducibility, performance, and tests. Use when reviewing PRs, git diffs, sparse method integrations, cache-manager or scheduler changes, benchmark/evaluation scripts, or when the user asks for a code review; if no range is specified, diff the current branch against main.
---

# Code Review

## Workflow

### Step 1: Determine the Diff

Use the user's SHAs, patch, or PR when provided. Otherwise:

```bash
git fetch origin main --quiet
CURRENT_BRANCH=$(git branch --show-current)
MERGE_BASE=$(git merge-base origin/main HEAD)
git diff --stat "$MERGE_BASE..HEAD"
git diff "$MERGE_BASE..HEAD"
```

If `CURRENT_BRANCH` is `main`, ask which commits or files to review.

### Step 2: Load Review Standards

Read [svllm-review-standards.md](references/svllm-review-standards.md).

For sparse-method additions or refactors, also read [`$add-sparse-method`](../add-sparse-method/SKILL.md). For policy changes, inspect `src/sparsevllm/method_registry.py`, `src/sparsevllm/engine/scheduler.py`, and `tests/test_prefill_schedule_policy.py`.

For OpenAI-compatible serving or client changes, inspect `src/sparsevllm/entrypoints/openai/api_server.py`, `src/sparsevllm/entrypoints/openai/client.py`, `src/sparsevllm/sampling_params.py`, `src/sparsevllm/layers/sampler.py`, and `tests/test_openai_api_server.py`.

### Step 3: Review

Prioritize:

- inference correctness and tensor/cache invariants
- Sparse-vLLM architecture boundaries
- prefill policy, long/short split, and `long_bs1full_short_batch`
- OpenAI-compatible request lifecycle, streaming, cancellation, and sampling contracts
- research reproducibility and fail-fast behavior
- hot-path performance and tests

### Step 4: Report

Lead with findings, highest severity first. Use absolute file paths and line numbers.

```text
[P1] Short title
File: /absolute/path/to/file.py:123
What: ...
Why: ...
How: ...
```

Severity:

- `P0`: invalidates inference, corrupts results, or hides experiment failure.
- `P1`: likely bug, architecture violation, scheduler/policy regression, or serious performance regression.
- `P2`: meaningful test, reproducibility, edge-case, or maintainability gap.
- `P3`: minor clarity, style, or docs issue.

End with:

- `Open Questions` only when they affect correctness or confidence.
- `Summary` in 1-3 sentences.
- `Validation Notes` with commands run and missing coverage.
- `Assessment`: `Ready to merge? Yes / No / With fixes`, plus concise reasoning.

## Rules

- Do not approve without reading relevant code and tests.
- Do not comment broadly on files outside the diff unless needed to explain the changed behavior.
- Do not treat benchmark results as trustworthy if method, policy, config, command, checkpoint, sample status, or outputs are missing.
- If no issues are found, say so clearly and mention residual test or benchmark risk.
