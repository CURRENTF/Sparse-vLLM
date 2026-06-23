---
name: sparse-change-review
description: Review Sparse-vLLM code, config, benchmark, scheduler, cache-manager, attention, or evaluation changes with a mandatory independent subagent pass. Use after any repo change that may affect sparse method behavior, long/short prefill scheduling, `long_bs1full_short_batch`, cache-manager ownership, runtime correctness, benchmark results, or research-result integrity.
---

# Sparse Change Review

Use this as a review gate after Sparse-vLLM changes. This skill does not replace implementation skills such as `$add-sparse-method`; it checks that the completed diff still follows the repo architecture and research-code reliability rules.

## Mandatory Subagent Gate

If this skill is invoked and there are any code, config, test, benchmark, script, or documentation changes to review, an independent subagent review is required before final response.

- Use the available subagent or multi-agent tool. If no subagent tool is visible, search for one with `tool_search` before giving up.
- Give the subagent only the minimum review context: changed files, diff, and the checklist below. Do not pass your intended fix, prior conclusions, or a defensive explanation.
- Instruct the subagent to review only and not edit files.
- Treat subagent output as review evidence. Verify each finding yourself before changing code or reporting it.
- If no subagent tool is available, do not claim the review gate passed. Report that the review is blocked by missing subagent capability.

## Review Inputs

Collect the current change surface before review.

```bash
git status --short
git diff --stat
git diff
git diff --cached --stat
git diff --cached
git ls-files --others --exclude-standard
```

For untracked files, include file contents or add intent-to-add with `git add -N <path>` before collecting `git diff`.

Every changed path must be visible to the subagent. If the diff is too large, send all changed paths plus targeted diffs or content summaries for each path, prioritizing runtime, scheduler, cache manager, attention, benchmark, and evaluation files for full diffs. If any changed path cannot be covered, the review gate is blocked by incomplete subagent coverage.

## Sparse-vLLM Checklist

Check every applicable item.

1. `src/sparsevllm/layers/attention.py` remains generic. Method-specific branches are allowed only when they introduce a reusable hook that cannot live in cache-manager or controller code.
2. Method-specific runtime state lives in `src/sparsevllm/engine/cache_manager/`, not in `src/sparsevllm/utils/` or ad-hoc attention/controller fields.
3. `src/sparsevllm/engine/sparse_controller.py` owns scheduling and cross-layer coordination only, not persistent per-method cache metadata.
4. `src/sparsevllm/method_registry.py` remains the source of truth for method prefill policy.
5. `long_bs1full_short_batch` is used only when the method requires a complete long-prefill pass. Long requests must run full-prefill with batch size 1; short requests must still use chunked batching.
6. Long/short split logic preserves method semantics for mixed batches and does not silently downgrade long-prefill correctness.
7. Benchmark scripts and one-off configs do not override a method's prefill policy unless the user explicitly requested an ablation.
8. Any prefill policy change includes a focused test update, especially `tests/test_prefill_schedule_policy.py` when applicable.
9. Evaluation and benchmark code keeps raw outputs, parsed outputs, per-sample results, and aggregate metrics separate when it produces research results.
10. Failures remain explicit: no silent defaults, unbounded retries, warning-only missing data, or metric inclusion changes hidden behind fallback logic.

## Review Output

Lead with concrete findings, ordered by severity, with file and line references. If there are no findings, say so directly and mention any validation or subagent-review limitations.

Before final response, ensure one of these is true:

- The subagent review completed and all accepted findings were addressed or explicitly reported.
- There were no changes to review.
- The review is blocked because no subagent capability is available.
