# Sparse-vLLM Review Standards

Use these checks for Sparse-vLLM's Python/Triton research engine.

## Architecture

- Method-specific runtime state belongs in `src/sparsevllm/engine/cache_manager/`.
- `src/sparsevllm/layers/attention.py` should stay generic and use hooks such as `on_kv_stored(...)` and `build_decode_view(...)`.
- `SparseController` owns cross-layer observation, attention-score collection, and scheduler-facing coordination, not method-owned cache metadata.
- Public APIs should use `sparse_method`; Sparse-vLLM normalizes to internal `vllm_sparse_method`.
- New first-class methods should update config, `method_registry.py`, cache-manager routing, exports, docs, and policy tests.

Flag method logic hidden in `utils/`, method branches added directly to `attention.py`, or benchmark scripts redefining method semantics.

## Scheduling

Prefill policy is registry-owned in `src/sparsevllm/method_registry.py`.

- `all_chunked`: all prefill requests are capped by `chunk_prefill_size` and normal scheduler limits.
- `long_bs1full_short_batch`: long requests run one complete prefill with batch size 1; short requests remain chunked and batchable.

Review long/short split carefully. Use `long_bs1full_short_batch` only for methods that need complete long-prefill before sparse/cache transformation, such as PyramidKV and DeltaKV-family methods. Policy, bucket, admission, or decode-priority changes need focused coverage in `tests/test_prefill_schedule_policy.py`.

## Correctness and Cache Accounting

Check tensor shapes, dtypes, devices, strides, `slot_mapping`, `active_slots`, `req_indices`, `context_lens`, and `cu_seqlens_q` across prefill/decode.

Cache-manager capacity hooks must agree: `num_free_slots`, `reserved_prefill_slots(...)`, `prompt_admission_costs(...)`, `prompt_admission_budgets(...)`, and `prefill_step_free_slots(...)`. Partial prefill, preemption, temp slots, staging views, and decode scratch must not leak slots or drop sequences.

Prefer fail-fast errors with method/cache-manager name, prompt length, needed/free slots, and budgets.

## Research Reliability

Evaluation and benchmark code must not hide failures.

- Every evaluated sample needs a status: `success`, `invalid_input`, `model_failed`, `parse_failed`, `metric_failed`, or `skipped_by_policy`.
- Save raw outputs, parsed outputs, per-sample results, and aggregate metrics separately.
- Save enough run metadata: config, command, model, split, prompt, decoding params, seed, sample count, method, prefill policy, chunk size, prompt length, batch size, and checkpoint path when relevant.
- Keep retries, loops, API calls, parsing, and benchmark runs bounded.
- Do not silently change metric definitions or sample inclusion.

## Performance and Validation

Hot paths should avoid unnecessary CPU/GPU sync, host transfers, broad logging, repeated allocation, large tensor copies, and Python per-token loops. Kernel changes must preserve masks, bounds, block sizes, GQA/MHA behavior, and score collection.

Expected validation:

- Python changes: `python -m py_compile` touched files.
- Scheduler/policy changes: `tests/test_prefill_schedule_policy.py`.
- Runtime parameter changes: `tests/test_runtime_param_normalization.py`.
- Research fail-fast changes: `tests/test_research_fail_fast.py`.
- Sparse method changes: one correctness-oriented run before throughput benchmarks.

Before GPU work, check device idleness and use an idle device when available.
