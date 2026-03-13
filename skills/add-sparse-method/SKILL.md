---
name: add-sparse-method
description: Add or refactor a first-class Sparse-vLLM sparse method alongside vanilla, SnapKV, OmniKV, QuEST, and DeltaKV. Use when Codex needs to introduce a new `vllm_sparse_method`, move method logic out of `attention.py` or `utils/`, add method-specific cache metadata or decode-time view building, wire config and registration, and preserve the repo's cache-manager-first architecture.
---

# Add Sparse Method

Implement new Sparse-vLLM methods as explicit runtime methods, not as ad-hoc helpers. Keep `attention.py` generic, let `cache_manager` own method state, and let `SparseController` keep scheduling and cross-layer coordination responsibilities.

## Start Here

Read [references/file-map.md](references/file-map.md) before changing code.

Read [references/quest-pattern.md](references/quest-pattern.md) when the new method:
- stores persistent metadata
- depends on decode-time `q`
- needs a custom `build_decode_view(...)`
- is being added as a full peer to `omnikv`, `snapkv`, `quest`, or `deltakv`

## Placement Rules

Follow this placement order.

1. Put the method's core runtime logic in `src/sparsevllm/engine/cache_manager/<method>.py` when the method owns any persistent state.
2. Put persistent page, chunk, token, or compressed metadata in the cache manager, not in `utils/`.
3. Use `CacheManager.on_kv_stored(...)` when metadata must be updated after KV is written.
4. Use `CacheManager.build_decode_view(...)` when the method needs the current layer's decode-time `q`.
5. Keep `src/sparsevllm/layers/attention.py` method-agnostic. It may call generic hooks, but should not grow method-specific branches unless adding a new reusable hook.
6. Put cross-layer observation, attention-score collection, or scheduler-facing sparse orchestration in `src/sparsevllm/engine/sparse_controller.py`.
7. Use `src/sparsevllm/utils/` only for truly generic helpers shared by multiple methods. Do not place an entire method implementation there.
8. Add custom kernels under `src/sparsevllm/triton_kernel/` or another explicit runtime module, then call them through the method's cache manager or shared decode path.

## Decision Rules

Use these rules before editing code.

- If the method only changes logical token selection and has no persistent state, reuse the existing cache manager if possible and prefer `SparseController`.
- If the method keeps method-specific metadata across steps, create a dedicated cache manager file.
- If the method needs the current `q`, do not try to force the logic into `prepare_forward()` or `get_read_view()` alone. Implement a decode-time hook through `build_decode_view(...)`.
- If the method changes physical KV allocation or page ownership, keep that logic in the cache manager and register it as a first-class method.
- If the method is meant to become a repo-supported algorithm, do not hide it behind a one-off helper in `attention.py`.

## Editing Workflow

1. Add config knobs to `src/sparsevllm/config.py`.
2. Register the method in `src/sparsevllm/engine/cache_manager/base.py` and `src/sparsevllm/engine/cache_manager/__init__.py` if needed.
3. Create or update `src/sparsevllm/engine/cache_manager/<method>.py`.
4. Touch `src/sparsevllm/engine/sparse_controller.py` only for controller responsibilities.
5. Touch `src/sparsevllm/layers/attention.py` only to use generic hooks or shared kernels.
6. Update README and benchmark examples after the method runs.
7. Compile touched Python files with `python -m py_compile`.
8. Run at least one correctness-oriented task and one throughput benchmark.

## Architecture Guardrails

Do not do these things.

- Do not put a new method's main logic in `src/sparsevllm/utils/`.
- Do not add method-specific decode branches directly inside `Attention.forward()` when a cache-manager hook can express the same behavior.
- Do not make `SparseController` own persistent cache metadata that belongs to one method.
- Do not couple README claims to behavior that the code does not implement.

## Validation

Compile first.

```bash
python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/engine/cache_manager/__init__.py \
  src/sparsevllm/engine/cache_manager/<method>.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/layers/attention.py
```

Benchmark after correctness is established.

```bash
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL_PATH> \
  --methods <method> \
  --lengths 128000 \
  --batch_sizes 8 \
  --output_len 128 \
  --hyper_params '{"gpu_memory_utilization":0.9,"chunk_prefill_size":4096,"tensor_parallel_size":1}'
```

When the user asks to add a new method, follow this skill first and only then invent method-specific details.
