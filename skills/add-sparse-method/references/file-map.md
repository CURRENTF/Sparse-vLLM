# File Map

Use this map to decide which repo files must change when adding a new Sparse-vLLM method.

## Always Review

- `src/sparsevllm/config.py`
- `src/sparsevllm/engine/cache_manager/base.py`
- `src/sparsevllm/layers/attention.py`
- `src/sparsevllm/engine/sparse_controller.py`
- `README.md`

## Add a First-Class Method

Touch these files when the method becomes a supported `vllm_sparse_method`.

- `src/sparsevllm/config.py`
  Add config fields, validation, and defaults.
- `src/sparsevllm/engine/cache_manager/<method>.py`
  Put method state, metadata, cache layout, and decode-time hooks here.
- `src/sparsevllm/engine/cache_manager/base.py`
  Register `CacheManager.create(...)` routing and add generic hooks only if the existing hooks are insufficient.
- `src/sparsevllm/engine/cache_manager/__init__.py`
  Export the new cache manager when appropriate.
- `README.md`
  Document method semantics, knobs, and benchmark examples.

## Touch `SparseController` Only for Controller Work

Edit `src/sparsevllm/engine/sparse_controller.py` when the method:

- reuses observed attention scores
- needs cross-layer propagation
- shares dynamic logical views with other layers
- changes scheduler-facing sparse state

Do not move method-owned cache metadata here.

## Touch `attention.py` Only for Generic Hooks

Edit `src/sparsevllm/layers/attention.py` when you need to:

- call a new generic cache-manager hook
- wire a new shared kernel path
- keep the store-view, read-view, and decode-view call sequence consistent

Do not bury a full method implementation in `attention.py`.

## Add Kernel Code Only When Needed

Touch `src/sparsevllm/triton_kernel/` or another explicit kernel module when:

- the existing decode or prefill kernels are the bottleneck
- the method requires a new layout-aware fused operator
- the method cannot be expressed as view selection plus existing kernels

## Minimum Validation Set

After editing code:

1. Compile touched Python files with `python -m py_compile`.
2. Run one small correctness task.
3. Run one throughput benchmark.
4. Compare against at least one existing method on the same machine.
