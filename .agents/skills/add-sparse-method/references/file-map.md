# Architecture And Integration Map

Use this map to locate architectural owners. "Inspect" does not mean "edit": a new method should touch only the boundaries whose contracts actually change.

## Public Configuration

| Concern | Primary source | Expected use |
| --- | --- | --- |
| Public parameter normalization | `src/sparsevllm/configs/runtime_params.py` | Map public `sparse_method` to canonical internal state; reject legacy/conflicting fields. |
| Config composition | `src/sparsevllm/configs/groups.py` | Understand how focused config groups are assembled. |
| Sparse semantics | `src/sparsevllm/configs/sparse.py` | Validate sparse-specific values and derived settings. |
| Model/layout compatibility | `src/sparsevllm/configs/model.py` | Inspect model-derived storage and attention properties. |
| Scheduling semantics | `src/sparsevllm/configs/scheduling.py` | Inspect prefill/decode scheduling constraints. |
| Prefix and graph settings | `src/sparsevllm/configs/prefix_cache.py`, `src/sparsevllm/configs/cuda_graph.py` | Validate lifecycle combinations. |
| Compatibility facade | `src/sparsevllm/config.py` | Inspect imports only; do not place new config ownership here. |

## Static Method Contract

Inspect `src/sparsevllm/method_registry.py` for:

- aliases and canonical names;
- prefill schedule defaults;
- prefill/decode attention-score contracts;
- model and parallel-topology compatibility;
- prefix-cache and decode CUDA Graph support;
- external asset requirements.

The registry describes static capabilities. Do not put mutable runtime state or allocation policy there.

## Runtime Construction And Control Plane

| Owner | Typical responsibilities |
| --- | --- |
| `src/sparsevllm/engine/model_runner.py` | Construct and connect generic runtime components. Inspect wiring; avoid method-name branches. |
| `src/sparsevllm/engine/runtime_state.py` | Expose runtime capacity, lifecycle, and scheduling-facing state. |
| `src/sparsevllm/engine/cache_manager/factory.py` | Select the registered cache manager implementation. |
| `src/sparsevllm/engine/cache_manager/base.py` | Shared allocation, typed view, and lifecycle contracts. Extend narrowly. |
| `src/sparsevllm/engine/cache_manager/` | Persistent method state, physical mutation, metadata, and view materialization. |
| `src/sparsevllm/engine/sparse_controller.py` | Per-step/cross-layer score and selection orchestration. |
| `src/sparsevllm/engine/activation_controller.py` | Hidden-state capture and activation-based method state. |
| `src/sparsevllm/engine/scheduler.py` via `MemoryOracle` | Consume generic capacity, reservation, batching, and execution-mode contracts. |

Search for the nearest method's canonical name across these owners before assuming that one factory registration is sufficient.

## Attention Data Plane And Storage

Inspect the definitions and call sites of:

- `SparseSelection`;
- `AttentionViewMeta`;
- `PrefillComputeView` and `DecodeComputeView`;
- `ExplicitKVPayload` and `MlaLatentPayload`;
- typed cache-write records;
- `AttentionCacheStorage` and its explicit, heterogeneous, and MLA implementations.

Relevant locations include `src/sparsevllm/engine/cache_manager/base.py`, cache-storage modules, `src/sparsevllm/layers/attention.py`, MLA attention code, and model attention-runtime adapters.

Attention should consume a typed compute view and payload without knowing the sparse-method name. Cache layout comes from model/runtime layout, not from a sparse-method-specific tensor convention.

## Operators And Kernels

Inspect `src/sparsevllm/operators/`, `src/sparsevllm/platforms/`, and the relevant kernel tree when the method changes compute.

- Attention compute alternatives belong behind `OpRegistry`, a typed `*OpSpec`, `DeviceCaps`, and provider selection.
- Selection, metadata, compaction, compression, and reconstruction kernels belong to the state owner and may remain method-specific.
- Provider capability and dependency validation must happen before execution or CUDA Graph capture.

Use `$review-operator-organization` for nontrivial provider/layout changes and `$optimize-sparsevllm-kernel` for implementation, tuning, profiling, or external kernel integration.

## Lifecycle Coordinators

Inspect the prefix/radix or chain coordinators, `RuntimeState`, CacheManager lifecycle methods, and CUDA Graph preparation paths when advertising any of these capabilities:

- sequence allocation, append, fork, restore, rollback, and free;
- prefix match, attach, detach, eviction, and ownership transfer;
- eager-to-captured buffer stability and replay-safe metadata updates;
- CPU/offload movement and capacity accounting.

Every persistent method tensor or metadata object must participate in the same lifecycle as the sequence/cache state it describes.

## Documentation, Tests, And Benchmarks

Inspect before editing:

- focused unit tests near config, registry, cache manager, scheduler, storage, and operators;
- method evaluation scripts/configs and documented model/assets;
- `docs/en/benchmarking/efficiency.md` or the Chinese equivalent for performance runs;
- public method documentation and support matrices.

Avoid a single method-only smoke test. Test the contracts that other components depend on.
