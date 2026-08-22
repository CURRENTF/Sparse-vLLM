---
name: add-sparse-method
description: Add or refactor a first-class Sparse-vLLM sparse method across configuration, capability registration, cache and controller ownership, scheduler admission, typed attention views, model and storage compatibility, CUDA Graph and prefix-cache lifecycles, operators, and reproducible validation. Use for a new canonical sparse_method or an integration whose sparse runtime semantics change; do not use for a kernel/provider optimization that leaves method semantics unchanged.
---

# Add Sparse Method

Treat a sparse method as a runtime capability contract, not as a file template. Start from the method's semantics, route each responsibility to its existing owner, and add only the narrow contracts the method actually needs.

Always read [references/file-map.md](references/file-map.md) before editing. It maps the current architecture and distinguishes files to inspect from files that usually need modification.

## 1. Classify The Change

Use this skill when the change introduces or materially changes a canonical `sparse_method`, including its state, selection, scheduling, storage, lifecycle, or compatibility semantics.

Do not create a new sparse method merely to add:

- an attention compute implementation with unchanged semantics; use the operator/provider and kernel workflow;
- a storage layout required by several methods; extend the storage/model layout contract;
- a model integration that only adapts an existing method;
- a config-only experiment or a private ablation with no first-class runtime contract.

If classification is ambiguous, write the semantic difference from the nearest existing method first. If there is no difference in runtime behavior, it is not a new method.

## 2. Declare The Capability Contract

Before editing code, complete the contract in [references/runtime-contract.md](references/runtime-contract.md). At minimum, decide:

- identity: public name, canonical internal name, aliases, defaults, and external assets;
- state semantics: logical views, physical eviction, paged metadata, compression, activation state, or combinations;
- selection semantics: when scores are produced, which queries/keys they describe, and whether state crosses layers or steps;
- storage compatibility: explicit KV, heterogeneous explicit KV, MLA latent storage, and actual-key materialization needs;
- scheduling and admission: prefill mode, batch compatibility, capacity budgets, reservation costs, and offload behavior;
- lifecycle support: eager execution, CUDA Graph, radix prefix cache, chain prefix cache, and restore/fork/free behavior;
- model and topology support: model types, TP/EP/DP assumptions, and provider requirements;
- validation evidence: correctness oracle, quality workload, performance workload, and unsupported combinations.

Do not fill unknowns with permissive defaults. Unsupported combinations must be represented explicitly and fail during validation or initialization, not in a later kernel.

## 3. Route Responsibilities By Owner

Preserve these boundaries:

- Normalize the public `sparse_method` API and reject legacy or conflicting inputs in `configs/runtime_params.py`. Keep composed config validation in the relevant config group, not the compatibility facade `config.py`.
- Register static method capabilities, aliases, schedule defaults, score contracts, model/topology compatibility, prefix-cache support, graph support, and assets in `method_registry.py`.
- Keep persistent cache-coupled method state, physical allocation, eviction/compaction, and attention-view materialization in `engine/cache_manager/`.
- Keep per-step and cross-layer score/selection orchestration in `engine/sparse_controller.py`.
- Keep hidden-state capture and activation reuse in `engine/activation_controller.py`.
- Expose admission budgets, reservation costs, execution modes, and lifecycle state through `RuntimeState` and `MemoryOracle`; the scheduler consumes this generic contract.
- Pass typed selections, views, payloads, and writes through the cache/attention boundary. Keep `layers/attention.py` and model attention call sites method-agnostic.
- Let `ModelSpec`, `RuntimeLayout`, `ParallelTopology`, and storage protocols own physical cache layout compatibility. A sparse method does not own the model's KV representation.

Do not add method-name branches to Attention, Scheduler, or ModelRunner. If a method requires behavior not expressible by an existing generic contract, add the smallest typed hook at the owning boundary. Do not introduce a broad strategy framework solely to avoid one local branch.

## 4. Choose A Mechanism, Not An Inheritance Parent

Read [references/method-families.md](references/method-families.md), then select reference implementations by shared mechanics: logical view, physical compaction, scored selection, query-aware paged selection, activation reuse, or compressed/multi-pool storage.

Existing Python inheritance often represents incidental code reuse rather than a stable method taxonomy. Reuse a base only after verifying that allocation, state, admission, graph, prefix, and storage semantics are all compatible.

For QuEST-like query-aware selection over explicit paged KV, also read [references/quest-pattern.md](references/quest-pattern.md). Do not apply that pattern to MLA latent or custom payload methods without proving the same view contract is valid.

## 5. Preserve Typed Data-Plane Contracts

Prefer the existing data types and protocols:

- `SparseSelection` for selection results;
- `AttentionViewMeta` for logical attention metadata;
- `PrefillComputeView` and `DecodeComputeView` for execution-time views;
- `ExplicitKVPayload` and `MlaLatentPayload` for storage-specific payloads;
- typed cache writes and `AttentionCacheStorage` for cache mutation and storage access.

Use `build_decode_view` when the method only changes the logical explicit-KV view. Use `build_decode_compute_view` when execution requires a different payload or materialization step. Do not pass method-specific tuples, hidden tensor side channels, or config objects through attention.

## 6. Route Kernels At The Semantic Boundary

- Put alternative attention computation behind `OpRegistry`, a typed `*OpSpec`, `DeviceCaps`, and an operator provider. Prepare and bind providers before capture when CUDA Graph is supported.
- Invoke method-internal selection, metadata, compaction, compression, or reconstruction kernels from the state owner, usually CacheManager or SparseController.
- Keep kernel adapters thin and keep policy, allocation, and lifecycle decisions out of kernel modules.
- Do not silently fall back to another provider or dense behavior. Any fallback must be an explicit registered capability with tested semantics.

## 7. Integrate In Dependency Order

Implement the smallest vertical slice in this order:

1. Public normalization and static registry contract.
2. Model/storage/topology validation.
3. State ownership, allocation, and lifecycle operations.
4. Score/selection orchestration and typed view construction.
5. Scheduler admission and prefill execution semantics.
6. Operator/provider or method-internal kernels, if required.
7. Prefix-cache, CUDA Graph, and offload integration for every advertised capability.
8. Documentation, evaluation configuration, and reproducibility artifacts.

Inspect all layers, but edit only the owners affected by the declared contract. Research-code scope matters: do not combine a method addition with an unrelated architecture rewrite.

## 8. Validate Claims, Not Just Imports

Read and follow [references/validation.md](references/validation.md). Validation must cover every capability advertised by the registry and every relevant storage/layout path.

Run cheap static and unit checks first, then targeted runtime correctness, lifecycle matrices, quality evaluation, and matched performance benchmarks. Fail fast on missing models, assets, datasets, unsupported layouts, or unavailable providers. Preserve raw outputs and per-sample status so experimental results remain auditable.
