# QuEST-Like Explicit Paged Selection

Use this reference only when the method performs query-aware logical selection over explicit paged KV and maintains persistent per-page metadata. It is not a universal template for sparse methods.

## Required Semantics

A QuEST-like integration usually has three distinct pieces:

1. CacheManager owns page metadata and keeps it synchronized with allocation, append, prefix attach/fork/restore, eviction, and free.
2. The method's runtime or cache/provider-owned selection path computes query-dependent page/token selection with an explicit score contract; `SparseController` only delegates the generic lifecycle.
3. CacheManager turns the selection into a typed decode view consumed by generic attention.

Keep query-dependent scratch state separate from persistent page metadata. Document tensor shapes, head grouping, page indexing, and whether scores describe logical tokens, physical slots, or pages.

## View Boundary

Use `build_decode_view` only when the selected result can still be represented as a logical view over the existing explicit-KV payload.

Use `build_decode_compute_view` when the method must materialize a different execution payload, reconstruct keys, or adapt a storage-specific representation. Do not force MLA latent storage or a custom compressed payload into an explicit paged-KV tuple.

The resulting view must preserve:

- sequence boundaries and token order expected by the attention operator;
- valid physical slot/page indices after compaction or prefix reuse;
- query-head to KV-head mapping;
- batch and graph-stable shapes where CUDA Graph support is advertised;
- deterministic behavior for ties or bounded approximate selection.

## Metadata Lifecycle Audit

For every persistent metadata tensor, verify:

- initialization on sequence allocation;
- updates on prefill append and decode append;
- behavior on partial pages;
- fork/copy semantics for chain and radix prefix reuse;
- rollback/restore semantics;
- cleanup on free and eviction;
- device/offload movement;
- stable buffers or replay-safe updates under CUDA Graph.

Metadata that cannot be restored or forked correctly means the corresponding prefix mode is unsupported and must be rejected by the capability contract.

## Storage And Topology Compatibility

Do not infer compatibility from the QuEST name or from a shared base class. Verify:

- explicit homogeneous versus heterogeneous KV storage;
- availability and semantics of actual-key materialization;
- TP head partitioning and any cross-rank reductions;
- model-specific rotary/key transformations;
- page size and block-table assumptions;
- whether the selected attention provider accepts the produced view.

Treat MLA latent compatibility as unsupported unless a dedicated latent-aware score and compute-view path is implemented and validated.

## Performance Rules

- Vectorize metadata updates and selection; avoid Python loops on the decode path.
- Reuse scratch buffers when their ownership and graph lifetime are explicit.
- Separate selection time, view-materialization time, and attention time in microbenchmarks.
- Compare against a correctness oracle before tuning approximate kernels.
- Do not bypass operator/provider preparation when selection changes the attention compute implementation.
