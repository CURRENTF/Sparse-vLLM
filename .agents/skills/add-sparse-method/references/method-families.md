# Method Families By Runtime Mechanism

Use these families to choose code to study. They describe mechanisms, not a required class hierarchy.

## Logical View Methods

Examples include standard/vanilla-like and OmniKV-like paths where physical cache ownership remains stable and the method primarily changes the logical tokens visible to attention.

Inspect these when the method can be expressed with existing storage plus `AttentionViewMeta` and a typed prefill/decode view. Prefer this path over physical mutation when semantics allow it.

## Windowed Physical Retention

StreamingLLM-like methods enforce a persistent sink/window policy and physically bound retained KV.

Audit append-time eviction, position semantics, capacity accounting, prefix interaction, and whether compaction changes physical slot identities.

## Posthoc Scored Compaction

SnapKV/PyramidKV-like methods score completed or chunked prefill state, then compact physical KV according to a budget.

Audit the prefill score contract, observation window, per-head/per-layer budgets, compaction writes, scheduler reservation peak, and restoration/prefix semantics.

## Cumulative Decode Scoring

H2O-like methods maintain scores or statistics across decode steps and use them for repeated retention decisions.

Audit score update order, new-token protection, numerical accumulation, deterministic ties, graph-safe mutation, and rollback/fork behavior.

## Query-State Reuse

RKV-like methods retain or reuse query-derived state in addition to KV metadata.

Audit ownership of query state, cross-layer lifetime, batch reordering, topology semantics, and any additional temporary admission cost. Do not assume SnapKV lifecycle behavior merely because code is inherited.

## Activation Reuse Or Layer Skipping

SkipKV-like methods use hidden-state or activation signals to choose cache behavior or skip work.

Use ActivationController for persistent/cross-layer activation state and SparseController for per-step decisions. Keep model layers generic and make layer-to-layer dependencies explicit.

## Query-Aware Paged Selection

QuEST-like methods maintain persistent page metadata and select pages/tokens using the current query.

Read [quest-pattern.md](quest-pattern.md). Verify that explicit paged storage and the selected attention provider share the same indexing semantics.

## Compressed Or Multi-Pool Storage

DeltaKV-like methods may split state across full-precision, compressed, residual, or reconstructed pools and may require a custom compute payload.

Audit each pool's capacity and reservation accounting, movement between pools, typed reconstruction/materialization, provider compatibility, and lifecycle operations. Use `build_decode_compute_view` rather than encoding custom payloads in ordinary view metadata.

## Choosing Reuse Safely

Before subclassing or copying an existing method, compare:

- persistent state shape and ownership;
- physical allocation and compaction semantics;
- prefill/decode scoring and selection order;
- admission peak and scheduler-visible budget;
- storage representation and actual-key requirements;
- prefix fork/restore/free behavior;
- CUDA Graph buffer lifetime;
- supported models and topologies.

Reuse is safe only when these mechanics match. Otherwise share a small helper or protocol implementation rather than inheriting incompatible lifecycle assumptions.
