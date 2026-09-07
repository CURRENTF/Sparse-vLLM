# Sparse Method Runtime Contract

Write this contract before implementation. It can be a short design note, issue comment, or working checklist, but every row must have an explicit answer.

## Contract Template

| Axis | Questions to answer | Likely owner |
| --- | --- | --- |
| Identity | What is the public name? Canonical internal name? Aliases? Defaults? Required assets? | Runtime params and method registry |
| Semantic delta | How does behavior differ from the nearest existing method? What is kept, selected, evicted, compressed, or reused? | Design note and method implementation |
| Persistent state | Which tensors/objects survive a step? Are they per sequence, layer, head, page, token, or pool? | CacheManager or ActivationController |
| Score contract | Is scoring required in prefill or decode? What are its shape, dtype, indexing domain, and producer/consumer? | Registry and SparseMethodRuntime; CacheManager/provider when score production is physical-view owned |
| Selection contract | Is selection posthoc, query-aware, cross-layer, or cumulative? Is it logical or physically mutating? | SparseMethodRuntime for logical orchestration; CacheManager for physical resolution/mutation |
| Storage | Does it support explicit, heterogeneous explicit, or MLA latent cache? Are actual keys required? | Model/runtime layout and storage protocol |
| Attention view | Can it use a logical view, or must it construct a custom compute payload? | CacheManager typed view builders |
| Prefill execution | Full, chunked, raw-offload, or another registered mode? Can requests batch together? | Registry, RuntimeState, scheduler contract |
| Admission | What physical capacity and temporary reservations are required at prefill and decode? | CacheManager, MemoryOracle, RuntimeState |
| Lifecycle | What happens on allocate, append, fork, prefix attach, rollback, restore, evict, offload, and free? | CacheManager and prefix/runtime coordinators |
| CUDA Graph | What is the shared eager/replay compute path? Which buffers and launch shapes are stable, which metadata updates stay outside capture, and which providers are prepared beforehand? If deferred, what blocks support and how will it be adapted? | Runtime preparation, cache manager, operators |
| Topology | Which TP/EP/DP layouts work? Are reductions or replicated state required? | Registry and implementation validation |
| Time cost | What are prefill, per-step decode, and per-selection/mutation costs, including trigger frequency and amortization? | Cost and reuse review |
| Space cost | What are persistent, temporary, and peak live bytes, including batch/layer/head replication and graph workspaces? | Cost review and allocation/admission owners |
| Kernel reuse | Which existing symbols can be reused or narrowly extended? What exact semantic gap requires a new kernel? | State owner and operator/provider |
| Validation | What is the dense/reference oracle? Which quality and matched performance workloads prove the claims? | Tests and benchmark artifacts |

## Eager And CUDA Graph Design

Resolve these points before the first serving implementation:

- Share algorithm kernels, typed payloads, and cache/state-transition semantics
  between eager and graph execution. Keep mode-specific code at the execution
  boundary; justify any separate compute path with a concrete contract or
  measured need, not convenience. Prefill and decode remain separate phases;
  supporting decode graphs does not require capturing prefill.
- Define the graph boundary. CPU scheduling, page allocation/free, and metadata
  preparation may remain outside the graph. Publish each step's rows, lengths,
  slot/page maps, and mutation triggers into stable device buffers before replay,
  with explicit ordering and ownership. Python decisions made during capture do
  not execute again during replay.
- Plan bounded workspace capacity, buffer addresses/lifetimes, provider
  preparation, graph keepalive/reset, and batch/context launch buckets. Use
  device-side validity masks or counters for changing lengths and page-boundary
  updates; padded requests must not mutate live cache state. Account for these
  allocations in the existing memory budget.
- A shared path need not use identical launch grids in both modes. A tighter
  eager grid and a fixed graph grid may consume the same metadata and kernels;
  keep any plan differences explicit and preserve numerical/mutation semantics.
- If graph support is deferred, name the unsupported phase/contract, the actual
  blocker, and a bounded adaptation/validation plan. Separate implementation
  debt and unavailable validation hardware from algorithm or upstream-provider
  restrictions. Retain explicit rejection rather than silent eager execution.

## Cost And Reuse Review

Include this review in the pre-implementation contract; a short table with
formulas and concrete code pointers is sufficient. Do not substitute "sparse",
"vectorized", or "uses Triton" for a cost analysis.

1. Define the relevant dimensions: batch and request lengths, query/KV heads,
   head dimension, layers, page size/count, retained budget, and selection
   frequency. Distinguish original context length from physically retained KV
   length and logically selected length. State per-rank replication or sharding.
2. Give time complexity separately for prefill, one decode step, and a
   selection/eviction event. Include score production, reductions, selection,
   metadata/view building, KV movement/reconstruction, and attention. Report
   worst-event cost and amortized cost over the stated trigger interval; do not
   hide a full-context scan or sort behind the retained attention budget.
3. Give symbolic byte counts using actual shapes and dtypes for persistent
   metadata, scores, indices, temporary KV, and provider/graph workspaces. Derive
   peak memory from simultaneously live buffers, including source/destination
   overlap during compaction and batch/layer/head or graph-bucket replication.
   Include a representative target-size estimate and connect allocations to
   admission/reservation accounting; KV reduction alone is not a memory result.
4. Search existing operators, kernel trees, and method call sites for each
   required primitive, including scoring, reduction/top-k, page mapping,
   gather/compaction, and compression/reconstruction where relevant. Record
   candidate symbols/paths, their shape/layout/dtype and semantic compatibility,
   and the decision to reuse, extend, or implement. Check indexing, masking,
   normalization, tie behavior, and graph-buffer contracts before reuse. Share
   a narrow primitive when appropriate; do not copy a kernel under a new method
   name or create a generic framework solely to claim reuse.
   Include vLLM and SGLang (including `sgl-kernel`) in this search. Prefer direct
   reuse through a compatible existing interface when semantics, layout, device,
   dependencies, and graph lifecycle match. Otherwise, their kernels may guide
   a narrow adaptation: record the upstream source/revision and the contract
   differences, preserve applicable license/attribution when incorporating code,
   and validate the adapted implementation. Do not import an entire serving
   runtime solely to access a primitive that is available independently.
5. Trace the hot path for device-to-host reads (`item`, `tolist`, `cpu`),
   synchronization, Python loops launching work per token/head/page, repeated
   allocations, layout conversions, scans, sorts, and full KV copies. Remove
   avoidable costs; justify remaining ones with semantic needs or measurements.
   Consider incremental state, batched kernels, reusable workspaces, direct
   paged access, and fused/tiled reductions only where semantics permit them.
   Do not materialize full attention-score or pairwise-similarity matrices when
   an equivalent tiled/streaming computation can avoid that storage. If such
   materialization is necessary, bound and account for it explicitly.
6. State where saved attention/cache work is expected to outweigh added method
   work, and choose measurements that can falsify that expectation. Include
   short-context or small-batch overhead as well as the intended long-context
   regime. Treat this as a hypothesis until measured; do not invent a universal
   speedup threshold or silently switch to dense behavior below it.

Resolve missing formulas, unaccounted allocations, and unexplained duplicate
kernels before coding. Runtime-dependent constants may remain hypotheses with
an explicit validation plan; no user approval checkpoint is implied by this
review. Preserve scoring math, selection domain, update order, and KV lifecycle
when optimizing, and validate against an independent algorithm reference.

## Static Registration Audit

The registry must make all static decisions visible before model execution:

- canonicalization and aliases;
- method availability and external dependency/assets;
- default prefill schedule and score requirements;
- supported models and parallel topologies;
- prefix-cache modes;
- eager and CUDA Graph support;
- storage/provider prerequisites where they can be known statically.

Do not advertise capability because a code path happens not to crash. Each positive capability must have targeted evidence.

## Control-Plane Audit

The scheduler must not know the method name. Express method effects through generic runtime data:

- allocatable capacity by relevant pool/layout;
- request-specific persistent and temporary reservation cost;
- prefill execution mode and batch compatibility;
- decode append cost and reclaimable memory;
- offload or prefix ownership state;
- graph eligibility already validated by runtime construction.

If this cannot describe a method, extend the narrow memory/execution contract. Do not add another method-name branch in scheduling code.

## Data-Plane Audit

Trace one prefill step and one decode step end to end:

1. Identify the score producer and exact indexing domain.
2. Identify who converts scores into `SparseSelection`.
3. Identify who mutates persistent state or physical allocation.
4. Identify who builds the typed compute view and payload.
5. Identify the operator/provider consuming that view.
6. Identify how writes update storage and method metadata.

Also identify the runtime lifecycle hook for each logical transition:
`prepare_step`, prefill/decode selection, attention end, layer end, or
`finish_step`. `SparseController` should only translate the engine lifecycle into
these typed requests/events.

Any untyped tuple, global side channel, or direct config inspection in attention is a boundary violation unless no existing typed contract can represent the data. In that case, extend the type at its owner.

## Unsupported Combination Policy

List unsupported combinations explicitly, such as MLA latent storage, heterogeneous layers, radix prefix cache, decode graphs, TP greater than one, or a missing provider. Reject them during config/runtime validation with a precise reason.

Do not silently disable the sparse method, switch schedules, change providers, or fall back to dense attention.
