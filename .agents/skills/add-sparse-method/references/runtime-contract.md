# Sparse Method Runtime Contract

Write this contract before implementation. It can be a short design note, issue comment, or working checklist, but every row must have an explicit answer.

## Contract Template

| Axis | Questions to answer | Likely owner |
| --- | --- | --- |
| Identity | What is the public name? Canonical internal name? Aliases? Defaults? Required assets? | Runtime params and method registry |
| Semantic delta | How does behavior differ from the nearest existing method? What is kept, selected, evicted, compressed, or reused? | Design note and method implementation |
| Persistent state | Which tensors/objects survive a step? Are they per sequence, layer, head, page, token, or pool? | CacheManager or ActivationController |
| Score contract | Is scoring required in prefill or decode? What are its shape, dtype, indexing domain, and producer/consumer? | Registry and SparseController |
| Selection contract | Is selection posthoc, query-aware, cross-layer, or cumulative? Is it logical or physically mutating? | SparseController and CacheManager |
| Storage | Does it support explicit, heterogeneous explicit, or MLA latent cache? Are actual keys required? | Model/runtime layout and storage protocol |
| Attention view | Can it use a logical view, or must it construct a custom compute payload? | CacheManager typed view builders |
| Prefill execution | Full, chunked, raw-offload, or another registered mode? Can requests batch together? | Registry, RuntimeState, scheduler contract |
| Admission | What physical capacity and temporary reservations are required at prefill and decode? | CacheManager, MemoryOracle, RuntimeState |
| Lifecycle | What happens on allocate, append, fork, prefix attach, rollback, restore, evict, offload, and free? | CacheManager and prefix/runtime coordinators |
| CUDA Graph | Are buffers stable? Which metadata updates occur outside capture? Which providers are bound before capture? | Runtime preparation, cache manager, operators |
| Topology | Which TP/EP/DP layouts work? Are reductions or replicated state required? | Registry and implementation validation |
| Validation | What is the dense/reference oracle? Which quality and matched performance workloads prove the claims? | Tests and benchmark artifacts |

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

Any untyped tuple, global side channel, or direct config inspection in attention is a boundary violation unless no existing typed contract can represent the data. In that case, extend the type at its owner.

## Unsupported Combination Policy

List unsupported combinations explicitly, such as MLA latent storage, heterogeneous layers, radix prefix cache, decode graphs, TP greater than one, or a missing provider. Reject them during config/runtime validation with a precise reason.

Do not silently disable the sparse method, switch schedules, change providers, or fall back to dense attention.
