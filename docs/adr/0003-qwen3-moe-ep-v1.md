# Qwen3-MoE EP v1 uses TP=1 and eager MoE

Status: accepted and implemented for the first correctness milestone on 2026-06-26. Real `Qwen3-30B-A3B-Instruct-2507` smoke matched EP=1 and EP=2 greedy token ids for one short prompt. This ADR does not make throughput claims.

Sparse-VLLM will add first-stage Qwen3-MoE expert-parallel inference with Tensor Parallel size fixed to 1. Expert Parallelism is treated as an MoE-layer execution strategy, not as a replacement for the TP process group. Dense layers, attention, LM head, scheduler state, and cache-manager state must continue to observe TP size 1 when EP is enabled.

Before Qwen3-MoE EP execution is implemented, Sparse-VLLM must introduce an explicit parallel context that separates process-world rank and size from TP rank and size. `LLMEngine` launches `parallel_world_size = tensor_parallel_size * expert_parallel_size` model-runner processes. `ModelRunner` initializes distributed state with global rank and world size, while dense layers, embedding/head, cache manager, and TP loader sharding use only TP rank and TP size. MoE execution uses EP rank and EP group.

The cache manager remains EP-unaware in v1. Each EP rank owns the same request stream and a full copy of the non-expert model state and KV cache. Only routed expert computation is partitioned across EP ranks. Data-parallel replicas may wrap EP groups later, but native DP scheduling is not required for this first stage.

The first EP backend is correctness-first local-expert execution followed by an all-reduce over the MoE output contribution tensor. Expert ownership defaults to contiguous expert shards. This avoids token-dispatch complexity while establishing Qwen3-MoE loading, routing, and parity against a single-rank oracle. All-to-all and DeepEP-style dispatch are deferred backend options.

MoE execution should be factored behind a reusable executor interface, while model-specific adapters keep Qwen3-MoE configuration, routing options, and checkpoint key mapping explicit. The abstraction is for MoE execution mechanics, not a claim that all MoE checkpoints share identical routing or expert layouts. Layer-local MoE math belongs under `layers/`; distributed EP context and backends belong under `engine/expert_parallel/`.

Qwen3-MoE loading uses a model-owned checkpoint adapter before generic packed-module substring matching. Local expert tensors are loaded only on their owner rank; non-local expert tensors are explicitly skipped. Fused expert tensors should be read through owned expert slices when possible, not loaded fully on every EP rank and then sliced locally.

Decode CUDA graph capture is disabled for EP v1. Current Sparse-VLLM decode graph capture wraps the full decode forward. EP graph rejection must happen after decode graph aliases and implicit graph-enabling legacy paths are normalized. EP graph support should be added later as piecewise capture: non-MoE decode segments can be captured and replayed, while MoE execution stays eager until a graph-safe distributed MoE backend is validated.

Correctness validation should use `Qwen/Qwen3-30B-A3B-Instruct-2507` because it can run as a single-rank oracle. Validation compares TP=1, EP=1 against TP=1, EP>1 under identical prompts and greedy decoding, then checks first-step logits, generated token ids, and final text within explicit numeric tolerance.

## Consequences

- EP v1 has a smaller implementation and debugging surface than TP+EP or DeepEP-first designs.
- EP v1 duplicates non-expert weights and KV cache across EP ranks, so it is not the final memory-efficiency target.
- Existing TP CUDA graph behavior remains intact for non-EP runs.
- Qwen3-MoE EP graph acceleration and efficient token dispatch are follow-up work, not hidden fallbacks.
- Current dense-layer and cache-manager uses of distributed world size must be corrected before EP workers are launched.

## Rejected alternatives

- Supporting TP+EP in v1 was rejected because current dense layers and cache-manager sizing still treat distributed world size as TP size in several places.
- Making the cache manager EP-aware was rejected because EP does not change KV-cache semantics for a single request stream.
- Implementing all-to-all or DeepEP first was rejected because it would combine checkpoint loading, routing correctness, communication correctness, and efficiency work in one step.
- Letting Qwen3-MoE expert keys pass through generic packed-module matching was rejected because expert `gate`/`up`/`down` names can collide with dense packed-module rules.
