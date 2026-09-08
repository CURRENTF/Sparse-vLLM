# Startup prefill memory profiling

Startup retains the normal scheduler-driven prefill workload, which fills the
current-token budget with fresh prompts. It also measures one suffix chunk at
`max_model_len - 1` visible tokens, leaving room for the sampled token. The suffix
size is bounded by both `engine_prefill_chunk_size` and
`max_num_batched_tokens`. No additional probe runs if that chunk already covers
the entire context.

The history probe uses the ordinary StandardCacheManager allocation and view
paths, a dense SparseController, and the existing model forward and bound
attention providers. A startup-only cache manager seeds zero-valued history
without evaluating the preceding tokens. Same-shape layers share physical KV
storage: each layer overwrites the current chunk while the synthetic history
remains zero. MLA retains its latent/RoPE storage layout; explicit KV supports
both uniform and heterogeneous layer shapes.

History storage is allocated before the memory profile begins and remains live
until the profile finishes, so it does not count as transient model memory.
Afterward the temporary sequence and cache bindings are released and the
original runtime is restored, including on forward failure. The larger of the
fresh-prefill and history-prefill peaks feeds the existing per-rank KV capacity
decision. Logs report the history, context and chunk lengths and the measured
history-prefill peak.

This is a single-request allocation probe, not a worst-case workload search or
a numerical/throughput benchmark. It does not reconstruct sparse-method history,
long-history compression/score state, recurrent history, or simultaneous long
requests. Those additional costs rely on the existing workload measurement and
the memory left outside `gpu_memory_utilization`; no extra analytical reserve
or method-specific seeding is introduced. Existing workspace limits still apply,
so the probe can expose a maximum-context MLA workspace failure during startup.
