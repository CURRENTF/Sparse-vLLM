# ADR 0004: Linear chain prefix cache

## Status

Accepted.

## Context

SnapKV, PyramidKV, R-KV, and SkipKV physically delete KV positions. A radix
prefix node cannot describe their resident payload because logical token
positions and physical per-layer row lengths diverge after compression. These
methods still benefit from the common multi-turn case where one conversation
has exactly one continuation writer.

## Decision

`enable_prefix_caching` remains the feature switch. `prefix_cache_mode` selects
`auto`, `radix`, or `chain`. Auto resolves to radix for vanilla, OmniKV, and
QuEST, and to chain for SnapKV, PyramidKV, R-KV, and SkipKV. Incompatible
explicit combinations fail during config construction.

The chain implementation is independent of `RadixPrefixIndex`:

- `ChainCacheIndex` owns opaque IDs, ACTIVE/IDLE lifecycle, processed-token
  digest, compact driver-side logical token history, strict IDLE-only LRU
  metadata, and bounded tombstones.
- `ChainCacheCoordinator` owns logical coordination only.
- Cache managers own KV rows, physical slots, R-KV queries, SkipKV sentence
  state, and all other method metadata.
- `RuntimeState` is the payload reclamation entrypoint.

An omitted, null, or empty `chain_id` creates an opaque server ID. Reusing an
ID requires an IDLE record, the same method/config fingerprint, and an exact
SHA-256 match for the logical input prefix through the persisted processed
boundary. Rank 0 also retains that exact logical prefix in a compact unsigned
32-bit array. Text APIs need the original token identity because decoding and
re-encoding a BPE prefix is not generally token-stable; the stored prefix lets
the server tokenize only the appended text. This history is bounded by
`max_model_len * max_num_seqs_in_gpu` tokens, reclaimed with the chain, and
reported in chain-cache token/byte statistics. TP admission validation and
completion RPCs still carry only token count plus the 32-byte digest, so long
prompts do not consume the fixed-size shared-memory command buffer and worker
ranks do not duplicate the driver token history.

One chain has one ACTIVE writer. Normal EOS and length completion retain the
resident row and transition to IDLE. The final sampled token has not run a
forward pass, so the stored boundary is `seq.num_tokens - 1`; that token
belongs to the next suffix. A server-detected text stop invalidates the chain:
the hidden stop text can contain already-processed tokens that are absent from
the client-visible continuation, and the compressed physical layout cannot be
rolled back generically. Disconnect, failure, preemption, cancellation, and
parse failure also invalidate the chain and free all payload.

LRU eviction considers IDLE chains only and orders by
`(last_access, chain_id)`. ACTIVE chains are pinned. Rank 0 supplies the exact
victim plan through the TP RPC path and every rank executes and checks the
same lifecycle result. Admission plans also reserve their per-layer physical
peak and resident row before prefill allocation, preventing concurrently
queued chains from overcommitting the same free slots.

## HTTP and routing contract

Chat Completions, Completions, and Responses accept `chain_id`. Admission
finishes before a streaming response is constructed, so chain errors retain
their 404/409/410/503 status instead of appearing after HTTP 200. Successful
responses expose `X-SparseVLLM-Chain-ID`; JSON/SSE objects expose `chain_id`
and reused-token usage.

Workers expose read-only `/v1/chain_cache/routing_match`. The smart router
probes workers in parallel for a non-empty ID. A unique IDLE owner wins
regardless of load; ACTIVE, missing, tombstoned, and duplicate ownership map
to 409, 404, 410, and 500. An explicitly empty `chain_id` selects only a
chain-capable worker and creates the new owner there. No router-local
ownership map is authoritative.

## Consequences

The first version deliberately has no branching, public release endpoint, or
radix compatibility shim. Abandoned chains are reclaimed by automatic LRU.
Applications that need branching must use radix-capable methods or create
independent chains and recompute their first turn.
