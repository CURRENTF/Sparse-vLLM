# Prefix cache pruning

Vanilla and OmniKV radix prefix caches can compact the physical KV payload of
an existing, idle tree path without changing its logical token route or stable
block IDs. QuEST prefix caching and offload remain supported, but QuEST pages
cannot be physically pruned.

Start a maintenance job with a block-aligned half-open interval `[L, R)`:

```http
POST /v1/prefix_cache/prune
Content-Type: application/json

{
  "token_ids": [1, 2, 3, 4],
  "range_start": 0,
  "range_end": 4,
  "keep_tokens": 2,
  "policy": "snapkv_global"
}
```

The endpoint returns HTTP 202 with a `prune_id`. Query
`GET /v1/prefix_cache/prune/{prune_id}` until the status is `completed`,
`blocked`, or `failed`. `snapkv_global` and `kvzip_global` reduce scores across
layers, heads, and tensor-parallel ranks into one deterministic token mask.

Pruning is committed only when every affected block is unreferenced and has no
in-flight transfer. A completed prune adds an inherited `quality_degraded`
record at the `[L, L + block_size)` subtree root. Device allocation and
eviction accounting use retained token slots. Offload continues to own one
fixed host page per logical block, while D2H/H2D transfers include only retained
block offsets.

The request also accepts `text` or a complete OpenAI `chat` request instead of
`token_ids`. The `chat` selector reuses the server's chat-template and reasoning
rendering so an agent can address the exact tree path written by its last turn.
SnapKV accepts `observation_tokens`; KVzip accepts `score_chunk_size` plus
`prev_postfix_size` for KVzip. `allow_recompress` is reserved and currently
fails explicitly because already-dropped KV cannot be rescored from the
compacted tree alone.
