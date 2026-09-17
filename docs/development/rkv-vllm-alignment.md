# R-KV vLLM selection contract

The reference is Zefan-Cai/R-KV commit
`6715468b9872442be72e5c97322e4d9c9a2abf55`, `vLLM/rkv/algo.py` and
`vLLM/rkv/integration.py`. This is the cross-head-mean, cross-layer-sum serving
variant, not the original per-head eviction algorithm.

## Runtime contract

- Identity stays `rkv`. The total budget remains `sink_keep_tokens +
  decode_keep_tokens + recent_keep_tokens` for configuration compatibility.
  These three fields contribute to the total only: RKV reserves no sink and
  retains exactly `rkv_observation_tokens` trailing entries.
- Score each KV head using GQA max logits, probability averaging over the decode
  query window, token max-pooling, and per-head key cosine redundancy. Average
  joint scores across heads, sum in layer order, then SUM over attention TP.
  Select once per request and apply the sorted kept indices to every KV layer.
- Compression runs after forward, outside CUDA Graph capture, on positive
  decode counts divisible by `rkv_compression_interval`, when physical length
  reaches total budget plus interval. Prefill/recompute queries do not count.
- Query tensors and validity positions belong to the cache manager. Graph
  replay writes into fixed-address rings. Compaction invalidates ring positions;
  interval >= observation length ensures the next decision uses fresh queries.
  Prefill append invalidates observations, and chain snapshots preserve them.
- Explicit KV and actual-key materializers use the same scorer. Every KV layer
  must have the same ordered token domain. Layer dimensions may differ because
  layer scores are reduced separately. Attention providers remain unchanged.
- Prefix/chain lifecycle retains per-layer slot ownership. Global indices do
  not require moving KV: reuse SnapKV's sorted page-table compaction and free
  slot stack. Cache mutation happens only after all request groups are scored.
- TP ranks reduce the joint scores before top-k. DP requests remain independent;
  MoE EP does not participate in attention score reduction.
- Fail explicitly on incomplete required observations, non-finite scores,
  mismatched layer lengths, or a workspace too small for one score tile. Unlike the
  reference port's first-compression memory skip, never silently retain Full-KV.

## Cost and reuse

For layers M, requests B, local KV/query heads H/G, resident length L, query
window W and dimension D, an event costs O(M B (H L² D + G W L D)), plus top-k,
TP reduction of B L scores, and M B L mapping compaction. Amortize by the decode
interval. Ordinary attention and prefill are unchanged; prefill does not score.
Query storage is M B W G D elements plus int32 ring positions. Layer/request
scoring is bounded by `rkv_score_chunk_mb`, reserved separately from the KV pool;
per-unit admission includes pairwise matrices, masks/indices, QK and gathered K.
For H=8, L=4224, bf16, the official conservative pairwise estimate alone is
about 2.39 GiB: the scorer tiles heads and pairwise rows rather than allocating
that full matrix. Streaming accumulates column sums in float32, so numerical
parity is tolerance-based, not a bitwise promise for different tile shapes.

Reuse Torch matmul/softmax/max-pool/top-k and `ParallelGroup.all_reduce`.
Reuse `materialize_attention_keys` and `free_part_slots_batch_layers` for
storage. Existing `prefill_score_fwd` reduces heads after softmax and lacks the
required pooling, so it cannot implement this score unchanged. Inspected the
pinned vLLM and SGLang RKV Torch scorers: neither requires a dedicated external
kernel; importing an entire engine or sgl-kernel adds no matching primitive.
No attention provider dispatch or custom GPU kernel is changed.

## Validation

Compare per-head scores and final indices against the pinned official scorer;
exercise GQA, correlated keys, multiple layers, mixed lengths, TP reduction,
repeated eviction and query refill. Check actual CUDA Graph ring replay with
padded rows and row reuse, and model generation with repeated compression.
CPU results do not establish GPU parity or performance. Any quality or speed
claim requires a separate matched workload and recorded artifacts.
