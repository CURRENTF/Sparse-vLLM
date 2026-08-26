# Core Sparse Methods

Sparse-vLLM is built around a cache-manager-first sparse runtime. The engine
supports physical eviction, logical masking, and hybrid compression without
forcing `attention.py` to own method-specific state.

## Supported Methods

Set `sparse_method` to one of the following method names.

| Method | Family | Description | Main Runtime Knobs |
| --- | --- | --- | --- |
| `vanilla` | Dense baseline | Full attention baseline. Use it to verify correctness and measure the non-sparse engine path. | Common engine knobs only. |
| `streamingllm` | Physical eviction | StreamingLLM-style fixed sink plus recent-window cache. Tokens outside the retained prefix/tail policy are physically evicted from the active KV cache. | `sink_keep_tokens`, `recent_keep_tokens` |
| `attention-sink` | Physical eviction | Alias-style attention-sink policy with the same sink-token and recent-window retention model. It is useful for comparing sink-window behavior against other physical eviction methods. | `sink_keep_tokens`, `recent_keep_tokens` |
| `snapkv` | Physical eviction | SnapKV-style token selection keeps a compact set of important historical tokens after prefill. It reduces cache footprint by physically retaining only selected KV positions. | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `h2o` | Physical eviction | H2O maintains an independent cumulative attention-importance vector for every KV layer and physical row. Prefill scores and physically evicts after every chunk, and the final prefill chunk contracts to the decode budget. Decode scoring and periodic eviction are currently disabled: decode is score-free and its physical row grows with generated tokens. Prefill eviction retains heavy hitters plus a recent suffix. Sparse-vLLM v1 shares one selected token set across KV heads within a layer, but never shares scores or selected indices across layers. | `h2o_decode_budget`, `h2o_prefill_budget`, `h2o_recent_ratio`, `h2o_prefill_score_window`, `sparse_prefill_score_mode` |
| `pyramidkv` | Physical eviction | PyramidKV-style layer-dependent KV retention. It allocates sparse budgets across layers and physically stores the selected context tokens. | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `omnikv` | Logical masking | OmniKV keeps the physical cache available but constructs sparse attention views for selected layers. This is useful when the method should avoid rewriting cache storage while still reducing attention work. | `full_attention_layers`, `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens` |
| `quest` | Query-aware page selection | QuEST selects token pages from persistent min/max page summaries. Prefill stays dense. Explicit-KV models score in key coordinates; GLM-4.7-Flash scores the fused MLA latent/RoPE cache with the matching absorbed decode query while keeping the compute payload latent. | `quest_chunk_size`, `quest_skip_layers`, `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens` |
| `deltakv` | Hybrid compression | Slim compressor-backed DeltaKV runtime. Legacy `deltakv-less-memory*` names normalize here for older configs, but real benchmark runs still require a matching compressor checkpoint. | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

Sparse-vLLM uses `sparse_method` unchanged in public commands, `LLM(...)`, the
runtime config, and internal consumers.

> [!NOTE]
> Decode scoring and eviction for `snapkv` and `h2o` are future work. In the
> current runtime, both methods use score-free decode and their physical KV rows
> grow with generated tokens. This behavior must remain explicit until the
> score-producing eager/CUDA Graph paths and eviction lifecycle are implemented
> and validated.

SnapKV, PyramidKV, and H2O default `sparse_prefill_score_mode` to `probability`.
For H2O this is the canonical path: every KV layer independently sums its
normalized softmax attention probabilities over the full current query chunk,
then accumulates that attention mass across prefill chunks. Decode score
collection and eviction are intentionally disabled. Sparse-vLLM
reuses FA3's softmax LSE and performs one additional QK sweep because FlashAttention
does not materialize its probability matrix. `h2o_prefill_score_window=0` selects
the full current chunk and is the canonical default. A nonzero window in `[1, 128]`
or explicit `logits` mode is a non-canonical approximation; neither changes the
requirement that every H2O KV layer computes and retains its own prefill score.

## Prefill Scheduling Policies

Prefill scheduling is method-specific and registry-owned. The source of truth
is `src/sparsevllm/method_registry.py`; benchmark scripts and user configs
should not redefine method semantics.

| Policy | Runtime Semantics | Current Default Methods |
| --- | --- | --- |
| `all_chunked` | Every prefill request is capped by `engine_prefill_chunk_size` and normal scheduler batch limits; `long_prefill_offload_threshold` is ignored. | `vanilla`, `streamingllm`, `attention-sink`, `snapkv`, `h2o`, `quest`, `omnikv` |
| `long_bs1full_short_batch` | After supported prefix attachment, residuals at or below `long_prefill_offload_threshold` use atomic full prefill and may batch. Larger residuals are isolated and use RawKV offload chunks capped by `engine_prefill_chunk_size`. | `pyramidkv` and DeltaKV-family methods |

DeltaKV-family methods and PyramidKV keep `long_bs1full_short_batch` as the only
public policy. The threshold defaults to `65536` tokens (64K). If
`engine_prefill_chunk_size` is omitted, it defaults to that threshold; explicit
values must be positive and no larger than the threshold. `Config` raises
`max_num_batched_tokens` to fit one threshold-sized full prefill when necessary.
With full-layer KIVI enabled, DeltaKV keeps a small resident raw tail pool for
decode and a separate `max_model_len`-sized prefill staging buffer. Batched
short prefills share that staging buffer through disjoint per-request ranges;
the resident raw-tail slot count is not a prefill batch limit.
PyramidKV classifies the residual after chain-prefix attachment. DeltaKV does
not support prefix caching and rejects attached-prefix prefill before mutating
compressed or quantized row metadata.

## Prefix cache modes

`enable_prefix_caching=true` supports two deliberately separate layouts.
`prefix_cache_mode=auto` chooses radix for vanilla/OmniKV/QuEST and a linear
chain for SnapKV/H2O/PyramidKV/R-KV/SkipKV. `radix` and `chain` can be
requested explicitly, but incompatible method/mode pairs fail fast.
GLM-4.7-Flash QuEST is a storage-specific exception: its latent QuEST path does
not yet support prefix caching or prefix offload, and configuration rejects both.

The chain layout keeps one resident `seq_id` across turns and never branches.
Callers send the complete logical context plus the returned `chain_id`; only
the suffix after the verified processed boundary is forwarded. Method KV and
metadata remain in the cache manager. Idle chains are reclaimed by strict
LRU, while active writers are pinned. Rank 0 keeps the processed logical token
IDs in compact 32-bit storage so text continuations preserve the resident BPE
tokenization. This CPU history is bounded by
`max_model_len * max_num_seqs_in_gpu` and reclaimed with the chain.

`Config` resolves `None`, empty string, and `auto` to the registry default. An
explicit policy that does not match the method default fails fast so experiments
do not silently change scheduler semantics. Treat any policy override as an
explicit ablation and document it with the benchmark result.

## Runtime Ownership

- Method-specific runtime state belongs in `src/sparsevllm/engine/cache_manager/`.
- Cross-layer observation or scheduling coordination belongs in
  `src/sparsevllm/engine/sparse_controller.py`.
- `src/sparsevllm/layers/attention.py` should stay generic and call shared
  hooks.
- New first-class methods must register their default prefill policy in
  `src/sparsevllm/method_registry.py` and cover it in
  `tests/test_prefill_schedule_policy.py`.

## Query-Aware Knobs

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens
- `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens`: QuEST derives
  its decode-time token budget once during config construction by summing these
  three values
- `quest_skip_layers`: keep the first N layers dense during decode

`quest_token_budget` is no longer a runtime input. Passing it fails fast; remove
it and configure the three common keep-token fields instead.
