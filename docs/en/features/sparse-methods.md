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
| `snapkv` | Physical eviction | SnapKV-style token selection uses an end-of-prompt observation window to keep a compact set of important prompt KV positions before generation. The current paper-aligned decode path is score-free and appends generated tokens without another SnapKV selection pass. | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `h2o` | Physical eviction | Intermediate prefill chunks can be compacted to `h2o_prefill_budget`; the final prompt is compacted to `h2o_decode_budget`. Decode is score-free by default and grows with generated tokens. Optional `h2o_decode_eviction` accumulates decode probabilities and periodically evicts physical KV. | `h2o_decode_eviction`, `h2o_decode_budget`, `h2o_decode_eviction_interval`, `h2o_prefill_budget`, `h2o_recent_ratio`, `h2o_prefill_score_window`, `sparse_prefill_score_mode` |
| `pyramidkv` | Physical eviction | PyramidKV-style layer-dependent KV retention. It allocates sparse budgets across layers and physically stores the selected context tokens. | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `omnikv` | Logical masking with optional offload | Cross-layer token selection; optionally keep sparse-layer history in pinned CPU memory and fetch the exact selected KV for decode. | `full_attention_layers`, `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `enable_omnikv_offload` |
| `quest` | Query-aware page selection | QuEST selects token pages from persistent min/max page summaries. Prefill stays dense. Explicit-KV models score in key coordinates; GLM-4.7-Flash scores the fused MLA latent/RoPE cache with the matching absorbed decode query while keeping the compute payload latent. | `quest_chunk_size`, `quest_skip_layers`, `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens` |
| `deltakv` | Hybrid compression | Slim compressor-backed DeltaKV runtime. Legacy `deltakv-less-memory*` names normalize here for older configs, but real benchmark runs still require a matching compressor checkpoint. | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

Sparse-vLLM uses `sparse_method` unchanged in public commands, `LLM(...)`, the
runtime config, and internal consumers.


## OmniKV KV offload

Set `sparse_method="omnikv", enable_omnikv_offload=True` in `LLM(...)` or
its runtime configuration. The boolean defaults to `False`; enabling it for
another sparse method is an error. Keep the model's existing full-layer profile
and token budgets.

Offload supports CUDA uniform FP16/BF16 explicit KV and the existing BF16 MLA
512-dimensional latent plus 64-dimensional RoPE layout. Full-attention layers
retain complete GPU KV. Sparse layers retain complete pinned-host history and
bounded, request-private GPU decode buffers. Every step fetches the exact
selected history, including sink/recent tokens; there is no LRU or historical
GPU hot cache. MLA stays compressed. Existing model TP, EP and TP+EP semantics
apply, with independent backing per rank.

Prefix caching and decode CUDA Graph can remain enabled within the model's
existing compatibility limits. Qwen3-MoE currently rejects OmniKV prefix caching,
including with active offload. Prefix hits share
history; suffix prefill sees the complete prefix, and generated suffixes remain
private. `enable_prefix_cache_offload` is a separate option for backing up and
demoting idle prefix blocks, including full-attention KV. With active OmniKV
offload it also supports MLA; configure a positive `prefix_cache_host_size_gb`
large enough for the configured prefix block capacity.

This is a capacity option, with a substantial PCIe/host-memory bandwidth cost.
Fixed-batch decode latency can increase. Full-attention KV and one full-history
prefill buffer still grow with context. Chunked prefill reloads sparse-layer
history and may increase TTFT. Use matched BenchProbe measurements for the
intended model, context and concurrency before enabling it in a latency-sensitive
workload.

Prefill acceleration is selected separately with `prefill_sparse_method`.
Sparse-vLLM currently supports `h2o_prefill` for intermediate-chunk KV
compaction and `flashprefill_v2` for sparse prefill attention computation. They
are alternatives on one axis and can each be combined with a compatible
cache/decode method. See
[runtime parameter semantics](../configuration/runtime-parameter-semantics.md#prefill-sparsity)
for the H2O prefill/decode combination matrix and the omitted-versus-empty
compatibility rule.

> [!NOTE]
> The two score-free decode contracts have different paper provenance. The
> [SnapKV paper](https://arxiv.org/abs/2404.14469) selects prompt KV from an
> observation window at the end of the prompt; adding decode-time rescoring and
> eviction would be a Sparse-vLLM extension. The
> [H2O paper](https://arxiv.org/abs/2306.14048) instead defines dynamic retention
> over successive decode steps. Sparse-vLLM's intermediate-chunk H2O compaction
> is its own prefill extension. Final-prompt compaction instead belongs to the
> decode contract because it creates the shorter cache used during generation,
> even though the mutation executes at the final-prefill boundary. Optional
> online score updates move toward the original H2O algorithm; periodic batched
> eviction and bounded prefill observation remain system/algorithm variants.

SnapKV defaults `sparse_prefill_score_mode` to `logits`, while PyramidKV defaults
to `probability`. H2O (including standalone `h2o_prefill`) defaults to
`sparse_prefill_score_mode="logits"` and `h2o_prefill_score_window=128`.
Explicit score-mode and window settings override these defaults, except that
`h2o_decode_eviction=True` forces `sparse_prefill_score_mode="probability"`.

The H2O default is an approximation using a bounded query window. To select
full-chunk normalized attention-mass scoring, explicitly set
`sparse_prefill_score_mode="probability"` and `h2o_prefill_score_window=0`.
In that mode, every KV layer independently sums normalized attention
probabilities over the full current query chunk and accumulates attention mass
across prefill chunks. H2O probability mode emits a performance warning because
it needs additional QK scoring even when attention LSE is reused. Both modes
retain independent scores for every H2O KV layer.

`h2o_decode_eviction` defaults to `False`. Enable it with `sparse_method="h2o"`
to accumulate normalized attention mass at every decode step and physically
retain heavy hitters plus recent tokens. Eviction returns each triggered row
to `h2o_decode_budget` at `budget + h2o_decode_eviction_interval`; memory pressure
can trigger earlier eviction of over-budget active rows. The switch forces
probability scoring even if `logits` was requested, with a warning. It preserves
`h2o_prefill_score_window`: probability mode accepts 0 through 128, and a nonzero
window is allowed. MLA latent models use an explicit approximation: apply
`softmax(scale * RAW_QK_REDUCED)` to head-max decode logits, then accumulate and
evict. This is not equivalent to normalizing each head before reduction and is
not fully aligned with original H2O; enabling it emits a once-per-process warning.
MLA prefill scoring and the default score-free decode behavior are unchanged.

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
Existing vanilla/OmniKV radix trees can be physically compacted with the
SnapKV- or KVzip-scored maintenance API described in
[Prefix cache pruning](prefix-cache-pruning.md); QuEST trees reject pruning.

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

- Persistent physical cache state and prefix-coupled metadata belong in
  `src/sparsevllm/engine/cache_manager/`.
- Per-step/per-layer logical state, score orchestration, cross-layer selection,
  and mutation triggers belong in a `SparseMethodRuntime` under
  `src/sparsevllm/engine/sparse_methods/`.
- `src/sparsevllm/engine/sparse_controller.py` is the stable method-agnostic
  facade and must not contain method-name hot-path branches.
- `src/sparsevllm/layers/attention.py` should stay generic and call shared
  hooks.
- New first-class methods must register their default prefill policy in
  `src/sparsevllm/method_registry.py` and cover it in
  `tests/test_prefill_schedule_policy.py`.

See the [sparse method runtime architecture](../design/sparse-method-runtime.md)
for the complete interface, ownership, prefix-cache, CUDA Graph, and extension
contract.

## Query-Aware Knobs

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens
- `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens`: QuEST derives
  its decode-time token budget once during config construction by summing these
  three values
- `quest_skip_layers`: keep the first N layers dense during decode

`quest_token_budget` is no longer a runtime input. Passing it fails fast; remove
it and configure the three common keep-token fields instead.
