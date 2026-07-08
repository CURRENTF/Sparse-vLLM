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
| `snapkv` | Physical eviction | SnapKV-style token selection keeps a compact set of important historical tokens after prefill. It reduces cache footprint by physically retaining only selected KV positions. | `decode_keep_tokens`, `prefill_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens` |
| `pyramidkv` | Physical eviction | PyramidKV-style layer-dependent KV retention. It allocates sparse budgets across layers and physically stores the selected context tokens. | `decode_keep_tokens`, `prefill_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens` |
| `omnikv` | Logical masking | OmniKV keeps the physical cache available but constructs sparse attention views for selected layers. This is useful when the method should avoid rewriting cache storage while still reducing attention work. | `full_attention_layers`, `decode_keep_tokens`, `prefill_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `chunk_prefill_accel_omnikv` |
| `quest` | Query-aware page selection | QuEST selects token pages based on the decode query. Prefill stays dense, and sparse selection happens in decode through page/chunk budgets. | `quest_chunk_size`, `quest_token_budget`, `quest_skip_layers` |
| `deltakv` | Hybrid compression | Slim compressor-backed DeltaKV runtime. Legacy `deltakv-less-memory*` names normalize here for older configs, but real benchmark runs still require a matching compressor checkpoint. | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

Sparse-vLLM internally stores this as `vllm_sparse_method`, but public commands
and `LLM(...)` kwargs should use `sparse_method`.

## Prefill Scheduling Policies

Prefill scheduling is method-specific and registry-owned. The source of truth
is `src/sparsevllm/method_registry.py`; benchmark scripts and user configs
should not redefine method semantics.

| Policy | Runtime Semantics | Current Default Methods |
| --- | --- | --- |
| `all_chunked` | Every prefill request is capped by `chunk_prefill_size` and normal scheduler batch limits. | `vanilla`, `streamingllm`, `attention-sink`, `snapkv`, `quest`, `omnikv` |
| `long_bs1full_short_batch` | Long requests run as one complete prefill with batch size 1; short requests still use chunked batching. This is for methods whose intended sparse/cache transformation depends on a complete long-prefill representation. | `pyramidkv` and DeltaKV-family methods |

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
- `quest_token_budget`: decode-time token budget before page rounding
- `quest_skip_layers`: keep the first N layers dense during decode
