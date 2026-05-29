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
| `deltakv` | Hybrid compression | Compressor-backed DeltaKV keeps selected full-precision references and reconstructs compressed historical KV. It requires a compressor checkpoint trained for the same base model. | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits` |
| `deltakv-triton` / `deltakv-triton-v2` / `deltakv-triton-v3` / `deltakv-triton-v4` | Hybrid compression | Sparse-vLLM DeltaKV variants with Triton reconstruction and related kernel optimizations. Use these when benchmarking engine-side DeltaKV performance. | DeltaKV knobs plus method-specific Triton controls documented in [Runtime Parameter Semantics](../configuration/runtime-parameter-semantics.md). |
| `deltakv-delta-quant` | Hybrid ablation | No-checkpoint DeltaKV-style ablation. It reuses DeltaKV center selection but stores token-space residuals directly, optionally with int4 residual quantization. | `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `deltakv_full_pool_reserve_ratio` |

Sparse-vLLM internally stores this as `vllm_sparse_method`, but public commands
and `LLM(...)` kwargs should use `sparse_method`.

## Runtime Ownership

- Method-specific runtime state belongs in `src/sparsevllm/engine/cache_manager/`.
- Cross-layer observation or scheduling coordination belongs in
  `src/sparsevllm/engine/sparse_controller.py`.
- `src/sparsevllm/layers/attention.py` should stay generic and call shared
  hooks.

For new first-class methods, use the repo-local
[`$add-sparse-method`](../../skills/add-sparse-method/SKILL.md) skill.

## Query-Aware Knobs

`quest` runtime knobs:

- `quest_chunk_size`: QuEST page/chunk size in tokens
- `quest_token_budget`: decode-time token budget before page rounding
- `quest_skip_layers`: keep the first N layers dense during decode
