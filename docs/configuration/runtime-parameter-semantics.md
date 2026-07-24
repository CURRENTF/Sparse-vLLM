# Runtime Parameter Semantics And Audit

This document is the source of truth for runtime and benchmark parameters in
this repository. It is written for two readers:

- Humans who need to run or compare experiments without silently changing the
  method.
- Maintainers who need to edit this repo later without repeating old parameter
  mistakes.

The scope is runtime and experiment parameters that affect inference behavior,
accuracy, throughput, capacity, or model loading. It covers the repo-owned
DeltaKV/HF path, Sparse-vLLM path, LLaVA-OneVision visual-cache path, and the
main benchmark entrypoints. Vendored baseline internals are only documented
where this repo exposes their parameters.

## 1. High-Level Rules

Use canonical names in new configs when possible:

```json
{
  "sparse_method": "deltakv",
  "deltakv_checkpoint_path": "/path/to/compressor",
  "decode_keep_tokens": 2048,
  "sink_keep_tokens": 8,
  "recent_keep_tokens": 128,
  "full_attention_layers": "0,1,2,8,18",
  "deltakv_neighbor_count": 4,
  "deltakv_center_ratio": 0.1,
  "deltakv_latent_dim": 256
}
```

Then add backend-specific speed and capacity knobs:

```json
{
  "engine_prefill_chunk_size": 4096,
  "gpu_memory_utilization": 0.9,
  "max_num_batched_tokens": 8192,
  "max_num_seqs_in_batch": 8,
  "max_decoding_seqs": 8
}
```

For HF DeltaKV, use the explicit HF name instead:

```json
{
  "hf_prefill_chunk_size": 32768
}
```

Important rules:

- Legacy runtime names are rejected at public runtime/API/CLI boundaries. Do not
  use `chunk_prefill_size`, `num_top_tokens`, `model_cls`, `compressor_path`,
  `vllm_sparse_method`, `deltakv_path`, `seq_chunk_size`, or `k_neighbors` in
  new runtime configs.
- `engine_prefill_chunk_size` is Sparse-vLLM scheduler chunking.
  `hf_prefill_chunk_size` is HF wrapper/model chunking.
- `decode_keep_tokens=0.17` is a ratio in some HF paths but is invalid for
  Sparse-vLLM. Convert ratios to explicit token counts before Sparse-vLLM runs.
- `sparse_method` is the public method selector for both HF and Sparse-vLLM.
- `deltakv_checkpoint_path` is the public DeltaKV checkpoint path for both HF
  and Sparse-vLLM.
- Sparse-vLLM prefix-cache controls are direct engine config fields:
  `enable_prefix_caching`, `prefix_cache_block_size`,
  `prefix_cache_max_blocks`, and `prefix_cache_salt`. They are not HF
  parameters.
- `compressor_token_group_size` is for compressor token grouping.
  `deltakv_neighbor_count` is for selected reference/prototype count.
- LLaVA `--deltakv_checkpoint_path none` plus `visual_uniform_keep` is not
  learned DeltaKV. It is a visual-token uniform-pruning baseline.

## 2. Runtime Parameter Flow

There are five main runtime entry paths.

| Entry | Parameter container | Normalization | Main consumers |
| --- | --- | --- | --- |
| `scripts/benchmarks/bench_sparse_vllm.py` | `--hyper_params` JSON | `normalize_runtime_params(..., backend="sparsevllm")` | `sparsevllm.Config`, `Scheduler`, `CacheManager`, `SparseController` |
| `sparsevllm-openai-server` / `python -m sparsevllm.entrypoints.openai.api_server` | CLI flags plus OpenAI JSON request body | CLI engine kwargs are passed to `LLM(..., **kwargs)`, which normalizes via `normalize_runtime_params(..., backend="sparsevllm")`; request sampling params build `SamplingParams` directly | `LLMEngine`, `AsyncEngineDispatcher`, `/v1/completions` |
| `benchmark/long_bench/pred.py` and `benchmark/math_bench/pred.py` | `--hyper_param` JSON or file | `get_generate_api(...)` normalizes after merge | HF wrappers or Sparse-vLLM engine |
| `benchmark/scbench/run_scbench.py` DeltaKV branch | `--hyper_param` JSON dict | `get_generate_api(...)` normalizes | HF wrappers |
| `benchmark/multimodal/visual_cache/run_visual_cache.py` | dedicated CLI args | no global normalizer; builds `config.deltakv_infer_config` | LLaVA wrapper and `KVQwen2Config` |

Core files:

- `src/deltakv/configs/runtime_params.py`: canonical alias mapping and conflict
  checks.
- `src/deltakv/configs/model_config_cls.py`: HF custom config defaults and
  `set_infer_args`.
- `src/deltakv/get_chat_api.py`: routes to HF or Sparse-vLLM and resolves
  `sparse_method`.
- `src/sparsevllm/config.py`: Sparse-vLLM dataclass defaults and engine config
  validation.
- `src/sparsevllm/engine/scheduler.py`: prefill/decode scheduling and admission.
- `src/sparsevllm/engine/cache_manager/base.py`: method to cache-manager routing.
- `src/deltakv/modeling/kv_cache.py`: HF cache behavior for standard DeltaKV.
- `src/deltakv/modeling/origin_residual_quant_cache.py`: partial direct
  residual-quant ablation.
- `src/deltakv/modeling/all_origin_residual_quant_cache.py`: all-layer direct
  residual-quant ablation.

## 3. Canonical Alias Map

The normalizer accepts only canonical public runtime names. It maps them to
backend-native internal fields and rejects legacy public keys with `ValueError`.
Internal fields still exist in HF config objects, but they are not valid
user-facing runtime parameters.

| Canonical key | HF target | Sparse-vLLM target | Meaning |
| --- | --- | --- | --- |
| `sparse_method` | `model_cls` | `vllm_sparse_method` | Method selector. |
| `deltakv_checkpoint_path` | top-level `compressor_path` | `deltakv_path` | DeltaKV checkpoint directory. |
| `decode_keep_tokens` | `num_top_tokens` | `decode_keep_tokens` | Decode-time important-token budget. |
| `prefill_keep_tokens` | `num_top_tokens_in_prefill` | not supported | HF prefill/finalization important-token budget. Sparse-vLLM uses `decode_keep_tokens` for prefill-related budgets. |
| `sink_keep_tokens` | `num_sink_tokens` | `num_sink_tokens` | Prefix tokens always kept. |
| `recent_keep_tokens` | `num_recent_tokens` | `num_recent_tokens` | Recent tail tokens always kept. |
| `full_attention_layers` | `full_attn_layers` | `full_attn_layers` | Layers that stay full, or observation anchors depending on method. |
| `deltakv_neighbor_count` | same | `deltakv_k_neighbors` | Number of reference/prototype neighbors. |
| `deltakv_center_ratio` | `cluster_ratio` | `cluster_ratio` | Fraction or stride-derived rate for reference centers. |
| `deltakv_latent_dim` | `kv_compressed_size` | `kv_compressed_size` | DeltaKV latent width. |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `kv_quant_bits` | Quantization bits for the cached DeltaKV-style state. |
| `hf_prefill_chunk_size` | `chunk_prefill_size` | none | HF wrapper/model chunk size. |
| `engine_prefill_chunk_size` | none | `chunk_prefill_size` | Sparse-vLLM scheduler chunk size. |
| `visual_token_prune_only` | same | none | LLaVA visual-token-only cache dropping/pruning. |
| `visual_token_keep_ratio` | same | none | LLaVA ratio of eligible visual tokens to keep. |
| `enable_prefix_caching` | none | same | Enables Sparse-vLLM prefix KV reuse for supported methods. |
| `prefix_cache_block_size` | none | same | Prefix-cache hash/materialization block size. Defaults to 16 except QuEST. |
| `prefix_cache_max_blocks` | none | same | Optional cap on live prefix-cache blocks; evicts only unreferenced leaf blocks. |
| `prefix_cache_salt` | none | same | Extra fingerprint salt to isolate otherwise compatible cache entries. |

Rejected legacy runtime names:

| Legacy key | Replacement | Problem with legacy name |
| --- | --- | --- |
| `model_cls`, `vllm_sparse_method` | `sparse_method` | Backend-specific method selector names leaked into shared configs. |
| `compressor_path`, `deltakv_path` | `deltakv_checkpoint_path` | Backend-specific checkpoint names made cross-backend configs ambiguous. |
| `chunk_prefill_size` | `hf_prefill_chunk_size` or `engine_prefill_chunk_size` | Same spelling meant different speed/capacity behavior on each backend. |
| `num_top_tokens`, `num_top_tokens_in_prefill` | `decode_keep_tokens`; HF-only `prefill_keep_tokens` | Count/ratio/per-layer semantics differed by backend. |
| `num_sink_tokens`, `num_recent_tokens`, `tail_token_size` | `sink_keep_tokens`, `recent_keep_tokens` | Internal cache names leaked into experiment configs. |
| `full_attn_layers` | `full_attention_layers` | Internal layer-routing names leaked into shared configs. |
| `seq_chunk_size` | `compressor_token_group_size` | It described token grouping but was also used as a cluster-neighbor fallback. |
| `k_neighbors`, `deltakv_k_neighbors` | `deltakv_neighbor_count` | Backend/internal names hid the selected-reference meaning. |
| `cluster_ratio` | `deltakv_center_ratio` | DeltaKV-specific center/prototype rate. |
| `kv_compressed_size` | `deltakv_latent_dim` | Latent width, not a token count. |
| `kv_quant_bits` | `deltakv_latent_quant_bits` | The quantized object is method-dependent; name must be explicit in shared configs. |
| `deltakv_visual_compress_only` | `visual_token_prune_only` | Says "DeltaKV" and "compress", but the no-checkpoint path is uniform pruning. |
| `deltakv_visual_keep_ratio` | `visual_token_keep_ratio` | Tied to old naming. |

## 4. Method Routing

### 4.1 HF `sparse_method`

`get_generate_api(..., backend="hf")` uses public `sparse_method` and maps it to
an internal HF wrapper class. Supported repo-owned or routed values include:

| `sparse_method` | Main behavior | Checkpoint requirement |
| --- | --- | --- |
| `auto` | Plain HF `AutoModelForCausalLM`; optional chunked-forward monkey patch. | No compressor. |
| `deltakv` | Standard DeltaKV HF wrapper. | Optional, but learned compressor requires checkpoint. |
| `full_deltakv` | Full-layer DeltaKV compression wrapper. | Usually checkpoint-backed. |
| `origin_residual_quant` | Direct token-space residual quant for full-attention layers; sparse layers use standard path. | Checkpoint optional depending config, but cluster metadata can be loaded from checkpoint config. |
| `all_origin_residual_quant` | Direct token-space residual quant for every layer; requires `use_cluster=True`. | No learned compressor needed for reconstruction path. |
| `snapkv` | HF SnapKV wrapper. | No DeltaKV checkpoint. |
| `pyramidkv` | HF PyramidKV wrapper. | No DeltaKV checkpoint. |
| `omnikv` | Loads DeltaKV wrapper with `use_compression=False` and `use_cluster=False`. | No DeltaKV checkpoint. |
| `quest` | Patches Quest baseline attention. | No DeltaKV checkpoint. |
| `palu`, `kivi`, `adakv`, `kvzip` | Baseline adapters. | Baseline-specific. |

For example, `sparse_method="deltakv"` maps to the standard HF
DeltaKV wrapper, because the Triton variants are Sparse-vLLM-specific
implementation choices.

### 4.2 Sparse-vLLM `sparse_method`

`backend="sparsevllm"` uses public `sparse_method`. The engine stores the
normalized value internally as `vllm_sparse_method`.

Known Sparse-vLLM method strings:

| Method | Cache manager behavior |
| --- | --- |
| `""` or canonical `vanilla` | Standard dense cache manager. |
| `streamingllm`, `attention-sink`, `attention_sink` | StreamingLLM cache manager. Aliases normalize to `streamingllm`. |
| `snapkv` | SnapKV cache manager. |
| `pyramidkv` | SnapKV cache manager with PyramidKV layer budgets. |
| `omnikv` | OmniKV cache manager. |
| `quest` | Quest cache manager. |
| `rkv`, `r-kv`, `r_kv` | R-KV cache manager with physical decode eviction, query-cache attention importance scoring, and key-redundancy scoring. |
| `skipkv`, `skip-kv` | SkipKV cache manager with physical decode eviction and sentence-aware redundancy signals. |
| `deltakv` | Maintained compressor-backed DeltaKV runtime. |
| `deltakv-less-memory`, `deltakv-less-memory-cudagraph` | Legacy aliases retained for old configs and regression manifests. They normalize to `deltakv`; the cudagraph alias also requests decode CUDA graph. |

Sparse-vLLM has text-only Qwen3 support for the current DeltaKV-family paths
used in this repo's validation runs. Treat Qwen3 DeltaKV changes as
alignment-sensitive: qk-norm, RoPE theta/dtype, sparse-reference storage, and
full-layer KIVI views must be validated with HF-vs-Sparse logits checks before
reporting benchmark results.

Use `sparse_method="deltakv"` in new commands. Real Sparse-vLLM DeltaKV runs
require a matching `deltakv_checkpoint_path`; missing checkpoints are only for
construction-only tests that explicitly set `allow_missing_deltakv_path=True`.

### 4.3 Sparse-vLLM Prefix Cache

Prefix cache is a Sparse-vLLM engine feature, not a generic HF runtime feature.
It reuses request-context KV blocks across requests within the same live engine process.
It does not persist across process restarts.

Supported methods:

| `sparse_method` | Prefix-cache storage unit | Notes |
| --- | --- | --- |
| `vanilla` / `""` | token block | Uses `StandardCacheManager`. |
| `omnikv` | token block | Reuses `StandardCacheManager`; fingerprint still includes `omnikv` settings. |
| `quest` | QuEST page | Requires `prefix_cache_block_size == quest_chunk_size`. |

Unsupported methods fail fast when `enable_prefix_caching=true`: StreamingLLM,
attention-sink aliases, SnapKV, PyramidKV, and all DeltaKV-family methods. The
unsupported methods physically prune, compress, reconstruct, or remap KV in
ways that are not equivalent to reusing a complete request-context KV prefix.

Parameter semantics:

| Parameter | Meaning |
| --- | --- |
| `enable_prefix_caching` | Boolean or explicit true/false string. Enables scheduler lookup plus cache-manager attach/materialize/free/evict hooks. |
| `prefix_cache_block_size` | Positive integer or `null`. Defaults to 16 for vanilla/OmniKV. For QuEST it resolves to `quest_chunk_size`; any different explicit value is rejected. |
| `prefix_cache_max_blocks` | Optional positive integer cap. When set, insertions evict unreferenced leaf blocks only; referenced blocks are never evicted. |
| `prefix_cache_salt` | String folded into the cache fingerprint. Use it to intentionally isolate runs that should not share cache entries. |

Correctness constraints:

- Cache keys include model path, model type, dtype, tensor-parallel size, sparse
  method, block size, salt, and method-specific sparse settings.
- Full-prompt hits are not used directly; at least one suffix token is recomputed
  so logits for the first generated token are produced normally.
- Blocks are materialized only after all model layers have written KV for the
  corresponding forward step. Prompt and decode input tokens append to the same
  block-size buffer; complete blocks are inserted into the prefix cache and
  incomplete trailing blocks are discarded when the request is freed.
- Active cached blocks are refcounted and cannot be evicted or returned to the
  free-slot/page pool while referenced.
- `decode_cuda_graph=true` with prefix cache is supported for `vanilla`,
  `omnikv`, and `quest` when `decode_cuda_graph_capture_sampling=false`.
  With `tensor_parallel_size>1`, every rank keeps a rank-local mirrored prefix
  cache with stable logical block ids and rank-local KV payloads.

For API serving, pass these as `--kebab-case` engine flags:

```bash
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=2346 \
PYTHONPATH=src .venv/bin/python -m sparsevllm.entrypoints.openai.api_server \
  --model /models/Qwen2.5-7B-Instruct-1M \
  --served-model-name qwen25-7b-1m \
  --sparse-method vanilla \
  --enable-prefix-caching true \
  --prefix-cache-block-size 16 \
  --engine-prefill-chunk-size 4096 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768
```

For benchmark JSON, use the same snake_case field names:

```json
{
  "sparse_method": "vanilla",
  "enable_prefix_caching": true,
  "prefix_cache_block_size": 16,
  "prefix_cache_max_blocks": 4096,
  "engine_prefill_chunk_size": 4096
}
```

To measure prefix-cache benefit, send repeated requests with an identical prompt
prefix to the same engine process. A benchmark that constructs a new engine for
each sample will not measure cache reuse.

## 5. Unknown-Key Behavior

This matters because "the command ran" does not mean "the parameter was used".

| Backend | Current behavior |
| --- | --- |
| HF DeltaKV custom configs | `set_infer_args` applies keys only if the config has that attribute. Unknown keys log `There is NO <key> in Custom Config!` and are usually ignored. |
| Sparse-vLLM | `LLMEngine.__init__` filters kwargs to dataclass fields in `sparsevllm.Config`. Unknown keys raise `ValueError` by default; they are only logged and ignored when `allow_unknown_config_keys=True`. |
| LLaVA visual script | Builds a selected `infer_config`; unrelated CLI args are not in the config. |
| SCBench DeltaKV branch | Copies `hyper_param`, pops `sparse_method` and `cuda_device`, then sends the rest to `get_generate_api`. |

Do not use a shared mega-config across backends unless it has first been
normalized and inspected for ignored keys.

## 6. Accuracy-Affecting Parameters

### 6.1 Token Keep Budgets

| Parameter | HF behavior | Sparse-vLLM behavior | Risk |
| --- | --- | --- | --- |
| `decode_keep_tokens` | In token-selection helpers, may be integer count or float ratio `<= 1.0`. Also may be list/tuple or comma string in some HF wrappers for per-observation-layer budgets. | Must be explicit integer-like count. The normalizer rejects ratio-style floats. | Copying `0.17` from HF to Sparse-vLLM is wrong. |
| `prefill_keep_tokens` | Prefill/finalization token selection budget. May also use list/comma-string in HF wrappers. | Not supported. Sparse-VLLM prefill/finalization budgets reuse `decode_keep_tokens`. | Passing it to Sparse-VLLM is an unknown config key by default. |
| `sink_keep_tokens` | Prefix tokens kept by cache wrappers and sparse methods. | Prefix tokens kept by cache managers and sparse controller. | Usually same intuition, but storage layout differs. |
| `recent_keep_tokens` | Also copied into internal recent/tail buffers by `CustomConfigMixin`. | Directly consumed by scheduler/cache managers. | Keep this synchronized across backends. |

HF ratio semantics come from `src/deltakv/modeling/token_select.py`. Sparse-vLLM
uses engine/cache planning and cannot safely interpret ratios without a target
context length. Convert explicitly:

```text
int(131072 * 0.17) = 22282
```

### 6.2 Layer Routing

| Parameter | HF behavior | Sparse-vLLM behavior |
| --- | --- | --- |
| `full_attention_layers` | Parsed into a list in `CustomConfigMixin`. Standard DeltaKV uses these as full-attention layers and selection anchors. | Parsed into a list internally. Sparse-vLLM derives observation layers from this value. |
| `snapkv_num_full_layers` | HF SnapKV-related knob. | SnapKV manager can reserve early full layers. |

Special case:

- Matching the string value of `full_attention_layers` is not enough for backend
  parity. Check how the chosen method interprets it.

OmniKV full layers can be selected automatically offline with:

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/analysis/select_omnikv_full_layers.py \
  --model-path <MODEL_DIR> \
  --longbench-root <LONGBENCH_ROOT> \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir <OUTPUT_DIR> \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

The selector writes `<OUTPUT_DIR>/selected_full_layers.json`; use its
`full_attention_layers` value in OmniKV Sparse-vLLM configs, for example:

```json
{
  "sparse_method": "omnikv",
  "full_attention_layers": "0,2,4,11,16,22",
  "decode_keep_tokens": 4096,
  "recent_keep_tokens": 32,
  "sink_keep_tokens": 0,
  "engine_prefill_chunk_size": 512
}
```

This is an offline calibration step, not an automatic `LLM(...)` runtime mode.
After setting `full_attention_layers`, Sparse-vLLM derives its internal
observation layers from it. `observation_layers` is not a supported runtime key.

### 6.3 SnapKV, PyramidKV, OmniKV, Quest

| Parameter | Main consumer | Meaning |
| --- | --- | --- |
| `snapkv_window_size` | HF SnapKV/PyramidKV and Sparse-vLLM SnapKV/DeltaKV-SnapKV | Local observation/recent window. |
| `pool_kernel_size` | HF token selection and some Sparse-vLLM methods | Score smoothing kernel. The consumer differs by method. |
| `pyramid_layer_ratios` | Sparse-vLLM PyramidKV | Explicit per-KV/full-attention-layer keep ratios. A legacy full Transformer-layer list is projected onto KV layers for mixed-attention models. |
| `pyramidkv_start_layer`, `pyramidkv_start_ratio`, `pyramidkv_least_layer`, `pyramidkv_least_ratio` | HF and Sparse-vLLM PyramidKV-style paths | Auto-generate the budget schedule; layer positions count KV/full-attention layers. |
| `quest_chunk_size` | Sparse-vLLM Quest | Quest page size. |
| `quest_token_budget` | Sparse-vLLM Quest | Quest token budget. |
| `chunk_size` | HF Quest adapter | Quest chunk/page size on HF. |
| `decode_keep_tokens` | HF Quest adapter | Quest token budget on HF. |

Quest is a good example of "same research idea, different surface":

```json
{"backend": "hf", "sparse_method": "quest", "decode_keep_tokens": 1024, "chunk_size": 16}
```

is not equivalent to:

```json
{"backend": "sparsevllm", "sparse_method": "quest", "quest_token_budget": 1024, "quest_chunk_size": 16}
```

### 6.4 R-KV

Sparse-vLLM R-KV keeps a small per-layer query cache in the cache manager and
uses it to score candidate KV tokens during decode eviction.

| Parameter | Main consumer | Meaning |
| --- | --- | --- |
| `rkv_compression_interval` | SparseController | Generated-token buffer size between R-KV decode evictions. |
| `rkv_observation_tokens` | RKVCacheManager | Number of recent query states used as the R-KV observation window. It is separate from `rkv_compression_interval`, defaults to `8`, and must be `<= 128` and `<= rkv_compression_interval`. |
| `rkv_alpha` | RKVCacheManager | Paper-style joint score lambda: `alpha * importance - (1 - alpha) * redundancy`. |
| `rkv_redundancy_window` | RKVCacheManager | Candidate window scored for key redundancy. `0` scores the full candidate set; positive values opt into a trailing-window approximation. |

R-KV importance scores are computed from cached observation queries and the
current K cache via the shared prefill score kernel. The decode attention-score
buffer is not the R-KV source of truth.

## 7. DeltaKV HF Cache Semantics

HF DeltaKV behavior is controlled by `KVQwen2Config`, `KVQwen3Config`, and
`KVLlamaConfig`, all using `CustomConfigMixin`.

Important defaults from `CustomConfigMixin`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `deltakv_latent_dim` | `128` | Public name for latent KV width. Internally stored as `kv_compressed_size`. |
| `compressor_token_group_size` | `1` | Token group size for non-cluster compressor references. |
| `deltakv_neighbor_count` | `1` | Cluster/ref neighbor count. It no longer falls back from token group size. |
| `layer_chunk_size` | `1` | Historical layer grouping. Standard runtime asserts it remains `1` in compression paths. |
| `recon_mode` | `delta_in_latent` | Standard compressor residual mode. |
| `ref_mode` | `avg` | Chunk reference mode for non-cluster compression. |
| `use_compression` | `False` | Whether to use learned compressor in standard cache. Checkpoint configs may override. |
| `use_cluster` | `True` | Whether cluster/prototype path is selected. |
| `deltakv_center_ratio` | `0.1` | Center sampling ratio, implemented internally as `cluster_step=max(1, int(1/ratio))` in HF cluster paths. |
| `stride_alpha` | `0.0` | Dynamic center stride growth. `0.0` is fixed stride. |
| `deltakv_latent_quant_bits` | `0` | `2` or `4` enables packed quantized storage in supported paths. |
| `hf_prefill_chunk_size` | `100000000` | HF wrapper chunk size. The large default effectively avoids manual chunking for many prompts. |

### 7.1 Standard `CompressedKVCache`

File: `src/deltakv/modeling/kv_cache.py`.

When `use_cluster=False`:

- `use_compression=True` stores learned latent residuals.
- `compress()` groups tokens by `compressor_token_group_size`, uses a chunk base such as mean
  reference, and stores `compressor_down(kv) - compressor_down(base)`.
- Reconstruction uses `compressor_up(comp_kv) + base`.
- If `use_compression=False` and `deltakv_latent_quant_bits=2` or `4`, the historical KV stored
  from the buffer is direct quantized raw KV, not learned DeltaKV latent.
- If `use_compression=False` and `deltakv_latent_quant_bits=0`, historical key/value tensors
  are stored directly.

In full-attention layers, standard `CompressedKVCache` currently just keeps
sink plus buffer without compressing that layer in the main update path.

### 7.2 Standard `ClusterCompressedKVCache`

File: `src/deltakv/modeling/kv_cache.py`.

When `use_cluster=True`:

- Sink tokens are inserted as initial centers.
- More centers are sampled from history using `deltakv_center_ratio`, optionally with
  dynamic stride from `stride_alpha`.
- Each token selects up to `deltakv_neighbor_count` centers using `cluster_metric` and
  `cluster_on_kv`.
- When `use_compression=True`, cluster compression stores learned latent residual:

```text
compressor_down(token_kv) - compressor_down(mean(selected_centers))
```

- If `deltakv_latent_quant_bits=2` or `4`, this quantizes the latent `comp_kv`, not the original
  full KV and not token-space residual.
- When `use_compression=False`, the same cache can run the direct residual-quant
  path:

```text
residual = token_kv - mean(selected_centers)
```

- In that direct path, `deltakv_latent_quant_bits=2` or `4` quantizes the token-space
  residual itself. No learned `compress_down` or `compress_up` module is used
  for compression or reconstruction.
- The direct path is pad-aware for batched left-padding runs: padding tokens are
  stored with invalid positions and cannot become valid reference centers.

The current LLaVA benchmark exposes this through method `deltakv` with a real
checkpoint. No-checkpoint multimodal runs are the separate `visual_uniform_keep`
baseline, not DeltaKV.

### 7.3 `origin_residual_quant`

Files:

- `src/deltakv/modeling/origin_residual_quant_cache.py`
- `src/deltakv/modeling/qwen2/qwen2_origin_residual_quant_inference.py`
- Equivalent Qwen3/Llama wrappers.

Behavior:

- Full-attention layers store token-space residuals directly:

```text
residual = token_kv - reference
```

- If `deltakv_latent_quant_bits=2` or `4`, the residual is quantized.
- Sparse layers continue to use the original DeltaKV cache path by delegating to
  `super().update(...)`.
- In the clustered variant, references are selected cluster centers, but the
  stored value is token-space residual rather than learned latent residual.

This path is useful for ablation, but it is not "no compressor everywhere" when
non-full sparse layers still route through the standard path.

### 7.4 `all_origin_residual_quant`

Files:

- `src/deltakv/modeling/all_origin_residual_quant_cache.py`
- `src/deltakv/modeling/qwen2/qwen2_all_origin_residual_quant_inference.py`
- Equivalent Qwen3/Llama wrappers.

Behavior:

- Requires `use_cluster=True`.
- Applies token-space residual quantization to every layer.
- Deletes/ignores `compressor_down` and `compressor_up` in the cache update
  path.
- Reconstructs as:

```text
reconstructed_kv = dequantized_residual + mean(selected_centers)
```

This is the path that best matches "use cluster/ref tokens but directly quantize
residuals without a learned compressor".

## 8. Removed `seq_chunk_size` And Split Semantics

`seq_chunk_size` was the most misleading DeltaKV HF parameter and has been
removed from public runtime/training parameters.

It had at least four meanings or historical uses:

| Code path | Actual use |
| --- | --- |
| Non-cluster compression in `CompressedKVCache.compress()` | Token group size for reshaping KV into chunks and computing chunk reference bases. Now `compressor_token_group_size`. |
| E2E compressor training/inference helpers | Token group size for sequence references. |
| `origin_residual_quant` full-layer path | Token group size for building token-space residual bases. |
| `CustomConfigMixin.finalize_cluster_args()` | Legacy fallback for cluster neighbors. This fallback has been removed. |

The last use is semantically wrong by name. A parameter named `seq_chunk_size`
looks like "how many sequence tokens are in a compression chunk", but in cluster
inference it may silently become "how many centers to average per token".

Current behavior:

```python
if config.use_cluster and config.deltakv_neighbor_count is None:
    raise ValueError("deltakv_neighbor_count is required")
```

Practical rule:

- Use `compressor_token_group_size` for token grouping.
- Use `deltakv_neighbor_count` for cluster/reference neighbor count.
- Historical checkpoint `config.json` files may be migrated internally at load
  time so old trained weights remain testable. That is artifact schema
  migration, not accepted public runtime compatibility.

## 9. DeltaKV Sparse-vLLM Semantics

Sparse-vLLM DeltaKV is not a direct line-by-line port of the HF wrapper. It has
engine-owned scheduling, physical cache slots, compressed slot maps, and
method-specific cache managers.

Key Sparse-vLLM public names and internal fields:

| Public parameter | Internal field | Default | Behavior |
| --- | --- | --- | --- |
| `sparse_method` | `vllm_sparse_method` | `vanilla` / `""` | Sparse method selector. |
| `deltakv_checkpoint_path` | `deltakv_path` | `None` | DeltaKV checkpoint path for compressor weights/config. |
| `deltakv_neighbor_count` | `deltakv_k_neighbors` | `4` | Number of centers used for reconstruction. |
| `deltakv_center_ratio` | `cluster_ratio` | `0.1` | Center/prototype rate and capacity input. |
| `cluster_metric` | same | `l2` | Center scoring metric. |
| `deltakv_latent_dim` | `kv_compressed_size` | `128` | Latent dimension. |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `4` | Quantization bits for DeltaKV-style state. |
| `deltakv_full_pool_reserve_ratio` | same | `0.1` | Fraction of KV memory reserved for full-KV pool. |
| `deltakv_sparse_decode_backend` | same | `auto` | Sparse decode backend. `auto` resolves at `Config` construction to `fa2` when `flash_attn` is installed, otherwise `custom`. |
| `deltakv_triton_gather_heads_per_program`, `deltakv_triton_reconstruct_heads_per_program` | same | `4` | Triton grouping controls for the gather/reconstruct kernels. They do not control the materialized sparse-view kernel. |
| `deltakv_triton_materialize_block_tokens` | same | `16` | Token block size for the materialized sparse-view kernel. |
| `allow_unknown_config_keys` | same | `False` | Explicit opt-in for ignoring unknown Sparse-vLLM config keys. |
| `allow_missing_deltakv_path` | same | `False` | Construction-only test escape hatch for missing compressor weights. Do not use for reportable benchmark runs. |

`bitsandbytes` is a declared package dependency because 4-bit and 8-bit
loading paths import it at runtime. A normal `pip install -e .` should install
it; manually curated environments need to include it explicitly.

`benchmark/microbench.py` records both input `engine_hyper_params` and
post-construction `resolved_engine_config` in each result row. Use
`resolved_engine_config.deltakv_sparse_decode_backend` to audit whether an
`auto` backend run actually used `fa2` or `custom`.

### 9.1 Compressor-Backed DeltaKV

Method `deltakv` is the maintained compressor-backed DeltaKV runtime. It
requires `deltakv_checkpoint_path` for real runs.

### 9.2 Legacy `deltakv-less-memory*` Aliases

Files:

- `src/sparsevllm/engine/cache_manager/deltakv_runtime.py`
- `src/sparsevllm/engine/cache_manager/deltakv_less_memory.py`
- `src/sparsevllm/engine/cache_manager/deltakv_less_memory_cuda_graph.py`

The historical `deltakv-less-memory` names are retained so older configs and
regression manifests still load. They normalize to the public `deltakv`
runtime, which initializes compressor modules and requires `deltakv_path`.
The `deltakv-less-memory-cudagraph` alias additionally sets
`decode_cuda_graph=True`.

The slim runtime supports two storage combinations:

| Full-layer storage | Sparse-layer storage |
| --- | --- |
| `full_layer_kv_quant_bits=0` | `deltakv_latent_quant_bits=0` |
| `full_layer_kv_quant_bits=4` with full-layer KIVI enabled | `deltakv_latent_quant_bits=4` |

Other bit combinations fail fast. `deltakv_neighbor_count`,
`deltakv_center_ratio`, `full_attention_layers`, `sink_keep_tokens`,
`recent_keep_tokens`, and `decode_keep_tokens` still affect center/reference
selection, full-layer routing, and sparse-view budgets.

Example Sparse-vLLM smoke command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_checkpoint_path":"<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

Important differences from HF:

- `deltakv_neighbor_count` maps to `deltakv_k_neighbors`, not old `k_neighbors`.
- `deltakv_center_ratio` influences algorithmic behavior and memory planning.
- `decode_keep_tokens` must be an integer budget.
- `full_attention_layers` may derive observation layers.
- Unknown Sparse-vLLM config keys and missing DeltaKV compressor paths fail
  fast by default. Use `allow_missing_deltakv_path` only for construction-only
  tests, not reportable runs.
- Sparse-vLLM methods can physically evict or re-map cache slots during prefill
  and decode, unlike HF `DynamicCache` wrappers.

## 10. Speed And Capacity Parameters

These do not only affect speed. In Sparse-vLLM, they can change admission,
queueing, and whether a benchmark measures the intended batch.

| Parameter | Backend | Meaning |
| --- | --- | --- |
| `engine_prefill_chunk_size` | Sparse-vLLM | Max prefill chunk scheduled per sequence per step for `all_chunked`. Do not set it for `long_bs1full_short_batch`; that policy derives the chunk size from `long_prefill_offload_threshold`. |
| `hf_prefill_chunk_size` | HF | Wrapper/model chunk size for long input forwarding. Often a large value means "do not chunk". |
| `max_model_len` | Sparse-vLLM | Hard engine capacity for prompt plus generated tokens. Affects allocation and request validation. |
| `long_prefill_offload_threshold` | Sparse-vLLM | Exact boundary between complete batched short prefill and isolated chunked RawKV offload under `long_bs1full_short_batch`. Defaults to `98304` tokens (96K) and also becomes that policy's `chunk_prefill_size`. |
| `max_num_batched_tokens` | Sparse-vLLM | Aggregate scheduler cap for tokens in one step; memory heuristics may reduce it. `all_chunked` permits this cap to be smaller than `engine_prefill_chunk_size`. `long_bs1full_short_batch` normalizes it to at least `long_prefill_offload_threshold` so the boundary prompt fits atomically. |
| `max_num_seqs_in_batch` | Sparse-vLLM | Max active sequences in a prefill/decode step. |
| `max_decoding_seqs` | Sparse-vLLM | Max sequences in decode queue. |
| `gpu_memory_utilization` | Sparse-vLLM | Fraction of total GPU memory used for cache planning. |
| `tensor_parallel_size` | Sparse-vLLM | Number of TP ranks/processes. |
| `num_kvcache_slots` | Sparse-vLLM | Optional explicit KV slot override. |
| `enable_prefix_caching` | Sparse-vLLM | Enables prefix KV reuse for vanilla, OmniKV, and QuEST only. |
| `prefix_cache_block_size` | Sparse-vLLM | Prefix-cache block size; must equal `quest_chunk_size` for QuEST. |
| `prefix_cache_max_blocks` | Sparse-vLLM | Optional live-block cap for prefix cache. |
| `prefix_cache_salt` | Sparse-vLLM | Additional fingerprint salt for cache isolation. |
| `admission_wave_size` | `scripts/benchmarks/bench_sparse_vllm.py` | Benchmark-only staged admission. |
| `wave_decode_gap_steps` | `scripts/benchmarks/bench_sparse_vllm.py` | Benchmark-only delay before adding next wave. |
| `max_decode_steps_after_full` | `scripts/benchmarks/bench_sparse_vllm.py` | Benchmark-only decode window cap after full admission. |
| `enable_profiler` | Sparse-vLLM | Enables repo profiler. Also enabled by `PROFILER_SVLLM` env var. |
| `throughput_log_interval_s` | Sparse-vLLM | Periodic throughput logging interval. |

### 10.1 Why `chunk_prefill_size` Must Be Split

HF meaning:

- Used by wrappers to split long prompt forwarding.
- Very large values often mean "avoid wrapper chunking".
- Does not reserve a scheduler-owned KV pool.

Sparse-vLLM meaning:

- Controls scheduler step size.
- Participates in warmup length.
- Affects long/short bucket classification.
- Affects max batched tokens and cache admission.
- Can make the engine assert if memory heuristics estimate insufficient
  activation headroom.

Do not copy the same numeric value across backends.

### 10.2 Benchmark Queueing And Wave Admission

`scripts/benchmarks/bench_sparse_vllm.py` reports:

- TTFT.
- Prefill throughput.
- Decode throughput.
- ITL.
- Average active batch size.
- Peak memory.
- Speedup against vanilla for the same length and batch size.

If a run has queued requests, the printed BS column gets a `*`. That means the
requested batch size was not fully active at the same time for the entire
measurement.

Wave admission is intended for methods that can host more decode requests after
prefill eviction:

```bash
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --max_decode_steps_after_full 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096}'
```

For fair speed comparisons, compare:

- same model and tokenizer,
- same prompt length,
- same output length,
- same sampling settings,
- same batch admission policy,
- same actual full-admission status,
- same decode measurement scope.

## 11. `scripts/benchmarks/bench_sparse_vllm.py` Specific Notes

The benchmark sets stable defaults before applying `--hyper_params`:

```json
{
  "enforce_eager": true,
  "gpu_memory_utilization": 0.8,
  "engine_prefill_chunk_size": 4096,
  "tensor_parallel_size": 1
}
```

It validates canonical names for backend `sparsevllm`, then passes canonical
kwargs into `LLM(...)`.

Important behavior:

- `method=vanilla` forces `sparse_method="vanilla"`.
- `max_model_len` from `--hyper_params` is popped and overwritten as
  `length + output_len + 100` to keep benchmark cases consistent.
- `max_num_seqs_in_batch` and `max_decoding_seqs` default to the requested batch
  size if not supplied.
- `--hyper_params` accepts either inline JSON or `@file.json`.
- Old helper flags such as `--gpu_util`, `--chunk_size`, `--tp`, and
  `--enforce_eager` were removed. Use `--hyper_params`.

## 12. Model Loading And Quantization Parameters

HF `get_generate_api` routes model-loading settings through
`src/deltakv/quantization.py`.

| Parameter | HF behavior | Sparse-vLLM behavior through `infer_config` |
| --- | --- | --- |
| `torch_dtype` | Pops from runtime config and controls model dtype. | Ignored unless a Sparse-vLLM config field of that name is later added. |
| `load_in_4bit` | Builds BitsAndBytes 4-bit load config. | Ignored by Sparse-vLLM config filtering. |
| `load_in_8bit` | Builds BitsAndBytes 8-bit load config. | Ignored by Sparse-vLLM config filtering. |
| `quant_skip_modules` / `llm_int8_skip_modules` | Additional modules excluded from BnB quantization. | Ignored. |
| `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant`, `bnb_4bit_quant_type`, `bnb_4bit_quant_storage` | BnB 4-bit options. | Ignored. |
| `llm_int8_threshold`, `llm_int8_enable_fp32_cpu_offload`, `llm_int8_has_fp16_weight` | BnB 8-bit options. | Ignored. |

The default skip list intentionally includes compressor modules such as
`compress_down`, `compress_up`, `k_compress_down`, and `v_compress_up`.

Do not confuse base-model quantization with KV-cache quantization:

- `load_in_4bit` quantizes model weights at load time.
- `deltakv_latent_quant_bits=2` or `4` quantizes cached KV/latent/residual state in method-specific
  ways.

## 13. LLaVA-OneVision Visual-Cache Parameters

Main files:

- `benchmark/multimodal/visual_cache/run_visual_cache.py`
- `src/deltakv/modeling/llava_onevision_deltakv.py`
- `docs/benchmarking/multimodal/README.md`

Current implemented benchmark methods:

| Method label | Actual behavior |
| --- | --- |
| `vanilla` | Standard `LlavaOnevisionForConditionalGeneration`. |
| `deltakv` | LLaVA-OneVision DeltaKV wrapper with a real learned compressor checkpoint. Uses the checkpoint config plus CLI keep budgets. |
| `visual_uniform_keep` | DeltaKV wrapper infrastructure, but no compressor, no cluster, no ref tokens. Uniformly keeps visual tokens and drops the rest. |
| `visual_uniform_keep_int4` | Same uniform visual token keep path, plus direct int4 storage of kept visual KV. |

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `visual_token_prune_only` | Restricts cache dropping/pruning to visual tokens. |
| `visual_token_keep_ratio` / CLI `--visual_keep_ratio` | Fraction of eligible visual tokens kept by uniform subsampling. |
| `--quantize_visual_kv` | Sets `deltakv_latent_quant_bits=4` in no-checkpoint fallback. |
| `--deltakv_center_ratio` | Center/prototype sampling ratio for compressor-backed `deltakv`. |
| `--deltakv_neighbor_count` | Number of selected ref centers for compressor-backed `deltakv`. |
| `recent_keep_tokens`, `sink_keep_tokens`, `full_attention_layers` | Passed into text config and affect cache buffer behavior. |
| `decode_keep_tokens`, `prefill_keep_tokens` | Present for wrapper compatibility; current visual uniform path does not use SnapKV-style attention scoring for pruning. |

Current limitations:

- `visual_token_prune_only` currently raises for batch size greater than 1 in
  HF cache update.
- `visual_uniform_keep` explicitly sets `use_compression=False` and
  `use_cluster=False`.
- "LLaVA visual keep10" without checkpoint is still not using cluster/ref
  tokens. It is uniform pruning.

## 14. OpenAI-Compatible Serving

The OpenAI-compatible online serving entrypoint is:

```bash
sparsevllm-openai-server \
  --model /path/to/local/Qwen2.5-1.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

If the console script has not been refreshed in the active virtual
environment, the equivalent module entrypoint is:

```bash
python -m sparsevllm.entrypoints.openai.api_server \
  --model /path/to/local/Qwen2.5-1.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

`--model` is the local model directory passed to `sparsevllm.Config.model`.
`--served-model-name` is the external OpenAI API model id accepted in request
JSON. They may differ; requests must use the served name.

### 14.1 Serving CLI Parameters

The serving entrypoint has dedicated server flags:

| CLI flag | Default | Meaning |
| --- | --- | --- |
| `--model` | required | Local model directory loaded by Sparse-vLLM. |
| `--served-model-name` | `--model` value | Model id exposed through `/v1/models` and accepted by `/v1/completions`. |
| `--host` | `0.0.0.0` | Uvicorn bind host. |
| `--port` | `8000` | Uvicorn bind port. |
| `--engine-kwargs` | unset | JSON object or path to a JSON object with Sparse-vLLM engine kwargs. |
| `--request-log-dir` | unset | Optional directory for per-request JSON logs. |
| `--response-parser` | unset | Optional Chat Completions and Responses output parser. `qwen3` and `minimax_m2` split model output into reasoning, content, and tool calls; the loaded tokenizer selects the matching response template. |

The `/v1/models` entry also advertises the engine's effective
`max_model_len`. vLLM-compatible clients use this extension to discover the
real context window instead of treating it as unknown. A smart router reports
the smallest context window among healthy workers serving the same model.

`/livez` reports process liveness. `/health` and `/readyz` report traffic
readiness and return HTTP 503 after a fatal engine-step error. The CLI server
then exits with status 1 so an external supervisor can replace the process and
its CUDA context. The smart router probes worker readiness before routing,
removes failed workers, and automatically admits restarted workers once they
are ready. It deliberately does not replay interrupted requests. See
`deploy/systemd/README.md` for per-GPU worker and router service templates.

Additional `--kebab-case` flags are parsed as Sparse-vLLM engine kwargs. Use
the canonical semantic keys accepted by
`normalize_runtime_params(..., backend="sparsevllm")` for public runtime
controls. Non-legacy `src/sparsevllm/config.py` fields such as
`max_model_len`, `max_num_seqs_in_batch`, `gpu_memory_utilization`, and
`throughput_log_interval_s` may also be passed. Legacy public names listed in
Section 3 are still rejected during engine initialization even if the serving
parser can recognize their spelling. If `--engine-kwargs` and explicit CLI
engine flags set the same key, startup fails instead of silently choosing one
value.

Example:

```bash
sparsevllm-openai-server \
  --model /models/Qwen2.5-1.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 32768 \
  --max-num-seqs-in-batch 8 \
  --gpu-memory-utilization 0.9 \
  --sparse-method snapkv \
  --sink-keep-tokens 64 \
  --recent-keep-tokens 512
```

Important serving defaults:

| Engine parameter | Serving default | Notes |
| --- | --- | --- |
| `sparse_method` / `vllm_sparse_method` | `""` | Dense/vanilla path unless explicitly set. |
| `tensor_parallel_size` | `1` | Use `CUDA_VISIBLE_DEVICES=...` plus `--tensor-parallel-size N` for multi-GPU TP. |
| `gpu_memory_utilization` | `0.8` | Inherited from `Config`. |
| `max_model_len` | `128000` | Inherited from `Config`; prompt length plus `max_tokens` must fit. |
| `max_num_batched_tokens` | `65536` | Inherited from `Config`. |
| `max_num_seqs_in_batch` | `32` | Inherited from `Config`. |
| `max_decoding_seqs` | `64` | Inherited from `Config`. |
| `engine_prefill_chunk_size` / `chunk_prefill_size` | `8192` | `all_chunked` only. Use the semantic `--engine-prefill-chunk-size` on the CLI. |
| `long_prefill_offload_threshold` | `98304` | `long_bs1full_short_batch` only. Also determines that policy's chunk size. |
| `enable_prefix_caching` | `false` | Pass `--enable-prefix-caching true` to enable prefix KV reuse. |
| `prefix_cache_block_size` | `16` for vanilla/OmniKV, `quest_chunk_size` for QuEST | Use `--prefix-cache-block-size`; QuEST rejects values different from `quest_chunk_size`. |
| `prefix_cache_max_blocks` | unset | Optional cache capacity cap. |
| `prefix_cache_salt` | `""` | Optional cache fingerprint isolation salt. |
| `throughput_log_interval_s` | `0.0` in serving | The server disables periodic `Avg TP` logs by default and logs per request instead. Pass `--throughput-log-interval-s 10` to re-enable periodic throughput logs. |

DeltaKV-family sparse methods are intentionally not exposed through the OpenAI
server yet. Values whose normalized method starts with `deltakv` fail fast at
server startup. Use offline experiment entrypoints for those methods until
serving correctness and memory behavior are validated.

### 14.2 `/v1/completions` Request Parameters

The implemented endpoint is OpenAI-style text completions:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'
```

Streaming uses SSE:

```bash
sparsevllm-openai-client \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "San Francisco is a" \
  --max-tokens 7 \
  --temperature 0
```

The raw HTTP stream remains standard SSE (`data: {...}` frames ending with
`data: [DONE]`). The helper client parses those frames and prints only the
incremental text.

Online serving requires a Hugging Face fast tokenizer backend with
`DecodeStream` support. Sparse-vLLM keeps independent request-local visible and
raw incremental decoders, so byte-level tokens that split a multi-byte Unicode
character are buffered until the character is complete. This applies uniformly
to Completions, Chat Completions, and Responses streaming; the concatenated
text deltas match the corresponding non-streaming final text. Unsupported slow
tokenizers fail explicitly instead of falling back to unsafe per-token decoding
or replacement-character filtering.

Chat completions are also exposed:

```bash
sparsevllm-openai-client \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --chat \
  --prompt "Explain Sparse-vLLM in one sentence."
```

Supported JSON fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | required | Must equal `--served-model-name` if provided, otherwise the `--model` value. |
| `prompt` | required | String, token id list, list of strings, or list of token id lists. |
| `max_tokens` | `16` | Maps to `SamplingParams.max_tokens`; must be positive. |
| `max_completion_tokens` | `null` | Chat-only OpenAI-compatible alias for `max_tokens`; requests that explicitly set both to different values fail fast. |
| `temperature` | `1.0` | Maps to `SamplingParams.temperature`; `0` means greedy sampling. |
| `top_p` | `1.0` | Maps to `SamplingParams.top_p`; must be in `(0, 1]`. |
| `top_k` | `0` | Maps to `SamplingParams.top_k`; `0` disables top-k filtering. |
| `n` | `1` | Only `1` is currently supported. |
| `stream` | `false` | `true` returns `text/event-stream` chunks ending with `data: [DONE]`. |
| `ignore_eos` | `false` | Continue until `max_tokens` even if EOS is generated. |
| `stop` | `null` | String or list of strings. Stop text is omitted from the returned completion. |
| `logprobs` | `null` | Non-negative integer up to 5 for `/v1/completions`; returns sampled-token logprobs and up to this many top logprobs. |

Unknown JSON fields are rejected instead of silently ignored. This is stricter
than some OpenAI-compatible servers, but it avoids accepting parameters that do
not affect research results.

`stop` and `logprobs` are supported independently. Requests that set both fail
fast because text-level stop trimming can otherwise make returned token logprobs
disagree with the visible output.

`/v1/chat/completions` supports the same sampling fields plus `messages`.
Messages must use `developer`, `system`, `user`, `assistant`, or `tool` roles.
String content and text-only content-part lists are supported; unknown nested
message fields are rejected. Assistant messages may include the compatible
`reasoning_content` extension and OpenAI function `tool_calls`; tool result
messages must use the `tool` role and a matching `tool_call_id`. These fields
are passed through to the Hugging Face chat template so historical reasoning,
calls, and results round-trip into the next prompt. Role-specific fields and
malformed function call objects fail validation instead of being ignored.
The `developer` role is rendered as `system` for Hugging Face chat templates
because most local tokenizer templates do not define a separate developer
role. When the loaded tokenizer exposes a chat template, the server renders
messages with `apply_chat_template(..., add_generation_prompt=True)`;
otherwise it uses a simple role-prefixed prompt.

Chat requests may set `reasoning_effort` to `none`, `minimal`, `low`,
`medium`, `high`, or `xhigh`; `none` maps to `enable_thinking=false` and every
other value maps to `true`. The direct `enable_thinking` field and
`"chat_template_kwargs": {"enable_thinking": false}` remain available for
Qwen3-style templates. Following vLLM's Chat API contract,
`chat_template_kwargs` is an open JSON object whose values are passed directly
to the tokenizer template. The compatible top-level `preserve_thinking` field
is normalized into `chat_template_kwargs.preserve_thinking`; this lets local
Qwen-family clients replay historical reasoning when the loaded template
supports that switch. Duplicate controls with the same value are accepted,
while conflicting values, non-boolean known thinking controls, or template
kwargs without a tokenizer chat template fail fast.

Chat function tools accept OpenAI nested function schemas and the compatible
flat Responses form. Effective tools are passed through the tokenizer's
`tools` kwarg. `tool_choice` supports `null`, `"auto"`, and `"none"`; `none`
omits tools from the generation prompt. Named/required choices and
`parallel_tool_calls=false` fail explicitly because their generation
constraints are not implemented. The server parses Qwen-style
`<tool_call>`/`<tool_calls>` and MiniMax M2
`<minimax:tool_call><invoke ...>` output only when tools are effective and
never executes tools itself. Enable the matching `--response-parser` when
thinking and tool calling are both active; without it, reasoning-only output
remains raw `content`.

With `--response-parser qwen3` or `--response-parser minimax_m2`,
non-streaming Chat responses split local raw
reasoning into the Sparse-vLLM `message.reasoning_content` extension and place
the visible answer in `message.content`. Function calls use OpenAI
`message.tool_calls` and `finish_reason="tool_calls"`. Streaming uses
`delta.reasoning_content` for local raw reasoning and standard indexed
`delta.tool_calls` chunks for function name and argument deltas. Cross-chunk
reasoning tags and tool JSON are parsed by state machines; malformed or
unclosed output is reported explicitly. Without the reasoning parser, raw
reasoning text remains in `content`, preserving the previous behavior.

Chat `logprobs=true` enables sampled-token logprobs, and `top_logprobs`
controls the number of top alternatives up to 20. Logprobs are rejected when
reasoning or tool output parsing is active because raw generated token
positions cannot be represented truthfully against split/hidden Chat fields.
`/v1/completions` remains a raw prompt endpoint and does not add a server-side
thinking switch; clients can include prompt-level markers such as `/think` or
`/no_think` themselves if needed.

`/v1/responses` is exposed as a separate endpoint for item-based input and
output. The first implementation supports text input, text-only message items,
`function_call_output` input items, function tool schemas, `reasoning.effort`,
non-streaming responses, and Responses SSE streaming. `max_output_tokens` maps to
`SamplingParams.max_tokens`; `temperature`, `top_p`, and `top_k` map directly
to sampling parameters. `tool_choice` is limited to `null` or `"auto"`;
`parallel_tool_calls=false` and `reasoning.summary` fail explicitly until those
semantics are implemented. `stream=true` returns Responses semantic SSE events
instead of Chat Completions chunks.

For client compatibility, `store=false` (or omission) and a non-empty
`prompt_cache_key` are accepted. Sparse-vLLM does not persist response objects,
so `store=true` fails explicitly. `prompt_cache_key` is retained in request
logs as a cache-grouping hint but does not alter the rendered model prompt or
replace Sparse-vLLM's exact-prefix cache matching.

When `--response-parser qwen3` or `--response-parser minimax_m2` is enabled,
`/v1/responses` parses model output that starts with `<think>` into a
Sparse-vLLM extension reasoning item followed by the assistant message or
function call item. This extension exposes local model reasoning text for
reproducibility; it is not claiming equivalence with OpenAI-hosted reasoning
tokens, which are not exposed as raw text. If the parser is not enabled,
generated text is returned unchanged as `output_text`.
`reasoning.effort="none"` maps to `enable_thinking=false`; other effort values
map to `enable_thinking=true`. Conflicts with explicit
`chat_template_kwargs.enable_thinking` fail fast. In streaming mode,
`response.reasoning_text.delta` is a Sparse-vLLM extension event for local raw
reasoning text, not an OpenAI-hosted raw reasoning token field.

Function tools are passed to tokenizer chat templates through the `tools`
kwarg when supported. The server normalizes OpenAI function tool schemas,
adapts them to the loaded tokenizer's Qwen or MiniMax template shape, and
parses explicit Qwen `<tool_call>...</tool_call>` or MiniMax
`<minimax:tool_call><invoke ...>` output into Responses `function_call` items.
It does not execute tools; applications must execute tools and send results
back as `function_call_output` input items. Streaming tool calls emit a
`function_call` output item plus `response.function_call_arguments.delta` and
`response.function_call_arguments.done` events.

Prefix-cache matching accepts full `chat` and `response` selectors. The worker
renders them with the same endpoint prompt helpers used for real generation,
so messages, instructions, tools, reasoning controls, and chat template kwargs
participate in the cache-match key consistently. The smart router uses these
full selectors rather than approximating Chat requests from messages alone.

The server logs one request-start line and one request-finish or request-cancel
line per `/v1/completions` request. It does not log every generated token.

## 15. Benchmark Entrypoints

### 15.1 LongBench And MathBench

`benchmark/long_bench/pred.py` and `benchmark/math_bench/pred.py`:

- Start with `infer_config={"max_model_len": args.max_model_len}`.
- Merge `--hyper_param`, either JSON file or inline JSON.
- Call `get_generate_api(...)`.
- Pass generation parameters such as `temperature`, `top_p`, and `top_k`.

Backend differences after `get_generate_api` returns:

| Generation kwarg | HF path | Sparse-vLLM wrapper path |
| --- | --- | --- |
| `max_new_tokens` | Used. | Maps to `SamplingParams.max_tokens`. |
| `max_tokens` | Not primary. | Fallback if `max_new_tokens` absent. |
| `do_sample` | Used. | Only decides whether temperature becomes `0.0`. |
| `temperature` | Used. | Used. Greedy sets `0.0`; tiny sampling values clamp to `1e-5`. |
| `top_p` | Used by HF generation. | Not forwarded by current wrapper. |
| `top_k` | Used by HF generation. | Not forwarded by current wrapper. |
| `eos_token_id` | Used by HF generation. | Not forwarded by current wrapper. |
| `past_key_values` | Manual HF path can use it. | Accepted for signature compatibility but ignored. |

### 15.2 NIAH

`benchmark/niah/test_niah.py` builds an `infer_config` manually from function
arguments. It now uses canonical keys such as `sparse_method`,
`hf_prefill_chunk_size`, `engine_prefill_chunk_size`,
`gpu_memory_utilization`, and `use_cluster`.

This is convenient for quick experiments, but it increases the chance that one
backend ignores keys intended for the other. Use canonical names when adding new
NIAH runs.

### 15.3 SCBench

`benchmark/scbench/run_scbench.py` has a DeltaKV branch for:

```text
deltakv, full_deltakv, origin_residual_quant, all_origin_residual_quant,
snapkv, pyramidkv, palu, quest
```

It copies `hyper_param`, extracts `deltakv_checkpoint_path`, pops `sparse_method` and
`cuda_device`, and calls `get_generate_api(..., return_model=True)`. This path
is HF-oriented. Sparse-vLLM-style engine parameters in this branch usually do
not apply.

## 16. Known Ambiguities And Recommended Names

| Removed legacy name | Problem | Canonical name |
| --- | --- | --- |
| `seq_chunk_size` | Token grouping in some paths, legacy `k_neighbors` fallback in cluster paths. | `compressor_token_group_size` for grouping; `deltakv_neighbor_count` for neighbors. |
| `chunk_prefill_size` | Same name means HF wrapper chunking or Sparse-vLLM scheduler chunking. | `hf_prefill_chunk_size` or `engine_prefill_chunk_size`. |
| `num_top_tokens` | Count, ratio, or per-layer list depending on HF path; removed as a Sparse-vLLM config field. | `decode_keep_tokens`, with explicit ratio conversion notes. |
| `num_top_tokens_in_prefill` | Removed Sparse-vLLM prefill budget; still exists as an HF internal field. | HF-only `prefill_keep_tokens`; Sparse-vLLM uses `decode_keep_tokens`. |
| `compressor_path` | HF top-level path; ignored by Sparse-vLLM unless normalized. | `deltakv_checkpoint_path`. |
| `deltakv_path` | Sparse-vLLM path; ignored by HF unless normalized. | `deltakv_checkpoint_path`. |
| `model_cls` | HF method selector only. | `sparse_method` for portable intent. |
| `vllm_sparse_method` | Sparse-vLLM method selector only. | `sparse_method` for portable intent. |
| `tail_token_size` | Historical HF recent-buffer name. | `recent_keep_tokens`. |
| `kv_quant_bits` | Quantizes different objects in different paths. | `deltakv_latent_quant_bits`, plus document object being quantized. |
| `deltakv_visual_compress_only` | Says DeltaKV/compress even for uniform pruning. | `visual_token_prune_only`. |
| `deltakv_visual_keep_ratio` | Tied to misleading visual compress name. | `visual_token_keep_ratio`. |

## 17. Alignment Workflow

### 17.1 Accuracy Alignment

Before comparing accuracy, match:

- model path,
- tokenizer path,
- backend,
- method family,
- checkpoint path,
- keep budgets,
- sink/recent budgets,
- full/observation layers,
- cluster/ref settings,
- latent width and quantization mode,
- sampling parameters,
- prompt formatting and truncation.

Use greedy decoding for deterministic checks where possible.

HF example:

```bash
python benchmark/long_bench/pred.py \
  --model_path <MODEL> \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path <COMPRESSOR> \
  --hyper_param '{"decode_keep_tokens":0.17,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","hf_prefill_chunk_size":32768}'
```

Sparse-vLLM example with explicit count:

```bash
python benchmark/long_bench/pred.py \
  --model_path <MODEL> \
  --backend sparsevllm \
  --hyper_param '{"sparse_method":"deltakv","deltakv_checkpoint_path":"<COMPRESSOR>","decode_keep_tokens":22282,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","engine_prefill_chunk_size":512,"gpu_memory_utilization":0.9}'
```

### 17.2 Speed Alignment

Before comparing speed, separate three classes of settings:

| Class | Examples | Should be matched? |
| --- | --- | --- |
| Sparse policy | `decode_keep_tokens`, `recent_keep_tokens`, `full_attention_layers`, `deltakv_center_ratio` | Yes for method parity. |
| Engine capacity | `gpu_memory_utilization`, `engine_prefill_chunk_size`, `max_num_batched_tokens` | Match only when measuring same engine regime; tune when measuring max capacity. |
| Benchmark policy | `batch_sizes`, `admission_wave_size`, `max_decode_steps_after_full` | Must be reported with results. |

Full-attention baseline example:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization":0.95,"engine_prefill_chunk_size":4096}'
```

Sparse wave-admission example:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --max_decode_steps_after_full 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096}'
```

## 18. Environment Variables

Environment variables are not normalized by `normalize_runtime_params`. Treat
them as process-level switches and always record them with benchmark results.

| Variable | Scope | Effect |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | all GPU runs | Selects physical GPUs. Multi-worker scripts map local rank into this list. |
| `PYTHONPATH` | all repo runs | Should include `$PWD/src` when running from source. |
| `LOG_LEVEL` | repo logging | Controls `deltakv` and `sparsevllm` logger verbosity. |
| `PROFILER_SVLLM` | Sparse-vLLM | Enables the repo profiler, equivalent to setting `enable_profiler=True`. |
| `CUDA_SYNC_SVLLM` | Sparse-vLLM profiler | Synchronizes CUDA around profiler timing points. Adds overhead; use only for profiling. |
| `SPARSEVLLM_MASTER_PORT` | Sparse-vLLM TP | Master port for spawned tensor-parallel workers. |
| `SPARSEVLLM_DEBUG_SLOTS` | Sparse-vLLM scheduler/cache | Prints extra cache-slot admission/debug information. |
| `SPARSEVLLM_DELTAKV_L2_BLOCK_N/M/D/NUM_WARPS` | Sparse-vLLM DeltaKV kernels | Overrides Triton block/warp tuning for DeltaKV L2 selection. |
| `SPARSEVLLM_DELTAKV_STANDALONE_TEMP_SLOTS` | DeltaKV standalone/snapkv | Overrides temporary reconstruction slot reservation. |
| `SPARSEVLLM_DELTAKV_STANDALONE_DECOMPRESS_CHUNK_TOKENS` | DeltaKV standalone/snapkv | Overrides reconstruction chunking size. |
| `OMNIKV_ASSERT` | OmniKV fused kernel | Enables extra assertions. |
| `USE_ADVSEL` | Sparse-vLLM DeltaKV | Enables experimental advanced selection paths. |
| `MANUAL_GEN_CHUNK_PREFILL_SIZE` | HF generation wrappers | Forces manual prompt chunking in selected HF paths. |
| `BAN_EOS` | HF generation wrappers | Masks EOS token during generation where implemented. |
| `NOT_SKIP_SPECIAL_TOKENS` | HF generation decode | Keeps special tokens in decoded output. |
| `ENABLE_HF_GEN` | HF generation path | Forces the model `.generate(...)` path in `get_generate_api`. |
| `KVZIP_DEBUG` | KVzip adapter | Prints KVzip debug memory/cache information. |
| `DEBUG` | benchmark/HF paths | Enables extra prompt/cache debug prints in several scripts. |
| `DELTAKV_OUTPUT_DIR` | LongBench/MathBench/SCBench | Output root for benchmark predictions/logs. |
| `DELTAKV_DATA_DIR` | LongBench/MathBench | Dataset root fallback. |
| `DELTAKV_LONGBENCH_DATA_DIR` | LongBench | LongBench-specific dataset root override. |
| `DELTAKV_OUTPUT_BASE` | NIAH | NIAH output root. |
| `ENABLE_THINKING` | MathBench | Controls tokenizer chat template thinking mode. |
| `LOCAL_RANK` | compressor training | Distributed training local rank override. |
| `FORCE_QWEN` | compressor training | Forces Qwen2 model branch in `train_compressor.py`. |
| `ANALYSIS` | compressor training/analysis | Enables extra analysis outputs in cluster training code. |
| `MSE_DETACH`, `NTP_DETACH` | compressor training | Ablation switches for loss-gradient detaching. |
| `REMOVE_COMP`, `REMOVE_REF` | HF cache ablation | Removes compressor or reference components in selected cache paths. |
| `COPY_ON_GPU` | SCBench DeltaKV wrapper | Keeps copies on GPU in `DeltaKVGreedySearch`. |

Profiling example:

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=$PWD/src:$PYTHONPATH \
LOG_LEVEL=DEBUG \
PROFILER_SVLLM=1 \
CUDA_SYNC_SVLLM=1 \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods deltakv \
  --hyper_params '{"sparse_method":"deltakv","deltakv_checkpoint_path":"<COMPRESSOR>","engine_prefill_chunk_size":4096,"decode_keep_tokens":4096,"gpu_memory_utilization":0.9}'
```

Benchmark output/data example:

```bash
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_DATA_DIR=<DATA_ROOT> \
PYTHONPATH=$PWD/src:$PYTHONPATH \
python benchmark/long_bench/pred.py \
  --model qwen25-deltakv \
  --model_path <MODEL> \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path <COMPRESSOR> \
  --hyper_param '{"hf_prefill_chunk_size":32768,"decode_keep_tokens":0.17,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18"}'
```

## 19. Safe Config Checklist

Before launching a run:

- Use `sparse_method` and `deltakv_checkpoint_path` in shared configs.
- Use `hf_prefill_chunk_size` or `engine_prefill_chunk_size`; do not use bare
  `chunk_prefill_size` in new shared configs.
- If `use_cluster=True`, set `deltakv_neighbor_count` explicitly.
- If using Sparse-vLLM, convert all ratio budgets to token counts.
- If using prefix cache, keep `sparse_method` in `vanilla`, `omnikv`, or
  `quest`; generated decode input tokens are cached by default once they
  complete full prefix-cache blocks; and keep
  `decode_cuda_graph_capture_sampling=false` when using `decode_cuda_graph`.
- For QuEST prefix cache, set `prefix_cache_block_size` equal to
  `quest_chunk_size` or omit it.
- If using LLaVA no-checkpoint path, label it as visual uniform pruning, not
  DeltaKV compressor inference.
- Record whether `deltakv_latent_quant_bits=2` or `4` quantized latent state, raw KV, or residual.
- Check logs for ignored unknown keys.
- For throughput, record whether the run queued and whether decode throughput
  came from full admission or fallback scope.
