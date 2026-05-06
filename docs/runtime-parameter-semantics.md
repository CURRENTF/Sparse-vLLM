# Runtime Parameter Semantics And Audit

This document is the source of truth for runtime and benchmark parameters in
this repository. It is written for two readers:

- Humans who need to run or compare experiments without silently changing the
  method.
- Agents that need to edit this repo later without repeating old parameter
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
  "prefill_keep_tokens": 4096,
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
- `compressor_token_group_size` is for compressor token grouping.
  `deltakv_neighbor_count` is for selected reference/prototype count.
- LLaVA `--deltakv_checkpoint_path none` plus `visual_uniform_keep` is not
  learned DeltaKV. It is a visual-token uniform-pruning baseline.

## 2. Runtime Parameter Flow

There are four main runtime entry paths.

| Entry | Parameter container | Normalization | Main consumers |
| --- | --- | --- | --- |
| `scripts/bench_sparse_vllm.py` | `--hyper_params` JSON | `normalize_runtime_params(..., backend="sparsevllm")` | `sparsevllm.Config`, `Scheduler`, `CacheManager`, `SparseController` |
| `benchmark/long_bench/pred.py` and `benchmark/math_bench/pred.py` | `--hyper_param` JSON or file | `get_generate_api(...)` normalizes after merge | HF wrappers or Sparse-vLLM engine |
| `benchmark/scbench/run_scbench.py` DeltaKV branch | `--hyper_param` JSON dict | `get_generate_api(...)` normalizes | HF wrappers |
| `scripts/bench_llava_onevision_visual_prune.py` | dedicated CLI args | no global normalizer; builds `config.deltakv_infer_config` | LLaVA wrapper and `KVQwen2Config` |

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
Internal fields still exist in HF config objects and `sparsevllm.Config`, but
they are not valid user-facing runtime parameters.

| Canonical key | HF target | Sparse-vLLM target | Meaning |
| --- | --- | --- | --- |
| `sparse_method` | `model_cls` | `vllm_sparse_method` | Method selector. |
| `deltakv_checkpoint_path` | top-level `compressor_path` | `deltakv_path` | DeltaKV checkpoint directory. |
| `decode_keep_tokens` | `num_top_tokens` | `num_top_tokens` | Decode-time important-token budget. |
| `prefill_keep_tokens` | `num_top_tokens_in_prefill` | `num_top_tokens_in_prefill` | Prefill/finalization important-token budget. |
| `sink_keep_tokens` | `num_sink_tokens` | `num_sink_tokens` | Prefix tokens always kept. |
| `recent_keep_tokens` | `num_recent_tokens` | `num_recent_tokens` | Recent tail tokens always kept. |
| `full_attention_layers` | `full_attn_layers` | `full_attn_layers` | Layers that stay full, or observation anchors depending on method. |
| `observation_layers` | not supported | `obs_layer_ids` | Explicit Sparse-vLLM observation layers. |
| `deltakv_neighbor_count` | same | `deltakv_k_neighbors` | Number of reference/prototype neighbors. |
| `deltakv_center_ratio` | `cluster_ratio` | `cluster_ratio` | Fraction or stride-derived rate for reference centers. |
| `deltakv_latent_dim` | `kv_compressed_size` | `kv_compressed_size` | DeltaKV latent width. |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `kv_quant_bits` | Quantization bits for the cached DeltaKV-style state. |
| `hf_prefill_chunk_size` | `chunk_prefill_size` | none | HF wrapper/model chunk size. |
| `engine_prefill_chunk_size` | none | `chunk_prefill_size` | Sparse-vLLM scheduler chunk size. |
| `visual_token_prune_only` | same | none | LLaVA visual-token-only cache dropping/pruning. |
| `visual_token_keep_ratio` | same | none | LLaVA ratio of eligible visual tokens to keep. |

Rejected legacy runtime names:

| Legacy key | Replacement | Problem with legacy name |
| --- | --- | --- |
| `model_cls`, `vllm_sparse_method` | `sparse_method` | Backend-specific method selector names leaked into shared configs. |
| `compressor_path`, `deltakv_path` | `deltakv_checkpoint_path` | Backend-specific checkpoint names made cross-backend configs ambiguous. |
| `chunk_prefill_size` | `hf_prefill_chunk_size` or `engine_prefill_chunk_size` | Same spelling meant different speed/capacity behavior on each backend. |
| `num_top_tokens`, `num_top_tokens_in_prefill` | `decode_keep_tokens`, `prefill_keep_tokens` | Count/ratio/per-layer semantics differed by backend. |
| `num_sink_tokens`, `num_recent_tokens`, `tail_token_size` | `sink_keep_tokens`, `recent_keep_tokens` | Internal cache names leaked into experiment configs. |
| `full_attn_layers`, `obs_layer_ids` | `full_attention_layers`, `observation_layers` | Internal layer-routing names leaked into shared configs. |
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
| `deltasnapkv` | DeltaKV plus SnapKV-style method. Requires empty `full_attention_layers`. | Requires compressor checkpoint. |
| `pyramidkv` | HF PyramidKV wrapper. | No DeltaKV checkpoint. |
| `omnikv` | Loads DeltaKV wrapper with `use_compression=False` and `use_cluster=False`. | No DeltaKV checkpoint. |
| `quest` | Patches Quest baseline attention. | No DeltaKV checkpoint. |
| `palu`, `kivi`, `adakv`, `kvzip` | Baseline adapters. | Baseline-specific. |

For example, `sparse_method="deltakv-triton-v4"` maps to the standard HF
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
| `deltakv` | DeltaKV cache manager. |
| `deltakv-triton` | DeltaKV logic with Triton reconstruction. Internally rewrites method to `deltakv`. |
| `deltakv-triton-v2` | Triton reconstruction plus eviction. Internally rewrites method to `deltakv`. |
| `deltakv-triton-v3` | Adds blockwise L2 top-k. Internally rewrites method to `deltakv`. |
| `deltakv-triton-v4` | Adds more kernel fusions. Internally rewrites method to `deltakv`. |
| `deltakv-triton-v3-offload` | DeltaKV V3 with CPU latent offload. |
| `deltakv-triton-v3-cuda-offload` | DeltaKV V3 offload with custom CUDA gather path. |
| `deltakv-standalone` | DeltaKV standalone manager; clears full-attention and observation-layer routing. |
| `deltakv-snapkv` | DeltaKV plus SnapKV-style cache manager; clears full-attention and observation-layer routing. |
| `dsa` | DeepSeek sparse attention placeholder. Currently restricted and disabled for most model types. |

Sparse-vLLM currently rejects Qwen3 plus DeltaKV in `CacheManager.create(...)`
because of qk-norm/runtime mismatch. Use HF for Qwen3 DeltaKV runs in this repo.

## 5. Unknown-Key Behavior

This matters because "the command ran" does not mean "the parameter was used".

| Backend | Current behavior |
| --- | --- |
| HF DeltaKV custom configs | `set_infer_args` applies keys only if the config has that attribute. Unknown keys log `There is NO <key> in Custom Config!` and are usually ignored. |
| Sparse-vLLM | `LLMEngine.__init__` filters kwargs to dataclass fields in `sparsevllm.Config`. Unknown keys are logged as ignored. |
| LLaVA visual script | Builds a selected `infer_config`; unrelated CLI args are not in the config. |
| SCBench DeltaKV branch | Copies `hyper_param`, pops `sparse_method` and `cuda_device`, then sends the rest to `get_generate_api`. |

Do not use a shared mega-config across backends unless it has first been
normalized and inspected for ignored keys.

## 6. Accuracy-Affecting Parameters

### 6.1 Token Keep Budgets

| Parameter | HF behavior | Sparse-vLLM behavior | Risk |
| --- | --- | --- | --- |
| `decode_keep_tokens` | In token-selection helpers, may be integer count or float ratio `<= 1.0`. Also may be list/tuple or comma string in some HF wrappers for per-observation-layer budgets. | Must be explicit integer-like count. The normalizer rejects ratio-style floats. | Copying `0.17` from HF to Sparse-vLLM is wrong. |
| `prefill_keep_tokens` | Prefill/finalization token selection budget. May also use list/comma-string in HF wrappers. | Integer-like prefill budget used by sparse controller, warmup, and capacity planning. Defaults to decode budget if unset internally. | It affects both accuracy and memory/speed in Sparse-vLLM. |
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
| `full_attention_layers` | Parsed into a list in `CustomConfigMixin`. Standard DeltaKV uses these as full-attention layers and selection anchors. | Parsed into a list internally. If `observation_layers` is absent, Sparse-vLLM derives observation layers from full layers. |
| `observation_layers` | Not a generic HF custom config field. | Explicit Sparse-vLLM observation-layer override. |
| `snapkv_num_full_layers` | HF SnapKV-related knob. | SnapKV manager can reserve early full layers. |

Special cases:

- HF `deltasnapkv` asserts that `full_attention_layers` is empty.
- Sparse-vLLM `deltakv-standalone` and `deltakv-snapkv` forcibly clear both
  full-attention and observation-layer routing.
- Matching the string value of `full_attention_layers` is not enough for backend
  parity. Check how the chosen method interprets it.

### 6.3 SnapKV, PyramidKV, OmniKV, Quest

| Parameter | Main consumer | Meaning |
| --- | --- | --- |
| `snapkv_window_size` | HF SnapKV/PyramidKV and Sparse-vLLM SnapKV/DeltaKV-SnapKV | Local observation/recent window. |
| `pool_kernel_size` | HF token selection and some Sparse-vLLM methods | Score smoothing kernel. The consumer differs by method. |
| `pyramid_layer_ratios` | Sparse-vLLM PyramidKV | Explicit per-layer keep ratios. Length must equal number of layers. |
| `pyramidkv_start_layer`, `pyramidkv_start_ratio`, `pyramidkv_least_layer`, `pyramidkv_least_ratio` | HF and Sparse-vLLM PyramidKV-style paths | Auto-generate layer budget schedule. |
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
| `deltakv_latent_quant_bits` | `0` | `4` enables int4 storage in supported paths. |
| `hf_prefill_chunk_size` | `100000000` | HF wrapper chunk size. The large default effectively avoids manual chunking for many prompts. |

### 7.1 Standard `CompressedKVCache`

File: `src/deltakv/modeling/kv_cache.py`.

When `use_cluster=False`:

- `use_compression=True` stores learned latent residuals.
- `compress()` groups tokens by `compressor_token_group_size`, uses a chunk base such as mean
  reference, and stores `compressor_down(kv) - compressor_down(base)`.
- Reconstruction uses `compressor_up(comp_kv) + base`.
- If `use_compression=False` and `deltakv_latent_quant_bits=4`, the historical KV stored
  from the buffer is direct int4 quantized raw KV, not learned DeltaKV latent.
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

- If `deltakv_latent_quant_bits=4`, this quantizes the latent `comp_kv`, not the original
  full KV and not token-space residual.
- When `use_compression=False`, the same cache can run the direct residual-quant
  path:

```text
residual = token_kv - mean(selected_centers)
```

- In that direct path, `deltakv_latent_quant_bits=4` quantizes the token-space
  residual itself. No learned `compress_down` or `compress_up` module is used
  for compression or reconstruction.
- The direct path is pad-aware for batched left-padding runs: padding tokens are
  stored with invalid positions and cannot become valid reference centers.

This is the no-compressor variant used by the LLaVA `deltakv_delta_quant`
benchmark. The residual-quant ablation paths below still have their own wrapper
routing.

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

- If `deltakv_latent_quant_bits=4`, the residual is int4 quantized.
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
| `cluster_metric` | `l2` | Center scoring metric. |
| `deltakv_latent_dim` | `kv_compressed_size` | `128` | Latent dimension. |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `4` | Quantization bits for DeltaKV-style state. |
| `deltakv_full_pool_reserve_ratio` | `0.1` | Fraction of KV memory reserved for full-KV pool. |
| `deltakv_offload_latent` | `False` | CPU latent offload switch. |
| `deltakv_offload_*` | varies | Offload prefetch/threading/gather controls. |
| `deltakv_triton_*_heads_per_program` | `4` | Triton grouping controls. |
| `allow_unknown_config_keys` | `False` | Explicit opt-in for ignoring unknown Sparse-vLLM config keys. |
| `allow_raw_config_fallback` | `False` | Explicit opt-in for raw `config.json` fallback when `AutoConfig` fails. Currently restricted to validated DeepSeek configs. |
| `allow_missing_deltakv_path` | `False` | Explicit opt-in for no-checkpoint DeltaKV ablations that intentionally omit compressor weights. |

Important differences from HF:

- `deltakv_neighbor_count` maps to `deltakv_k_neighbors`, not old `k_neighbors`.
- `deltakv_center_ratio` influences algorithmic behavior and memory planning.
- `decode_keep_tokens` and `prefill_keep_tokens` must be integer budgets.
- `full_attention_layers` may derive observation layers.
- `deltakv-standalone` and `deltakv-snapkv` clear full/obs layer routing.
- Unknown Sparse-vLLM config keys, raw config fallback, and missing DeltaKV
  compressor paths fail fast by default. Use the `allow_*` switches only for
  explicitly documented compatibility or ablation runs.
- Sparse-vLLM methods can physically evict or re-map cache slots during prefill
  and decode, unlike HF `DynamicCache` wrappers.

## 10. Speed And Capacity Parameters

These do not only affect speed. In Sparse-vLLM, they can change admission,
queueing, and whether a benchmark measures the intended batch.

| Parameter | Backend | Meaning |
| --- | --- | --- |
| `engine_prefill_chunk_size` | Sparse-vLLM | Max prefill chunk scheduled per sequence per step; used in warmup, long-text bucket, admission, memory estimates. |
| `hf_prefill_chunk_size` | HF | Wrapper/model chunk size for long input forwarding. Often a large value means "do not chunk". |
| `max_model_len` | Sparse-vLLM | Hard engine capacity for prompt plus generated tokens. Affects allocation and request validation. |
| `max_num_batched_tokens` | Sparse-vLLM | Scheduler cap for tokens in one step. Auto-raised to at least `2 * engine_prefill_chunk_size` after normalization. May also be reduced by memory heuristics. |
| `max_num_seqs_in_batch` | Sparse-vLLM | Max active sequences in a prefill/decode step. |
| `max_decoding_seqs` | Sparse-vLLM | Max sequences in decode queue. |
| `gpu_memory_utilization` | Sparse-vLLM | Fraction of total GPU memory used for cache planning. |
| `tensor_parallel_size` | Sparse-vLLM | Number of TP ranks/processes. |
| `num_kvcache_slots` | Sparse-vLLM | Optional explicit KV slot override. |
| `admission_wave_size` | `scripts/bench_sparse_vllm.py` | Benchmark-only staged admission. |
| `wave_decode_gap_steps` | `scripts/bench_sparse_vllm.py` | Benchmark-only delay before adding next wave. |
| `max_decode_steps_after_full` | `scripts/bench_sparse_vllm.py` | Benchmark-only decode window cap after full admission. |
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

`scripts/bench_sparse_vllm.py` reports:

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
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --max_decode_steps_after_full 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096,"prefill_keep_tokens":4096}'
```

For fair speed comparisons, compare:

- same model and tokenizer,
- same prompt length,
- same output length,
- same sampling settings,
- same batch admission policy,
- same actual full-admission status,
- same decode measurement scope.

## 11. `scripts/bench_sparse_vllm.py` Specific Notes

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
- `deltakv_latent_quant_bits=4` quantizes cached KV/latent/residual state in method-specific
  ways.

## 13. LLaVA-OneVision Visual-Cache Parameters

Main files:

- `scripts/bench_llava_onevision_visual_prune.py`
- `src/deltakv/modeling/llava_onevision_deltakv.py`
- `docs/llava-onevision-visual-cache-benchmarks.md`

Current implemented benchmark methods:

| Method label | Actual behavior |
| --- | --- |
| `vanilla` | Standard `LlavaOnevisionForConditionalGeneration`. |
| `deltakv` | LLaVA-OneVision DeltaKV wrapper with a real learned compressor checkpoint. Uses the checkpoint config plus CLI keep budgets. |
| `deltakv_delta_quant` | LLaVA-OneVision DeltaKV wrapper with no learned compressor checkpoint. Uses cluster/ref reconstruction, stores token-space residuals, and int4 quantizes those residuals. |
| `visual_uniform_keep` | DeltaKV wrapper infrastructure, but no compressor, no cluster, no ref tokens. Uniformly keeps visual tokens and drops the rest. |
| `visual_uniform_keep_int4` | Same uniform visual token keep path, plus direct int4 storage of kept visual KV. |

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `visual_token_prune_only` | Restricts cache dropping/pruning to visual tokens. |
| `visual_token_keep_ratio` / CLI `--visual_keep_ratio` | Fraction of eligible visual tokens kept by uniform subsampling. |
| `--quantize_visual_kv` | Sets `deltakv_latent_quant_bits=4` in no-checkpoint fallback. |
| `--delta_quant_bits` | Sets residual quantization bits for `deltakv_delta_quant`; currently only `4` is implemented. |
| `--deltakv_center_ratio` | Public center/prototype sampling ratio for `deltakv_delta_quant`. |
| `--deltakv_neighbor_count` | Number of selected ref centers for each compressed token in `deltakv_delta_quant`. |
| `recent_keep_tokens`, `sink_keep_tokens`, `full_attention_layers` | Passed into text config and affect cache buffer behavior. |
| `decode_keep_tokens`, `prefill_keep_tokens` | Present for wrapper compatibility; current no-checkpoint visual uniform path does not use SnapKV-style attention scoring for pruning. |

Current limitations:

- `visual_token_prune_only` currently raises for batch size greater than 1 in
  HF cache update.
- `visual_uniform_keep` explicitly sets `use_compression=False` and
  `use_cluster=False`.
- `deltakv_delta_quant` is a no-checkpoint DeltaKV-style path, but it is not
  visual-only pruning. It compresses the eligible text-backbone KV stream; in
  image VQA prompts most eligible tokens are visual tokens.
- "LLaVA visual keep10" without checkpoint is still not using cluster/ref
  tokens. It is uniform pruning.

## 14. Benchmark Entrypoints

### 14.1 LongBench And MathBench

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

### 14.2 NIAH

`benchmark/niah/test_niah.py` builds an `infer_config` manually from function
arguments. It now uses canonical keys such as `sparse_method`,
`hf_prefill_chunk_size`, `engine_prefill_chunk_size`,
`gpu_memory_utilization`, and `use_cluster`.

This is convenient for quick experiments, but it increases the chance that one
backend ignores keys intended for the other. Use canonical names when adding new
NIAH runs.

### 14.3 SCBench

`benchmark/scbench/run_scbench.py` has a DeltaKV branch for:

```text
deltakv, full_deltakv, origin_residual_quant, all_origin_residual_quant,
snapkv, pyramidkv, palu, quest
```

It copies `hyper_param`, extracts `deltakv_checkpoint_path`, pops `sparse_method` and
`cuda_device`, and calls `get_generate_api(..., return_model=True)`. This path
is HF-oriented. Sparse-vLLM-style engine parameters in this branch usually do
not apply.

## 15. Known Ambiguities And Recommended Names

| Removed legacy name | Problem | Canonical name |
| --- | --- | --- |
| `seq_chunk_size` | Token grouping in some paths, legacy `k_neighbors` fallback in cluster paths. | `compressor_token_group_size` for grouping; `deltakv_neighbor_count` for neighbors. |
| `chunk_prefill_size` | Same name means HF wrapper chunking or Sparse-vLLM scheduler chunking. | `hf_prefill_chunk_size` or `engine_prefill_chunk_size`. |
| `num_top_tokens` | Count, ratio, or per-layer list depending on HF path; count only on Sparse-vLLM. | `decode_keep_tokens`, with explicit ratio conversion notes. |
| `num_top_tokens_in_prefill` | Accuracy budget and Sparse-vLLM capacity/warmup input. | `prefill_keep_tokens`. |
| `compressor_path` | HF top-level path; ignored by Sparse-vLLM unless normalized. | `deltakv_checkpoint_path`. |
| `deltakv_path` | Sparse-vLLM path; ignored by HF unless normalized. | `deltakv_checkpoint_path`. |
| `model_cls` | HF method selector only. | `sparse_method` for portable intent. |
| `vllm_sparse_method` | Sparse-vLLM method selector only. | `sparse_method` for portable intent. |
| `tail_token_size` | Historical HF recent-buffer name. | `recent_keep_tokens`. |
| `kv_quant_bits` | Quantizes different objects in different paths. | `deltakv_latent_quant_bits`, plus document object being quantized. |
| `deltakv_visual_compress_only` | Says DeltaKV/compress even for uniform pruning. | `visual_token_prune_only`. |
| `deltakv_visual_keep_ratio` | Tied to misleading visual compress name. | `visual_token_keep_ratio`. |

## 16. Alignment Workflow

### 16.1 Accuracy Alignment

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
  --hyper_param '{"sparse_method":"deltakv-triton-v4","deltakv_checkpoint_path":"<COMPRESSOR>","decode_keep_tokens":22282,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","engine_prefill_chunk_size":512,"gpu_memory_utilization":0.9}'
```

### 16.2 Speed Alignment

Before comparing speed, separate three classes of settings:

| Class | Examples | Should be matched? |
| --- | --- | --- |
| Sparse policy | `decode_keep_tokens`, `recent_keep_tokens`, `full_attention_layers`, `deltakv_center_ratio` | Yes for method parity. |
| Engine capacity | `gpu_memory_utilization`, `engine_prefill_chunk_size`, `max_num_batched_tokens` | Match only when measuring same engine regime; tune when measuring max capacity. |
| Benchmark policy | `batch_sizes`, `admission_wave_size`, `max_decode_steps_after_full` | Must be reported with results. |

Full-attention baseline example:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization":0.95,"engine_prefill_chunk_size":4096}'
```

Sparse wave-admission example:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --max_decode_steps_after_full 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096,"prefill_keep_tokens":4096}'
```

## 17. Environment Variables

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
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods deltakv-triton-v4 \
  --hyper_params '{"sparse_method":"deltakv-triton-v4","deltakv_checkpoint_path":"<COMPRESSOR>","engine_prefill_chunk_size":4096,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"gpu_memory_utilization":0.9}'
```

Benchmark output/data example:

```bash
DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs \
DELTAKV_DATA_DIR=/data2/haojitai/datasets \
PYTHONPATH=$PWD/src:$PYTHONPATH \
python benchmark/long_bench/pred.py \
  --model qwen25-deltakv \
  --model_path <MODEL> \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path <COMPRESSOR> \
  --hyper_param '{"hf_prefill_chunk_size":32768,"decode_keep_tokens":0.17,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18"}'
```

## 18. Safe Config Checklist

Before launching a run:

- Use `sparse_method` and `deltakv_checkpoint_path` in shared configs.
- Use `hf_prefill_chunk_size` or `engine_prefill_chunk_size`; do not use bare
  `chunk_prefill_size` in new shared configs.
- If `use_cluster=True`, set `deltakv_neighbor_count` explicitly.
- If using Sparse-vLLM, convert all ratio budgets to token counts.
- If using LLaVA no-checkpoint path, label it as visual uniform pruning, not
  DeltaKV compressor inference.
- Record whether `deltakv_latent_quant_bits=4` quantized latent state, raw KV, or residual.
- Check logs for ignored unknown keys.
- For throughput, record whether the run queued and whether decode throughput
  came from full admission or fallback scope.
