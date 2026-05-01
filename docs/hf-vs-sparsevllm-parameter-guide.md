# HF vs Sparse-vLLM Backend Parameter Guide

This document explains how this repository currently interprets parameters when you run through the `hf` backend versus the `sparsevllm` backend.

For the current canonical parameter aliases, conflict checks, and accuracy/speed
alignment workflow, see [`runtime-parameter-semantics.md`](runtime-parameter-semantics.md).

It is intentionally implementation-specific. The goal is not to describe an ideal API. The goal is to describe what the code in this repo actually does today, especially in places where:

- the same parameter name means different things,
- one backend ignores a parameter that the other backend uses,
- a method exists on both sides but is configured differently,
- a benchmark script passes one shared `infer_config` dictionary into two very different runtimes.

## Scope

This guide covers the backend split exposed by:

- `src/deltakv/get_chat_api.py`
- benchmark entrypoints such as `benchmark/long_bench/pred.py`, `benchmark/math_bench/pred.py`, and `benchmark/niah/test_niah.py`

The benchmark scripts all build a single `infer_config` dictionary and then pass it into `get_generate_api(...)`. That shared front-end is convenient, but it can hide very different backend semantics.

## 1. Mental Model

Use this as the first approximation:

| Question | `backend="hf"` | `backend="sparsevllm"` |
| --- | --- | --- |
| Who chooses the method? | `model_cls` | `infer_config["vllm_sparse_method"]` |
| Who loads the compressor? | top-level `compressor_path` | `infer_config["deltakv_path"]` |
| Who owns scheduling and KV memory layout? | the HF model wrapper / adapter | the Sparse-vLLM engine and cache manager |
| Who owns tokenizer/device setup? | `get_generate_api(...)` | Sparse-vLLM internally |
| How are unknown `infer_config` keys handled? | usually logged as unknown and then ignored | silently filtered out before `Config(...)` is built |

The most important consequence is this:

> The same `infer_config` dictionary is not portable by default. Treat backend switching as a configuration migration, not as a flag flip.

## 2. Top-Level Arguments to `get_generate_api(...)`

The function signature is shared, but the two backends do not consume the same top-level arguments.

| Argument | HF backend | Sparse-vLLM backend | Notes |
| --- | --- | --- | --- |
| `model_path` | used | used | Both need it. |
| `infer_config` | used | used | Same container, different meaning. |
| `compressor_path` | used by DeltaKV-family HF paths | ignored | Sparse-vLLM expects `deltakv_path` inside `infer_config`. |
| `tokenizer_path` | used | ignored by backend construction | Benchmarks may still load a tokenizer separately for truncation, but Sparse-vLLM itself does not use this top-level argument here. |
| `model_cls` | required and semantic | ignored | Sparse-vLLM method selection does not come from `model_cls`. |
| `use_cache` | required to be `True` | ignored | HF asserts cache-based inference. |
| `cuda_device` | used for `device_map` and some compressor loading paths | ignored | Sparse-vLLM handles devices inside its own engine. |
| `backend` | selects route | selects route | This is the actual backend switch. |
| `return_kv_cache` | partially supported | shape-compatible but always returns `None` for cache | See the generation section below. |
| `return_model` | supported on HF | unsupported | Sparse-vLLM raises if `return_model=True`. |

## 3. Generation-Time Keyword Arguments

After `get_generate_api(...)` returns, you call the generated function with prompts and generation kwargs. This is another place where the two backends diverge.

### 3.1 What the current wrappers actually forward

| Generation kwarg | HF manual path | HF `model.generate(...)` path | Sparse-vLLM wrapper path | Notes |
| --- | --- | --- | --- | --- |
| `max_new_tokens` | used | used | used | Sparse-vLLM maps it to `SamplingParams.max_tokens`. |
| `max_tokens` | not primary | not primary | used as fallback | Sparse-vLLM accepts `max_new_tokens` first, then falls back to `max_tokens`. |
| `do_sample` | used | used | used indirectly | Sparse-vLLM only uses it to decide whether `temperature` becomes `0.0`. |
| `temperature` | used | used when sampling | used | Sparse-vLLM clamps tiny positive values to `1e-5` when sampling. |
| `top_p` | used | used when sampling | ignored by current wrapper | Even if Sparse-vLLM could support more sampling knobs internally later, this wrapper does not forward `top_p` today. |
| `top_k` | used | used when sampling | ignored by current wrapper | Same caveat as `top_p`. |
| `eos_token_id` | used | used | ignored by current wrapper | Sparse-vLLM wrapper does not map it into `SamplingParams`. |
| `num_beams` | effectively unsupported in generic HF path | effectively unsupported in generic HF path | ignored | `kvzip` explicitly requires `num_beams=1`. Do not assume beam search support. |
| `past_key_values` | supported | explicitly rejected | ignored | Sparse-vLLM wrapper accepts the arg for call-shape compatibility but does not use it. |

### 3.2 Return-side behavior

| Behavior | HF backend | Sparse-vLLM backend |
| --- | --- | --- |
| `return_kv_cache=True` | best-effort only | always returns `None` as the cache payload |
| actual cache object returned | only on the manual HF generation path | never |
| `return_model=True` | returns `(generate_fn, model)` | raises an error |

In other words:

- If you need external `past_key_values` reuse, you are in HF-only territory.
- If you need a portable interface across both backends, assume only text output is portable.

## 4. Method Selection Is Different

Do not try to use the same method-selection fields on both sides.

| Intent | HF backend | Sparse-vLLM backend |
| --- | --- | --- |
| Select DeltaKV | `model_cls="deltakv"` or a related HF branch | `vllm_sparse_method="deltakv"` or a Sparse-vLLM DeltaKV variant |
| Select OmniKV | `model_cls="omnikv"` | `vllm_sparse_method="omnikv"` |
| Select SnapKV | `model_cls="snapkv"` | `vllm_sparse_method="snapkv"` |
| Select PyramidKV | `model_cls="pyramidkv"` | `vllm_sparse_method="pyramidkv"` |
| Select Quest | `model_cls="quest"` | `vllm_sparse_method="quest"` |

The supported method sets are not identical.

### HF `model_cls` branches in this repo

- `deltakv`
- `full_deltakv`
- `origin_residual_quant`
- `all_origin_residual_quant`
- `snapkv`
- `deltasnapkv`
- `pyramidkv`
- `omnikv`
- `auto`
- `quest`
- `palu`
- `kivi`
- `adakv`
- `kvzip`

### Sparse-vLLM `vllm_sparse_method` strings in this repo

- `""` for dense/default engine behavior
- `streamingllm`
- `attention-sink` / `attention_sink` as aliases for `streamingllm`
- `snapkv`
- `omnikv`
- `quest`
- `deltakv`
- `deltakv-triton`
- `deltakv-triton-v2`
- `deltakv-triton-v3`
- `deltakv-triton-v4`
- `deltakv-triton-v3-offload`
- `deltakv-triton-v3-cuda-offload`
- `deltakv-standalone`
- `deltakv-snapkv`
- `pyramidkv`
- `dsa`

Some methods only exist on one side. `kvzip`, `palu`, `kivi`, and `adakv` are HF-side concepts in this repository. `dsa` is Sparse-vLLM-side.

## 5. `infer_config` Keys: Same Name, Different Meaning

These are the parameters that most deserve extra attention.

| Key | HF backend meaning | Sparse-vLLM meaning | Recommendation |
| --- | --- | --- | --- |
| `chunk_prefill_size` | mostly a model-side or wrapper-side chunking knob | an engine scheduling and memory-budget knob | Never copy the same numeric value blindly across backends. |
| `max_model_len` | mainly a limit seen by model-side logic or outer benchmark truncation | a hard engine capacity parameter that affects allocation and admission control | Re-tune for Sparse-vLLM instead of treating it as metadata. |
| `num_top_tokens` | may be an integer count or a float ratio in HF token selection code | treated as an integer-like token budget by Sparse-vLLM | Convert float ratios into explicit counts before moving to Sparse-vLLM. |
| `num_top_tokens_in_prefill` | model-side prefill keep budget | engine-visible keep budget that also participates in warmup and capacity planning | Revisit together with `chunk_prefill_size` and batching limits. |
| `cluster_ratio` | mainly an algorithmic ratio in HF wrappers | both an algorithmic ratio and a memory/capacity input in Sparse-vLLM | Treat it as performance-sensitive and memory-sensitive on Sparse-vLLM. |
| `full_attn_layers` | wrapper config used by DeltaKV / OmniKV style logic | mixed-layer routing signal that can auto-derive `obs_layer_ids` | Do not assume the same layer list leads to the same runtime behavior. |
| `num_recent_tokens` | exposed as `num_recent_tokens`, but many HF wrappers internally use `tail_token_size` | directly consumed by engine cache managers and scheduler math | Same intuition, different plumbing. |
| `pool_kernel_size` | smoothing kernel in HF token-selection helpers | method-dependent engine-side sparse control knob | Same name, but not the same consumer path. |

### 5.1 `chunk_prefill_size`

This is one of the most misleading "shared" names.

On HF:

- it usually controls how a wrapper chunks long prefills,
- some methods effectively disable chunking with very large sentinel values,
- it does not mean "reserve this chunk size inside a persistent scheduler-owned KV pool".

On Sparse-vLLM:

- it participates in warmup length,
- it constrains prefill scheduling,
- it influences `max_num_batched_tokens`,
- it feeds directly into cache sizing and OOM avoidance logic.

Practical rule:

> A large HF `chunk_prefill_size` often means "avoid wrapper chunking". A large Sparse-vLLM `chunk_prefill_size` means "change engine behavior and memory planning".

### 5.2 `num_top_tokens`

On HF, token-selection helpers accept `Union[int, float]` and interpret floats `<= 1.0` as ratios of candidate tokens.

Examples:

- `num_top_tokens=2048` means "keep 2048 tokens"
- `num_top_tokens=0.11` means "keep about 11% of the candidates"

On Sparse-vLLM, this parameter is used as an integer-like budget in multiple engine and cache-manager paths. Passing `0.11` there is not a portable way to express "11%". It becomes an invalid or misleading budget.

Practical rule:

> Float ratios are an HF-side convenience. Sparse-vLLM expects explicit budgets.

### 5.3 `full_attn_layers`

The same list can still imply different behavior.

On HF:

- it is passed into wrapper config,
- it helps define which layers stay fully attentive in DeltaKV/OmniKV-style methods,
- `deltasnapkv` explicitly requires it to be empty.

On Sparse-vLLM:

- it is parsed into a list and may drive mixed-layer routing,
- `obs_layer_ids` may be auto-derived from it,
- for `deltakv-standalone` and `deltakv-snapkv`, Sparse-vLLM forcibly clears `full_attn_layers` and `obs_layer_ids`.

Practical rule:

> Matching the string value is not enough. Check the method-specific normalization rules on each backend.

## 6. `infer_config` Keys: Same Concept, Different Name

Some ideas exist on both sides but are spelled differently.

| Concept | HF-side parameter | Sparse-vLLM-side parameter | Notes |
| --- | --- | --- | --- |
| DeltaKV checkpoint path | top-level `compressor_path` | `infer_config["deltakv_path"]` | This is the most common migration mistake. |
| DeltaKV nearest-center count | `k_neighbors` with legacy fallback from `seq_chunk_size` | `deltakv_k_neighbors` | Similar idea, not a shared key. |
| Quest page/chunk size | `chunk_size` | `quest_chunk_size` | HF Quest uses its own adapter naming. |
| Quest token budget | `num_top_tokens` | `quest_token_budget` | Same high-level intent, different key. |
| Method selection | `model_cls` | `vllm_sparse_method` | Top-level versus `infer_config`. |

If you switch backends without renaming these, the run may still start, but it may be using defaults rather than your intended configuration.

## 7. `infer_config` Keys: Mostly Shared Intuition

These are closer to portable, but still not guaranteed to be numerically equivalent across runtimes.

| Key | High-level meaning |
| --- | --- |
| `num_sink_tokens` | number of sink/prefix tokens kept by sink-style sparse methods |
| `snapkv_window_size` | local recent-window budget in SnapKV-like logic |
| `kv_compressed_size` | latent/compressed KV width in DeltaKV-style paths |
| `cluster_metric` | distance/similarity metric for DeltaKV clustering / matching |
| `kv_quant_bits` | quantization/compression control for DeltaKV-style state |

These keys are easier to reason about than the previous sections, but you should still validate method-specific code paths before assuming one-to-one performance or memory behavior.

## 8. HF-Only or Primarily HF-Oriented Parameters

If you pass these into the Sparse-vLLM backend, they are usually ignored because they are not Sparse-vLLM `Config` fields.

### 8.1 HF model-loading and quantization parameters

| Key | Meaning on HF | Sparse-vLLM behavior |
| --- | --- | --- |
| `torch_dtype` | controls HF model load dtype | ignored by Sparse-vLLM `infer_config` filtering |
| `load_in_4bit` / `load_in_8bit` | BitsAndBytes load-time quantization | ignored |
| `quant_skip_modules` | exclude modules from low-bit loading | ignored |
| `llm_int8_threshold` | BitsAndBytes int8 tuning | ignored |
| `llm_int8_enable_fp32_cpu_offload` | BitsAndBytes int8 offload | ignored |
| `llm_int8_has_fp16_weight` | BitsAndBytes int8 option | ignored |
| `bnb_4bit_compute_dtype` | BitsAndBytes 4-bit compute dtype | ignored |
| `bnb_4bit_use_double_quant` | BitsAndBytes 4-bit option | ignored |
| `bnb_4bit_quant_type` | BitsAndBytes 4-bit option | ignored |
| `bnb_4bit_quant_storage` | BitsAndBytes 4-bit storage dtype | ignored |

### 8.2 HF DeltaKV-wrapper parameters

These are meaningful to the HF wrapper family, not to Sparse-vLLM engine construction:

- `use_compression`
- `use_cluster`
- `collect_kv_before_rope`
- `split_kv`
- `recon_mode`
- `ref_mode`
- `cluster_on_kv`
- `stride_alpha`
- `cluster_temp`
- `cluster_soft_assignment`
- `k_neighbors`
- `seq_chunk_size`
- `layer_chunk_size`

### 8.3 HF baseline-specific parameters

These exist because the HF backend can dispatch to repo-local baseline adapters.

| Method | Representative parameters |
| --- | --- |
| `palu` | `lt_bits`, `lt_group_size`, `lt_sym`, `lt_clip_ratio`, `lt_hadamard` |
| `kivi` | `k_bits`, `v_bits`, `group_size`, `residual_length` |
| `adakv` | `use_adaptive`, `kernel_size`, `pooling`, `floor_alpha`, `pyram_mode`, `pyram_beta`, `gqa_support`, `gqa_func` |
| `kvzip` | `kv_type`, `prefill_chunk_size`, `ratio`, `level`, `load_score`, `do_score`, `update_cache`, `iterative_expand_chunk_size`, `iterative_keep_tokens`, `iterative_decode_chunk_size`, `iterative_decode_keep_tokens` |

These are not Sparse-vLLM settings, even if a similar research idea exists elsewhere.

## 9. Sparse-vLLM-Only Parameters

If you pass these into the HF backend, the outcome is usually one of:

- they are logged as unknown custom config keys,
- they are ignored,
- they only matter if the chosen HF `model_cls` happens to consume them manually.

### 9.1 Sparse-vLLM engine sizing and scheduling

| Key | Meaning on Sparse-vLLM |
| --- | --- |
| `gpu_memory_utilization` | fraction of GPU memory budgeted for the engine's KV/cache planning |
| `max_num_batched_tokens` | scheduler-wide token throughput cap |
| `max_num_seqs_in_batch` | scheduler batch cardinality limit |
| `max_decoding_seqs` | maximum concurrent decode sequences |
| `tensor_parallel_size` | number of Sparse-vLLM worker ranks |
| `enforce_eager` | engine execution policy knob |
| `num_kvcache_slots` | explicit KV slot override |
| `enable_profiler` | engine profiler switch |
| `throughput_log_interval_s` | throughput logging interval |

### 9.2 Sparse-vLLM method routing and cache-manager parameters

| Key | Meaning on Sparse-vLLM |
| --- | --- |
| `vllm_sparse_method` | primary method selector |
| `deltakv_path` | DeltaKV checkpoint path |
| `obs_layer_ids` | explicit observation-layer override |
| `chunk_prefill_accel_omnikv` | OmniKV-prefill acceleration switch |
| `quest_chunk_size` | Quest page size |
| `quest_token_budget` | Quest token budget |
| `quest_skip_layers` | Quest layer-skipping control |
| `snapkv_num_full_layers` | number of full-attention layers before SnapKV eviction starts |
| `pyramid_layer_ratios` | explicit per-layer PyramidKV ratios |
| `pyramidkv_start_layer`, `pyramidkv_start_ratio`, `pyramidkv_least_layer`, `pyramidkv_least_ratio` | PyramidKV automatic ratio generation controls |
| `deltakv_full_pool_reserve_ratio` | reserve ratio for Sparse-vLLM DeltaKV full-KV pool |
| `deltakv_offload_latent` and related offload knobs | Sparse-vLLM DeltaKV offload strategy |
| `dsa_*` fields | DeepSeek Sparse Attention / FlashMLA engine knobs |

## 10. Unknown-Key Behavior

This matters because "the run did not crash" does not mean "the parameter was applied".

### HF backend

For many HF paths, custom config objects call `set_infer_args(**infer_config)`.

If a key is unknown:

- the code logs an error like `There is NO <key> in Custom Config!`,
- the key is generally ignored,
- one special compatibility case maps `num_recent_tokens` into `tail_token_size`.

### Sparse-vLLM backend

The engine builds:

- `config_fields = {field.name for field in fields(Config)}`
- `config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}`

That means:

- unknown keys are silently dropped before `Config(...)` is constructed,
- typos do not fail fast,
- HF-only keys can sit in the same benchmark config without causing an error, but they also do nothing.

Practical rule:

> Sparse-vLLM is quieter. HF is noisier. Neither side guarantees that a typo will stop your run.

## 11. Migration Patterns

Switching backends usually requires three steps:

1. Remove parameters that only exist on the old backend.
2. Rename parameters whose concept exists on both sides but under different names.
3. Re-tune parameters whose names match but semantics do not.

### 11.1 Example: HF DeltaKV to Sparse-vLLM DeltaKV

Typical HF-style setup:

```python
backend = "hf"
model_cls = "deltakv"
compressor_path = "/path/to/compressor"
infer_config = {
    "num_top_tokens": 0.17,
    "num_top_tokens_in_prefill": 4096,
    "num_recent_tokens": 128,
    "num_sink_tokens": 8,
    "full_attn_layers": "0,1,2,8,18",
    "use_compression": True,
    "use_cluster": True,
    "cluster_ratio": 0.1,
    "chunk_prefill_size": 2048000,
}
```

Typical Sparse-vLLM-style setup:

```python
backend = "sparsevllm"
model_cls = "deltakv"  # ignored
compressor_path = "/path/to/compressor"  # ignored
infer_config = {
    "vllm_sparse_method": "deltakv-triton-v4",
    "deltakv_path": "/path/to/compressor",
    "num_top_tokens": 2048,
    "num_top_tokens_in_prefill": 4096,
    "num_recent_tokens": 128,
    "num_sink_tokens": 8,
    "full_attn_layers": "0,1,2,8,18",
    "cluster_ratio": 0.1,
    "chunk_prefill_size": 512,
    "max_model_len": 131000,
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 8192,
}
```

Important differences in that migration:

- `compressor_path` must become `deltakv_path`
- `model_cls` stops mattering
- float-ratio `num_top_tokens` should become an explicit token budget
- `chunk_prefill_size` must be re-tuned for engine scheduling, not copied
- engine sizing knobs now matter

### 11.2 Example: HF Quest to Sparse-vLLM Quest

HF-style Quest parameters:

```python
backend = "hf"
model_cls = "quest"
infer_config = {
    "num_top_tokens": 1024,
    "chunk_size": 16,
}
```

Sparse-vLLM-style Quest parameters:

```python
backend = "sparsevllm"
infer_config = {
    "vllm_sparse_method": "quest",
    "quest_token_budget": 1024,
    "quest_chunk_size": 16,
    "quest_skip_layers": 2,
}
```

This is the same high-level method family, but not the same configuration surface.

## 12. Backend-Specific Caveats

### Qwen3 + DeltaKV

Sparse-vLLM currently rejects Qwen3 DeltaKV inference in its cache-manager layer and tells you to use the HF backend instead.

Practical rule:

> If you need Qwen3 with DeltaKV in this repo today, use `backend="hf"`.

### `deltasnapkv` versus Sparse-vLLM mixed DeltaKV paths

HF `model_cls="deltasnapkv"` has a hard requirement that `full_attn_layers` be empty.

Sparse-vLLM has different mixed DeltaKV variants:

- `deltakv-snapkv`
- `deltakv-standalone`
- `deltakv-triton-*`

Do not treat HF `deltasnapkv` settings as directly portable to those engine methods.

## 13. Common Pitfalls Checklist

- `compressor_path` does nothing on `backend="sparsevllm"`.
- `deltakv_path` does nothing on `backend="hf"`.
- `model_cls` does not select the Sparse-vLLM method.
- `num_top_tokens=0.11` is meaningful on HF but not a portable Sparse-vLLM setting.
- `top_p`, `top_k`, and `eos_token_id` are currently ignored by the Sparse-vLLM wrapper in `get_chat_api.py`.
- `past_key_values` reuse is not portable across backends.
- `return_kv_cache=True` is not a portable backend contract.
- `return_model=True` is HF-only.
- `chunk_prefill_size` is not numerically portable.
- Sparse-vLLM silently drops unknown config keys.
- HF usually logs unknown config keys but still continues.

## 14. Recommended Workflow

When adding or comparing experiments across both backends:

1. Start from a backend-specific minimal config, not from a shared "mega-config".
2. Add only the parameters that the chosen backend actually consumes.
3. For any key that exists on both sides, verify whether it is a true shared semantic or only a shared name.
4. When moving from HF to Sparse-vLLM, convert ratio-style budgets into explicit counts unless the Sparse-vLLM code clearly accepts ratios.
5. Treat `chunk_prefill_size`, `max_model_len`, batching limits, and memory-utilization knobs as a joint tuning problem on Sparse-vLLM.

If you follow only one rule from this document, use this one:

> Backend switching in this repo is a runtime migration, not just a method toggle.
