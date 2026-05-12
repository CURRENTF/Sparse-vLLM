# Repo Review: multimodal-deltakv-adaptation

Date: 2026-05-12
Branch: `multimodal-deltakv-adaptation`
Review environment: `/root/autodl-tmp/conda-envs/svllm-review`

This document is the engineering review index for the current branch. It
summarizes repository structure, runtime parameters, supported method paths,
benchmark entrypoints, concrete findings, and maintenance priorities. Detailed
parameter semantics remain in `docs/runtime-parameter-semantics.md`.

## Scope

Reviewed areas:

| Area | Paths |
| --- | --- |
| Packaging and install | `pyproject.toml`, `README.md` |
| Public runtime entrypoints | `src/deltakv/get_chat_api.py`, `src/sparsevllm/llm.py`, `src/sparsevllm/engine/llm_engine.py` |
| Parameter normalization | `src/deltakv/configs/runtime_params.py`, `src/deltakv/configs/model_config_cls.py`, `src/sparsevllm/config.py` |
| HF DeltaKV caches | `src/deltakv/modeling/kv_cache.py`, `origin_residual_quant_cache.py`, `all_origin_residual_quant_cache.py`, `cache_factory.py` |
| Sparse-vLLM engine | `src/sparsevllm/engine/`, `src/sparsevllm/models/`, `src/sparsevllm/layers/`, `src/sparsevllm/triton_kernel/` |
| Benchmarks | `benchmark/long_bench/`, `benchmark/math_bench/`, `benchmark/scbench/`, `benchmark/niah/`, `benchmark/multimodal/` |
| Tests | `tests/` |
| Existing docs | `README.md`, `docs/*.md` |

Heavy model benchmarks were not run in this review. The verification here is
limited to static inspection, unit tests, import/compile checks, and focused
regression tests.

## Repository Map

| Path | Role | Review notes |
| --- | --- | --- |
| `src/deltakv/` | HF/Transformers-side DeltaKV integration, compressor training, custom model wrappers, cache implementations, runtime parameter normalization. | Main HF entrypoint is `get_generate_api`. Cache creation is centralized by `modeling/cache_factory.py`, which is good. `modeling/kv_cache.py` is still large and mixes buffer management, visual pruning, clustering, quantization, and reconstruction. |
| `src/sparsevllm/` | Sparse-first inference engine with its own scheduler, model runner, cache managers, sparse controllers, kernels, and model definitions. | Public `LLM(...)` kwargs are normalized in `LLMEngine.__init__`; unknown config keys fail fast by default. Cache-manager routing is explicit in `engine/cache_manager/base.py`. |
| `benchmark/` | LongBench, MathBench, SCBench, NIAH evaluation entrypoints. | LongBench/MathBench have been moved to canonical `sparse_method` and `deltakv_checkpoint_path` names. SCBench still includes several upstream-style silent fallback blocks that should be audited before publication-critical use. |
| `benchmark/multimodal/` | Multimodal image/video benchmark entrypoints plus shared helpers and model adapters. | StreamingBench/VideoMME/QA-Ego4D scripts preserve raw outputs, parsed outputs, per-sample files, metrics, and run info. Dataset/task logic is separated from model-specific adapters so Qwen3-VL can be added without duplicating Video-MME or StreamingBench logic. |
| `tests/` | Lightweight regression tests for runtime normalization, checkpoint sync, visual pruning, quant helpers, fail-fast behavior, and kernel helpers. | Coverage is useful for parameter boundary behavior. It does not exercise real model loading or long benchmark correctness. |
| `docs/` | Runtime parameter docs and benchmark-specific runbooks. | The parameter doc is the main source of truth. This review corrected one drift: Sparse-vLLM unknown keys now raise by default. |
| `baselines/` | Imported baseline method code. | Treat as third-party or research-adapter code unless a task explicitly targets it. |
| `skills/` | Local Codex skill material. | Not part of runtime package review. |

## Runtime Flow

### HF DeltaKV path

The high-level path is:

1. `get_generate_api(..., backend="hf", sparse_method=...)`
2. `normalize_runtime_params(..., backend="hf")`
3. HF model class selection through `sparse_method`
4. config load from compressor checkpoint or base model
5. `config.set_native_args(...)`
6. model wrapper load from `src/deltakv/modeling/{qwen2,qwen3,llama}/`
7. cache implementation selected by `deltakv_cache_impl`

Strengths:

- Public parameter names are centralized in `runtime_params.py`.
- Legacy names such as `model_cls`, `deltakv_path`, and `k_neighbors` are rejected at API boundaries.
- Compressor-backed and residual-quant cache variants are selected through `cache_factory.py`, not by scattered `isinstance` checks.

Risks:

- `set_native_args` still logs unknown config keys instead of raising. The public
  API protects normal use, but direct config mutation or checkpoint-driven paths
  can still hide typos.
- `manual_generate` is intentionally custom and only lightly covered by tests.
  Treat it as a research inference helper, not a drop-in replacement for all HF
  generation behavior.

### Sparse-vLLM path

The high-level path is:

1. `sparsevllm.LLM(model, **kwargs)`
2. `LLMEngine.__init__`
3. `normalize_runtime_params(..., backend="sparsevllm")`
4. dataclass filter into `sparsevllm.Config`
5. `Config.__post_init__` validates model/config/checkpoint compatibility
6. `CacheManager.create(...)` selects the cache manager
7. `Scheduler`, `ModelRunner`, `SparseController`, kernels and cache manager execute inference

Strengths:

- Unknown Sparse-vLLM config keys raise `ValueError` by default.
- Checkpoint-backed DeltaKV methods require `deltakv_checkpoint_path` unless a
  no-checkpoint ablation explicitly opts in.
- Unsupported combinations, such as sparsevllm Qwen3 plus DeltaKV, fail early.

Risks:

- DeltaKV Triton/offload methods rewrite `config.vllm_sparse_method` to
  `deltakv` during cache-manager selection. This keeps the controller path
  simple, but downstream logs can lose the originally requested method unless it
  is recorded before routing.
- Several cache managers are very large. `engine/cache_manager/deltakv.py`,
  `deltakv_standalone.py`, and `deltakv_snapkv.py` should be split only when a
  functional change already touches the relevant area.

## Method Matrix

| Public `sparse_method` | HF backend | Sparse-vLLM backend | Notes |
| --- | --- | --- | --- |
| `vanilla` or empty | AutoModel baseline | Standard cache manager | In Sparse-vLLM, `vanilla` normalizes to empty internal method. |
| `deltakv` | Compressor-backed DeltaKV wrappers | `DeltaKVCacheManager` | Requires `deltakv_checkpoint_path` for normal compressor-backed runs. |
| `deltakv-triton` | Maps to HF `deltakv` | `DeltaKVCacheTritonManager` | Sparse-vLLM-specific runtime optimization. |
| `deltakv-triton-v2` | Maps to HF `deltakv` | `DeltaKVCacheTritonManagerV2` | Reconstruction plus eviction kernel path. |
| `deltakv-triton-v3` | Maps to HF `deltakv` | `DeltaKVCacheTritonManagerV3` | Adds blockwise L2 top-k path. |
| `deltakv-triton-v4` | Maps to HF `deltakv` | `DeltaKVCacheTritonManagerV4` | Adds grouped-head reconstruction and related fusions. |
| `deltakv-triton-v3-offload` | Maps to HF `deltakv` | `DeltaKVCacheTritonManagerV3WithOffload` | Enables CPU latent offload. |
| `deltakv-triton-v3-cuda-offload` | Maps to HF `deltakv` | `DeltaKVCacheTritonManagerV3WithCUDAOffload` | Requires custom CUDA extension under `src/sparsevllm/cuda_kernel/`. |
| `deltakv-delta-quant` or `deltakv_delta_quant` | Not a distinct HF path | `DeltaKVDeltaQuantCacheManager` | No-checkpoint direct residual quantization path. Prefer hyphenated spelling. |
| `deltakv-standalone` | Maps to HF `deltakv` | `DeltaKVStandaloneCacheManager` | Sparse-vLLM standalone variant. |
| `deltakv-snapkv` | HF `deltasnapkv` | `DeltaKVSnapKVCacheManager` | DeltaKV/SnapKV hybrid path. |
| `snapkv` | HF SnapKV wrapper | `SnapKVCacheManager` | `pyramidkv` shares Sparse-vLLM manager. |
| `pyramidkv` | HF PyramidKV wrapper | `SnapKVCacheManager` | Check script docs before comparing to SnapKV. |
| `streamingllm`, `attention-sink`, `attention_sink` | HF StreamingLLM mapping | `StreamingLLMCacheManager` | Alias forms are accepted. |
| `quest` | HF Quest mapping | `QuestCacheManager` | Sparse-VLLM token budget must be a token count, not a ratio. |
| `omnikv` | HF OmniKV adapter | `OmniKVCacheManager` | Includes kernel and tuning utilities. |
| `dsa` | Not a general HF path | Currently gated to disabled DeepSeek-V3.2 path | Sparse-vLLM raises for normal model types. |

## Parameter System

Use semantic public names in commands and docs:

| Semantic name | Backend native target |
| --- | --- |
| `sparse_method` | HF model-class mapping or Sparse-vLLM `vllm_sparse_method` |
| `deltakv_checkpoint_path` | HF compressor path or Sparse-vLLM `deltakv_path` |
| `decode_keep_tokens` | `num_top_tokens` |
| `prefill_keep_tokens` | `num_top_tokens_in_prefill` |
| `sink_keep_tokens` | `num_sink_tokens` |
| `recent_keep_tokens` | `num_recent_tokens` / `tail_token_size` depending on path |
| `full_attention_layers` | `full_attn_layers` |
| `deltakv_center_ratio` | `cluster_ratio` |
| `deltakv_neighbor_count` | Sparse-vLLM `deltakv_k_neighbors`; HF config compatibility key |
| `deltakv_latent_dim` | `kv_compressed_size` |
| `deltakv_latent_quant_bits` | `kv_quant_bits` |
| `hf_prefill_chunk_size` | HF `chunk_prefill_size` |
| `engine_prefill_chunk_size` | Sparse-vLLM `chunk_prefill_size` |

Rules that matter for experiment reliability:

- Do not pass legacy names such as `model_cls`, `vllm_sparse_method`,
  `deltakv_path`, `compressor_path`, `k_neighbors`, `seq_chunk_size`, or raw
  `chunk_prefill_size` through public APIs.
- Sparse-VLLM unknown keys fail by default. Use
  `allow_unknown_config_keys=True` only for explicitly validated compatibility
  runs.
- Sparse-VLLM token budgets are token counts. Ratio-style values are rejected
  for `num_top_tokens` and `num_top_tokens_in_prefill`.
- `bitsandbytes` is a package dependency because 4-bit and 8-bit loading paths
  import it at runtime.

## Benchmark and Script Matrix

| Entry point | Purpose | Backend support | Review status |
| --- | --- | --- | --- |
| `benchmark/long_bench/pred.py` | LongBench prediction generation | `hf`, `sparsevllm` | Uses canonical `sparse_method` and `deltakv_checkpoint_path`. Has a broad fallback around model max length detection that should be tightened. |
| `benchmark/long_bench/eval.py` | LongBench scoring | Output-file evaluator | Mostly standalone. Some parsing fallbacks should be audited before publication use. |
| `benchmark/math_bench/pred.py` | MathBench generation | `hf`, `sparsevllm` | Uses canonical method/checkpoint args and deterministic flags. |
| `benchmark/math_bench/eval.py` | MathBench scoring | Output-file evaluator | Lightweight parser with exception handling. Check failure cases when adding datasets. |
| `benchmark/scbench/run_scbench.py` | SCBench generation/eval driver | HF-oriented plus DeltaKV branch | Supports canonical DeltaKV args, but includes several silent `except/pass` blocks inherited from research code. |
| `benchmark/scbench/run_scbench_preprocessed.py` | SCBench preprocessed path | SCBench-specific | Needs separate validation if used as the main reporting path. |
| `benchmark/niah/test_niah.py` | Needle-in-a-haystack probe | `hf`, `sparsevllm` | Uses canonical args and backend-specific prefill chunk names. |
| `scripts/bench_sparse_vllm.py` | Sparse-VLLM throughput/latency driver | Sparse-VLLM | Normalizes params and rejects legacy runtime names. |
| `benchmark/multimodal/video_qa/streamingbench.py` | LLaVA-OneVision StreamingBench | LLaVA/HF model path | Stronger artifact discipline: raw, parsed, per-sample, metrics, run info. Missing videos fail fast unless opt-in. |
| `benchmark/multimodal/video_qa/videomme.py` | LLaVA-OneVision VideoMME | Wraps StreamingBench runner | Adds VideoMME prompt/eval layer and preserves dry-run metadata. |
| `benchmark/multimodal/video_qa/qaego4d.py` | LLaVA-OneVision QA-Ego4D/ReKV-style eval | LLaVA/HF model path | Supports `vanilla` and `deltakv_delta_quant` style visual-cache comparisons. |
| `benchmark/multimodal/visual_cache/run_visual_cache.py` | Visual token/cache pruning experiments | LLaVA/HF model path | Contains explicit checkpoint checks and ablation modes. |
| `benchmark/multimodal/video_qa/frame_cache.py` | Frame cache generation | Utility | Good separation from evaluation. Backend choices and fallback flags are explicit. |
| `benchmark/multimodal/video_qa/audit_livevlm_table4.py` | Audit StreamingBench/LiveVLM artifacts | Utility | Useful guardrail after long video runs. |

## Findings

| Priority | Status | Finding | Impact | Recommendation |
| --- | --- | --- | --- | --- |
| P0 | Fixed in this review | `pyproject.toml` advertised Python `>=3.8`, but the code uses modern typing syntax that requires Python 3.10+. | Fresh installs on Python 3.8/3.9 can fail before runtime. | Keep `requires-python = ">=3.10"`. |
| P0 | Fixed in this review | `bitsandbytes` was imported by quantized load paths but was not declared as an install dependency. | `pip install -e .` could produce an environment that fails during 4-bit/8-bit model loading. | Keep `bitsandbytes` in dependencies and README install notes. |
| P0 | Fixed in this review | `ClusterCompressedKVCache.update()` could reference `compress_lens` when `visual_token_prune_only=True`, `use_cluster=True`, and `use_compression=False`. | Visual-only no-compressor cluster ablation could crash during cache update. | Regression test added in `tests/test_visual_uniform_pruning.py`. |
| P1 | Fixed in this review | `unpack_tensor` used list-based multidimensional tensor indexing. | PyTorch emits a future warning and future releases may change behavior. | Convert list indices to tuple before indexing. |
| P1 | Fixed in docs | `docs/runtime-parameter-semantics.md` said Sparse-VLLM unknown keys are logged and ignored. Current code raises by default. | Misleading docs can hide experiment typo expectations. | Documentation now states default fail-fast behavior and the explicit opt-in flag. |
| P1 | Open | HF `set_native_args` logs unknown config keys instead of raising. | Direct internal callers can silently ignore experiment parameter typos. | Add a strict mode or switch public experiment scripts to a strict setter after checking legacy checkpoint compatibility. |
| P1 | Open | SCBench and parts of LongBench contain broad `except` or `except/pass` fallback blocks. | Evaluation setup errors can be hidden, especially around tokenizer/model metadata, repo QA parsing, or optional components. | Tighten exceptions in the exact paths used for reported results; write failed-case artifacts instead of swallowing errors. |
| P1 | Open | Sparse-VLLM method routing mutates `config.vllm_sparse_method` for DeltaKV Triton/offload variants. | Downstream logs and artifacts can lose the originally requested method. | Add a separate immutable `requested_sparse_method` or record the original method in run info before mutation. |
| P2 | Fixed in this review | `compileall` reported invalid escape sequence warnings in analysis/benchmark helper scripts. | Non-fatal, but noisy under newer Python versions and easy to confuse with real warnings. | Regex strings now use raw strings and LaTeX-style plot labels escape backslashes intentionally. |
| P2 | Open | Large implementation files concentrate too many responsibilities. | Harder review and higher regression risk when changing cache or benchmark behavior. | Split only along real ownership boundaries during future feature work: buffer state, compression, visual pruning, reconstruction, artifact writing, and dataset adapters. |
| P2 | Open | Some scripts are local experiment utilities with hardcoded paths, defaults, or GPU assumptions. | New users can mistake them for stable public entrypoints. | In docs, classify scripts as stable entrypoints, utilities, or local experiments before adding new commands. |

## Verification

Commands run in `/root/autodl-tmp/conda-envs/svllm-review` during this review:

```bash
conda run --prefix /root/autodl-tmp/conda-envs/svllm-review --no-capture-output python -m pytest tests/test_visual_uniform_pruning.py tests/test_quantization_helpers.py -q
```

Result:

```text
5 passed
```

```bash
conda run --prefix /root/autodl-tmp/conda-envs/svllm-review --no-capture-output python -m pytest tests -q
```

Result after the tuple-index fix:

```text
34 passed, 15 subtests passed
```

The previous PyTorch list-indexing future warning in
`src/sparsevllm/triton_kernel/quant.py` no longer appears.

```bash
conda run --prefix /root/autodl-tmp/conda-envs/svllm-review --no-capture-output python -m compileall -q src scripts benchmark tests
```

Result:

```text
compileall succeeded
```

No warnings were emitted by the final compile check.

## Maintenance Priorities

1. Keep `docs/runtime-parameter-semantics.md` as the canonical parameter doc.
   Every new runtime flag should be documented there before being used in a
   benchmark command.
2. Prefer semantic public names in README, docs, scripts, and benchmark JSON:
   `sparse_method`, `deltakv_checkpoint_path`, `decode_keep_tokens`,
   `prefill_keep_tokens`, `hf_prefill_chunk_size`, and
   `engine_prefill_chunk_size`.
3. Keep fail-fast behavior for experiment parameters. Compatibility flags such
   as `allow_unknown_config_keys`, `allow_raw_config_fallback`, and
   `allow_missing_deltakv_path` should remain explicit opt-ins.
4. Preserve benchmark artifact separation: raw model output, parsed answer,
   per-sample records, aggregate metrics, and run info should not overwrite one
   another.
5. Before reporting SCBench/LongBench numbers, pin the exact script path and
   remove or audit silent exception handling in that path.
6. Avoid broad cache refactors unless paired with focused tests. The current
   cache implementations are complex enough that purely cosmetic edits are
   higher risk than useful.
