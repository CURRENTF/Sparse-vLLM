# DeltaKV Modeling Refactor - 2026-05-12

## Background

This change reduces the complexity of `src/deltakv/modeling/` and removes the
HF batched-inference attempt. HF DeltaKV now intentionally supports only
batch size 1, while batched inference and future benchmark work should use the
Sparse-vLLM backend.

The previous modeling tree had many near-duplicate cache and model files for
different experimental branches. The refactor keeps the active research paths,
removes deleted experiments from public imports, and makes cache semantics
explicit in the class and config names.

## Scope

- Branch: `refactor/deltakv-cache-pipeline`
- Modeling file count after refactor: 22 Python files
- `pygount --format=summary src/deltakv/modeling/`: 3141 Python code lines
- Main retained HF model families: Qwen2, Qwen3, Llama
- HF inference policy: batch size 1 only; batched or padded inputs raise
  `NotImplementedError`
- Training policy: only `cluster_e2e_big` compressor training is supported

## Cache Naming

The old names such as `origin_residual_quant`, `all_origin_residual_quant`,
`full_deltakv`, and classes such as `LlamaAllOriginResidualQuant` were removed.
They were ambiguous because "all origin" did not clearly describe the formula
or the full-layer behavior.

The four active cache implementations are now:

| Config value | Meaning |
| --- | --- |
| `delta_compressed_latent_wo_full` | Store delta in compressor latent space; do not compress full-attention layers. |
| `delta_compressed_latent_w_full` | Store delta in compressor latent space; also compress full-attention layers. |
| `delta_origin_wo_full` | Store delta in original KV space; do not compress full-attention layers. |
| `delta_origin_w_full` | Store delta in original KV space; also compress full-attention layers. |

The default remains the learned latent DeltaKV path:
`delta_compressed_latent_wo_full`.

## Public Class Layout

Inference files:

- `qwen2_inference.py`
  - `Qwen2KVCompress`
  - `Qwen2DeltaCompressedLatentWFull`
  - `Qwen2DeltaOriginWoFull`
  - `Qwen2DeltaOriginWFull`
- `qwen3_inference.py`
  - `Qwen3KVCompress`
  - `Qwen3DeltaCompressedLatentWFull`
  - `Qwen3DeltaOriginWoFull`
  - `Qwen3DeltaOriginWFull`
- `llama_inference.py`
  - `LlamaKVCompress`
  - `LlamaDeltaCompressedLatentWFull`
  - `LlamaDeltaOriginWoFull`
  - `LlamaDeltaOriginWFull`

Training files:

- `qwen2_training.py`
  - `Qwen2KVClusterCompress`
- `qwen3_training.py`
  - `Qwen3KVClusterCompress`
- `llama_training.py`
  - `LlamaKVClusterCompress`

Shared implementation files:

- `cache_pipeline.py`: cluster cache pipeline, SnapKV cache, and the four cache
  implementation classes.
- `cache_factory.py`: validates `deltakv_cache_impl` and constructs the matching
  cache class.
- `hf_common.py`: shared HF inference/training builders for Qwen2/Qwen3/Llama.
- `compressor.py`: compressor construction and Qwen3 Q/K norm helper.
- `token_select.py`: OmniKV token selection helper retained from the previous
  implementation.

## Removed Paths

Removed HF modeling implementations:

- non-cluster e2e training/inference files
- `cluster_e2e`
- `cluster_e2e_big` per-model duplicate files, replaced by shared training
  builders plus thin model-family modules
- chunk `avg` / `first` reference-mode training behavior
- old cache files:
  - `kv_cache.py`
  - `full_deltakv_compress_cache.py`
  - `origin_residual_quant_cache.py`
  - `all_origin_residual_quant_cache.py`

`train_compressor.py` now raises for removed training modes and only accepts
`model_type="cluster_e2e_big"`. It also raises for `ref_mode="avg"` and
`ref_mode="first"`.

## Retained Paths

Baselines retained:

- Qwen2 SnapKV: `modeling/qwen2/qwen2_snapkv.py`
- Qwen2 PyramidKV: `modeling/qwen2/qwen2_pyramidkv.py`
- Qwen2 DeltaSnapKV: `modeling/qwen2/qwen2_deltasnapkv.py`
- Llama SnapKV: `modeling/llama/llama_snapkv.py`
- Llama PyramidKV: `modeling/llama/llama_pyramidkv.py`
- Llama DeltaSnapKV: `modeling/llama/llama_deltasnapkv.py`

Multimodal DeltaKV retained and moved to:

- `modeling/llava_ov/llava_onevision_deltakv.py`
- `modeling/llava_ov/__init__.py`

## Entrypoints

- `src/deltakv/get_chat_api.py` now routes DeltaKV HF inference to the new
  `qwen2_inference.py`, `qwen3_inference.py`, and `llama_inference.py` files.
- SCBench argument and loader code now exposes the new cache names instead of
  the removed names.
- The visual-cache benchmark imports LLaVA OneVision DeltaKV from
  `deltakv.modeling.llava_ov`.
- `src/deltakv/analysis/analyze_comp_kv_range.py` now imports the new Qwen2
  training class and no longer depends on removed `qwen2_e2e_cluster`.

## Tests Added Or Updated

Added:

- `tests/test_deltakv_baseline_imports.py`
- `tests/test_deltakv_training_entrypoints.py`
- `tests/test_hf_deltakv_modeling.py`
- `tests/test_longbench_deltakv_contracts.py`

Updated:

- `tests/test_hf_deltakv_cache_factory.py`
- `tests/test_visual_uniform_pruning.py`

Removed:

- `tests/test_qwen2_left_padding_batch.py`, replaced by the three-model HF
  modeling test.

## Verification

Commands run:

```bash
conda run -p <REVIEW_CONDA_ENV> \
  env PYTHONPATH=src python -m compileall -q \
  src/deltakv/modeling src/deltakv/get_chat_api.py \
  src/deltakv/train_compressor.py \
  src/deltakv/analysis/analyze_comp_kv_range.py

conda run -p <REVIEW_CONDA_ENV> \
  env PYTHONPATH=src:. python -m pytest -q \
  tests/test_hf_deltakv_cache_factory.py \
  tests/test_hf_deltakv_modeling.py \
  tests/test_deltakv_training_entrypoints.py \
  tests/test_deltakv_baseline_imports.py \
  tests/test_longbench_deltakv_contracts.py \
  tests/test_visual_uniform_pruning.py

conda run -p <REVIEW_CONDA_ENV> \
  env PYTHONPATH=src:. python -m pytest -q \
  tests/test_deltakv_checkpoint_config_sync.py \
  tests/test_quantization_helpers.py \
  tests/test_runtime_param_normalization.py \
  tests/test_research_fail_fast.py

conda run -p <REVIEW_CONDA_ENV> \
  env PYTHONPATH=src:. python -m pytest -q \
  tests/test_deltakv_delta_quant_kernel.py
```

Results:

- `compileall`: passed
- Target DeltaKV tests: 18 passed, 36 subtests passed
- Existing lightweight tests: 26 passed, 9 subtests passed
- DeltaKV Triton kernel test: 1 passed
- Import check for Qwen2/Qwen3/Llama inference, training, and LLaVA OV: passed

## Notes

- Old cache implementation names are not supported for creation or public import.
- No `unsupported_class` wrapper layer is kept.
- LongBench hotpotqa/trec tests are contract tests for metric/routing/fail-fast
  behavior. Full LongBench score reproduction still requires explicit dataset,
  model, checkpoint, and runtime configuration.
