# HF / Sparse-VLLM Logits Alignment Check

Date: 2026-05-16

## Purpose

Compare HF and Sparse-VLLM logits on local GPU 6 with the same prompt,
tokenizer, model, DeltaKV compressor checkpoint, and explicit runtime sparse
parameters. Decode logits are the primary signal; prefill logits are recorded
as a localization aid.

## Code

- Commit at run time: `44e26e8`
- Dirty tree at run time:
  - `src/deltakv/modeling/cache_pipeline.py`
  - `tests/test_hf_deltakv_modeling.py`
  - `scripts/debug/compare_logits_hf_sparsevllm.py`

## Command

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_logits_hf_sparsevllm.py \
  --cases short,long \
  --methods vanilla,deltakv \
  --output_dir /data2/haojitai/outputs/sparsevllm_logits_align/20260516_0304_qwen25_7b_q0
```

## Key Config

- Model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- Compressor: `/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- Sparse-VLLM DeltaKV method request: `deltakv-triton-v4`
- Resolved cache manager: `DeltaKVCacheTritonManagerV4`
- `decode_keep_tokens`: `4096`
- `prefill_keep_tokens`: `4096`
- `sink_keep_tokens`: `8`
- `recent_keep_tokens`: `128`
- `full_attention_layers`: `0,1,2,4,7,14`
- `deltakv_center_ratio`: `0.1`
- `deltakv_latent_dim`: `256`
- `deltakv_latent_quant_bits`: `0`
- `deltakv_neighbor_count`: `4`
- HF prefill chunk size: `100000000`
- Sparse-VLLM prefill chunk size: `4096`

The HF side explicitly normalizes `decode_keep_tokens` to `num_top_tokens` and
`prefill_keep_tokens` to `num_top_tokens_in_prefill`; both are integer `4096`.

## Output

Main output directory:

```text
/data2/haojitai/outputs/sparsevllm_logits_align/20260516_0304_qwen25_7b_q0
```

Files:

- `run_info.json`: command environment, git state, args, full results.
- `summary.json`: same run-level payload after successful completion.
- `short_vanilla.json`, `long_vanilla.json`
- `short_deltakv.json`, `long_deltakv.json`

## Decode Results

| Case | Prompt tokens | Method | max abs diff | mean abs diff | argmax match | top-1 | top-10 | top-50 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| short | 5 | vanilla | 0.21875 | 0.032235 | true | 1.00 | 1.00 | 0.96 |
| short | 5 | DeltaKV | 0.21875 | 0.032235 | true | 1.00 | 1.00 | 0.96 |
| long | 9000 | vanilla | 0.1875 | 0.027787 | true | 1.00 | 0.90 | 1.00 |
| long | 9000 | DeltaKV | 1.390625 | 0.156811 | true | 1.00 | 1.00 | 0.98 |

## Notes

- The short DeltaKV case is intentionally below the sparse threshold and does
  not exercise DeltaKV compression; it matches the vanilla short baseline.
- The long DeltaKV prefill metrics match the vanilla long prefill metrics
  (`max_abs_diff=0.25`, `mean_abs_diff=0.031788`, argmax match true). The
  larger DeltaKV difference appears only on decode, after prefill compression
  and decode reconstruction.
- Running non-Triton `deltakv` did not improve decode alignment, so the observed
  long DeltaKV residual is not specific to `DeltaKVCacheTritonManagerV4`.
- The remaining gap is most likely in DeltaKV cache compression/reconstruction
  details between the HF cache pipeline and Sparse-VLLM cache manager, rather
  than in top-token parameter parsing or full-cache attention numerics.
- A short OmniKV HF attempt failed before logits comparison because
  `load_omnikv_model()` sets `use_cluster=False`, while the shared HF
  `Qwen2KVCompress.forward()` path now creates a cluster-only DeltaKV cache.
  The failing run info is under
  `/data2/haojitai/outputs/sparsevllm_logits_align/quick_short_omnikv`.

## HF OmniKV Fix

Branch: `fix/hf-omnikv-alignment`

The HF OmniKV path now uses an explicit `OmniKVRawCache` instead of trying to
instantiate the cluster-only DeltaKV cache. The cache stores raw sink, exact
history, and recent tail KV, and the HF selection path uses raw QK logits for
decode to match Sparse-VLLM's observation kernels. The logits comparison script
also now runs Sparse-VLLM prefill according to the resolved
`prefill_schedule_policy`; for OmniKV this is `all_chunked` with
`chunk_prefill_size=4096`.

Final OmniKV alignment run:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_logits_hf_sparsevllm.py \
  --cases short,long \
  --methods omnikv \
  --output_dir /data2/haojitai/outputs/sparsevllm_logits_align/quick_omnikv_fixed_policychunk_exact_rawscore
```

Long vanilla policy-chunk baseline:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_logits_hf_sparsevllm.py \
  --cases long \
  --methods vanilla \
  --output_dir /data2/haojitai/outputs/sparsevllm_logits_align/quick_long_vanilla_policychunk
```

Output directories:

- `/data2/haojitai/outputs/sparsevllm_logits_align/quick_omnikv_fixed_policychunk_exact_rawscore`
- `/data2/haojitai/outputs/sparsevllm_logits_align/quick_long_vanilla_policychunk`

Decode results from the final OmniKV run:

| Case | Prompt tokens | Method | max abs diff | mean abs diff | p99 abs diff | argmax match | top-1 | top-10 | top-50 |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| short | 5 | OmniKV | 0.21875 | 0.032235 | 0.109375 | true | 1.00 | 1.00 | 0.96 |
| long | 9000 | OmniKV | 0.28125 | 0.041316 | 0.125 | true | 1.00 | 0.90 | 1.00 |
| long | 9000 | vanilla baseline | 0.1875 | 0.027787 | 0.09375 | true | 1.00 | 0.90 | 1.00 |

The remaining long OmniKV decode gap is close to the vanilla HF/Sparse-VLLM
kernel baseline and no longer shows the earlier sparse-view-scale mismatch
(`mean_abs_diff` was about `0.174466` before the raw-QK decode scoring fix).

## Fixes Added

- Added `scripts/debug/compare_logits_hf_sparsevllm.py`.
- Fixed HF DeltaKV short prompts with `prompt_len < num_sink_tokens`: the cache
  view now exposes only filled sink slots, avoiding RoPE position/view length
  mismatch.
- Added a regression test for that short-prompt sink-budget boundary.
- Added HF OmniKV raw-cache construction, exact-history recent-tail handling,
  and raw-QK OmniKV selection tests.

## Verification

```bash
conda run -n svllm python -m compileall -q \
  src tests scripts/debug/compare_logits_hf_sparsevllm.py

conda run -n svllm python -m unittest tests.test_hf_deltakv_modeling

git diff --check
```
