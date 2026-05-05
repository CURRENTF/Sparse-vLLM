# LLaVA-OneVision Visual Cache Benchmarks

This note keeps the LLaVA-OneVision benchmark naming explicit.

## Method Names

The benchmark intentionally separates three different ideas:

| Method | Checkpoint | Actual behavior |
| --- | --- | --- |
| `vanilla` | None | Standard HF LLaVA-OneVision generation. |
| `deltakv` | Required | Standard DeltaKV wrapper with a learned compressor checkpoint. |
| `deltakv_delta_quant` | Forbidden | DeltaKV-style cluster/ref path, but stores token-space delta residuals directly with int4 quantization. No learned compressor is loaded or used. |
| `visual_uniform_keep` | Forbidden | Uniform visual-token pruning baseline. No cluster, no ref tokens, no compressor. |
| `visual_uniform_keep_int4` | Forbidden | Same uniform visual keep path plus int4 storage of kept visual KV. |

## Direct Delta Quant Path

When the benchmark is run with:

```bash
--deltakv_checkpoint_path none
--methods deltakv_delta_quant
--delta_quant_bits 4
--deltakv_center_ratio 0.1
--deltakv_neighbor_count 1
```

the method is:

```text
llava_deltakv_delta_quant
```

This is the no-compressor DeltaKV path:

- no learned DeltaKV compressor checkpoint,
- cluster/prototype centers are used as ref tokens,
- the stored value is `token_kv - mean(selected_ref_tokens)`,
- the residual is direct int4 quantized,
- generation supports `--batch_size > 1` with left padding.

It compresses the eligible text-backbone KV stream. In image VQA prompts, most
eligible prompt tokens are visual tokens, but this is not a visual-only pruning
method.

## Visual Uniform No-Checkpoint Path

When the benchmark is run with:

```bash
--deltakv_checkpoint_path none
--methods vanilla,visual_uniform_keep
--visual_keep_ratio 0.1
```

the method is:

```text
visual_uniform_keep
```

It is a visual-token uniform-pruning baseline:

- no DeltaKV learned compressor,
- no DeltaKV cluster/prototype selection,
- no ref-token residual path,
- no SnapKV attention-score top-k,
- no KV quantization unless `--quantize_visual_kv` is set.

The implementation uniformly samples visual-token positions from the eligible
visual-token span and drops the remaining eligible visual tokens. Text tokens
remain in the raw KV cache.

## Optional Int4 Variant

With:

```bash
--deltakv_checkpoint_path none
--quantize_visual_kv
```

the method label becomes:

```text
visual_uniform_keep_int4
```

This still does not use DeltaKV cluster/ref/compressor logic. It stores kept
visual KV tokens using direct min/max int4 packing.

## Standard DeltaKV Compressor Path

Supplying a real `--deltakv_checkpoint_path` with `--methods deltakv` runs
through the LLaVA-OneVision DeltaKV wrapper and labels the method as:

```text
llava_deltakv
```

Whether this uses cluster/ref behavior depends on the checkpoint config. Treat
results from this path separately from both `visual_uniform_keep` and
`deltakv_delta_quant`.

## Entry Points

Use the explicit benchmark entrypoint for new runs:

```bash
python scripts/bench_llava_onevision_visual_prune.py
```

The older entrypoint remains as a legacy script-name wrapper:

```bash
python scripts/bench_llava_onevision_deltakv.py
```

It delegates to the visual-pruning script and prints a deprecation warning.
