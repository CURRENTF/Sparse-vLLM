# LLaVA-OneVision Visual Cache Benchmarks

This note keeps the LLaVA-OneVision benchmark naming explicit.

## Current No-Checkpoint Path

When the benchmark is run with:

```bash
--compressor_path none
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

The implementation uniformly samples visual token positions from the eligible
visual-token span and drops the remaining eligible visual tokens. Text tokens
remain in the raw KV cache.

## Optional Int4 Variant

With:

```bash
--compressor_path none
--quantize_visual_kv
```

the method label becomes:

```text
visual_uniform_keep_int4
```

This still does not use DeltaKV cluster/ref/compressor logic. It stores kept
visual KV tokens using direct min/max int4 packing.

## Experimental Compressor Path

Supplying a real `--compressor_path` runs through the LLaVA-OneVision DeltaKV
wrapper and labels the method as:

```text
visual_deltakv_compressor
```

Whether this uses cluster/ref behavior depends on the checkpoint config. Treat
results from this path separately from `visual_uniform_keep`.

## Entry Points

Use the explicit benchmark entrypoint for new runs:

```bash
python scripts/bench_llava_onevision_visual_prune.py
```

The older entrypoint remains as a compatibility wrapper:

```bash
python scripts/bench_llava_onevision_deltakv.py
```

It delegates to the visual-pruning script and prints a deprecation warning.
