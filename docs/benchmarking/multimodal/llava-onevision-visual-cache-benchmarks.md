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

Use this benchmark entrypoint for new runs:

```bash
python benchmark/multimodal/visual_cache/run_visual_cache.py
```

Older local notes may mention legacy wrapper names. Treat
`benchmark/multimodal/visual_cache/run_visual_cache.py` as the canonical
entrypoint in new commands.

## Reproducibility Artifacts

`benchmark/multimodal/visual_cache/run_visual_cache.py` now writes experiment artifacts
under `--output_dir`. If `--output_dir` is omitted, it writes to `--dataset_dir`
for compatibility with older local commands.

Each method writes:

- `<method>_raw_outputs.jsonl`: raw generations.
- `<method>_parsed_outputs.jsonl`: parsed text, labels, VQA score, and explicit
  per-sample status.
- `<method>_per_sample_results.jsonl`: full per-sample records.
- `<method>_aggregate_metrics.json`: aggregate speed, memory, status counts,
  and VQA metrics.
- `run_info.json`: command, git commit, model, dataset, decoding parameters,
  seed, runtime parameters, and evaluated sample count.
- `last_benchmark_result.json`: method summaries plus artifact paths.

When exactly two methods are run and one is `vanilla`, the candidate aggregate
metrics include `speedup_vs_vanilla`, `memory_delta_gb_vs_vanilla`,
`vqa_score_delta_vs_vanilla`, and `contains_answer_delta_vs_vanilla`.
