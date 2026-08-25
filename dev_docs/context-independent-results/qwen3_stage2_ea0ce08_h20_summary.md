# Qwen3 batch-only decode graph stage 2 validation

Date: 2026-08-25

Commit: `ea0ce08f261e6b80facd6500244ea723538d1f0b`

Environment: NVIDIA H20, CUDA 13.0, PyTorch 2.11.0, Triton 3.6.0,
`sglang-kernel` 0.4.5, Qwen3-8B BF16, TP=1.

## Scope

- Dense Qwen3 GQA (`32` query heads, `8` KV heads, head dimension `128`).
- Prompt lengths `1024` and `8192`, output length `32`.
- Fixed batches and oversubscribed churn at concurrency `1` and `4`.
- Two measured iterations after one warmup iteration.
- Matched scheduler budget, traces, hardware, model and commit for batch-only,
  bucketed and eager runs.

The fixed and churn probes use separate processes because `bench_probe.py` creates
one LLM per churn concurrency. Churn concurrency `1` declares batch bucket `[1]`;
concurrency `4` declares `[1, 2, 4]`. Together they cover every declared stage 2
batch bucket without asking a one-row runtime to capture larger batch graphs.

## Graph identity

| Policy | Concurrency | Startup cached graphs | Measured replays | New captures | Recaptures | Eager-static steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch-only | 1 | 1 | 217 | 0 | 0 | 0 |
| batch-only | 4 | 3 | 236 | 0 | 0 | 0 |
| bucketed | 1 | 5 | 217 | 0 | 0 | 0 |
| bucketed | 4 | 15 | 236 | 0 | 0 | 0 |
| eager | 1 | 0 captured | 0 | 0 | 0 | 217 |
| eager | 4 | 0 captured | 0 | 0 | 0 | 236 |

The batch-only startup keys contain only batch buckets `[1, 2, 4]`, use the
single `dense` topology path, and store `context_capacity=0` in graph identity.
All three graphs share the configured capture capacity `8352`. The bucketed
baseline captures the Cartesian product of the same batch buckets and context
buckets `[1024, 2048, 4096, 8192, 8352]`.

## Matched end-to-end results

Output throughput is tokens per second. Ratios compare batch-only with the
matched baseline in the same row.

| Scenario | Prompt | Concurrency | Batch-only output tok/s | vs bucketed | vs eager | Batch-only TPOT ms | Bucketed TPOT ms | Eager TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 1024 | 1 | 100.76 | 1.015x | 2.876x | 6.54 | 6.69 | 25.78 |
| fixed | 1024 | 4 | 201.26 | 1.013x | 1.949x | 6.94 | 7.22 | 26.41 |
| fixed | 8192 | 1 | 25.91 | 1.004x | 1.466x | 7.16 | 7.31 | 25.75 |
| fixed | 8192 | 4 | 31.24 | 1.016x | 1.131x | 8.58 | 10.70 | 26.03 |
| churn | 1024 | 1 | 98.19 | 1.007x | 2.848x | 6.54 | 6.61 | 26.02 |
| churn | 1024 | 4 | 175.04 | 1.008x | 1.876x | 13.03 | 13.21 | 31.97 |
| churn | 8192 | 1 | 23.79 | 1.004x | 1.437x | 7.04 | 7.22 | 26.11 |
| churn | 8192 | 4 | 27.65 | 1.013x | 1.129x | 87.14 | 89.14 | 105.00 |

These measurements establish a no-regression gate for this Qwen3/H20 contract;
they are not a universal performance claim for other hardware, models or sparse
methods.

## Provider binding

The batch-only binding report selects `triton_context_independent` and rejects
the other providers because their planning or launch topology depends on context
length. Its capture-time launch plan is:

```text
plan_id = portable_context_independent_v1
context_capacity = 8352
max_kv_splits = 16
target_tokens_per_split = 256
block_n = 64
stage1 = 2 warps, 2 stages
stage2 = 4 warps, 2 stages
workspace_owner = provider
```

The independent CUDA oracle also passed BF16 GQA head dimensions `128` and
`256`, FP16 MHA head dimension `64`, CUDA Graph replay with changed context
lengths, and raw per-head attention-score output.

## Artifacts

- `qwen3_stage2_ea0ce08_h20_batch_only_fixed/`
- `qwen3_stage2_ea0ce08_h20_batch_only_churn_c1/`
- `qwen3_stage2_ea0ce08_h20_batch_only_churn_c4/`
- `qwen3_stage2_ea0ce08_h20_bucketed_fixed/`
- `qwen3_stage2_ea0ce08_h20_bucketed_churn_c1/`
- `qwen3_stage2_ea0ce08_h20_bucketed_churn_c4/`
- `qwen3_stage2_ea0ce08_h20_eager_fixed/`
- `qwen3_stage2_ea0ce08_h20_eager_churn_c1/`
- `qwen3_stage2_ea0ce08_h20_eager_churn_c4/`

Each directory contains the run manifest, raw iteration and request samples,
summary, operator binding report, hardware samples and run status.

## Remaining gate

This artifact set validates TP=1. A multi-GPU Qwen3 run remains required before
claiming the stage 2 production TP matrix is complete; no two allowed local GPUs
were idle during this run, and the exclusive H20 environment exposes one GPU.
