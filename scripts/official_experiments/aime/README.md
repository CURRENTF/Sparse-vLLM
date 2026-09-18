# AIME 2024: methods without auxiliary checkpoints

This recipe calls `benchmark/math_bench/pred.py` and its canonical `eval.py`
scorer. It covers Vanilla, StreamingLLM, SnapKV, H2O, PyramidKV, OmniKV,
QuEST, RKV, KIVI, TurboQuant and FP8 KV. DeltaKV and SkipKV are excluded
because this campaign does not provision method-specific learned assets.
Aliases and independent `prefill_sparse_method` combinations are not separate
methods in this campaign.

## Settings

`setting.json` is the experiment configuration. Use the same
Qwen3-30B-A3B-Thinking-2507-FP8 checkpoint for every method (FP8 weights,
BF16 activations; KV representation depends on the method). GLM-4.7-Flash from the MiniSWE
recipe cannot cover the full matrix: quantized KV requires explicit KV and
does not support GLM MLA. Model compatibility is based on current source;
this recipe has not been GPU validated.

Inherited from `../chain_cache_miniswe/setting.json`: TP2/EP2/DP1, memory
fraction .95, engine concurrency limits 64, prefill chunk 4096, batch token
budget 65536, decode reservation 1024, CUDA Graph enabled, FP32 attention
scores, temperature .7, top_p 1 and maximum output 16384 tokens. Existing
SnapKV/H2O/OmniKV/QuEST budgets are retained.

Explicit differences: context capped at 40960 for this campaign;
30 independent single-turn problems in one batch; prefix caching disabled
for every method (quantized KV does not support it). There are no tools,
agent retries, chain continuations or preserved previous thinking turns.
Thinking uses the tokenizer chat template with `enable_thinking=true`.
The MathBench `deepseek` problem prompt is retained, while synthetic output
think-prefix insertion and the extra user instruction to emit that prefix
are disabled. Seed is 42, top_k is 0. These are borrowed MiniSWE sampling
settings, not a claim to reproduce a published AIME score.

| Method | Budget / representation |
|---|---|
| Vanilla | Dense KV |
| StreamingLLM | 64 sink + 1984 recent |
| SnapKV | 64 sink + 512 recent + 15808 selected; window 32 |
| H2O | Prefill 16384, decode 8192; eviction interval 128 |
| PyramidKV | Base budget 16384; layer ratios .6 to .01 |
| OmniKV | Selection 2048; model-profile full layers |
| QuEST | Selection 2048; chunks 16; first 2 layers full |
| RKV | Budget 16384; compression interval 128; observation 8 |
| KIVI | 4-bit, page size 32 |
| TurboQuant | 4-bit, page size 32, rotation seed 0 |
| FP8 KV | FP8, page size 32 |

These settings are not equal physical-memory budgets. Short AIME prompts
may not trigger prefill compression; retain this limitation when interpreting
SnapKV/PyramidKV scores. Decode compression differs by method.

## Run

Activate the inference environment first (including its compiler executables).
It needs the usual MathBench dependencies and `math-verify==0.9.0`.
Export `Maxwell-Jia/AIME_2024`, split `train`, to local JSON/JSONL with
`Problem` and `Answer` fields. Supply exactly 30 distinct problems; the driver
saves the normalized dataset and its checksum. Verify the export provenance;
row count alone cannot authenticate the dataset.

From the repository root, set `MODEL_PATH`, `AIME_DATA`, `RUN_ROOT` and
`GPU_PAIR` for your machine, then preview:

```bash
python scripts/official_experiments/aime/run.py \
  --model "$MODEL_PATH" --data "$AIME_DATA" \
  --output "$RUN_ROOT" --gpus "$GPU_PAIR"
```

Add `--execute` to run all 11 methods sequentially. `--methods vanilla quest`
selects a subset; use a new output directory for each invocation. Run long
jobs in tmux. The driver checks the selected GPUs for compute processes,
memory use and activity before each method, and stops on a busy GPU or failed
method. It does not retry or overwrite existing runs. `--timeout` bounds each
method (default 43200 seconds). Perform GPU smoke validation before treating
the full matrix as runnable on a new device/model/toolchain.

## Artifacts and acceptance

The run root contains settings, normalized inputs, a Git commit/dirty-state
manifest, exact commands and per-method logs/engine configs. MathBench saves
raw predictions, parsed outputs, per-sample results and `result.json` under
each method's `benchmark/math_bench/pred/aime/` directory.

`final_summary.json` is written only after every selected method has all 30
unique sample IDs and consistent aggregate/per-sample results. Parse failures
count as incorrect in the denominator; invalid inputs or metric failures
reject the run. A failed run retains its manifest, logs and partial artifacts.
The wrapper checks artifacts even if the underlying entrypoint exits zero
after a scorer failure. This is one sampled answer per problem (pass@1),
not pass@k or a multi-seed estimate.

MathBench performance fields count re-tokenized output text over generation
call time. They are diagnostic and are not HTTP throughput, exact generated
token counts or isolated decode-stage throughput.
