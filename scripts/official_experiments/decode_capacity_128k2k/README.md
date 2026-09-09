# 128K input / 2K output decode capacity

This package preserves the experiment recipe and all 44 accepted points across
10 curves. Original raw outputs and historical records are retained separately;
the JSON/CSV copies here replace private paths with run-root-relative artifact IDs.

The original export uses OmniKV selected-history budget 2048 (total 2624).
The `config.omnikv-total2048.json` variant aligns OmniKV with QuEST at total
2048: 1472 selected + 64 sink + 512 recent. It retains auto full-attention layers.
The updated 44-point comparison is in `data/omnikv-total2048/`; the original
`data/plot_data.json`, CSV, config and provenance are preserved unchanged.

`data/sm-parallel/` is the context-independent SM-scaled MLA profile update.
Only GLM OmniKV is remeasured; the other nine curves are preserved exactly.
It includes portable plot data, actual per-rank launch plans and comparison
data. Both previous exports remain available unchanged.

`data/tangram-hisparse/` extends that comparison with Tangram SnapKV and the
HiSparse QuEST PR-series on Qwen3-30B FP8. Their GLM MLA combinations are explicit
N/A entries. The original 44 points remain unchanged and were revalidated from
raw steps and complete outputs before export. The separate Triton MLA SM rule
does not change these original attention paths. See
[`../triton_mla_sm_schedule/`](../triton_mla_sm_schedule/) for the external stage
adapters, their timing/algorithm differences, tuning recipe and validation.

## Contents

- `sweep_decode_capacity.py`: idle-GPU guarded sweep and integer boundary search,
  calling the selected checkout's canonical `benchmark/microbench.py`.
- `decode_capacity_guard.py`: keeps the reservation between cases; foreign GPU
  contention aborts this queue without terminating foreign processes.
- `config.json`: measured model topologies, method parameters, and runtime path
  placeholders. Resolved configuration is saved as `campaign.json` in each attempt.
- `plot_decode_capacity.py`, `palettes/fresh_modern.json`: standalone Matplotlib /
  Seaborn plotting with native constrained layout and editable method colors.
- `data/plot_data.json`: authoritative replot input, including capacity attempts,
  stage sums, all accepted points, and configuration; `data/points.csv` is its flat
  export. `data/environment.json` records the measured software/hardware environment.
- `data/provenance.json`: original source and artifact identity; no private paths.

## Replot without models or GPUs

Run from the repository root using an environment with Matplotlib and Seaborn
(measured plotting versions: Matplotlib 3.11.1, Seaborn 0.13.2):

```bash
python scripts/official_experiments/decode_capacity_128k2k/plot_decode_capacity.py \
  --plot-data scripts/official_experiments/decode_capacity_128k2k/data/tangram-hisparse/plot_data.json \
  --output-dir scripts/official_experiments/decode_capacity_128k2k/plots
```

This creates `decode_capacity_128k2k` and `decode_capacity_128k2k_logy` in PNG,
PDF, and SVG, plus per-model figures. Both use a base-2 concurrency axis; y is
linear or logarithmic respectively. Override colors with `--palette FILE.json`.
Figures omit titles and protocol annotations for use with an external caption.
Overview canvases are 7 × 2.8 inches; per-model canvases are 4 × 2.9 inches.
Font sizes remain unchanged. Legends use fixed columns (3 in the overview,
2 per model), with explicit `Tangram (SnapKV)` and `HiSparse (QuEST)` labels,
not automatic width-based wrapping; constrained layout positions the artists.
Export validation checks completeness and token/time arithmetic; it cannot
revalidate raw steps or token outputs that are not bundled here.

## Measurement contract

Input is 131072 tokens; every request completes 2048 output tokens. Qwen3-30B-A3B-
Instruct-2507-FP8 uses TP1/EP1; GLM-4.7-Flash BF16 uses TP2/EP2, **not DP2**.
Both use H100 80GB and memory utilization 0.9. The five lanes are native vanilla,
vLLM vanilla, SnapKV, QuEST, and OmniKV.

- SnapKV retains 8192 prompt tokens (64 sink + 512 recent + 7616 selected), with
  wave size 1 and decode gap 1 for admission.
- QuEST uses total budget 2048 (64 sink + 512 recent + 1472 selected).
- OmniKV uses auto full-attention layers and 2048 selected tokens in addition to
  64 sink + 512 recent tokens. Its budget convention differs from QuEST's.
- Pure decode throughput is actual computed decode tokens divided by accumulated
  synchronized full-batch engine-step time, including engine bookkeeping. Exclude
  mixed/prefill steps, admission, the first 32 full-batch decode steps, and tails
  after batch occupancy falls. This is not request TPOT or end-to-end throughput.
- Sweep powers of two, then binary-search the integer maximum and verify failure
  at maximum + 1. Preserve successful boundary-search points and failed attempts.
  Maxima are conditional on this configuration, not theoretical device limits.
- One workload per concurrency; no repeated-run confidence intervals or smoothing.

## Run a new sweep

GPU execution needs a compatible benchmark checkout, explicitly selected by
`--repo`. The repository's canonical microbench supports synchronized full-batch
stage adapters for both engines, and includes the SnapKV bootstrap capacity
repair. Set `BENCHMARK_REPO` to this checkout for new runs. For exact historical
source reproduction, use the archive identified in `data/provenance.json`; the
recorded base commit alone is insufficient to reconstruct that source.

Set these environment variables to your actual absolute paths:

| Variable | Meaning |
| --- | --- |
| `BENCHMARK_REPO` | Compatible benchmark checkout |
| `CONDA_EXE` | Conda executable |
| `SPARSE_VLLM_ENV`, `VLLM_ENV` | Native and vLLM conda environment prefixes |
| `QWEN3_MODEL`, `GLM47_MODEL` | Local model directories |
| `DECODE_OUTPUT_ROOT` | New persistent campaign output directory |
| `DECODE_SCRATCH_ROOT` | Scratch directory on a disk with sufficient space |

Alternatively, make an ignored `config.local.json` with literal absolute paths
and pass it instead. Never overwrite the original campaign. The example uses
automatic idle-device selection; explicit GPU indices are also accepted.

```bash
python scripts/official_experiments/decode_capacity_128k2k/sweep_decode_capacity.py \
  --config scripts/official_experiments/decode_capacity_128k2k/config.json \
  --repo "$BENCHMARK_REPO" --model qwen3-30b-fp8 --gpus auto:1 --check-only

python scripts/official_experiments/decode_capacity_128k2k/sweep_decode_capacity.py \
  --config scripts/official_experiments/decode_capacity_128k2k/config.json \
  --repo "$BENCHMARK_REPO" --model qwen3-30b-fp8 --gpus auto:1 --attempt run1

python scripts/official_experiments/decode_capacity_128k2k/sweep_decode_capacity.py \
  --config scripts/official_experiments/decode_capacity_128k2k/config.json \
  --repo "$BENCHMARK_REPO" --model glm4.7-flash --gpus auto:2 --attempt run1
```

Use tmux for the long-running sweep commands. `--check-only` checks paths and
required adapter option presence without touching GPUs; it is not a CUDA
correctness test or a proof of measurement-contract equivalence. The sweep also
validates complete raw outputs and full-batch step sums for every accepted point.

After both models complete, use `plot_decode_capacity.py --config PATH` with an
attempt's resolved `campaign.json` to validate the new campaign's raw artifacts
and regenerate both plots. This requires exactly one completed sweep per
model/lane; incomplete or ambiguous campaigns fail explicitly.

## Rerun OmniKV with the aligned total budget

Use the same measured runtime source and environment as the original campaign;
do not combine new-runtime OmniKV measurements with old-runtime baselines.
Set the same path variables above, but choose a **new** `DECODE_OUTPUT_ROOT`.
Pass `config.omnikv-total2048.json` to the canonical sweep with
`--lanes svllm-omnikv --attempt total2048`, once per model. The existing sweep
runs a fresh smoke, preserves its GPU reservation, and searches through the exact
integer capacity boundary. `run_aligned_omnikv.sh BENCHMARK_REPO RESOLVED_CONFIG
OUTPUT_ROOT` runs these two queues serially under tmux; its config must already
contain absolute runtime paths and its output root must match the third argument.

After both queues finish, merge only the replacement curves and revalidate all
ten curves against raw data:

```bash
python scripts/official_experiments/decode_capacity_128k2k/merge_omnikv_rerun.py \
  --original-root "$ORIGINAL_CAMPAIGN_ROOT" \
  --rerun-root "$DECODE_OUTPUT_ROOT" \
  --output-dir "$DECODE_OUTPUT_ROOT/plots-combined"
```

The merger rejects additional config, model-config, runtime-source, or
measurement-source changes. It retains every other method's original numerical
values, validates the rerun's boundary, and writes new PNG/PDF/SVG plus JSON/CSV.
It refuses an existing output directory; previous figures and measurements are
never overwritten.
The `portable/` subdirectory contains path-independent JSON/CSV. Artifact IDs
start with `original/` or `omnikv-total2048/`, identifying the original campaign
root or the rerun campaign root respectively. Replotting does not require either
raw root; full revalidation does.

## MLA split-profile ablation

### Context-independent SM-scaled profile rerun

`sm_parallel_nearest_v1` chooses the closest split count in `[4, 8, 16, 32]`
to `SM_count / (batch * head_tiles)`, with lower splits winning exact ties.
The target is one CTA per SM; context length and capacity do not select splits.
Other launch parameters and provider eligibility remain unchanged. This is a
hardware-scaled heuristic, not cross-GPU performance validation.

Use the original measured runtime with **only** the reviewed MLA runtime and
provider-binding patch. `update_mla_profile_results.py prepare --repo
"$BENCHMARK_REPO" --base-plot "$BASE_PLOT_DATA" --run-root "$PROFILE_RUN_ROOT"
--scratch-root "$DECODE_SCRATCH_ROOT"`
rejects unrelated runtime/measurement changes and preserves source/config hashes
and an archive. `BASE_PLOT_DATA` is the full-path, raw-validated aligned-budget
export, not its portable copy. Use a short absolute scratch path on the output
volume (at most 65 characters) for multiprocessing Unix sockets.
Set `CONDA_EXE` and `SPARSE_VLLM_ENV`, then run
`run_sm_parallel_profile.sh "$BENCHMARK_REPO" "$PROFILE_RUN_ROOT"
"$VALIDATION_ROOT" auto:2` under tmux. `VALIDATION_ROOT` comes from the successful
`tilelang_mla_split_profiles/validate_rule.sh` correctness run on the patched
implementation. No additional split search or context benchmark is performed.

The canonical sweep reruns GLM OmniKV to its integer capacity boundary. Export
checks every run's runtime identity, hyperparameters and bound split plans,
revalidates all raw steps/outputs, and retains the other nine curves only after
checking they do not select the changed provider. New linear/log-y PNG/PDF/SVG,
JSON/CSV and actual generated launch plans are written to `PROFILE_RUN_ROOT/export`.
The old aligned-budget and original-budget results are never overwritten.

### Historical fixed-32 control

`profile_ablation.json` and `run_mla_profile_ablation.py` reproduce the GLM
TP2/EP2 BS4/BS5 ABBA control: the existing split table versus fixed 32 splits.
This is **not autotuning**. Other provider choices and method parameters remain
unchanged. The runner reserves an idle GPU pair across numerical checks, smokes,
and all formal cases, then uses the same canonical microbench and raw validator.

The temporary runtime switch is not a supported engine setting. Historical
reproduction requires the original measured source plus the archived
`data/mla-split-profile-ablation/profile-bypass.patch`; apply it only to an
explicitly selected experimental checkout. The runner rejects an unpatched
checkout and checks the numerical oracle's actual split count before benchmarking.
Remove the patch again after the experiment. Do not apply it to unrelated or
newer runtime code without reviewing the changed contract.

Set `BENCHMARK_REPO`, `CONDA_EXE`, `SPARSE_VLLM_ENV`, `GLM47_MODEL`,
`DECODE_SCRATCH_ROOT`, and a fresh `ABLATION_OUTPUT_ROOT`, then run under tmux:

```bash
python3 scripts/official_experiments/decode_capacity_128k2k/run_mla_profile_ablation.py \
  --config scripts/official_experiments/decode_capacity_128k2k/profile_ablation.json

python3 scripts/official_experiments/decode_capacity_128k2k/summarize_mla_profile_ablation.py \
  --run-root "$ABLATION_OUTPUT_ROOT" --output-dir "$ABLATION_OUTPUT_ROOT/export"
```

The exporter requires all eight formal runs, validates raw tokens and stage
sums, checks both ranks' launch plans, and compares operator metadata after
excluding split counts. Its portable JSON/CSV retain every repetition; two-run
means are descriptive, not confidence intervals. This focused ablation does not
replace the ten capacity curves or establish a universal split-count default.
