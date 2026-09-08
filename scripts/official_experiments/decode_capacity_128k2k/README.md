# 128K input / 2K output decode capacity

This package preserves the experiment recipe and all 44 accepted points across
10 curves. Original raw outputs and historical records are retained separately;
the JSON/CSV copies here replace private paths with run-root-relative artifact IDs.

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
  --plot-data scripts/official_experiments/decode_capacity_128k2k/data/plot_data.json \
  --output-dir scripts/official_experiments/decode_capacity_128k2k/plots
```

This creates `decode_capacity_128k2k` and `decode_capacity_128k2k_logy` in PNG,
PDF, and SVG, plus per-model figures. Both use a base-2 concurrency axis; y is
linear or logarithmic respectively. Override colors with `--palette FILE.json`.
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
`--repo`. It must include the synchronized full-batch stage adapters for both
engines and the SnapKV bootstrap capacity repair. This package contains the
orchestration layer, not those engine changes. Use a compatible checkout or the
exact archived source identified in `data/provenance.json`; the recorded base
commit alone is insufficient to reconstruct that source.

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
