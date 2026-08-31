# Efficiency and Throughput Benchmark Suite

[简体中文](../../zh/benchmarking/efficiency.md) | English

## Purpose

The efficiency suite compares Sparse-vLLM methods with the vLLM full-attention
baseline under matched synthetic request traces. It covers prompt length,
concurrency, fixed batches, oversubscribed request churn, tensor parallelism,
request latency, throughput, GPU activity, and peak memory.

The synthetic workload uses deterministic random token IDs. Every measured
iteration receives a new trace, systems receive the same trace for the same
seed and case, prompt lengths vary within batches when concurrency is greater
than one, and prefix caching is disabled. The churn scenario submits more
requests than the configured maximum concurrency so that sequence replacement
and scheduler behavior are included in the measurement.

Use the synthetic suite to locate suspicious settings. Use the separate Nsight
diagnostic only after a suspicious case has been identified.

## Prerequisites

Run commands from the repository root. Before a run, verify that:

- Sparse-vLLM and its benchmark dependencies are installed in the selected
  Python environment.
- The same `PYTHON_BIN` can import `vllm` when `vllm-vanilla` is requested.
- `nvidia-smi` is available and every selected physical GPU is idle.
- The model path or Hugging Face model ID is accessible.
- The output root has enough free space for JSONL traces and hardware
  timelines.
- `jq` is available if you want to use the JSON verification snippets below.

Useful checks:

```bash
cd "<SPARSE_VLLM_REPO>"

PYTHON_BIN=python3
"${PYTHON_BIN}" -c 'import sparsevllm'
"${PYTHON_BIN}" -c 'import vllm'  # Required only for the vLLM baseline.
nvidia-smi
```

The shell wrappers reject GPUs that already have compute processes. The
standalone Python CLI does not perform that idle-device preflight, so check the
devices manually before using it.

## Quick Smoke Test

This small fixed-batch run verifies model loading, inference, hardware sampling,
and artifact generation on Qwen3-30B with TP=2. It is not a representative
throughput result.

```bash
cd "<SPARSE_VLLM_REPO>"

PROMPT_LENS=4096 \
OUTPUT_LENS=32 \
BATCH_SIZES=1 \
BENCH_SCENARIO=fixed \
NUM_WARMUPS=0 \
NUM_ITERS=1 \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_efficiency_probe.sh \
  "svllm-vanilla" \
  "qwen3_30b" \
  "0,1"
```

The wrapper creates a timestamped directory below `<OUTPUT_ROOT>`. A successful
run contains `svllm-vanilla/run_status.json` with `"status": "success"`.

## Matched Cross-System Sweep

The following command compares H2O, SnapKV, Sparse-vLLM full attention, and
vLLM full attention across prompt length and concurrency. `BENCH_SCENARIO=all`
runs both fixed batches and oversubscribed churn.

```bash
cd "<SPARSE_VLLM_REPO>"

PROMPT_LENS="8192,16384,32768" \
OUTPUT_LENS=512 \
BATCH_SIZES="1,4,8" \
BENCH_SCENARIO=all \
BENCH_SEED=42 \
NUM_WARMUPS=1 \
NUM_ITERS=3 \
MAX_NUM_BATCHED_TOKENS=8192 \
SPARSE_PREFILL_SCORE_MODE=probability \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_efficiency_probe.sh \
  "svllm-h2o,svllm-snapkv,svllm-vanilla,vllm-vanilla" \
  "qwen3_30b" \
  "0,1"
```

This is a substantial run: each system executes every prompt-length,
output-length, concurrency, scenario, warmup, and measured-iteration
combination. Start with the smoke test or a single prompt length before running
the complete matrix. A 512-token output is preferred for representative decode
metrics; shorter outputs are appropriate only for validation or time-bounded
exploration.

Supported wrapper system names are:

| Name | Engine | Method |
| --- | --- | --- |
| `svllm-vanilla` | Sparse-vLLM | `vanilla` |
| `svllm-h2o` | Sparse-vLLM | `h2o` |
| `svllm-snapkv` | Sparse-vLLM | `snapkv` |
| `svllm-omnikv` | Sparse-vLLM | `omnikv` |
| `svllm-deltakv` | Sparse-vLLM | `deltakv` |
| `vllm-vanilla` or `vllm` | vLLM | `vanilla` |

The probe wrapper recognizes these model aliases:

| Alias | Default model | Tensor parallel size |
| --- | --- | --- |
| `qwen3_30b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 2 |
| `qwen3_8b` | `Qwen/Qwen3-8B` | 1 |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct-1M` | 1 |

Sparse method parameters are resolved from
`benchmark/sparsevllm_regression/manifest.json`, including model-specific
OmniKV full-attention layers. An OmniKV run fails before model loading when the
model has no calibrated manifest entry. For an explicitly calibrated custom
model, set `BENCH_MANIFEST_MODEL_ID`; for a one-off external calibration, set
`OMNIKV_FULL_ATTENTION_LAYERS`. A single-layer OmniKV configuration is rejected
unless the standalone Python runner's explicit ablation flag is used.

`MODEL_PATH` can override the model ID associated with an alias. The probe
wrapper currently fixes TP from the alias and uses TP=2 for an arbitrary model
path. Use the standalone CLI for explicit TP sweeps.

## Tensor-Parallel Sweeps

Run `benchmark/efficiency/bench_probe.py` once per system and TP setting. Keep
the seed, prompt/output lengths, concurrency, scheduler token budget, jitter,
warmups, and iterations identical across systems. Use a unique output directory
for every run.

TP=1 Sparse-vLLM H2O example:

```bash
cd "<SPARSE_VLLM_REPO>"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$PWD/src" python3 \
  benchmark/efficiency/bench_probe.py \
  --engine sparsevllm \
  --sparse-method h2o \
  --model-path "<MODEL_PATH>" \
  --prompt-lens 8192,16384 \
  --output-lens 512 \
  --batch-sizes 1,4,8 \
  --scenario all \
  --seed 42 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 8192 \
  --monitor-gpus 0 \
  --output-dir "<OUTPUT_ROOT>/tp1-svllm-h2o"
```

TP=2 vLLM baseline example:

```bash
cd "<SPARSE_VLLM_REPO>"

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD:$PWD/src" python3 \
  benchmark/efficiency/bench_probe.py \
  --engine vllm \
  --sparse-method vanilla \
  --model-path "<MODEL_PATH>" \
  --prompt-lens 8192,16384 \
  --output-lens 512 \
  --batch-sizes 1,4,8 \
  --scenario all \
  --seed 42 \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 8192 \
  --monitor-gpus 0,1 \
  --output-dir "<OUTPUT_ROOT>/tp2-vllm-vanilla"
```

`--monitor-gpus` names physical GPU IDs. When using nonzero physical devices,
keep those IDs aligned with `CUDA_VISIBLE_DEVICES`; for example, use both
`CUDA_VISIBLE_DEVICES=6,7` and `--monitor-gpus 6,7`.

Run `python3 benchmark/efficiency/bench_probe.py --help` for the complete CLI.

## Unified Synthetic and LongBench Suite

`run_unified_efficiency_suite.sh` runs the matched synthetic suite and then a
matched LongBench lifecycle workload. It validates stage status, synthetic
trace identity, prefix-cache policy, per-request statuses, hardware samples,
LongBench sample counts, and source-ID coverage across systems.

LongBench data defaults to `data/LongBench`. Set
`SPARSEVLLM_LONGBENCH_DATA_DIR` to use another dataset root.

```bash
cd "<SPARSE_VLLM_REPO>"

SPARSEVLLM_LONGBENCH_DATA_DIR="<LONGBENCH_ROOT>" \
LONGBENCH_SAMPLES=10 \
PROMPT_LENS="8192,16384,32768" \
OUTPUT_LENS=512 \
BATCH_SIZES="1,4,8" \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_unified_efficiency_suite.sh \
  "0,1" \
  "svllm-h2o,svllm-snapkv,svllm-vanilla,vllm-vanilla" \
  "qwen3_30b"
```

Unlike `run_efficiency_probe.sh`, the unified runner's positional arguments are
`GPUS`, `SYSTEMS`, and `MODEL_NAME`, in that order. The final validator writes
`suite_status.json` at the timestamped run root and returns a nonzero exit code
when validation fails.

## Main Wrapper Parameters

`run_efficiency_probe.sh` accepts three positional arguments:

```text
run_efficiency_probe.sh SYSTEMS MODEL_NAME_OR_PATH PHYSICAL_GPU_IDS
```

The principal environment variables are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PYTHON_BIN` | `python3` | Python executable used by every requested system. |
| `MODEL_PATH` | Alias-specific | Override the model resolved from a known alias. |
| `SPARSEVLLM_OUTPUT_DIR` | `outputs` | Output root; the wrapper adds a timestamped run directory. |
| `PROMPT_LENS` | `8192,16384,32768` | Requested maximum prompt lengths. |
| `OUTPUT_LENS` | `512` | Requested generated-token lengths. |
| `BATCH_SIZES` | `1,4,8` | Fixed-batch size and churn maximum concurrency ladder. |
| `BENCH_SCENARIO` | `all` | `fixed`, `churn`, or `all`. |
| `BENCH_SEED` | `42` | Base seed shared across engines. |
| `PROMPT_LENGTH_JITTER` | `0.10` | Fraction below each requested prompt maximum used for variable lengths. |
| `OUTPUT_LENGTH_JITTER` | `0.25` | Fraction below the requested output length used by churn. |
| `CHURN_REQUEST_MULTIPLIER` | `4` | Churn request count is maximum concurrency multiplied by this value. |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Matched scheduler token budget for both engines. |
| `NUM_WARMUPS` | `1` | Warmup iterations per synthetic case. |
| `NUM_ITERS` | `3` | Measured iterations per synthetic case. |
| `SPARSE_PREFILL_SCORE_MODE` | `probability` | SnapKV prefill score mode: `probability` or `logits`. |
| `CUDA_HOME` | `/usr/local/cuda-13.0` | CUDA toolkit root used by the probe wrapper. |

Prefix caching is always disabled by this suite. There is no prefix-caching
benchmark mode in this entrypoint.

## Metrics and Interpretation

- Request throughput is computed over the complete measured workload. Prefill
  token throughput divides all prompt tokens by the wall-time window from
  submission through the last first token. Decode token throughput excludes
  each request's first generated token (which is produced by prefill) and divides
  the remaining generated tokens by the wall-time window from the first first
  token through the last completion. In churn workloads these phase windows can
  overlap because prefill and decode are interleaved. With `output_len=1`, the
  probe still reports TTFT and prefill throughput; decode throughput and TPOT
  are `skipped_by_policy`.
- GPU compute activity and memory I/O activity are directly sampled from
  `nvidia-smi`. They are not theoretical MFU/MBU, achieved FLOP/s, or achieved
  HBM GB/s.
- Coarse active duty is the fraction of samples above 10% GPU utilization. Its
  complement cannot attribute idle time to CPU scheduling or kernel launches.
- TTFT and TPOT are end-to-end request wall-clock metrics and include host
  scheduling, synchronization, and engine overhead.
- Churn metrics compare the oversubscribed workload with its matched fixed-batch
  setting, including throughput ratio and tail-TTFT change.

Use the same model checkpoint, benchmark trace/metric contract, engine
configuration, TP, seed, lengths, scheduler token budget, warmups, and
iteration count before treating rows as matched.

A successful vLLM sweep is an immutable baseline artifact. Reuse it for later
Sparse-vLLM candidates instead of rerunning vLLM on every code change. Create a
new baseline only when the GPU model, checkpoint, TP, request trace, scheduler
budget, graph/backend policy, or metric contract changes; record package
versions for provenance without silently overwriting an older baseline.

## Artifacts and Verification

Each standalone or per-system wrapper output contains:

| Artifact | Meaning |
| --- | --- |
| `run_manifest.json` | Command, Git state, arguments, package/GPU environment, model metadata, workload contract, and final status. |
| `run_status.json` | Terminal success or failure status. |
| `raw_samples.jsonl` | One record per measured synthetic iteration. |
| `request_samples.jsonl` | Per-request trace metadata and status. |
| `summary.json` | Aggregated rows and terminal status. |
| `comparison_report.md` | Human-readable metric table. |
| `case_hardware/*.json` | Per-case sampled GPU timeline and summary. |
| `operator_runtime_stats.json` | Sparse fixed-batch Provider bindings, rejection reasons, and observed runtime kernel paths. |

Do not report a run as complete from terminal output alone. Check at least:

```bash
jq -e '.status == "success"' "<RUN_DIR>/run_status.json"
jq -e '.status == "success"' "<RUN_DIR>/summary.json"
jq -e '.status == "success"' "<RUN_DIR>/run_manifest.json"
test -s "<RUN_DIR>/raw_samples.jsonl"
test -s "<RUN_DIR>/request_samples.jsonl"
```

For a unified run, also require:

```bash
jq -e '.status == "success"' "<UNIFIED_RUN_ROOT>/suite_status.json"
```

The probe refuses to write into a directory that already contains benchmark
artifacts. Choose a new output directory instead of mixing or overwriting runs.

## Nsight Diagnostic

Use the profiling wrapper only after the standard sweep identifies a suspicious
case. It currently supports `svllm-vanilla`, `svllm-snapkv`, and
`vllm-vanilla`, and uses the probe CLI's TP=2 default.

```bash
cd "<SPARSE_VLLM_REPO>"

PROMPT_LEN=16384 \
OUTPUT_LEN=512 \
CONCURRENCY=8 \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_efficiency_profile.sh \
  svllm-vanilla \
  "<MODEL_PATH>" \
  "0,1"
```

This command requires Nsight Systems (`nsys`) and permission to access NVIDIA
performance counters. It fails instead of substituting estimated hardware
metrics when counters are unavailable. The main output is `timeline.nsys-rep`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Wrapper reports a busy GPU | Wait for the listed PID to finish or select different idle physical GPU IDs. |
| `ModuleNotFoundError` for `sparsevllm` or `vllm` | Set `PYTHON_BIN` to an environment that can import every requested engine; use `PYTHONPATH="$PWD:$PWD/src"` for a source checkout. |
| Model configuration cannot be loaded | Verify `<MODEL_PATH>/config.json` or access to the Hugging Face model ID. |
| Out of memory at long context or high concurrency | Reduce `BATCH_SIZES` or `PROMPT_LENS`; record the changed matrix instead of silently dropping failed rows. |
| Output-directory collision | Select a new `--output-dir` or output root; do not append to an existing run. |
| Hardware metric status is `metric_failed` | Inspect the per-case hardware JSON for missing or failed `nvidia-smi` samples. |
| Nsight reports insufficient privilege | Enable NVIDIA performance counters on the host or skip the diagnostic; do not reinterpret coarse activity as hardware-counter data. |
