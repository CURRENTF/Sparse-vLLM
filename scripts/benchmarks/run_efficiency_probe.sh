#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Standardized Cross-System Efficiency & Hardware Utilization Probe Runner
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SYSTEMS="${1:-svllm-vanilla,svllm-snapkv,vllm-vanilla}" # comma-separated
MODEL_NAME="${2:-qwen3_30b}" # qwen3_30b | qwen3_8b | qwen25_7b | custom path
GPUS="${3:-0,1}"
PROMPT_LENS="${PROMPT_LENS:-8192,16384,32768}"
OUTPUT_LENS="${OUTPUT_LENS:-512}"
BATCH_SIZES="${BATCH_SIZES:-1,4,8}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
BENCH_SCENARIO="${BENCH_SCENARIO:-all}"
BENCH_SEED="${BENCH_SEED:-42}"
PROMPT_LENGTH_JITTER="${PROMPT_LENGTH_JITTER:-0.10}"
OUTPUT_LENGTH_JITTER="${OUTPUT_LENGTH_JITTER:-0.25}"
CHURN_REQUEST_MULTIPLIER="${CHURN_REQUEST_MULTIPLIER:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
SPARSE_PREFILL_SCORE_MODE="${SPARSE_PREFILL_SCORE_MODE:-probability}"
case "${SPARSE_PREFILL_SCORE_MODE}" in
  probability|tilelang_raw_qk) ;;
  *)
    echo "ERROR: SPARSE_PREFILL_SCORE_MODE must be probability or tilelang_raw_qk." >&2
    exit 2
    ;;
esac

# 1. Environment & Path Configurations
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_OUT="${SPARSEVLLM_OUTPUT_DIR:-outputs}/efficiency_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BASE_OUT}"

case "${MODEL_NAME}" in
  qwen3_30b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
    TP_SIZE=2
    ;;
  qwen3_8b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
    TP_SIZE=1
    ;;
  qwen25_7b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct-1M}"
    TP_SIZE=1
    ;;
  *)
    MODEL_PATH="${MODEL_NAME}"
    TP_SIZE=2
    ;;
esac

# 2. Idle GPU Preflight Check
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
for GPU_ID in "${GPU_ARR[@]}"; do
  if ! ACTIVE_APPS=$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits); then
    echo "ERROR: failed to query GPU ${GPU_ID}." >&2
    exit 2
  fi
  ACTIVE_APPS=$(printf '%s\n' "${ACTIVE_APPS}" | sed '/^[[:space:]]*$/d')
  if [ -n "${ACTIVE_APPS}" ]; then
    echo "ERROR: GPU ${GPU_ID} is currently busy! Active PIDs: ${ACTIVE_APPS}" >&2
    echo "Task running rule: Wait for idle device or select an idle GPU." >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TOKENIZERS_PARALLELISM="false"

echo "================================================================================="
echo "Starting Standardized Efficiency Benchmark Probe"
echo "================================================================================="
echo "Model:       ${MODEL_PATH} (TP=${TP_SIZE})"
echo "GPUs:        ${GPUS}"
echo "Systems:     ${SYSTEMS}"
echo "Prompt Lens: ${PROMPT_LENS}"
echo "Output Lens: ${OUTPUT_LENS}"
echo "Concurrency: ${BATCH_SIZES}"
echo "Scenarios:   ${BENCH_SCENARIO}"
echo "Output Dir:  ${BASE_OUT}"
echo "================================================================================="

IFS=',' read -ra SYS_ARR <<< "${SYSTEMS}"
for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(echo "${SYS}" | xargs)
  SYS_OUT="${BASE_OUT}/${SYS_TRIM}"
  mkdir -p "${SYS_OUT}"

  echo ""
  echo "---------------------------------------------------------------------------------"
  echo "Running System: [${SYS_TRIM}]"
  echo "---------------------------------------------------------------------------------"

  # 1. Determine Engine and Sparse Method
  case "${SYS_TRIM}" in
    svllm-vanilla)
      ENGINE="sparsevllm"
      SPARSE_METHOD="vanilla"
      ;;
    svllm-snapkv)
      ENGINE="sparsevllm"
      SPARSE_METHOD="snapkv"
      ;;
    svllm-h2o)
      ENGINE="sparsevllm"
      SPARSE_METHOD="h2o"
      ;;
    svllm-omnikv)
      ENGINE="sparsevllm"
      SPARSE_METHOD="omnikv"
      ;;
    svllm-deltakv)
      ENGINE="sparsevllm"
      SPARSE_METHOD="deltakv"
      ;;
    vllm-vanilla|vllm)
      ENGINE="vllm"
      SPARSE_METHOD="vanilla"
      ;;
    *)
      ENGINE="sparsevllm"
      SPARSE_METHOD="${SYS_TRIM}"
      ;;
  esac

  # 2. Execute Probe. The probe samples hardware only around measured cases.
  PROBE_ARGS=(
    --engine "${ENGINE}"
    --model-path "${MODEL_PATH}"
    --sparse-method "${SPARSE_METHOD}"
    --prompt-lens "${PROMPT_LENS}"
    --output-lens "${OUTPUT_LENS}"
    --batch-sizes "${BATCH_SIZES}"
    --scenario "${BENCH_SCENARIO}"
    --seed "${BENCH_SEED}"
    --prompt-length-jitter "${PROMPT_LENGTH_JITTER}"
    --output-length-jitter "${OUTPUT_LENGTH_JITTER}"
    --churn-request-multiplier "${CHURN_REQUEST_MULTIPLIER}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization 0.85
    --num-warmups "${NUM_WARMUPS}"
    --num-iters "${NUM_ITERS}"
    --monitor-gpus "${GPUS}"
    --output-dir "${SYS_OUT}"
  )
  if [ "${SPARSE_METHOD}" = "snapkv" ]; then
    PROBE_ARGS+=(--sparse-prefill-score-mode "${SPARSE_PREFILL_SCORE_MODE}")
  fi
  "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/efficiency/bench_probe.py" "${PROBE_ARGS[@]}"

  echo "[${SYS_TRIM}] Run completed."
done

# 3. Master Aggregator
"${PYTHON_BIN}" - <<PY
import json
import os
from pathlib import Path

base_out = Path("${BASE_OUT}")
sys_names = [s.strip() for s in "${SYSTEMS}".split(",") if s.strip()]

rows = []
for s in sys_names:
    summary_file = base_out / s / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            data = json.load(f)
            for r in data.get("summary", []):
                rows.append(r)

def fmt(value, precision):
    return "n/a" if value is None else f"{float(value):.{precision}f}"

print("\n" + "=" * 120)
print(f"MASTER STANDARDIZED EFFICIENCY COMPARISON REPORT ({base_out.name})")
print("=" * 120)
print(f"{'System':<16} | {'Scenario':<22} | {'Prompt':<12} | {'Out':<9} | {'Conc':<5} | {'Req/s':<9} | {'Out tok/s':<12} | {'Peak %':<7} | {'Scale %':<7} | {'TTFT p99':<10} | {'GPU act':<8} | {'Mem I/O':<8} | {'Peak VRAM':<10}")
print("-" * 120)
for r in rows:
    label = f"{r.get('engine')}-{r.get('sparse_method')}"
    prompt_range = f"{r.get('prompt_len_min')}-{r.get('prompt_len_max')}"
    output_range = f"{r.get('output_len_min')}-{r.get('output_len_max')}"
    print(f"{label:<16} | {r.get('scenario')!s:<22} | {prompt_range:<12} | {output_range:<9} | {r.get('concurrency')!s:<5} | {fmt(r.get('request_throughput_rps'), 2):<9} | {fmt(r.get('output_token_throughput_tps'), 1):<12} | {fmt(r.get('output_tps_pct_of_observed_sweep_peak'), 1):<7} | {fmt(r.get('output_tps_scaling_efficiency_pct_vs_min_concurrency'), 1):<7} | {fmt(r.get('ttft_ms_p99'), 1):<10} | {fmt(r.get('gpu_compute_activity_pct_mean'), 1):<7}% | {fmt(r.get('gpu_memory_io_activity_pct_mean'), 1):<7}% | {fmt(r.get('peak_vram_gb_max'), 2):<8}GB")
print("=" * 120 + "\n")
PY

echo "Master Report and individual logs saved under: ${BASE_OUT}"
