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
BATCH_SIZES="${BATCH_SIZES:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
NUM_ITERS="${NUM_ITERS:-3}"

# 1. Environment & Path Configurations
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_OUT="${SPARSEVLLM_OUTPUT_DIR:-outputs}/efficiency_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BASE_OUT}"

case "${MODEL_NAME}" in
  qwen3_30b)
    MODEL_PATH="${MODEL_PATH:-/data2/haojitai/models/Qwen3-30B-A3B-Instruct-2507}"
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
  ACTIVE_APPS=$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [ -n "${ACTIVE_APPS}" ]; then
    echo "ERROR: GPU ${GPU_ID} is currently busy! Active PIDs: ${ACTIVE_APPS}" >&2
    echo "Task running rule: Wait for idle device or select an idle GPU." >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="/home/haojitai/miniconda3/envs/svllm/bin:${CUDA_HOME}/bin:/data2/haojitai/conda_envs/sparse-vllm-glm47-torch211/bin:${PATH}"
export TOKENIZERS_PARALLELISM="false"

echo "================================================================================="
echo "Starting Standardized Efficiency Benchmark Probe"
echo "================================================================================="
echo "Model:       ${MODEL_PATH} (TP=${TP_SIZE})"
echo "GPUs:        ${GPUS}"
echo "Systems:     ${SYSTEMS}"
echo "Prompt Lens: ${PROMPT_LENS}"
echo "Output Lens: ${OUTPUT_LENS}"
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

  # 1. Start Hardware Monitor
  HW_LOG="${SYS_OUT}/gpu_timeline.json"
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/hardware_monitor.py" \
    --gpus "${GPUS}" \
    --interval_ms 200 \
    --output_file "${HW_LOG}" &
  MON_PID=$!
  echo "[Monitor] High-resolution GPU Hardware Monitor started (PID: ${MON_PID})."

  # 2. Determine Engine and Sparse Method
  case "${SYS_TRIM}" in
    svllm-vanilla)
      ENGINE="sparsevllm"
      SPARSE_METHOD="vanilla"
      ;;
    svllm-snapkv)
      ENGINE="sparsevllm"
      SPARSE_METHOD="snapkv"
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

  # 3. Execute Probe
  "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/efficiency/bench_probe.py" \
    --engine "${ENGINE}" \
    --model-path "${MODEL_PATH}" \
    --sparse-method "${SPARSE_METHOD}" \
    --prompt-lens "${PROMPT_LENS}" \
    --output-lens "${OUTPUT_LENS}" \
    --batch-sizes "${BATCH_SIZES}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization 0.85 \
    --num-warmups "${NUM_WARMUPS}" \
    --num-iters "${NUM_ITERS}" \
    --output-dir "${SYS_OUT}"

  # 4. Stop Hardware Monitor
  if kill -0 "${MON_PID}" 2>/dev/null; then
    kill -INT "${MON_PID}" 2>/dev/null || true
    wait "${MON_PID}" 2>/dev/null || true
  fi
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
    hw_file = base_out / s / "gpu_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            data = json.load(f)
            for r in data.get("summary", []):
                rows.append(r)

print("\n" + "=" * 120)
print(f"MASTER STANDARDIZED EFFICIENCY COMPARISON REPORT ({base_out.name})")
print("=" * 120)
print(f"{'System':<16} | {'Prompt':<8} | {'Out':<5} | {'TTFT (ms)':<11} | {'TPOT (ms)':<10} | {'Prefill MFU':<12} | {'Decode MBU':<12} | {'Peak VRAM':<10}")
print("-" * 120)
for r in rows:
    label = f"{r.get('engine')}-{r.get('sparse_method')}"
    print(f"{label:<16} | {r.get('prompt_len'):<8} | {r.get('output_len'):<5} | {r.get('ttft_ms_mean', 0.0):<11.2f} | {r.get('tpot_ms_mean', 0.0):<10.2f} | {r.get('prefill_mfu_pct_mean', 0.0):<11.1f}% | {r.get('decode_mbu_pct_mean', 0.0):<11.1f}% | {r.get('peak_vram_gb_max', 0.0):<8.2f}GB")
print("=" * 120 + "\n")
PY

echo "Master Report and individual logs saved under: ${BASE_OUT}"
