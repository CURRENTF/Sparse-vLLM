#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Unified End-to-End Efficiency, Sequence Lifecycle & Hardware Profiling Suite
# Robust orchestration without abrupt 'set -e' failure.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${1:-6,7}"
SYSTEMS="${2:-svllm-vanilla,svllm-snapkv,vllm-vanilla}"
MODEL_NAME="${3:-qwen3_30b}"
PROMPT_LENS="${PROMPT_LENS:-8192,16384,32768}"
OUTPUT_LENS="${OUTPUT_LENS:-512}"
LONGBENCH_SAMPLES="${LONGBENCH_SAMPLES:-10}" # 10 samples/task for fast dynamic sequence churn analysis

# 1. Environment & Paths
PYTHON_BIN="${PYTHON_BIN:-/data2/haojitai/conda_envs/sparse-vllm-glm47-torch211/bin/python}"
if [ ! -f "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
TMPDIR="${TMPDIR:-/data2/haojitai/tmp}"
DATA_ROOT="${SPARSEVLLM_LONGBENCH_DATA_DIR:-${SPARSEVLLM_DATA_DIR:-/data2/haojitai/datasets/LongBench}}"
BASE_OUT="${SPARSEVLLM_OUTPUT_DIR:-outputs}/unified_efficiency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BASE_OUT}" "${TMPDIR}"

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

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME}"
export TMPDIR="${TMPDIR}"
export PATH="/home/haojitai/miniconda3/envs/svllm/bin:${CUDA_HOME}/bin:/data2/haojitai/conda_envs/sparse-vllm-glm47-torch211/bin:${PATH}"
export TOKENIZERS_PARALLELISM="false"
export SPARSEVLLM_DATA_DIR="${DATA_ROOT}"
export SPARSEVLLM_LONGBENCH_DATA_DIR="${DATA_ROOT}"

echo "=========================================================================================="
echo "UNIFIED EFFICIENCY & SEQUENCE LIFECYCLE BENCHMARK"
echo "=========================================================================================="
echo "Model:       ${MODEL_PATH} (TP=${TP_SIZE})"
echo "GPUs:        ${GPUS}"
echo "Systems:     ${SYSTEMS}"
echo "Prompt Lens: ${PROMPT_LENS} | Output Len: ${OUTPUT_LENS}"
echo "Output Root: ${BASE_OUT}"
echo "=========================================================================================="

# ========================================================================================
# 1. SCENARIO A: SYNTHETIC LENGTH SWEED (STATIC UNIFORM BATCH)
# ========================================================================================
SCENARIO_A_DIR="${BASE_OUT}/scenario_a_synthetic"
mkdir -p "${SCENARIO_A_DIR}"
echo ""
echo ">>> [STAGE 1/2] Starting Scenario A (Synthetic Length Sweep: 8k/16k/32k -> ${OUTPUT_LENS}) <<<"

IFS=',' read -ra SYS_ARR <<< "${SYSTEMS}"
for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(echo "${SYS}" | xargs)
  SYS_OUT="${SCENARIO_A_DIR}/${SYS_TRIM}"
  mkdir -p "${SYS_OUT}"
  
  echo "--- Running Scenario A on System: [${SYS_TRIM}] ---"
  
  # Start Hardware Monitor
  HW_LOG="${SYS_OUT}/gpu_timeline.json"
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/hardware_monitor.py" \
    --gpus "${GPUS}" \
    --interval_ms 200 \
    --output_file "${HW_LOG}" >/dev/null 2>&1 &
  MON_PID=$!

  case "${SYS_TRIM}" in
    svllm-vanilla)
      ENGINE="sparsevllm"
      METHOD="vanilla"
      ;;
    svllm-snapkv)
      ENGINE="sparsevllm"
      METHOD="snapkv"
      ;;
    vllm-vanilla|vllm)
      ENGINE="vllm"
      METHOD="vanilla"
      ;;
    *)
      ENGINE="sparsevllm"
      METHOD="${SYS_TRIM}"
      ;;
  esac

  # Run probe without set -e
  "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/efficiency/bench_probe.py" \
    --engine "${ENGINE}" \
    --model-path "${MODEL_PATH}" \
    --sparse-method "${METHOD}" \
    --prompt-lens "${PROMPT_LENS}" \
    --output-lens "${OUTPUT_LENS}" \
    --batch-sizes 1 \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization 0.85 \
    --num-warmups 1 \
    --num-iters 3 \
    --output-dir "${SYS_OUT}" || echo "[WARNING] Scenario A failed on ${SYS_TRIM}, continuing..."

  # Stop monitor
  if kill -0 "${MON_PID}" 2>/dev/null; then
    kill -INT "${MON_PID}" 2>/dev/null || true
    wait "${MON_PID}" 2>/dev/null || true
  fi
done

# ========================================================================================
# 2. SCENARIO B: LONGBENCH DYNAMIC SEQUENCE CHURN & CONTINUOUS BATCHING
# ========================================================================================
SCENARIO_B_DIR="${BASE_OUT}/scenario_b_longbench"
mkdir -p "${SCENARIO_B_DIR}"
echo ""
echo ">>> [STAGE 2/2] Starting Scenario B (LongBench Dynamic Sequence Lifecycle & Churn) <<<"

TASKS="qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc"
for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(echo "${SYS}" | xargs)
  SYS_OUT="${SCENARIO_B_DIR}/${SYS_TRIM}"
  mkdir -p "${SYS_OUT}"
  
  echo "--- Running Scenario B on System: [${SYS_TRIM}] ---"
  HW_LOG="${SYS_OUT}/gpu_timeline.json"
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/hardware_monitor.py" \
    --gpus "${GPUS}" \
    --interval_ms 200 \
    --output_file "${HW_LOG}" >/dev/null 2>&1 &
  MON_PID=$!

  if [ "${SYS_TRIM}" = "vllm-vanilla" ] || [ "${SYS_TRIM}" = "vllm" ]; then
    "${PYTHON_BIN}" -u benchmark/long_bench/pred_vllm.py \
      --model_path "${MODEL_PATH}" \
      --task "${TASKS}" \
      --output_root "${SYS_OUT}" \
      --tensor_parallel_size "${TP_SIZE}" \
      --gpu_memory_utilization 0.85 \
      --max_model_len 32768 \
      --samples_per_task "${LONGBENCH_SAMPLES}" \
      --min_required_samples "${LONGBENCH_SAMPLES}" \
      --temperature 0.0 \
      --top_p 1.0 \
      --top_k 1 \
      --batch_size 16 || echo "[WARNING] LongBench vLLM failed on ${SYS_TRIM}, continuing..."
  else
    METHOD="vanilla"
    if [ "${SYS_TRIM}" = "svllm-snapkv" ]; then
      METHOD="snapkv"
      HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85, \"snapkv_window_size\": 64, \"sparse_prefill_score_mode\": \"tilelang_raw_qk\", \"sink_keep_tokens\": 64, \"decode_keep_tokens\": 2048, \"recent_keep_tokens\": 64, \"pool_kernel_size\": 7}"
    else
      HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85}"
    fi

    "${PYTHON_BIN}" -u benchmark/long_bench/pred.py \
      --model_path "${MODEL_PATH}" \
      --task "${TASKS}" \
      --output_root "${SYS_OUT}" \
      --sparse_method "${METHOD}" \
      --hyper_param "${HPARAMS}" \
      --max_model_len 32768 \
      --samples_per_task "${LONGBENCH_SAMPLES}" \
      --min_required_samples "${LONGBENCH_SAMPLES}" \
      --temperature 0.0 \
      --top_p 1.0 \
      --top_k 1 \
      --batch_size -1 || echo "[WARNING] LongBench svLLM failed on ${SYS_TRIM}, continuing..."
  fi

  if kill -0 "${MON_PID}" 2>/dev/null; then
    kill -INT "${MON_PID}" 2>/dev/null || true
    wait "${MON_PID}" 2>/dev/null || true
  fi
done

# ========================================================================================
# 3. MASTER COMPARISON & SEQUENCE LIFECYCLE BREAKDOWN REPORT
# ========================================================================================
"${PYTHON_BIN}" - <<PY
import json
import os
from pathlib import Path

root = Path("${BASE_OUT}")
scen_a = root / "scenario_a_synthetic"
scen_b = root / "scenario_b_longbench"

print("\n" + "=" * 120)
print(f"MASTER UNIFIED EFFICIENCY & SEQUENCE LIFECYCLE REPORT ({root.name})")
print("=" * 120)

print("\n[1] SCENARIO A: SYNTHETIC LENGTH LADDER (Static Uniform Batches)")
print("-" * 120)
print(f"{'System':<16} | {'Prompt':<8} | {'Out':<5} | {'TTFT (ms)':<11} | {'TPOT (ms)':<10} | {'Prefill MFU':<12} | {'Decode MBU':<12} | {'Peak VRAM':<10}")
print("-" * 120)
if scen_a.exists():
    for sys_dir in sorted(scen_a.iterdir()):
        sum_f = sys_dir / "summary.json"
        if sum_f.exists():
            try:
                with open(sum_f) as f:
                    d = json.load(f)
                    for r in d.get("summary", []):
                        label = f"{r.get('engine')}-{r.get('sparse_method')}"
                        print(f"{label:<16} | {r.get('prompt_len'):<8} | {r.get('output_len'):<5} | {r.get('ttft_ms_mean', 0.0):<11.2f} | {r.get('tpot_ms_mean', 0.0):<10.2f} | {r.get('prefill_mfu_pct_mean', 0.0):<11.1f}% | {r.get('decode_mbu_pct_mean', 0.0):<11.1f}% | {r.get('peak_vram_gb_max', 0.0):<8.2f}GB")
            except Exception as e:
                pass

print("\n[2] SCENARIO B: LONGBENCH DYNAMIC SEQUENCE LIFECYCLE & HARDWARE TIMELINE")
print("-" * 120)
print(f"{'System':<16} | {'Duration (s)':<13} | {'Avg Compute %':<14} | {'Active Duty %':<14} | {'Host Bubble %':<14} | {'Avg Power (W)':<14}")
print("-" * 120)
if scen_b.exists():
    for sys_dir in sorted(scen_b.iterdir()):
        hw_f = sys_dir / "gpu_summary.json"
        if not hw_f.exists():
            hw_f = sys_dir / "gpu_timeline_summary.json"
        if hw_f.exists():
            try:
                with open(hw_f) as f:
                    d = json.load(f)
                    agg = d.get("aggregate", {})
                    dur = d.get("duration_seconds", 0.0)
                    c_util = agg.get("mean_compute_util_pct", 0.0)
                    duty = agg.get("mean_active_duty_cycle_pct", 0.0)
                    bubble = agg.get("mean_host_launch_bubble_pct", 0.0)
                    pwr = agg.get("avg_total_power_w", 0.0)
                    print(f"{sys_dir.name:<16} | {dur:<13.1f} | {c_util:<14.1f}% | {duty:<14.1f}% | {bubble:<14.1f}% | {pwr:<14.1f}W")
            except Exception:
                pass
print("=" * 120 + "\n")
PY

echo "Unified report saved under: ${BASE_OUT}"
