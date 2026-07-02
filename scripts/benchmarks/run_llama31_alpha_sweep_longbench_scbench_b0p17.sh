#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export DELTAKV_OUTPUT_DIR="${DELTAKV_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
export DELTAKV_LONGBENCH_DATA_DIR="${DELTAKV_LONGBENCH_DATA_DIR:-${DELTAKV_DATA_DIR:-${REPO_ROOT}/data}/LongBench}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL_ROOT="${DELTAKV_MODEL_ROOT:-${REPO_ROOT}/models}"
COMPRESSOR_ROOT="${DELTAKV_COMPRESSOR_ROOT:-${REPO_ROOT}/checkpoints/compressor}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Llama-3.1-8B-Instruct}"
DELTAKV_CHECKPOINT_PATH="${DELTAKV_CHECKPOINT_PATH:-${COMPRESSOR_ROOT}/Llama-3.1-8B-Instruct-Compressor}"
PYTHON_BIN="${PYTHON:-python3}"
ALPHAS=("0.001" "0.02" "0.05" "0.1")
SCBENCH_TASKS="scbench_kv,scbench_qa_eng,scbench_summary_with_needles,scbench_many_shot"

for alpha in "${ALPHAS[@]}"; do
  alpha_label="${alpha//./p}"
  hyper_param=$(cat <<JSON
{"hf_prefill_chunk_size":32768,"prefill_keep_tokens":0.17,"chunk_prefill_accel_omnikv":false,"deltakv_use_omnikv_selection":true,"decode_keep_tokens":0.17,"full_attention_layers":"0,1,2,8,18","recent_keep_tokens":128,"sink_keep_tokens":8,"use_compression":true,"use_cluster":true,"deltakv_center_ratio":0.1,"stride_alpha":${alpha},"deltakv_latent_quant_bits":0}
JSON
)

  echo "[$(date '+%F %T')] alpha=${alpha} longbench start"
  "${PYTHON_BIN}" -u benchmark/long_bench/pred.py \
    --model "llama31-8b-hf-deltakv-longbench-b0p17-alpha${alpha_label}" \
    --model_path "${MODEL_PATH}" \
    --deltakv_checkpoint_path "${DELTAKV_CHECKPOINT_PATH}" \
    --ws 4 \
    --batch_size 1 \
    --backend hf \
    --sparse_method deltakv \
    --temperature 0 \
    --top_p 1 \
    --top_k 0 \
    --hyper_param "${hyper_param}"
  echo "[$(date '+%F %T')] alpha=${alpha} longbench done"

  echo "[$(date '+%F %T')] alpha=${alpha} scbench start"
  "${PYTHON_BIN}" -u benchmark/scbench/run_scbench.py \
    --task "${SCBENCH_TASKS}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${DELTAKV_OUTPUT_DIR}/benchmark/scbench_alpha_llama/llama31-8b-scbench-merged-b0p17-alpha${alpha_label}" \
    --attn_type deltakv \
    --kv_type dense \
    --use_chat_template \
    --trust_remote_code \
    --max_seq_length 131072 \
    --ws 4 \
    --hyper_param "{\"sparse_method\":\"deltakv\",\"deltakv_checkpoint_path\":\"${DELTAKV_CHECKPOINT_PATH}\",\"hf_prefill_chunk_size\":32768,\"prefill_keep_tokens\":0.17,\"chunk_prefill_accel_omnikv\":false,\"deltakv_use_omnikv_selection\":true,\"decode_keep_tokens\":0.17,\"full_attention_layers\":\"0,1,2,8,18\",\"recent_keep_tokens\":128,\"sink_keep_tokens\":8,\"use_compression\":true,\"use_cluster\":true,\"deltakv_center_ratio\":0.1,\"stride_alpha\":${alpha},\"deltakv_latent_quant_bits\":0}"
  echo "[$(date '+%F %T')] alpha=${alpha} scbench done"
done
