#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/Sparse-vLLM

export CUDA_VISIBLE_DEVICES=4,5,6,7
export DELTAKV_OUTPUT_DIR=/home/haojitai/outputs
export DELTAKV_LONGBENCH_DATA_DIR=/home/haojitai/datasets/LongBench
export PYTHONPATH=/home/haojitai/projects/Sparse-vLLM/src:${PYTHONPATH:-}

MODEL_PATH="/home/haojitai/models/Llama-3.1-8B-Instruct"
COMPRESSOR_PATH="/home/haojitai/checkpoints/compressor/cluster_e2e_cs512_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_cdownmlp_swiglud3072_cuplinear_0125_051527"
ALPHAS=("0.001" "0.02" "0.05" "0.1")
SCBENCH_TASKS="scbench_kv,scbench_qa_eng,scbench_summary_with_needles,scbench_many_shot"

for alpha in "${ALPHAS[@]}"; do
  alpha_label="${alpha//./p}"
  hyper_param=$(cat <<JSON
{"chunk_prefill_size":32768,"num_top_tokens_in_prefill":0.17,"chunk_prefill_accel_omnikv":false,"deltakv_use_omnikv_selection":true,"num_top_tokens":0.17,"full_attn_layers":"0,1,2,8,18","num_recent_tokens":128,"num_sink_tokens":8,"use_compression":true,"use_cluster":true,"cluster_ratio":0.1,"stride_alpha":${alpha},"kv_quant_bits":0}
JSON
)

  echo "[$(date '+%F %T')] alpha=${alpha} longbench start"
  /home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/long_bench/pred.py \
    --model "llama31-8b-hf-deltakv-longbench-b0p17-alpha${alpha_label}" \
    --model_path "${MODEL_PATH}" \
    --compressor_path "${COMPRESSOR_PATH}" \
    --ws 4 \
    --batch_size 1 \
    --backend hf \
    --model_cls deltakv \
    --temperature 0 \
    --top_p 1 \
    --top_k 0 \
    --hyper_param "${hyper_param}"
  echo "[$(date '+%F %T')] alpha=${alpha} longbench done"

  echo "[$(date '+%F %T')] alpha=${alpha} scbench start"
  /home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/scbench/run_scbench.py \
    --task "${SCBENCH_TASKS}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "/home/haojitai/outputs/benchmark/scbench_alpha_llama/llama31-8b-scbench-merged-b0p17-alpha${alpha_label}" \
    --attn_type deltakv \
    --kv_type dense \
    --use_chat_template \
    --trust_remote_code \
    --max_seq_length 131072 \
    --ws 4 \
    --hyper_param "{\"model_cls\":\"deltakv\",\"compressor_path\":\"${COMPRESSOR_PATH}\",\"chunk_prefill_size\":32768,\"num_top_tokens_in_prefill\":0.17,\"chunk_prefill_accel_omnikv\":false,\"deltakv_use_omnikv_selection\":true,\"num_top_tokens\":0.17,\"full_attn_layers\":\"0,1,2,8,18\",\"num_recent_tokens\":128,\"num_sink_tokens\":8,\"use_compression\":true,\"use_cluster\":true,\"cluster_ratio\":0.1,\"stride_alpha\":${alpha},\"kv_quant_bits\":0}"
  echo "[$(date '+%F %T')] alpha=${alpha} scbench done"
done
