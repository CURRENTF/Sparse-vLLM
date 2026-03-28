#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/Sparse-vLLM

export DELTAKV_OUTPUT_DIR=/home/haojitai/outputs
export DELTAKV_LONGBENCH_DATA_DIR=/home/haojitai/datasets/LongBench
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/haojitai/projects/Sparse-vLLM/src:${PYTHONPATH:-}

exec /home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/long_bench/pred.py \
  --task multi_news,passage_count,passage_retrieval_en,lcc,repobench-p \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --model_cls deltasnapkv \
  --model llama31-8b-hf-deltasnapkv-longbench-b0p175-w16 \
  --model_path /home/haojitai/models/Llama-3.1-8B-Instruct \
  --compressor_path /home/haojitai/checkpoints/compressor/cluster_e2e_cs512_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_cdownmlp_swiglud3072_cuplinear_0125_051527 \
  --hyper_param '{"deltasnapkv_total_budget":0.175,"chunk_prefill_size":4096,"snapkv_window_size":16,"full_attn_layers":""}' \
  --temperature 0 \
  --top_p 1 \
  --top_k 0
