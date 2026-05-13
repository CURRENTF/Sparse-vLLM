#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/Sparse-vLLM

export DELTAKV_OUTPUT_DIR=/home/haojitai/outputs
export DELTAKV_LONGBENCH_DATA_DIR=/home/haojitai/datasets/LongBench
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/haojitai/projects/Sparse-vLLM/src:${PYTHONPATH:-}

exec /home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/long_bench/pred.py \
  --task multi_news,passage_count,passage_retrieval_en,lcc,repobench-p \
  --ws 2 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltasnapkv \
  --model llama31-8b-hf-deltasnapkv-longbench-b0p175-w16 \
  --model_path /home/haojitai/models/Llama-3.1-8B-Instruct \
  --deltakv_checkpoint_path /data2/haojitai/checkpoints/compressor/Llama-3.1-8B-Instruct-Compressor \
  --hyper_param '{"deltasnapkv_total_budget":0.175,"hf_prefill_chunk_size":4096,"snapkv_window_size":16,"full_attention_layers":""}' \
  --temperature 0 \
  --top_p 1 \
  --top_k 0
