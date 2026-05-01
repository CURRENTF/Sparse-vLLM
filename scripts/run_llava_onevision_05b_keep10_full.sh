#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/Sparse-vLLM

export PYTHONPATH=/home/haojitai/projects/Sparse-vLLM/src:${PYTHONPATH:-}

/home/haojitai/miniconda3/envs/svllm/bin/python -u scripts/bench_llava_onevision_visual_prune.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-0.5b-ov-hf \
  --compressor_path none \
  --dataset_dir /data2/haojitai/datasets/llava_onevision_visual_uniform_keep10_full \
  --source_vqa_dir /data2/haojitai/datasets/VQAv2 \
  --num_samples -1 \
  --max_new_tokens 8 \
  --cuda_device 7 \
  --methods vanilla,visual_uniform_keep \
  --visual_keep_ratio 0.1 \
  --full_attn_layers "" \
  --attn_implementation flash_attention_2 \
  --log_every 500
