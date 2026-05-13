#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/Sparse-vLLM

CUDA_VISIBLE_DEVICES=6 \
DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs \
DELTAKV_LONGBENCH_DATA_DIR=/data2/haojitai/datasets/LongBench \
PYTHONPATH=/home/haojitai/projects/Sparse-vLLM/src:${PYTHONPATH:-} \
/home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-deltakv-main-results-longbench-cr30 \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path /data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor \
  --temperature 0 \
  --top_p 1 \
  --top_k 20 \
  --hyper_param '{"hf_prefill_chunk_size":32768,"prefill_keep_tokens":4096,"chunk_prefill_accel_omnikv":false,"deltakv_use_omnikv_selection":true,"decode_keep_tokens":0.11,"full_attention_layers":"0,1,2,4,7,14","recent_keep_tokens":128,"sink_keep_tokens":8,"use_compression":true,"use_cluster":true,"deltakv_center_ratio":0.1,"stride_alpha":0.0,"deltakv_latent_quant_bits":0}' \
  --output_root /data2/haojitai/outputs/benchmark/long_bench/main_results_on_the_longbench_benchmark/qwen25_7b_deltakv_cr30
