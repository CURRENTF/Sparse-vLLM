#!/usr/bin/env bash
set -euo pipefail

cd /home/haojitai/projects/Sparse-vLLM-svllm-cleanup

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7} \
SPARSEVLLM_MASTER_PORT=${SPARSEVLLM_MASTER_PORT:-26342} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DELTAKV_OUTPUT_DIR=/data2/haojitai/outputs \
DELTAKV_LONGBENCH_DATA_DIR=/data2/haojitai/datasets/LongBench \
PYTHONPATH=/home/haojitai/projects/Sparse-vLLM-svllm-cleanup/src:${PYTHONPATH:-} \
/home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/long_bench/pred.py \
  --task passage_retrieval_en \
  --model llama31-8b-deltakv-long-bs1full-short-batch-longbench-passage-retrieval-en \
  --model_path /data2/haojitai/models/Llama-3.1-8B-Instruct \
  --tokenizer_path /data2/haojitai/models/Llama-3.1-8B-Instruct \
  --ws 1 \
  --batch_size 4 \
  --backend sparsevllm \
  --sparse_method deltakv-triton-v4 \
  --deltakv_checkpoint_path /data2/haojitai/checkpoints/compressor/Llama-3.1-8B-Instruct-Compressor \
  --temperature 0 \
  --top_p 1 \
  --top_k 0 \
  --hyper_param '{"engine_prefill_chunk_size":8192,"prefill_schedule_policy":"long_bs1full_short_batch","mlp_seq_chunk_size":4096,"prefill_keep_tokens":4096,"decode_keep_tokens":1024,"full_attention_layers":"0,1,2,3,8,16,22","recent_keep_tokens":128,"sink_keep_tokens":8,"deltakv_center_ratio":0.1,"deltakv_latent_quant_bits":0,"deltakv_neighbor_count":4,"chunk_prefill_accel_omnikv":false,"max_num_seqs_in_batch":4,"max_decoding_seqs":4,"deltakv_full_pool_reserve_ratio":0.1}' \
  --output_root /data2/haojitai/outputs/benchmark/long_bench/prefill_schedule_policy/llama31_8b_deltakv_long_bs1full_short_batch_passage_retrieval_en
