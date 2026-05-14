#!/usr/bin/env bash
cd /home/haojitai/projects/Sparse-vLLM

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/haojitai/projects/Sparse-vLLM/src:${PYTHONPATH:-} \
/home/haojitai/miniconda3/envs/svllm/bin/python -u benchmark/multimodal/video_qa/streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_svllm_delta_quant_7b_official60_bs2_smoke \
  --tasks real \
  --methods svllm_deltakv_delta_quant \
  --num_samples 2 \
  --batch_size 2 \
  --streamingbench_profile official_60s \
  --max_new_tokens 8 \
  --cuda_device 0 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --recent_keep_tokens 128 \
  --sink_keep_tokens 8 \
  --decode_keep_tokens 1024 \
  --prefill_keep_tokens 4096 \
  --svllm_chunk_prefill_size 8192 \
  --svllm_max_num_batched_tokens 65536 \
  --svllm_max_num_seqs_in_batch 4 \
  --svllm_gpu_memory_utilization 0.72 \
  --log_every 1
