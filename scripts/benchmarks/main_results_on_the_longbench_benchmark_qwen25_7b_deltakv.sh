#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ROOT="${DELTAKV_MODEL_ROOT:-${REPO_ROOT}/models}"
COMPRESSOR_ROOT="${DELTAKV_COMPRESSOR_ROOT:-${REPO_ROOT}/checkpoints/compressor}"
OUTPUT_ROOT="${DELTAKV_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
LONGBENCH_ROOT="${DELTAKV_LONGBENCH_DATA_DIR:-${DELTAKV_DATA_DIR:-${REPO_ROOT}/data}/LongBench}"
PYTHON_BIN="${PYTHON:-python3}"

CUDA_VISIBLE_DEVICES=6 \
DELTAKV_OUTPUT_DIR="${OUTPUT_ROOT}" \
DELTAKV_LONGBENCH_DATA_DIR="${LONGBENCH_ROOT}" \
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${PYTHON_BIN}" -u benchmark/long_bench/pred.py \
  --model qwen25-7b-deltakv-main-results-longbench-cr30 \
  --model_path "${MODEL_ROOT}/Qwen2.5-7B-Instruct-1M" \
  --tokenizer_path "${MODEL_ROOT}/Qwen2.5-7B-Instruct-1M" \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "${COMPRESSOR_ROOT}/Qwen2.5-7B-Instruct-1M-Compressor" \
  --temperature 0 \
  --top_p 1 \
  --top_k 20 \
  --hyper_param '{"hf_prefill_chunk_size":32768,"prefill_keep_tokens":4096,"chunk_prefill_accel_omnikv":false,"deltakv_use_omnikv_selection":true,"decode_keep_tokens":0.11,"full_attention_layers":"0,1,2,4,7,14","recent_keep_tokens":128,"sink_keep_tokens":8,"use_compression":true,"use_cluster":true,"deltakv_center_ratio":0.1,"stride_alpha":0.0,"deltakv_latent_quant_bits":0}' \
  --output_root "${OUTPUT_ROOT}/benchmark/long_bench/main_results_on_the_longbench_benchmark/qwen25_7b_deltakv_cr30"
