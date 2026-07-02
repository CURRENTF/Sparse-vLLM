#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ROOT="${DELTAKV_MODEL_ROOT:-${REPO_ROOT}/models}"
DATA_ROOT="${DELTAKV_DATA_DIR:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -u scripts/bench_llava_onevision_visual_prune.py \
  --model_path "${MODEL_ROOT}/llava-onevision-qwen2-0.5b-ov-hf" \
  --deltakv_checkpoint_path none \
  --dataset_dir "${DATA_ROOT}/llava_onevision_visual_uniform_keep10_full" \
  --source_vqa_dir "${DATA_ROOT}/VQAv2" \
  --num_samples -1 \
  --max_new_tokens 8 \
  --cuda_device 7 \
  --methods vanilla,visual_uniform_keep \
  --visual_keep_ratio 0.1 \
  --full_attention_layers "" \
  --attn_implementation flash_attention_2 \
  --log_every 500
