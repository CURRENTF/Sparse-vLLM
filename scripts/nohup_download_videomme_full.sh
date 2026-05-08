#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOMME_ROOT:-/data2/haojitai/datasets/Video-MME_hf}"
LOG_DIR="${VIDEOMME_LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${LOG_DIR}/download_videomme_full_${timestamp}.log"

nohup bash scripts/download_videomme_full.sh >"${log_path}" 2>&1 &
pid="$!"

echo "${pid}" >"${LOG_DIR}/download_videomme_full.pid"
echo "[info] pid=${pid}"
echo "[info] log=${log_path}"
