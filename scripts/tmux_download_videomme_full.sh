#!/usr/bin/env bash
set -euo pipefail

SESSION="${VIDEOMME_TMUX_SESSION:-videomme_download}"
ROOT="${VIDEOMME_ROOT:-/data2/haojitai/datasets/Video-MME_hf}"
LOG_DIR="${VIDEOMME_LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

log_path="${LOG_DIR}/download_videomme_full_tmux.log"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[error] tmux session already exists: ${SESSION}" >&2
  echo "[info] attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${project_dir}' && VIDEOMME_ROOT='${ROOT}' bash scripts/download_videomme_full.sh >>'${log_path}' 2>&1"

echo "[info] session=${SESSION}"
echo "[info] log=${log_path}"
echo "[info] attach=tmux attach -t ${SESSION}"
