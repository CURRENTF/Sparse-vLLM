#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${STREAMINGBENCH_DOWNLOAD_LOG:-/data2/haojitai/datasets/logs/streamingbench_full_download.nohup.log}"
PID_PATH="${STREAMINGBENCH_DOWNLOAD_PID:-/data2/haojitai/datasets/logs/streamingbench_full_download.pid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "${LOG_PATH}")"

setsid nohup bash "${SCRIPT_DIR}/download_streamingbench_full.sh" > "${LOG_PATH}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${PID_PATH}"

echo "${pid}"
echo "[info] log=${LOG_PATH}"
echo "[info] pid_file=${PID_PATH}"
