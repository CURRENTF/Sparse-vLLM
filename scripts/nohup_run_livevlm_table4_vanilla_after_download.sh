#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="${LIVEVLM_TABLE4_RUN_LOG:-/data2/haojitai/datasets/logs/livevlm_table4_7b_vanilla_after_download.nohup.log}"
PID_PATH="${LIVEVLM_TABLE4_RUN_PID:-/data2/haojitai/datasets/logs/livevlm_table4_7b_vanilla_after_download.pid}"

mkdir -p "$(dirname "${LOG_PATH}")"

setsid nohup bash "${SCRIPT_DIR}/run_livevlm_table4_vanilla_after_download.sh" > "${LOG_PATH}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${PID_PATH}"

echo "${pid}"
echo "[info] log=${LOG_PATH}"
echo "[info] pid_file=${PID_PATH}"
