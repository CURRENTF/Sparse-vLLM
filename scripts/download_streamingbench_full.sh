#!/usr/bin/env bash
set -euo pipefail

ROOT="${STREAMINGBENCH_ROOT:-/data2/haojitai/datasets/StreamingBench_hf}"
CACHE_DIR="${HF_CACHE_DIR:-/data2/haojitai/.cache/huggingface}"
HF_BIN="${HF_BIN:-/home/haojitai/miniconda3/envs/svllm/bin/hf}"
REPO_ID="${STREAMINGBENCH_REPO_ID:-mjuicem/StreamingBench}"
PROXY_URL="${PROXY_URL:-http://localhost:7890}"
MAX_WORKERS="${HF_MAX_WORKERS:-1}"

if [[ ! -x "${HF_BIN}" ]]; then
  echo "[error] hf command not found or not executable: ${HF_BIN}" >&2
  exit 1
fi

mkdir -p "${ROOT}" "${CACHE_DIR}" "${ROOT}/videos"

export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${CACHE_DIR}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"

echo "[info] start_time=$(date -Is)"
echo "[info] repo=${REPO_ID}"
echo "[info] root=${ROOT}"
echo "[info] cache=${CACHE_DIR}"
echo "[info] proxy=${PROXY_URL}"
echo "[info] max_workers=${MAX_WORKERS}"

set +e
"${HF_BIN}" download "${REPO_ID}" \
  --repo-type dataset \
  --local-dir "${ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --max-workers "${MAX_WORKERS}"
download_rc="$?"
set -e
echo "[info] hf_download_rc=${download_rc}"
if [[ "${download_rc}" -ne 0 ]]; then
  exit "${download_rc}"
fi

echo "[info] download_done=$(date -Is)"
echo "[info] unzip downloaded shards"

while IFS= read -r -d '' zip_path; do
  base="$(basename "${zip_path}" .zip)"
  dest="${ROOT}/videos/${base}"
  mkdir -p "${dest}"
  echo "[info] unzip ${zip_path} -> ${dest}"
  unzip -n "${zip_path}" -d "${dest}"
done < <(find "${ROOT}" -type f -name '*.zip' ! -path "${ROOT}/videos/*" -print0 | sort -z)

echo "[info] unzip_done=$(date -Is)"
echo "[info] zip_count=$(find "${ROOT}" -type f -name '*.zip' ! -path "${ROOT}/videos/*" | wc -l)"
echo "[info] video_count=$(find "${ROOT}/videos" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' -o -iname '*.mov' \) | wc -l)"
echo "[info] done=$(date -Is)"
