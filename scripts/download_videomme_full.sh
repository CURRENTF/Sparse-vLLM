#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEOMME_ROOT:-/data2/haojitai/datasets/Video-MME_hf}"
CACHE_DIR="${HF_CACHE_DIR:-/data2/haojitai/.cache/huggingface}"
HF_BIN="${HF_BIN:-/home/haojitai/miniconda3/envs/svllm/bin/hf}"
REPO_ID="${VIDEOMME_REPO_ID:-lmms-lab/Video-MME}"
PROXY_URL="${PROXY_URL:-http://localhost:7890}"
MAX_WORKERS="${HF_MAX_WORKERS:-1}"
DOWNLOAD_SCOPE="${VIDEOMME_DOWNLOAD_SCOPE:-full}"

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
echo "[info] download_scope=${DOWNLOAD_SCOPE}"

declare -a DOWNLOAD_FILES=(
  "README.md"
  "videomme/test-00000-of-00001.parquet"
  "subtitle.zip"
)
if [[ "${DOWNLOAD_SCOPE}" == "full" ]]; then
  for idx in $(seq -w 1 20); do
    DOWNLOAD_FILES+=("videos_chunked_${idx}.zip")
  done
elif [[ "${DOWNLOAD_SCOPE}" != "metadata" ]]; then
  echo "[error] unsupported VIDEOMME_DOWNLOAD_SCOPE=${DOWNLOAD_SCOPE}; expected metadata or full" >&2
  exit 1
fi

set +e
"${HF_BIN}" download "${REPO_ID}" \
  "${DOWNLOAD_FILES[@]}" \
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
echo "[info] unzip downloaded archives"

if [[ -f "${ROOT}/subtitle.zip" ]]; then
  unzip -n "${ROOT}/subtitle.zip" -d "${ROOT}"
fi

while IFS= read -r -d '' zip_path; do
  base="$(basename "${zip_path}" .zip)"
  dest="${ROOT}/videos/${base}"
  mkdir -p "${dest}"
  echo "[info] unzip ${zip_path} -> ${dest}"
  unzip -n "${zip_path}" -d "${dest}"
done < <(find "${ROOT}" -maxdepth 1 -type f -name 'videos_chunked_*.zip' -print0 | sort -z)

echo "[info] unzip_done=$(date -Is)"
echo "[info] zip_count=$(find "${ROOT}" -maxdepth 1 -type f -name 'videos_chunked_*.zip' | wc -l)"
echo "[info] video_count=$(find "${ROOT}/videos" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' -o -iname '*.mov' \) | wc -l)"
echo "[info] done=$(date -Is)"
