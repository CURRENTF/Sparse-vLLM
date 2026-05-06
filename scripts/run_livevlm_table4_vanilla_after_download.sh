#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${STREAMINGBENCH_ROOT:-/data2/haojitai/datasets/StreamingBench_hf}"
VIDEO_DIR="${STREAMINGBENCH_VIDEO_DIR:-${DATASET_ROOT}/videos}"
DOWNLOAD_LOG="${STREAMINGBENCH_DOWNLOAD_LOG:-/data2/haojitai/datasets/logs/streamingbench_full_download.nohup.log}"
DOWNLOAD_PID="${STREAMINGBENCH_DOWNLOAD_PID:-/data2/haojitai/datasets/logs/streamingbench_full_download.pid}"
OUTPUT_DIR="${LIVEVLM_TABLE4_OUTPUT_DIR:-/data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla}"
PYTHON_BIN="${PYTHON_BIN:-/home/haojitai/miniconda3/envs/svllm/bin/python}"
MODEL_PATH="${LLAVA_ONEVISION_7B_MODEL_PATH:-/data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf}"
GPU_ID="${LIVEVLM_TABLE4_GPU_ID:-6}"
POLL_SECONDS="${LIVEVLM_TABLE4_POLL_SECONDS:-60}"
DOWNLOAD_MAX_POLLS="${LIVEVLM_TABLE4_DOWNLOAD_MAX_POLLS:-1440}"
GPU_MAX_POLLS="${LIVEVLM_TABLE4_GPU_MAX_POLLS:-720}"
GPU_MEMORY_READY_MIB="${LIVEVLM_TABLE4_GPU_MEMORY_READY_MIB:-2048}"

echo "[info] start_time=$(date -Is)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] dataset_root=${DATASET_ROOT}"
echo "[info] video_dir=${VIDEO_DIR}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] model_path=${MODEL_PATH}"
echo "[info] gpu_id=${GPU_ID}"

wait_for_download() {
  if [[ ! -f "${DOWNLOAD_PID}" ]]; then
    echo "[error] missing download pid file: ${DOWNLOAD_PID}" >&2
    exit 1
  fi
  local pid
  pid="$(cat "${DOWNLOAD_PID}")"
  for ((i = 1; i <= DOWNLOAD_MAX_POLLS; i++)); do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[info] download still running pid=${pid} poll=${i}/${DOWNLOAD_MAX_POLLS} time=$(date -Is)"
      sleep "${POLL_SECONDS}"
      continue
    fi
    if ! grep -q "\\[info\\] hf_download_rc=0" "${DOWNLOAD_LOG}"; then
      echo "[error] StreamingBench download did not report hf_download_rc=0. log=${DOWNLOAD_LOG}" >&2
      tail -n 120 "${DOWNLOAD_LOG}" >&2 || true
      exit 1
    fi
    if ! grep -q "\\[info\\] done=" "${DOWNLOAD_LOG}"; then
      echo "[error] StreamingBench download did not finish unzip/verification footer. log=${DOWNLOAD_LOG}" >&2
      tail -n 120 "${DOWNLOAD_LOG}" >&2 || true
      exit 1
    fi
    echo "[info] download complete time=$(date -Is)"
    return
  done
  echo "[error] timed out waiting for StreamingBench download after ${DOWNLOAD_MAX_POLLS} polls" >&2
  exit 1
}

verify_livevlm_table4_dataset() {
  DATASET_ROOT="${DATASET_ROOT}" VIDEO_DIR="${VIDEO_DIR}" "${PYTHON_BIN}" - <<'PY'
import csv
import os
import re
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
video_dir = Path(os.environ["VIDEO_DIR"])
csv_dir = root / "StreamingBench"
sample_re = re.compile(r"sample[_ -]?(\d+)", re.I)

video_index = {}
for path in video_dir.rglob("*"):
    if path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
        continue
    if "__MACOSX" in path.parts or path.name.startswith("._"):
        continue
    match = sample_re.search(str(path))
    if match:
        video_index.setdefault(int(match.group(1)), []).append(path)

checks = {
    "real": "Real_Time_Visual_Understanding.csv",
    "omni": "Omni_Source_Understanding.csv",
    "contextual": "Contextual_Understanding.csv",
}
task_hints = {
    "real": ("real", "visual", "real-time"),
    "omni": ("omni", "emotion", "alignment", "source", "scene"),
    "contextual": ("context", "anomaly", "misleading"),
}
task_type_hints = {
    "Emotion Recognition": ("emotion",),
    "Multimodal Alignment": ("multimodal", "alignment"),
    "Scene Understanding": ("scene",),
    "Source Discrimination": ("source",),
    "Anomaly Context Understanding": ("anomaly",),
    "Misleading Context Recognition": ("misleading",),
}
total_rows = 0
missing = []
for task, filename in checks.items():
    csv_path = csv_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing StreamingBench CSV: {csv_path}")
    rows = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            question_id = row["question_id"]
            match = sample_re.search(question_id)
            if not match:
                raise ValueError(f"Cannot parse sample id from question_id={question_id!r}")
            sample_id = int(match.group(1))
            candidates = video_index.get(sample_id, [])
            hints = task_type_hints.get(row["task_type"], task_hints[task])
            if not any(any(hint in str(path).lower() for hint in hints) for path in candidates):
                missing.append({"task": task, "question_id": question_id, "sample_id": sample_id})
    total_rows += rows
    print(f"[verify] {task} rows={rows}")

if missing:
    raise FileNotFoundError(f"Missing videos for LiveVLM Table 4 rows: first={missing[:10]} total={len(missing)}")
if total_rows != 4000:
    raise RuntimeError(f"Expected 4000 LiveVLM Table 4 overall rows, got {total_rows}")
print(f"[verify] video_sample_ids={len(video_index)} video_files={sum(len(v) for v in video_index.values())}")
print("[verify] LiveVLM Table 4 dataset coverage OK")
PY
}

wait_for_gpu() {
  for ((i = 1; i <= GPU_MAX_POLLS; i++)); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
    echo "[info] gpu=${GPU_ID} memory_used_mib=${used} poll=${i}/${GPU_MAX_POLLS} time=$(date -Is)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= GPU_MEMORY_READY_MIB )); then
      return
    fi
    sleep "${POLL_SECONDS}"
  done
  echo "[error] GPU ${GPU_ID} did not become ready below ${GPU_MEMORY_READY_MIB} MiB" >&2
  exit 1
}

wait_for_download
verify_livevlm_table4_dataset
wait_for_gpu

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "[info] launching LiveVLM Table 4 vanilla baseline time=$(date -Is)"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path "${MODEL_PATH}" \
  --dataset_dir "${DATASET_ROOT}" \
  --video_dir "${VIDEO_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --methods vanilla \
  --num_samples -1 \
  --batch_size 1 \
  --streamingbench_profile livevlm_table4 \
  --torch_dtype float16 \
  --attn_implementation sdpa \
  --max_new_tokens 8 \
  --choice_parse_mode official_first_char \
  --cuda_device 0 \
  --seed 0 \
  --log_every 50

echo "[info] baseline_done=$(date -Is)"
