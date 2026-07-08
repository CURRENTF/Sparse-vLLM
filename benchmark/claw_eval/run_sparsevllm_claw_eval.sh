#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ROOT="${DELTAKV_MODEL_ROOT:-${REPO_ROOT}/models}"
OUTPUT_BASE="${DELTAKV_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
CACHE_ROOT="${DELTAKV_CACHE_DIR:-${REPO_ROOT}/.cache}"
CONDA_ENVS_ROOT="${CONDA_ENVS_ROOT:-${HOME}/.conda/envs}"
RUN_NAME="${RUN_NAME:-claw_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}/Sparse-vLLM/claw-eval}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
TRACE_DIR="${RUN_DIR}/traces"
REQUEST_LOG_DIR="${RUN_DIR}/sparsevllm_requests"
ENGINE_KWARGS_FILE="${ENGINE_KWARGS_FILE:-${RUN_DIR}/engine_kwargs.json}"
RUN_MANIFEST="${RUN_MANIFEST:-${RUN_DIR}/run_manifest.json}"
mkdir -p "${LOG_DIR}" "${TRACE_DIR}" "${REQUEST_LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-7B-Instruct-1M}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sparsevllm-claw}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18000}"
SPARSEVLLM_OPENAI_BASE_URL="${SPARSEVLLM_OPENAI_BASE_URL:-http://${SERVER_HOST}:${SERVER_PORT}/v1}"
SPARSEVLLM_OPENAI_API_KEY="${SPARSEVLLM_OPENAI_API_KEY:-local-sparsevllm}"
SPARSEVLLM_CLAW_MODEL_ID="${SPARSEVLLM_CLAW_MODEL_ID:-${SERVED_MODEL_NAME}}"
SPARSEVLLM_CONTEXT_WINDOW="${SPARSEVLLM_CONTEXT_WINDOW:-131072}"
CLAW_EVAL_JUDGE_MODEL="${CLAW_EVAL_JUDGE_MODEL:-google/gemini-3-flash-preview}"
CLAW_EVAL_JUDGE_BASE_URL="${CLAW_EVAL_JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"

CLAW_EVAL_DIR="${CLAW_EVAL_DIR:-${REPO_ROOT}/../claw-eval}"
CLAW_EVAL_REF="${CLAW_EVAL_REF:-main}"
CLAW_EVAL_CONFIG_TEMPLATE="${CLAW_EVAL_CONFIG_TEMPLATE:-${REPO_ROOT}/benchmark/claw_eval/sparsevllm_config.yaml}"
CLAW_EVAL_CONFIG="${RUN_DIR}/sparsevllm_config.yaml"
CLAW_EVAL_ARGS="${CLAW_EVAL_ARGS:-batch --config ${CLAW_EVAL_CONFIG} --sandbox --trials 3 --parallel 1}"
SETUP_ONLY="${SETUP_ONLY:-0}"

SPARSEVLLM_CONDA_ENV="${SPARSEVLLM_CONDA_ENV:-${CONDA_ENVS_ROOT}/sparse-vllm-tf530}"
CLAW_EVAL_CONDA_ENV="${CLAW_EVAL_CONDA_ENV:-${CONDA_ENVS_ROOT}/claw-eval-py311}"
SPARSEVLLM_MASTER_PORT="${SPARSEVLLM_MASTER_PORT:-2333}"
if [[ -z "${ENGINE_KWARGS:-}" ]]; then
  ENGINE_KWARGS="{\"tensor_parallel_size\":1,\"gpu_memory_utilization\":0.88,\"max_model_len\":${SPARSEVLLM_CONTEXT_WINDOW},\"engine_prefill_chunk_size\":4096,\"sparse_method\":\"vanilla\"}"
fi
SPARSEVLLM_PYTHON_BIN="${SPARSEVLLM_PYTHON_BIN:-python}"

DOWNLOAD_HTTP_PROXY="${DOWNLOAD_HTTP_PROXY:-http://127.0.0.1:7898}"
DOWNLOAD_HTTPS_PROXY="${DOWNLOAD_HTTPS_PROXY:-${DOWNLOAD_HTTP_PROXY}}"
CLASH_DIR="${CLASH_DIR:-${HOME}/clash}"
CLASH_DOWNLOAD_CONFIG="${CLASH_DOWNLOAD_CONFIG:-${CLASH_DIR}/config.download.yaml}"
CLASH_DOWNLOAD_LOG="${LOG_DIR}/clash_download.log"

export RUN_NAME
export REPO_ROOT
export RUN_DIR
export TRACE_DIR
export REQUEST_LOG_DIR
export MODEL_PATH
export SERVED_MODEL_NAME
export CUDA_VISIBLE_DEVICES
export SPARSEVLLM_OPENAI_BASE_URL
export SPARSEVLLM_CLAW_MODEL_ID
export SPARSEVLLM_CONTEXT_WINDOW
export CLAW_EVAL_JUDGE_BASE_URL
export CLAW_EVAL_JUDGE_MODEL
export CLAW_EVAL_DIR
export CLAW_EVAL_CONFIG
export CLAW_EVAL_ARGS
export SPARSEVLLM_CONDA_ENV
export CLAW_EVAL_CONDA_ENV
export SPARSEVLLM_MASTER_PORT
export SETUP_ONLY
export DOWNLOAD_HTTP_PROXY
export ENGINE_KWARGS_FILE

require_file() {
  local path="$1"
  local what="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing ${what}: ${path}" >&2
    exit 2
  fi
}

start_download_clash_if_needed() {
  if ss -ltn 2>/dev/null | grep -qE '127\.0\.0\.1:7898|:7898 '; then
    return
  fi
  require_file "${CLASH_DIR}/clash" "Clash binary"
  require_file "${CLASH_DOWNLOAD_CONFIG}" "download-only Clash config"
  echo "[INFO] Starting download-only Clash on 127.0.0.1:7898"
  (
    cd "${CLASH_DIR}"
    nohup ./clash -d "${CLASH_DIR}" -f "${CLASH_DOWNLOAD_CONFIG}" >"${CLASH_DOWNLOAD_LOG}" 2>&1 &
  )
  for _ in $(seq 1 30); do
    if ss -ltn 2>/dev/null | grep -qE '127\.0\.0\.1:7898|:7898 '; then
      return
    fi
    sleep 1
  done
  echo "[ERROR] download-only Clash did not open port 7898. See ${CLASH_DOWNLOAD_LOG}" >&2
  exit 3
}

download_env() {
  http_proxy="${DOWNLOAD_HTTP_PROXY}" \
  https_proxy="${DOWNLOAD_HTTPS_PROXY}" \
  HTTP_PROXY="${DOWNLOAD_HTTP_PROXY}" \
  HTTPS_PROXY="${DOWNLOAD_HTTPS_PROXY}" \
  NO_PROXY="127.0.0.1,localhost,::1" \
  no_proxy="127.0.0.1,localhost,::1" \
  "$@"
}

activate_conda_env() {
  set +u
  conda activate "$1"
  set -u
}

deactivate_conda_env() {
  set +u
  conda deactivate
  set -u
}

prepare_claw_eval_repo() {
  start_download_clash_if_needed
  mkdir -p "$(dirname "${CLAW_EVAL_DIR}")"
  if [[ -d "${CLAW_EVAL_DIR}/.git" ]]; then
    echo "[INFO] Updating Claw-Eval repo at ${CLAW_EVAL_DIR}"
    download_env git -C "${CLAW_EVAL_DIR}" fetch origin "${CLAW_EVAL_REF}"
    download_env git -C "${CLAW_EVAL_DIR}" checkout "${CLAW_EVAL_REF}"
    download_env git -C "${CLAW_EVAL_DIR}" pull --ff-only
  else
    echo "[INFO] Cloning Claw-Eval repo to ${CLAW_EVAL_DIR}"
    download_env git clone https://github.com/claw-eval/claw-eval "${CLAW_EVAL_DIR}"
    download_env git -C "${CLAW_EVAL_DIR}" checkout "${CLAW_EVAL_REF}"
  fi
}

prepare_claw_eval_env() {
  start_download_clash_if_needed
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
  mkdir -p "${PIP_CACHE_DIR}"
  if [[ ! -x "${CLAW_EVAL_CONDA_ENV}/bin/python" ]]; then
    echo "[INFO] Creating Claw-Eval Python 3.11 env at ${CLAW_EVAL_CONDA_ENV}"
    download_env conda create -y -p "${CLAW_EVAL_CONDA_ENV}" python=3.11 pip
  fi

  activate_conda_env "${CLAW_EVAL_CONDA_ENV}"
  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("claw_eval") else 1)
PY
  then
    echo "[INFO] Installing Claw-Eval into ${CLAW_EVAL_CONDA_ENV}"
    download_env python -m pip install -U pip
    download_env python -m pip install -e "${CLAW_EVAL_DIR}[mock,sandbox]"
  fi
  deactivate_conda_env
}

render_config() {
  require_file "${CLAW_EVAL_CONFIG_TEMPLATE}" "Claw-Eval config template"
  export SPARSEVLLM_OPENAI_API_KEY
  export SPARSEVLLM_OPENAI_BASE_URL
  export SPARSEVLLM_CLAW_MODEL_ID
  export SPARSEVLLM_CONTEXT_WINDOW
  export CLAW_EVAL_TRACE_DIR="${TRACE_DIR}"
  export CLAW_EVAL_JUDGE_MODEL
  export CLAW_EVAL_JUDGE_BASE_URL
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
  "${CLAW_EVAL_CONDA_ENV}/bin/python" - "${CLAW_EVAL_CONFIG_TEMPLATE}" "${CLAW_EVAL_CONFIG}" <<'PY'
import os
import sys
from string import Template

src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as f:
    text = f.read()
rendered = Template(text).substitute(os.environ)
with open(dst, "w", encoding="utf-8") as f:
    f.write(rendered)
PY
}

write_engine_kwargs_file() {
  ENGINE_KWARGS_RAW="${ENGINE_KWARGS}" "${CLAW_EVAL_CONDA_ENV}/bin/python" - "${ENGINE_KWARGS_FILE}" <<'PY'
import json
import os
import sys
from pathlib import Path

raw = os.environ["ENGINE_KWARGS_RAW"].strip()
if raw.startswith("{"):
    data = json.loads(raw)
else:
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"ENGINE_KWARGS is neither JSON nor an existing path: {raw}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
if not isinstance(data, dict):
    raise ValueError("ENGINE_KWARGS must resolve to a JSON object")
dst = Path(sys.argv[1])
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
PY
}

resolve_claw_eval_args() {
  CLAW_EVAL_ARGS_TEMPLATE="${CLAW_EVAL_ARGS}" \
    CLAW_EVAL_CONFIG="${CLAW_EVAL_CONFIG}" \
    TRACE_DIR="${TRACE_DIR}" \
    "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY'
import os
from string import Template

print(Template(os.environ["CLAW_EVAL_ARGS_TEMPLATE"]).safe_substitute(os.environ))
PY
}

wait_for_server() {
  for _ in $(seq 1 180); do
    if curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] Sparse-vLLM OpenAI server exited early. See ${LOG_DIR}/sparsevllm_openai_server.log" >&2
      exit 4
    fi
    sleep 2
  done
  echo "[ERROR] Sparse-vLLM OpenAI server did not become healthy" >&2
  exit 5
}

write_run_manifest() {
  "${CLAW_EVAL_CONDA_ENV}/bin/python" - "${RUN_MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

engine_kwargs_file = Path(os.environ["ENGINE_KWARGS_FILE"])
with engine_kwargs_file.open("r", encoding="utf-8") as f:
    engine_kwargs = json.load(f)
manifest = {
    "run_name": os.environ["RUN_NAME"],
    "repo_root": os.environ["REPO_ROOT"],
    "claw_eval_dir": os.environ["CLAW_EVAL_DIR"],
    "sparsevllm_conda_env": os.environ["SPARSEVLLM_CONDA_ENV"],
    "claw_eval_conda_env": os.environ["CLAW_EVAL_CONDA_ENV"],
    "model_path": os.environ["MODEL_PATH"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "engine_kwargs_file": str(engine_kwargs_file),
    "engine_kwargs": engine_kwargs,
    "server_base_url": os.environ["SPARSEVLLM_OPENAI_BASE_URL"],
    "judge_base_url": os.environ["CLAW_EVAL_JUDGE_BASE_URL"],
    "judge_model": os.environ["CLAW_EVAL_JUDGE_MODEL"],
    "sparsevllm_master_port": os.environ["SPARSEVLLM_MASTER_PORT"],
    "claw_eval_args": os.environ["CLAW_EVAL_ARGS"],
    "setup_only": os.environ["SETUP_ONLY"],
    "download_http_proxy": os.environ["DOWNLOAD_HTTP_PROXY"],
    "trace_dir": os.environ["TRACE_DIR"],
    "request_log_dir": os.environ["REQUEST_LOG_DIR"],
}
dst = Path(sys.argv[1])
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
PY
}

main() {
  require_file "${MODEL_PATH}" "model path"
  if [[ "${SETUP_ONLY}" != "1" && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[ERROR] OPENROUTER_API_KEY is required for the default Claw-Eval judge config." >&2
    exit 2
  fi

  source "${HOME}/miniconda3/etc/profile.d/conda.sh"

  prepare_claw_eval_repo
  prepare_claw_eval_env
  render_config
  write_engine_kwargs_file
  CLAW_EVAL_ARGS="$(resolve_claw_eval_args)"
  export CLAW_EVAL_ARGS
  write_run_manifest

  if [[ "${SETUP_ONLY}" == "1" ]]; then
    echo "[INFO] SETUP_ONLY=1 finished. Run directory: ${RUN_DIR}"
    exit 0
  fi

  activate_conda_env "${SPARSEVLLM_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
  export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
  export http_proxy="${DOWNLOAD_HTTP_PROXY}"
  export https_proxy="${DOWNLOAD_HTTPS_PROXY}"
  export HTTP_PROXY="${DOWNLOAD_HTTP_PROXY}"
  export HTTPS_PROXY="${DOWNLOAD_HTTPS_PROXY}"
  export NO_PROXY="127.0.0.1,localhost,::1"
  export no_proxy="127.0.0.1,localhost,::1"

  echo "[INFO] Starting Sparse-vLLM OpenAI server on GPUs ${CUDA_VISIBLE_DEVICES}"
  "${SPARSEVLLM_PYTHON_BIN}" -u -m sparsevllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --engine-kwargs "${ENGINE_KWARGS_FILE}" \
    --request-log-dir "${REQUEST_LOG_DIR}" \
    >"${LOG_DIR}/sparsevllm_openai_server.log" 2>&1 &
  SERVER_PID="$!"
  echo "${SERVER_PID}" >"${RUN_DIR}/sparsevllm_openai_server.pid"
  trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT
  wait_for_server

  echo "[INFO] Running claw-eval ${CLAW_EVAL_ARGS}"
  (
    activate_conda_env "${CLAW_EVAL_CONDA_ENV}"
    cd "${CLAW_EVAL_DIR}"
    # shellcheck disable=SC2086
    "${CLAW_EVAL_CONDA_ENV}/bin/claw-eval" ${CLAW_EVAL_ARGS}
  ) 2>&1 | tee "${LOG_DIR}/claw_eval.log"
}

main "$@"
