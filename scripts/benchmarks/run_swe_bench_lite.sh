#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWE_BENCH_PYTHON="${SWE_BENCH_PYTHON:-python3}"
if [[ "${SWE_BENCH_PYTHON}" != */* ]]; then
  SWE_BENCH_PYTHON="$(command -v "${SWE_BENCH_PYTHON}")"
fi
SWE_BENCH_BIN_DIR="$(cd "$(dirname "${SWE_BENCH_PYTHON}")" && pwd)"
SWE_BENCH_PYTHON="${SWE_BENCH_BIN_DIR}/$(basename "${SWE_BENCH_PYTHON}")"
export PATH="${SWE_BENCH_BIN_DIR}:${PATH}"

cd "${REPO_ROOT}"
exec "${SWE_BENCH_PYTHON}" -m benchmark.swe_bench_lite.run "$@"
