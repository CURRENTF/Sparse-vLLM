#!/usr/bin/env bash
# Wait for independent queues, then use the canonical raw-artifact validator.
set -euo pipefail
source_flag=--config
if [[ ${1:-} == --grid ]]; then source_flag=--grid-config; shift; fi
if [[ $# -lt 5 ]]; then
    echo "Usage: $0 [--grid] CONFIG BENCHMARK_REPO EXPORT_DIR PLOT_RUN_DIR STATUS_FILES..." >&2
    exit 2
fi
config=$(realpath "$1")
repo=$(realpath "$2")
export_dir=$3
plot_run=$4
shift 4
mkdir "$plot_run"
status_file="$plot_run/status.tsv"
printf '%s\tplot\twaiting_for_queues\n' "$(date -Is)" > "$status_file"
deadline=$((SECONDS + 172800))
for status in "$@"; do
    until [[ -f "$status" ]] && rg -q '\tqueue\t(exit=[0-9]+$|completed(\t|$)|failed(\t|$))' "$status"; do
        if (( SECONDS >= deadline )); then
            printf '%s\tplot\texit=1\n' "$(date -Is)" >> "$status_file"
            echo "Timed out waiting for terminal queue status: $status" >&2
            exit 1
        fi
        sleep 30
    done
done
mapfile -t environment < <(python3 -c 'import json,os,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads(p.read_text()); c=json.loads((p.parent / os.path.expandvars(c["panels"][0]["config"])).read_text()) if sys.argv[2]=="--grid-config" else c; print(c["conda"]); print(c["native_env"])' "$config" "$source_flag")
command=("${environment[0]}" run --no-capture-output -p "${environment[1]}" python
    "$repo/scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py"
    "$source_flag" "$config" --output-dir "$plot_run/plots" --export-data-dir "$export_dir")
printf '%q ' "${command[@]}" > "$plot_run/command.sh"
printf '\n' >> "$plot_run/command.sh"
printf '%s\tplot\tvalidating_and_rendering\n' "$(date -Is)" >> "$status_file"
code=0
"${command[@]}" > "$plot_run/plot.log" 2>&1 || code=$?
printf '%s\tplot\texit=%s\n' "$(date -Is)" "$code" >> "$status_file"
exit "$code"
