#!/usr/bin/env bash
# Independent lane failures remain explicit without blocking unrelated methods.
set -euo pipefail
if [[ $# -lt 5 ]]; then
    echo "Usage: $0 RUN_ROOT MODEL GPUS LANES_CSV ATTEMPT [BENCHMARK_REPO [WAIT_FOR_STATUS]] [-- SWEEP_ARGS...]" >&2
    exit 2
fi
extra_args=()
if [[ $# -gt 7 ]]; then
    if [[ $8 != -- ]]; then echo "Expected -- before extra sweep arguments" >&2; exit 2; fi
    extra_args=("${@:9}")
fi
run_root=$(realpath "$1")
model=$2
gpus=$3
lanes_csv=$4
attempt=$5
source_repo=$(realpath "${6:-$run_root/source}")
wait_status=${7:-}
if [[ -n "$wait_status" ]]; then test -f "$wait_status"; fi
package="$source_repo/scripts/official_experiments/sparse_decode_efficiency"
data_root=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"])' "$run_root/launch.json")
status_file="$run_root/$attempt.status.tsv"
test ! -e "$status_file"
set -o noclobber
printf '%s\tqueue\tstarted\n' "$(date -Is)" > "$status_file"
if [[ -n "$wait_status" ]]; then
    printf '%s\tresource\twaiting_for_queue\t%s\n' "$(date -Is)" "$wait_status" >> "$status_file"
    deadline=$((SECONDS + 172800))
    until rg -q '\tqueue\texit=[0-9]+$' "$wait_status"; do
        if (( SECONDS >= deadline )); then
            printf '%s\tqueue\texit=1\n' "$(date -Is)" >> "$status_file"
            echo "Timed out waiting for queue terminal status: $wait_status" >&2
            exit 1
        fi
        sleep 30
    done
fi
failed=0
IFS=',' read -r -a lanes <<< "$lanes_csv"
for lane in "${lanes[@]}"; do
    command=(python3 "$package/sweep_decode_capacity.py" --config "$run_root/config.json"
             --repo "$source_repo" --model "$model" --gpus "$gpus" --lanes "$lane"
             --attempt "$attempt" --export-measurements-dir "$data_root/measurements"
             "${extra_args[@]}")
    printf '%q ' "${command[@]}" > "$run_root/$attempt.$lane.command.sh"
    printf '\n' >> "$run_root/$attempt.$lane.command.sh"
    printf '%s\t%s\tstarted\n' "$(date -Is)" "$lane" >> "$status_file"
    if "${command[@]}" > "$run_root/$attempt.$lane.log" 2>&1; then
        code=0
    else
        code=$?
        failed=1
    fi
    printf '%s\t%s\texit=%s\n' "$(date -Is)" "$lane" "$code" >> "$status_file"
done
printf '%s\tqueue\texit=%s\n' "$(date -Is)" "$failed" >> "$status_file"
exit "$failed"
