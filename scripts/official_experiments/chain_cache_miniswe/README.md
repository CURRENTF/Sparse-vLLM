# GLM-4.7-Flash MiniSWE: cross-request cache reuse

This recipe runs real mini-SWE-agent closed-loop tasks through
`benchmark/swe_bench_lite/run.py`, followed by the official SWE-bench Lite
Docker evaluator. It does not run synthetic or teacher-forced generation.
Use `setting.json` as the single experimental setting. No GPU result is implied
by this recipe; every method needs its own smoke and concurrency pilot.

## Frozen comparison

- Same BF16 GLM-4.7-Flash checkpoint, TP2/EP2/DP1, two GPUs, target 64 agent workers.
  Agent concurrency and server prefill/decode/resident limits are separate; record
  both, and select each through pilot evidence. Check the checkpoint dtype in the
  saved model config; do not substitute an FP8 checkpoint within the campaign.
- Context limit 202752; output limit 16384; temperature .7, top_p 1; 80 agent
  steps; 7200 seconds per task, including model and tool waits. These sampling
  values follow the [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash#evaluation-parameters)
  SWE-Bench Verified and Terminal Bench recommendation. All six methods read
  this one global agent setting and may not override it. The shared API adapter
  has no seed control. Keep the
  upstream `swebench.yaml` prompt identical and save its version through the
  canonical manifest. Timeouts remain failures in the score denominator.
- Thinking is enabled for every method. [Preserved Thinking](https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking)
  is also enabled for
  every method by returning unmodified `reasoning_content` and rendering prior
  assistant reasoning with `clear_thinking=false`. Do not rely on the model
  template's default or change thinking behavior for an individual method.
- On `finish_reason=length` followed by MiniSWE `FormatError`, the shared adapter
  retains the truncated assistant turn before the upstream correction for every
  method. Renderable tool calls receive explicit "not executed" tool results;
  truncated arguments are never executed by this recovery path. The original
  response stays on the correction only, with `truncated_response_preserved=true`,
  so usage is recorded once. This changes subsequent closed-loop prompts and
  requires a fresh run for comparisons with the previous recovery behavior.
  Chain continuation after `length` uses full token-prefix validation, without
  the append shortcut: incomplete tool serialization or tokenization differences
  may still require the server to release and recreate the chain safely.
- Prefill chunk 4096, batch token budget 65536, GPU memory utilization .95, and
  CUDA Graph on. CPU cache offload is disabled for the frozen comparison.
  Actual concurrent decoding varies while agents run tools. The default is
  **64 agent workers**, not proof of 64 resident requests or the maximum feasible
  server limit.
- Full selection is the canonical sorted Lite test set of 300. Pilot uses its
  first max(64, concurrency); smoke uses the first 1. Compare `instances.txt` across methods.
  `batch_size=300` avoids the old 50-task batches capping 64 workers at 50.

| Label | Cache | Budget and exceptions |
|---|---|---|
| snapkv-chain | chain | 64 sink + 512 recent + 15808 selected = 16384 |
| h2o-chain | chain | prefill 16384, decode 8192; online eviction every 128 steps |
| omnikv-prefix | radix | 64 + 512 + 1472 = 2048; model-profile full layers retained |
| quest-prefix | radix | total selection 2048; page size 16; first 2 layers full |
| vanilla-prefix | radix | dense KV |
| snapkv-no-chain | disabled | exactly the same SnapKV settings, cache reuse disabled |

SnapKV uses probability scoring/window 32. H2O uses probability scoring/window
128, recent ratio .5, FP32 scores and online cumulative updates. Current MLA H2O
reduces heads before softmax in decode: it is not official per-head H2O parity.
Do not silently switch to logits or disable online eviction to get a passing run.
OmniKV/Quest selection budgets are not physical KV storage limits, and full layers
are exceptions. These are useful operating points, not equal-quality algorithms.

## Preparation and execution

All paths below are supplied by the operator. Use a persistent data disk for
`RUN_ROOT`, check free space (including Docker storage), and record the choice.
Use another persistent data disk if the preferred disk has insufficient space. Do not download
images/models implicitly. The scripts refuse to overwrite invocation artifacts.
For a repaired attempt, prepare a new root and preserve the previous failure.

```bash
RECIPE=scripts/official_experiments/chain_cache_miniswe/run.py
python3 "$RECIPE" prepare --root "$RUN_ROOT" --concurrency "$CONCURRENCY"
```

`--concurrency` controls the number of concurrent MiniSWE agents. By default the
server prefill, decode, and resident-sequence limits match it. When the client
concurrency intentionally exceeds the number of sequences that must reside on
the GPUs at once, set the server limit explicitly, for example
`--concurrency 64 --engine-concurrency 48`. Record both values in the result;
they are different workload and engine controls.

For the 24-agent candidate with 64 resident rows and offload disabled, prepare
a fresh root with the shared `.95` memory setting:

```bash
python3 "$RECIPE" prepare --root "$RUN_ROOT" --concurrency 24 \
  --engine-concurrency 24 --engine-resident-concurrency 64
```

This keeps prefill/decode sequence limits at 24 and captures decode Graphs only
through batch 24. The 64 resident rows allow additional idle chains to retain
their GPU state. `gpu_memory_utilization=.95` is the memory planning fraction,
not a target for measured GPU compute utilization. This configuration still
needs a new pilot; earlier `.90` runs do not establish its stability.

The shared engine setting uses `decode_reservation_tokens=1024`, a rolling
reservation window for future decode capacity. It does not change the agent's
16384-token output limit. For the 32-agent, 64-resident-row candidate, use
`--concurrency 32 --engine-concurrency 32 --engine-resident-concurrency 64` in
a fresh root; retain `.95` memory utilization and disabled offload.

If 64 agents must be admitted while only 48 sequences decode in one engine
step, keep 64 resident rows without increasing the decode Graph batch:
`--concurrency 64 --engine-concurrency 48 --engine-resident-concurrency 64`.
This lets the scheduler queue work behind the 48-sequence execution limit.
Setting resident rows to 48 instead causes excess simultaneous chain admissions
to return HTTP 503 `chain_capacity_unavailable`; client retries are bounded and
do not make that configuration a stable 64-agent result.

Start with `snapkv-chain`, then `h2o-chain`, `omnikv-prefix`, `quest-prefix`,
`vanilla-prefix`, `snapkv-no-chain`. Finish smoke for each before its pilot;
complete pilots before choosing the formal queue. Run methods sequentially
on the same GPU pair. Each agent/server concurrency pair gets a fresh root, e.g.
`agents64-engine48`.
The prepared settings freeze both agent and engine limits; never edit an active root.
Changing the output limit from 8192 to 16384 invalidates earlier concurrency
pilots for capacity selection. Prepare a new root and rerun smoke/pilot before
using an old stable-concurrency claim; in particular, the prior c24/8192 pilot
does not establish c24 capacity under this protocol.

Concurrency plan: for radix methods start at 4 or 8 and double toward 64;
for SnapKV/H2O start at 16, then 32/48/64, and only if stable try 96/128.
A one-task smoke tests interfaces only. A complete pilot at the intended worker
count tests long-lived sessions and includes at least that many tasks. Inspect
actual running/decoding counts, cache reuse, preemption, idle-chain eviction and
HTTP failures, not just process exit. Short successful tasks do not establish a
worst-case cache capacity. Stop escalation at memory/admission failure; ordinary
bugs require diagnosis and must not be called capacity limits. Keep full logical
history and all length/budget settings unchanged during this scan.

Choose a common pilot-validated concurrency for a matched-concurrency comparison,
and optionally each method's largest pilot-validated concurrency for application
capacity results. Label the latter "largest tested", not an integer maximum
unless the next integer was actually tested under the same workload. SnapKV
on/off pilots for the slow-cost gate must be at the same concurrency and share
one prepared root. Differences in full-run concurrency must remain visible in
the results and cannot be attributed solely to cache reuse.

GPU host, in a persistent tmux session, after activating the server conda env
(`conda activate "$SERVER_ENV"`; do not invoke only a conda Python without
activation). `SERVER_PYTHON` should be `command -v python` in that environment:

```bash
python3 "$RECIPE" serve --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SERVER_PYTHON" --model "$GLM47_MODEL" --gpus "$GPU_PAIR" \
  --port 18147 --timeout 86400
```

The benchmark coordinator must install an exit trap before starting the stages,
so success, failure, timeout, and operator interruption all stop the task-owned
server. The stop command validates hostname, boot ID, PID start time, and private
process group before sending a signal; it never searches or kills by process name:

```bash
cleanup_server() {
  python3 "$RECIPE" stop --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE"
}
trap cleanup_server EXIT INT TERM
```

For a remote Docker worker, keep this trap in the GPU-host coordinator around
the SSH command that runs the remote benchmark. A trap on the Docker worker
cannot stop a process on the GPU host. Confirm `server.result.json` records
`stop_reason=requested_after_benchmark` and both selected GPUs have no task-owned
compute processes before advancing to the next phase.

`GPU_PAIR` must contain two distinct idle indices. The launcher checks compute
PIDs, used memory and utilization, and refuses an occupied port. Recheck ownership
during startup until both ranks attach; the preflight check is not a GPU
reservation. Never kill another process or pass occupied GPUs as "idle".
Record PID/UUID/start time. The foreground launcher owns only its child process
group; interrupting it stops that server group. Stop it after each phase and
start a fresh server for the next phase so pilot caches cannot warm full tasks.

Docker driver, in its activated environment, after `/readyz` returns success:

```bash
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage prepare --timeout 1800
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage generate --timeout 43200
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage evaluate --timeout 43200
python3 "$RECIPE" bench --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE" \
  --python "$SWE_PYTHON" --swe-bench-dir "$SWE_BENCH_DIR" \
  --api-base http://127.0.0.1:18147/v1 --stage summarize --timeout 1800
```

Use `PHASE=smoke`, then `pilot`, then `full`; pilot/full use the prepared concurrency.
The full campaign has one official 300-task evaluation per method. A paper claim
about stable performance needs repeated independent campaigns, not duplicated
aggregates. Preflight Docker images and host RAM/process/storage capacity for 64
containers before generation. `prepare` uses the canonical offline dataset/image
checks. Evaluation uses 16 workers and can run after stopping the GPU server.

If local Docker is unavailable, use the established Docker worker. Copy this
recipe, the same adapter revision, the prepared root's settings/engine JSONs,
and each phase's `server_manifest.json` to that host. Keep the directory structure;
paths to Python/SWE-bench and `--root` can differ. Establish an SSH reverse tunnel
to the local server and use the forwarded URL as `--api-base` consistently within
each run. Do not start a second GPU server remotely. Copy benchmark artifacts
back beside the original server logs before `collect`. Server readiness alone is
insufficient: verify both rank logs, actual Graph execution, and at least two
successful turns with nonzero reuse for the enabled cache modes.

## Slow baseline policy and failures

Run SnapKV-no-chain last. For its **64-task pilot generation only**, use
`--timeout 7200`. Generate the SnapKV-chain pilot first with the same agent
settings. The `full --stage generate` command automatically skips the no-chain
baseline if its pilot timed out, its elapsed-time ratio exceeds 4, or its rough
300-task projection exceeds 12 hours. Pilot selection scales to at least the
prepared concurrency. The decision is saved as
`slow_baseline_decision.json` with `skipped_by_policy`; this is a cost policy,
not a performance conclusion. Use separate `generate` and `evaluate` commands
for pilots, because the gate reads generation time, excluding evaluation.
If the pilot times out, preserve its timeout record, stop its server, and check
for task-owned leftover Docker containers before proceeding. Do not evaluate an
incomplete pilot as a complete score. To record the full skip, invoke the full
generation command; it checks the gate before contacting a server.

Ordinary crashes, HTTP errors, prefix mismatches, OOM, timeouts on other methods,
and invalid outputs are explicit failures. Do not call them capacity boundaries,
slow-baseline skips, or zero-quality results. Preserve logs and move to other
methods; report failures in the final campaign record. An idle-chain HTTP 410
can use only the adapter's existing bounded recovery, which must be counted as
recomputation. No new retry/fallback is enabled here.

## Evidence and interpretation

```bash
python3 "$RECIPE" collect --root "$RUN_ROOT" --method "$METHOD" --phase "$PHASE"
```

Collection validates selected IDs and exact smoke/pilot/full row coverage, preserves the
official summary, and exports `request_samples.jsonl` and `report.json` from
server request JSON. Also retain server logs, trajectories, predictions, official
reports, and environment manifests. `prepare` records the Git commit and working-tree
status in `source.json`; it does not copy source files or save a working-tree patch.
Use the Research-Vault workflow for the final dated record and compact data.

`timed_mini.py` delegates to the installed `mini-extra` console entrypoint and
wraps only its `process_instance` call. It preserves returned values/exceptions
and writes one `*.wall_time.json` per task. Unsupported upstream module/signature
changes fail explicitly. This uses the upstream
[SWE-bench instance boundary](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/benchmarks/swebench.py);
verify the installed version during smoke. Task time includes environment startup,
model waiting and tool execution, excludes executor-queue waiting and official
evaluation. An interrupted task may have no final timing: collection rejects
missing coverage. Keep timeout/failed tasks distinct from successful completions.
`task_samples.jsonl` joins these times to official outcomes; `report.json` reports
all-attempt and resolved-only P50/P95. Resolved-only subsets differ across methods
and are descriptive, not paired comparisons. Generate/evaluate wall times come
from separate `*.result.json` records; do not use a combined `all` stage.

Report resolved/300, all failure statuses, API calls, reused tokens, logical
uncached prompt tokens, generation wall time and summed server request duration.
Summed request durations overlap at concurrency 64: they are neither GPU time
nor workload wall time. Logical uncached tokens do not include unobserved
scheduler recompute and cannot be called exact executed-prefill tokens.

The existing nonstreaming API logs **do not measure TTFT/TPOT**. Collection
explicitly emits null/status instead of deriving them from full-response time.
It also does not claim peak KV usage or H2O state equality. For those additional
experiment-3 figures, add validated engine-event observation/fixed-trace replay
and an independent state-restoration oracle in a follow-up. This campaign tests
closed-loop application utility: same issues, but method-dependent generated
tokens, tool results and turn counts. Do not plot its latency ratios as a
same-input Chain Cache speedup. Inspect resets/reuse and separate generation
failures from official test failures before attributing score differences.
