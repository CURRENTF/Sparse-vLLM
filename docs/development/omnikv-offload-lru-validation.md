# OmniKV offload: layerwise prefetch and exact LRU reuse

This follows the initial full-selected-set offload implementation documented in
[the first-stage report](omnikv-offload-validation.md). The original research
plan remains untracked. Work stays on `feat/omnikv-offload`.

## Runtime behavior

`enable_omnikv_offload=True` is still the master switch. It now enables a
request-private GPU LRU pool by default. `omnikv_offload_cache_tokens=None`
rounds the total sink/keep/recent budget up to a power of two, capped by
`max_model_len`. A positive explicit capacity must cover the selected budget;
`0` retains the exact full-selected-set transfer path without historical reuse.

Neither option changes observer layers, attention scores, Top-K membership,
selection order, update frequency, MLA representation, or parallel semantics.
All sparse-layer history remains backed by pinned CPU memory. Full-attention
layers and the leading full-history layers remain on GPU. A layer's current
projection is written through to host and inserted directly into its GPU view.

The two requested directions are separate changes:

1. After observer selection, prefetch only the first sparse layer. When its
   compute view is acquired, start the next layer's transfer after the current
   stream dependency. Each layer waits for its own completion event; transfer
   can overlap the previous layer's attention/MLP. This replaces enqueueing the
   entire observation group immediately. It does not reduce transfer bytes.
2. Reuse GPU hits and fetch only misses from the exact selected set. Each
   persistent request row has an independent pool in each sparse layer.
   Observation groups share only lookup/eviction metadata because their token
   selections agree. They do not share layer KV values.

The final attention view points directly into the LRU pool in original selected
order. This removes the extra selected-KV copy and allocation. One immutable
zero slot per sparse layer supplies padded graph rows. The index-table width
retains the original attention-provider planning capacity, including the short
path; pool compaction must not change FA3 split planning.

## LRU and lifecycle details

Cache managers own all pool state and capacity. A per-request inverse directory
maps logical physical slots to GPU cache positions. CUDA kernels first touch
selected hits, then admit misses without evicting any selected hit. Entries
selected in one decode step have the same recency; equal-age victims are chosen
by ascending pool position. A binary search finds the oldest timestamp covering
the required number of victims, followed by prefix-sum compaction. No cache-wide
sort, CPU Top-K readback, or dynamic GPU allocation is needed during replay.
After admission, the victim workspace becomes a compact list of missing selected
token positions. Copy kernels iterate only that list. Their launch envelope is
fixed; the effective device loop length follows the replay's miss count.

Per observation group metadata costs, for `B` resident request rows, `N` logical
slots, `R` GPU cache entries and `C` selected entries:

- inverse directory: `4 * B * N` bytes;
- keys and timestamps: `12 * B * R` bytes;
- plans and victim workspace: `8 * B * C` bytes;
- clocks, per-batch miss counts and cumulative hit/miss counters: `12 * B + 16` bytes.

GPU KV costs `B * R` entries per sparse layer plus a padding entry; there is no
additional `B * C` selected-KV pool in LRU mode. The no-LRU path retains that
selected pool. These costs, full-layer storage, the full-history prefill pool,
and slot-table metadata feed the same pool-budget function for startup profiling
and production admission. Each rank participates in the existing minimum-capacity
reduction before publishing capacity.

Request release invalidates its private directory, keys, and ages before row
reuse. Shared prefix backing and its reference/transfer lifecycle are unchanged.
An LRU eviction only discards a GPU replica. Suffix prefill still restores all
required prefix history. Captured buffer addresses stay fixed; real request rows,
selected slots, lengths, and padding are runtime GPU metadata. The side stream
joins the main stream before forward completion and resource reuse.

## Validation and performance artifacts

The local bundle is `outputs/omnikv-offload-lru-20260911/`. It retains successful
runs and failed intermediate attempts separately. Model parity compares generated
token IDs and final per-layer/per-rank selection hashes against the GPU-only
reference. This is not a claim of a complete per-step selection trace comparison.

The independent copy/LRU oracle checks FP16 explicit KV, BF16 MLA components,
miss accounting, hit reuse, full-pool eviction, selected-order changes, current
KV in the middle or outside the selection, padded rows, persistent-row changes,
physical-slot reuse, and CUDA Graph replay. CPU tests protect configuration,
startup admission and attention-view metadata.

The microbenchmark script is
the archived `benchmark_omnikv_offload_copy.py` in the original experiment bundle
(`archived-development/`). This development-only probe and its runtime hit counters
were removed from the production tree after validation. Each timed graph replay
starts from the same seeded cache. Cache reseeding and output verification are
outside the event-timed region; lookup, eviction planning and both history
components are inside it. It excludes attention and current-token write-through.
Thus repeated iterations cannot silently turn a requested miss rate into all hits.

Nsight Systems identified about 95 microseconds per observation-group admission
in the initial full-sort implementation. Replacing sorting with timestamp
selection reduced the complete batch-4 copy/metadata microbenchmark at 95% hits
from about 206 to 126 microseconds, and at 99% hits from 177 to 97 microseconds.
Thirty measured iterations follow five warmups. The same microbenchmark's exact
full-set copy takes about 0.69–0.72 ms. These are microbenchmark results, not
serving TPOT. The final compact-miss microbenchmark (GPU 5 / NUMA 1) measures
0.691 ms for exact full-set copy, 0.066 ms at 95% hits and 0.038 ms at 99% hits.
The cold-miss case is 0.699 ms; LRU provides no bandwidth saving on cold entries.
Final Nsight Systems timing attributes about 7.5 microseconds to admission and
4.7 microseconds to victim selection in this diagnostic, excluding lookup.
Nsight Compute was attempted but failed with `ERR_NVGPUCTRPERM`;
no hardware-counter or theoretical MFU/MBU claim is made.

All serving performance uses `benchmark/efficiency/bench_probe.py`, matching
model, traces, selection budgets, topology, graph mode, warmups and iterations.
Measurements use H100 80GB GPUs with Torch 2.11/CUDA 13 and NUMA-local memory.
Every device was checked before use. Other users ran on other GPUs: these are
shared-host measurements. An initial launch raced with a new GPU-5 user job,
was terminated during startup, and supplies no performance evidence. The matched
Qwen ablations therefore use GPU 0 / NUMA node 0 throughout.

### Qwen 8K ablation

Sink 0, recent 32, keep 2048, cache 4096, original full-layer profile,
512 output tokens, one warmup and three measured iterations:

| Implementation | Batch 1 TPOT ms | Batch 4 TPOT ms |
| --- | ---: | ---: |
| GPU-only OmniKV | 5.645 | 6.516 |
| Original exact offload | 9.265 | 28.646 |
| One-layer-ahead exact offload | 9.290 | 27.858 |
| LRU with selected-buffer copy | 6.821 | 11.763 |
| Direct LRU attention view, full-sort admission | 6.663 | 9.874 |
| Timestamp-threshold admission | 6.018 | 9.049 |
| Final: compact misses before transfer | **5.848** | **7.368** |

Layerwise scheduling alone provides little benefit here. Avoiding host reads is
the dominant improvement; removing the redundant GPU copy improves it further.
The final implementation reduces TPOT by about 36.9% / 74.3% relative to the
original offload at batch 1 / 4. Relative to GPU-only OmniKV it remains about
3.6% / 13.1% slower. The no-obvious-regression objective is therefore not
universally met, particularly at batch 4. Prefill overhead is also still present.

For the same 16,384-slot prefix-validation capacity, the direct LRU manager's
GPU allocation is about 65.2% below native full-GPU KV for Qwen and 68.4% below
native MLA KV for GLM, including its pools and metadata. Model weights and
provider/graph allocations are outside that accounting. BenchProbe peak-VRAM
changes additionally reflect different auto-reserved capacities and should not
be presented as algorithmic KV savings.

### MLA fixed-batch comparison

GLM-4.7-Flash, batch 1, the original full-layer profile and the same 2080-token
selection / 4096-entry cache budgets, 512 output tokens, one warmup and three
iterations, GPU 0 / NUMA node 0:

| Prompt length | GPU-only TPOT ms | Original offload TPOT ms | Final LRU TPOT ms |
| --- | ---: | ---: | ---: |
| 8192 | 7.222 | 7.725 | 7.359 |
| 32768 | 7.525 | 8.073 | 7.814 |

The final LRU run is about 1.9% / 3.8% above GPU-only and about 4.7% / 3.2%
below the original offload. MLA's smaller compressed payload and compute overlap
already kept the original offload penalty small; its absolute gain is smaller
than Qwen's explicit-KV gain.

### Correctness evidence

- The final copy/LRU/configuration/startup/view suite passes 60 tests, with the
  two-GPU capacity case run separately (three capacity tests pass).
- Prefix transfer and observer-score lifecycle checks pass 11 tests.
- Final Qwen and GLM pressure runs each pass 12 cases, including an actual
  shared-prefix H2D restoration, shorter-request completion, ragged prompts,
  short-to-long transitions and 277 CUDA Graph replays with zero recaptures.
- Final Qwen TP2 and GLM TP2+EP2 outputs and selection hashes match the original
  GPU-only reference on both ranks. GLM TP2 and EP2 also pass with the timestamp
  planner before the final miss-list copy optimization.
- Qwen's final same-slot GPU cache allocation is 840,323,436 bytes versus
  2,415,919,104 native KV bytes. GLM's is 280,117,612 versus 887,095,296 bytes.
- The pressure traces show cumulative historical hit rates of 83.6% for Qwen
  and 81.7% for GLM. These include cold requests and use a 544-token selection /
  1024-entry cache, so they are not claimed as steady-state hit rates for the
  2080-token BenchProbe workload.

The initial model attempts with LRU failed during startup because the generic
profiling budget omitted the new fixed pool. The shared pool-budget function
fixes this; the failed JSON/log artifacts remain in the bundle. An early oracle
case also incorrectly changed shared host contents between checking two request
rows; its test input was corrected and the original failed log retained.

No request-selection approximation, reduced budget, less frequent Top-K update,
MLA expansion, provider substitution or cross-rank selection broadcast was used.

### Request churn

Qwen3-4B, prompts up to 8192 tokens, outputs up to 128, maximum concurrency 4,
16 requests per trace, one warmup and three iterations, GPU 0 / NUMA node 0:

| Metric | Original exact offload | Final LRU offload |
| --- | ---: | ---: |
| Request TPOT mean, ms | 31.132 | 15.287 |
| Observed request throughput, requests/s | 1.045 | 1.801 |
| TTFT p50, ms | 6083.66 | 3927.45 |

Throughput increases about 72%. Churn TPOT includes scheduling and intervening
prefill, so it is not a pure decode-kernel latency. Both runs complete all 48
measured requests. This pair compares the two offload implementations; it does
not establish parity with GPU-only serving under churn.

Final Qwen eager execution with cancellation of a shared-prefix peer also
matches the reference. Qwen3-30B-A3B TP2+EP2 matches tokens and selection hashes
on both ranks with prefix caching disabled, preserving its existing registry
restriction. An explicit `omnikv_offload_cache_tokens=0` run passes the same
12-case Qwen pressure reference, retaining the first requested exact-transfer
path. The default automatic pool capacity is exercised by the final model and
BenchProbe runs, rather than being replaced by a hidden benchmark override.

`audit.json` checks matched model/workload/selection configuration, trace
identity, per-request success, zero runtime captures/recaptures, required model
parity artifacts and the final microbenchmark statuses. It records hashes and
keeps failed intermediate model artifacts visible. The audit script is copied
into the local artifact bundle alongside the exact benchmark commands in each
run manifest. No GPU jobs are left running by this task.
