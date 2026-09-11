# OmniKV offload implementation and validation

Date: 2026-09-11. Base: `4cec18b4`. Branch: `feat/omnikv-offload`.

The active offload implementation is opt-in through
`sparse_method="omnikv", enable_omnikv_offload=True`. It preserves the existing
selector, full-layer profile, token budgets, native cache representation and
TP-local selection. It does not implement LRU, a historical GPU hot cache, or
miss-only fetching.

## Implementation

- `OmniKVStorage` allocates full-layer GPU backing and sparse-layer pinned host
  backing. Explicit KV retains local KV heads; MLA retains separate latent and
  RoPE tensors. Layers preceding the first observer retain GPU backing because
  GPU-only OmniKV already executes full-history attention there.
- The cache manager owns a bounded decode staging pair per sparse layer and a
  shared, single-layer full-history prefill buffer. It never allocates a full
  GPU cache for every sparse layer. GPU/host capacity is coupled through the
  same logical slot allocator; startup takes the minimum capacity across ranks.
  Host capacity uses a conservative fraction of whole-machine available memory,
  divided among workers and reduced by the configured prefix host reservation.
- The existing prefix payload keeps common logical slot IDs. The manager's
  device-resident map resolves those IDs to sparse host locations. This narrower
  physical adaptation reuses the logical radix API and reference counting; it
  does not add CPU pointers to shared logical prefix-cache interfaces.
- Prefix publication backs up full layers and rehomes sparse layers into the
  prefix host bank before marking the entire block host-present. Restore copies
  full layers to GPU and remaps sparse backing. Shared-prefix consumers have
  private decode staging; their generated suffix writes target private slots.
  Prefix counters distinguish full-layer H2D restore from sparse rehome reads.
- At observer attention-end, the side stream runs the original normalization,
  Top-K and cross-layer propagation, then gathers each target layer. The main
  stream continues projection/MLP and the next layer's QKV. Consumers wait for
  both components of their own layer; subsequent observers wait before reusing
  raw score storage. Capture records these dependencies and uses stable pools,
  pointer arrays, tables and graph inputs.
- GPU-indexed UVA kernels read the exact selected history. The current token is
  written through to host and inserted from its current GPU projection. With
  zero recent budget its actual selected position is used, including the case
  where it is absent. Padding rows cannot read host KV or overwrite live KV.

The current prefill implementation reloads each sparse layer's complete visible
history into the shared GPU buffer. Its traffic can grow quadratically with
prompt length divided by chunk size. The buffer is included in GPU accounting;
this implementation does not claim constant GPU memory with context length.

## Validation contracts

Run the model parity driver twice with the same model, topology, layer profile
and options, adding `--offload --reference BASELINE.json` to the second run:

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" .venv/bin/python \
  scripts/validation/validate_omnikv_offload.py \
  --model "$MODEL" --full-layers "$FULL_LAYERS" \
  --tp "$TP" --ep "$EP" --output "$BASELINE"

CUDA_VISIBLE_DEVICES="$GPU_IDS" .venv/bin/python \
  scripts/validation/validate_omnikv_offload.py \
  --model "$MODEL" --full-layers "$FULL_LAYERS" \
  --tp "$TP" --ep "$EP" --offload --prefix-offload \
  --reference "$BASELINE" --output "$OFFLOAD"
```

`--extended` adds a short-to-long transition and requests that finish at
different steps. `--pressure` adds enough unique prefixes to force active
OmniKV prefix demotion, then requires an actual completed H2D restore.
`--cancel` aborts one of two requests during shared-prefix decode and requires
the survivor to finish. `--eager` selects eager execution. For model combinations that prohibit prefix
caching, use `--no-prefix` on both runs and omit `--prefix-offload`.

Artifacts retain raw model outputs, token inputs, per-case status, topology,
configuration, revision/dirty state and per-rank debug snapshots. Comparison
checks output token IDs and the last step's logical selected-index SHA-256 for
every recorded sparse layer/rank. This is not a full per-step selection trace.
CUDA Graph runs require replay and zero recapture. Cold-versus-prefix-hit output
identity is recorded separately: GLM's GPU-only baseline itself can differ on
these synthetic token prompts, so that comparison is not substituted for the
same-request-order offload parity check.

The dedicated CUDA tests compare pinned copies with independent tensor indexing,
change graph replay metadata, exercise FP16/BF16 explicit KV and both MLA
components, protect padded rows, and delay the second component during prefix
restore after source-slot reuse. CPU tests cover exhausted GPU/host pool budgets
before allocation, plus existing configuration, score lifecycle and prefix
contracts.

## Completed correctness runs

| Model / topology | Mode | Evidence |
| --- | --- | --- |
| Qwen3-4B TP1 | eager + shared prefix | Extended cases and cancellation; survivor output and final selections match. |
| Qwen3-4B TP1 | graph + prefix offload | Extended/pressure cases; actual H2D restore, output/selection parity, zero recapture. |
| Qwen3-4B TP2 | graph + prefix offload | Both ranks replay; output and final selections match the same-topology baseline. |
| GLM TP1 | eager + prefix offload | Extended cases, including short-to-long; output/selection parity. |
| GLM TP1 | graph + prefix offload | Extended/pressure cases after the table-capacity fix; actual H2D restore and parity. |
| GLM TP2 | graph + prefix offload | Output/final-selection parity on both ranks, zero recapture. |
| GLM EP2 | graph + prefix offload | Replicated-attention topology; output/final-selection parity, zero recapture. |
| GLM TP2+EP2 | graph + prefix offload | Outer-TP topology; output/final-selection parity, zero recapture. |
| Qwen3-MoE TP2+EP2 | graph, prefix disabled | Output/final-selection parity on both ranks, zero recapture. |

The focused CPU suite passed 141 tests (one unrelated opt-in integration skipped),
with two additional view-contract tests passing. CUDA copy/prefix tests passed
12 cases. A real two-GPU NCCL test confirmed that one exhausted rank makes both
ranks reject pool admission before KV allocation. Two existing MLA tests passed
against the mathematical attention oracle and strided-copy contract. These
checks are correctness evidence, not throughput measurements.

## Measured results

The local artifact bundle is `outputs/omnikv-offload-20260911/` (ignored by Git).
It retains successful runs and failed diagnostic attempts separately. The
original research plan is intentionally not committed.

All performance numbers below come from `benchmark/efficiency/bench_probe.py`:
BF16, batch 1, decode CUDA Graph, sink 0, recent 32, history Top-K 2048, pool
kernel 1, 512 output tokens, one warmup and three measured iterations. The
baseline and offload request traces are matched. Qwen3-4B uses full layers
`0,1,3,9,13,16,21,28`; GLM uses `0,3,8,16,19,25,31,37,44`.

| Model | Prompt tokens | Baseline TPOT ms | Offload TPOT ms | Baseline TTFT p50 ms | Offload TTFT p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 8192 | 5.84 | 10.01 | 140.55 | 175.24 |
| Qwen3-4B-Instruct-2507 | 32768 | 6.29 | 10.55 | 959.21 | 1113.29 |
| GLM-4.7-Flash | 8192 | 7.26 | 7.58 | 183.72 | 196.62 |
| GLM-4.7-Flash | 32768 | 7.53 | 7.61 | 1299.12 | 1355.25 |

Qwen used GPU 5 with NUMA node 1 binding; GLM used GPU 0 with node 0 binding.
Devices were idle before launch. Other users were active on other devices;
these are local shared-host measurements, not isolated-machine results.

The no-obvious-regression objective is **not met for Qwen3-4B**: fixed-batch
TPOT increases about 68–71%. GLM's measured increases are about 4.4% and 1.1%.
Do not generalize the MLA result to explicit KV or to unmeasured concurrency.

For the same 16,384-slot capacity in the prefix validation, Qwen's manager GPU
allocation (including staging/metadata) is about 69.8% below its native full-GPU
KV bytes; GLM's is about 73.3% lower per rank. These exclude model weights and
provider/graph memory outside the cache manager. BenchProbe peak-VRAM differences
also include different auto-reserved pool sizes and cannot alone establish an
algorithmic KV saving.

### Long-context capacity case

GLM, batch 2, prompts up to 131072 tokens with the same jittered traces,
512 output tokens, one warmup and three iterations, GPU 5 / NUMA node 1:

| Metric | GPU-only | Offload |
| --- | ---: | ---: |
| Peak simultaneous decoding requests | 1 | 2 |
| Request throughput (requests/s) | 0.05704 | 0.05691 |
| Request TPOT mean (ms) | 8.77 | 27.31 |
| TTFT p50 (ms) | 21483.55 | 20708.94 |
| Batch decode-window throughput (tokens/s) | 46.90 | 48.47 |

The baseline defers admission because its KV slots cannot hold both requests.
Offload holds both, but the unchanged chunked-prefill scheduler interleaves
prefill with a request that has already emitted its first token. That work is
included in request TPOT and decode-window time. Capacity increases here; total
request throughput does not materially improve. Multiplying inverse TPOT by the
requested concurrency would misrepresent this workload.

### Request churn

Qwen3-4B, maximum concurrency 4, prompts up to 8192 tokens, output maximum
128 tokens, one warmup and three iterations, GPU 0 / NUMA node 0. Churn uses
16 requests with matched variable lengths. All measured requests succeeded;
all six offload samples report zero runtime captures and zero recaptures.

| Scenario | Baseline TPOT ms | Offload TPOT ms | Baseline requests/s | Offload requests/s |
| --- | ---: | ---: | ---: | ---: |
| Fixed batch 4 | 7.27 | 29.67 | 3.00 | 0.87 |
| Oversubscribed churn | 9.29 | 36.42 | 3.13 | 0.85 |

Exact full-selected-set traffic grows with concurrent requests. This explicit-KV
configuration has a large regression; freeing KV space does not make it a
throughput optimization on this hardware. The feature remains disabled by
default. `audit.json` in the local bundle verifies successful samples, matched
traces and graph counters for all four paired experiments.

## Optimization evidence and remaining limits

The initial Qwen offload implementation was slower. Limiting copy-kernel
occupancy improved the diagnostic decode graph from about 11.7 ms to 9.7 ms;
the production BenchProbe result is about 10.0 ms at 8K. The bound now applies
to the whole batch. Full GPU layers reuse the existing fused store kernels.

A pinned-host gather microbenchmark on local NUMA memory reached about 40 GB/s.
The installed SGL indexed copy kernel measured a similar bandwidth. Qwen's
28 sparse layers, 2079 historical rows and 4096 bytes per row require about
238 MB of H2D data per request/step. The earliest consumer after each observer
has only a short compute window. Nsight Systems confirms real overlap but also
exposed copy waits; adding a side stream cannot eliminate that dependency.
The report does not interpret sampled activity as theoretical MFU/MBU.

A real-model short-prefix regression exposed a provider-planning constraint:
FA3 derives split counts from index-table width. The offload view now preserves
the original table capacity, while the KV payload remains bounded. This restores
MLA output parity on short history and short-to-long transitions, in eager and
graph mode; a metadata contract regression test covers both table capacities.

An experimental fused score normalizer was rejected after FP16 rounding
mismatches; it is not in the implementation. No token budget, observation
frequency, selection precision or profile was changed to obtain performance
numbers. LRU/miss reuse, added wait layers, or different selection policies
would change the agreed first-stage scope and remain separate work.

Existing model compatibility limits remain. In particular, Qwen3-MoE rejects
OmniKV prefix caching in the base repository; its mixed-parallel validation uses
`--no-prefix`. The active MLA offload path does add the required split-layout
prefix transfer support; GPU-only MLA prefix offload still uses its existing
unsupported storage boundary.
