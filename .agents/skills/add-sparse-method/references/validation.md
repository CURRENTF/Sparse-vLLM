# Validation Ladder

Validate the declared contract in increasing cost order. Stop on failure; do not continue to quality or performance runs with a known semantic mismatch.

## 1. Static And Focused Unit Checks

- Run syntax/type/lint checks used by the touched modules.
- Test public-name normalization, aliases, conflicting/legacy parameter rejection, and invalid values.
- Test every new registry capability and unsupported combination.
- Test capacity, reservation, and prefill-mode calculations with boundary values.
- Test selection and view construction against small deterministic tensors.
- Test canonical method-to-`SparseMethodRuntime` construction and each lifecycle
  hook whose behavior differs from its runtime base.
- Test storage writes, compaction/reconstruction, and lifecycle state transitions independently of model execution where possible.

Use the repository's focused tests rather than relying on import success or a single end-to-end prompt.

## 2. Correctness Matrix

Build a matrix from the method contract. Include only supported combinations, but add negative tests for every explicitly unsupported one.

Relevant axes include:

- prefill and decode;
- batch size one and mixed request lengths;
- full and partial pages/chunks;
- explicit, heterogeneous, or MLA storage as advertised;
- eager and decode CUDA Graph execution;
- no prefix cache, radix prefix cache, and chain prefix cache;
- allocate, append, fork, restore/rollback, eviction, offload, and free;
- TP/EP/DP layouts that the registry advertises;
- required operator providers and optional dependencies.

Compare selections, logical lengths, physical ownership, cache payloads, and attention outputs with a deterministic reference. Check invariants after every lifecycle transition, not only final logits.

Verify that `SparseController`, Attention, ModelRunner, and Scheduler gained no
method-name hot-path branches, and that persistent prefix-coupled metadata did
not move from CacheManager into runtime state.

### CUDA Graph Replay Checks

For advertised graph support, compare eager execution with actual capture and
repeated replay from equivalent initial state. Change metadata between replays:
exercise growing lengths, relevant page/compaction boundaries, padded batches,
and request completion followed by row/slot reuse. Check cache contents,
selection, ownership, and attention outputs, not just successful capture.

Verify buffer identity/keepalive and that capture warmup/reset does not consume
live capacity or leave stale method state. Confirm that replay reads the new
metadata and padded rows cannot write to another request. Use the declared
numerical tolerance; changing algorithm semantics is not a tolerance adjustment.
If graph support is deferred, test explicit rejection and record the missing
replay cases. A config flag, CPU mock, or graph-configured eager fallback is not
evidence of graph execution; performance artifacts must identify actual replay.

## 3. Model-Level And Quality Validation

- Run a short deterministic generation that exercises prefill and multiple decode steps.
- Compare eager and graph outputs within the provider's documented tolerance.
- Run the method's intended quality evaluation with fixed model, dataset split, prompt, decoding parameters, seed, and sample count.
- Save raw outputs, parsed outputs, per-sample statuses, and aggregate metrics separately.
- Mark every sample as `success`, `invalid_input`, `model_failed`, `parse_failed`, `metric_failed`, or `skipped_by_policy`.

Missing checkpoints, datasets, assets, providers, or API keys are hard failures. Do not substitute data or silently reduce the evaluation set.

## 4. Performance Validation

Follow `docs/en/benchmarking/efficiency.md` or `docs/zh/benchmarking/efficiency.md` exactly.

- Check device idleness before every GPU run and select an idle device.
- Use matched traces, models, cache budgets, batch/request distributions, decoding parameters, and providers.
- Validate benchmark artifacts before interpreting metrics.
- Separate selection, metadata/view construction, cache mutation, and attention/operator time when investigating overhead.
- Report warmup, repetitions, variance, throughput, latency, and memory metrics required by the runbook.
- Use the documented Nsight diagnostic for kernel-timeline attribution; sampled GPU activity is not theoretical MFU/MBU.

Check the pre-implementation cost model against measured scaling. Use a bounded
set of representative context lengths, batch sizes, and retained budgets that
exercises the predicted bottleneck and overhead-dominated regime, not an
exhaustive cross-product. Record selection frequency and page size where they
affect the model. Include peak allocated/reserved device memory with the
measurement scope and graph/workspace residency stated; distinguish fixed model
and KV-pool reservations from method-specific live storage.

Investigate unexpected time or memory growth, repeated launches, synchronization,
and copying before calling the implementation efficient. Keep separately timed
kernel/stage diagnostics separate from unsynchronized request-latency runs, and
include all method work in end-to-end measurements. Validate optimized kernels
and selections against an independent numerical/algorithm oracle; routing mocks
and tests freezing a provider choice do not establish efficiency or correctness.

If hardware or assets prevent measurement, report the implementation and static
cost analysis separately from pending GPU correctness/performance evidence. Do
not claim verified efficiency or complete validation from formulas or CPU tests.

Do not claim a speedup from an unmatched schedule, a changed quality target, a different provider, or an invalid artifact.

## 5. Reproducibility Record

Record:

- exact command and repository revision;
- resolved config and canonical method name;
- model/checkpoint and external asset identifiers;
- dataset and split;
- prompt/template and decoding parameters;
- seed and evaluated sample count;
- device/provider/dependency versions;
- supported capability matrix exercised;
- raw outputs and benchmark/profiler artifacts.

Do not hardcode private local or remote paths in committed test/benchmark scripts. Pass them through arguments or variables.
