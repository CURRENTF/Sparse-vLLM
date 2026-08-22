# Validation Ladder

Validate the declared contract in increasing cost order. Stop on failure; do not continue to quality or performance runs with a known semantic mismatch.

## 1. Static And Focused Unit Checks

- Run syntax/type/lint checks used by the touched modules.
- Test public-name normalization, aliases, conflicting/legacy parameter rejection, and invalid values.
- Test every new registry capability and unsupported combination.
- Test capacity, reservation, and prefill-mode calculations with boundary values.
- Test selection and view construction against small deterministic tensors.
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
