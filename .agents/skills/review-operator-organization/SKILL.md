---
name: review-operator-organization
description: Review Sparse-vLLM operator architecture, provider selection, platform capability boundaries, kernel ownership, dependency compatibility, weight layouts, fallback semantics, and validation. Use for diffs touching src/sparsevllm/operators, src/sparsevllm/platforms, Triton or external kernels, model-to-operator call sites, quantized weight loading, CUDA Graph constraints, optional kernel dependencies, or backend removal and migration.
---

# Review Operator Organization

Review the complete selection path, not an isolated kernel:

```text
model semantic call
  -> OpSpec
  -> OpResolver(DeviceCaps)
  -> selected OperatorProvider
  -> weight preparation / workspace
  -> local or external kernel
```

Use the design in `dev_docs/plan/features/kernels.md` as additional context
when it exists, but judge the executable code and tests.

## Inspect the Change

Read all changed and directly coupled files in these areas:

- `src/sparsevllm/operators/`: specs, providers, registries, and resolvers.
- `src/sparsevllm/platforms/`: platform discovery and `DeviceCaps`.
- `src/sparsevllm/triton_kernel/`: repository-owned implementations.
- Model and loader call sites that construct specs or prepare physical weights.
- Dependency declarations and installation documentation for external providers.
- Resolver, kernel-equivalence, integration, and model tests.

Trace at least one supported configuration and every changed rejection or
fallback path from model construction through execution.

## Enforce the Boundaries

### Model and Operator

- Keep models expressed in stable operator semantics. Flag imports of concrete
  providers, external kernel packages, architecture checks, or backend names.
- Keep implementation choices out of `OpSpec`; fields describe dtype, shape,
  quantization, topology, graph, workspace, and mutation requirements.
- Do not add model-level or global backend configuration such as
  `moe_backend`. Tests and microbenchmarks may force a provider through an
  internal interface.

### Platform and DeviceCaps

- Treat `DeviceCaps` as an immutable snapshot produced and cached by the active
  `Platform`, never as a second platform abstraction.
- Keep device discovery, streams, synchronization, memory, and communication
  operations in `Platform`.
- Keep only stable selection facts in `DeviceCaps`. Do not place provider
  factories or optional-library availability there.
- Flag direct `torch.cuda` or architecture probing in common model/operator
  code when the platform can supply the fact.
- Treat ROCm or NPU placeholders as unsupported until an implementation and
  tests exist; a TODO must not advertise support.

### Provider and Resolver

- Require `supports(spec, caps)` to cover every condition needed by execution:
  platform, architecture, dtype, quantization format, shape alignment,
  TP/EP topology, CUDA Graph behavior, workspace, toolchain, and external API
  availability.
- Check optional dependencies lazily inside their provider. An unselected
  provider must not import or initialize a heavyweight dependency.
- Resolve once before weight preparation and bind the provider outside the
  forward hot path.
- Require deterministic priority among fully supported providers and useful
  rejection reasons when none is usable.
- Allow fallback only during resolution or preparation: reject the unsupported
  provider and select another provider for that operator. Once execution has
  begun, do not catch a kernel failure and silently switch implementations.
- Ensure one provider's build or JIT failure does not disable unrelated
  operators.

### Kernel Portfolio

- Prefer a verified, broadly applicable Triton provider as the production
  baseline.
- Give specialized FlashInfer, CUTLASS, or architecture-specific providers
  higher priority only under exact support conditions.
- Keep Torch or naive implementations as explicit correctness oracles, not
  automatic production candidates.
- Do not introduce Hub-downloaded or frozen kernels as implicit runtime
  dependencies.
- Keep kernel-specific layout conversions and workspaces behind the provider
  interface.

### Weights and Layouts

- Let checkpoint loading identify logical projections and scales.
- Let the selected provider own physical packing, ordering, shuffling,
  padding, and scale layout.
- Flag `[gate, up]` versus `[up, gate]` choices or provider-specific offsets in
  model classes.
- Resolve before finalizing weights, tag or otherwise preserve layout
  ownership, and prohibit provider switching without re-preparing weights.
- Flag repeated transpose, repack, or dequantization work in forward.

## Verify External API Compatibility

For every added external API call:

1. Find the declared minimum package version in `pyproject.toml` and docs.
2. Inspect the function signature and required artifacts at that exact minimum
   version, not only in the currently installed environment.
3. Verify every keyword, enum, dtype, layout, and return contract used by the
   provider.
4. Keep core Python/JIT-cache minimums compatible. Keep optional cubin
   packages out of required dependencies unless execution truly requires them.
5. Add a regression check when a parameter was introduced after an older
   supported release.

Treat a legal dependency environment that crashes on provider invocation as
P1 even when the developer environment has a newer working version.

## Require Validation

Match validation to the changed selection surface:

- Resolver tests: supported and rejected architectures, dtype/shape variants,
  dependency absent or too old, toolchain unavailable, graph mode, and TP/EP.
- Failure tests: aggregate actionable rejection reasons and prove fallback is
  local to the affected operator.
- Kernel tests: compare against an independent Torch oracle over boundary
  lengths, non-contiguous layouts, state mutation, padding, real model shapes,
  and every claimed dtype.
- Hardware tests: exercise each claimed architecture on an idle permitted
  device. Do not generalize an SM90 result to other architectures.
- Integration tests: run the actual model path so provider binding, weight
  layout, prefill/decode routing, and generation are covered.
- Quality checks: use the model chat template and deterministic decoding;
  non-empty output alone is not a quality assertion.
- Reproducibility: record selected providers, capability summary, dependency
  versions, model, command, device, and test result.

Do not accept a performance result before correctness equivalence. Do not
accept mocked provider-selection tests as proof that the minimum external
package version can execute the real call.

## Classify Findings

- P0: wrong output, corrupt state/cache, or incompatible physical weight
  layout that invalidates inference.
- P1: a legal environment crashes, the resolver selects an unsupported
  provider, runtime silently changes providers, or model semantics depend on a
  backend layout.
- P2: missing rejection coverage, incomplete observability, hot-path selection
  overhead, or unverified claimed hardware/shape support.
- P3: naming or documentation clarity that does not change execution.

Report the broken boundary, the concrete supported configuration that exposes
it, and the smallest architectural correction.
