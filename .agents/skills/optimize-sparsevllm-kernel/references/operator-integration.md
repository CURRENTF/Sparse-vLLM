# Sparse-vLLM Operator Integration

Read the sibling
[`review-operator-organization`](../../review-operator-organization/SKILL.md)
skill before changing production dispatch. Treat its architecture and
validation rules as authoritative.

## Keep Ownership Explicit

```text
model semantic call
  -> OpSpec
  -> OpResolver(DeviceCaps)
  -> selected OperatorProvider
  -> provider-owned preparation/workspace
  -> Triton, TileLang, JIT, or external kernel
```

- Keep repository-owned Triton kernels under `src/sparsevllm/triton_kernel/`.
- Keep repository-owned TileLang kernels under
  `src/sparsevllm/tilelang_kernel/`.
- Keep support checks, dependency availability, static launch selection,
  workspaces, and provider binding under `src/sparsevllm/operators/`.
- Keep models expressed in semantic operations rather than backend names,
  package imports, device probes, or physical weight layouts.
- Keep device discovery and stable capability facts under the platform layer.

## Resolve Before Execution

Make `supports(spec, caps)` cover platform, architecture, dtype, quantization,
shape and alignment, layouts, topology, graph behavior, workspace, toolchain,
and external API availability. Import optional compilers and kernel packages
lazily. Bind the provider outside the forward hot path.

Allow fallback only by rejecting an unsupported provider during resolution or
preparation. Once execution begins, surface compilation and launch failures.
Do not make one provider failure disable unrelated operators.

## Validate the Complete Path

Add or update:

- independent kernel-equivalence tests
- resolver selection and rejection tests
- missing and minimum-version dependency tests
- boundary shape, dtype, stride, padding, and workspace tests
- CUDA Graph capture/replay tests where supported
- actual model-path integration coverage
- matched performance evidence for every priority change

Record the selected provider and rejection reasons in reproducible artifacts.
After implementation, use `$review-operator-organization` for the focused
architecture review and `$code-review` for the complete diff.
