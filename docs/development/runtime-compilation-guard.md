# Runtime compilation guard

`runtime_compilation_limit` defaults to `500`. Set it to `0` to reject every
covered compilation after startup. The option is accepted by `LLM` and the
OpenAI server's engine configuration arguments, like other `Config` fields.
For example, pass `--runtime-compilation-limit 0` to the server or
`runtime_compilation_limit=0` to `LLM(...)` for strict mode.

The engine arms the guard on every model-runner rank after all startup warmup
and CUDA Graph capture has completed, before the engine constructor returns.
Each process has an independent lifetime budget shared across covered backends
and requests. The 501st compilation attempt with the default limit raises
`RuntimeCompilationError` before compilation begins. Allowed attempts log a
warning with rank, count, backend, kernel/module identity and Python stack.
Failed compiler attempts consume budget too; retries do not reset it.

Covered boundaries are Triton's backend `add_stages` (available in both 3.5
and 3.6), TileLang's `JITKernel._compile_and_create_adapter`, and FlashInfer's
module `build` methods. FlashInfer 0.6.15 uses `JitSpec`; newer releases use
`JitSpecNvcc` and, when available, `JitSpecCuteDsl`. The dependency minimums
remain unchanged.

Triton persistent-cache hits bypass `add_stages`. FlashInfer counts module
build entries, not individual source files: an NVCC `build` can invoke Ninja
to check an existing artifact, and that entry still consumes budget even if
Ninja has no work. In-memory module hits and AOT loads bypass the build hook.
Autotuning may consume multiple entries. Existing Triton inspection hooks are
left intact, and backend method patches are restored during runner shutdown.
Backends imported lazily after startup are instrumented when their modules load.

This is not an interception of all compiler activity in the process. Driver
PTX JIT, direct CuTe/NVRTC compilation outside these adapters, and unrelated
PyTorch extension builds are not covered. Upstream changes to the instrumented
APIs must be validated; missing expected APIs fail explicitly. The guard adds
no synchronization or Python hook to cached kernel launches.

Use the reported stack to add representative startup coverage or remove an
unintended dynamic specialization. Increasing the budget only changes the
failure threshold; it does not resolve the source of runtime compilation.

CPU tests exercise the budget, concurrent admission, lazy imports, hook
ownership, error propagation, and real offline Triton compilation/cache lookup.
They do not establish GPU numerical correctness, multi-rank failure recovery,
or end-to-end serving performance.
