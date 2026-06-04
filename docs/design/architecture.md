# Architecture

Sparse-vLLM is split into two runtime families:

- `src/sparsevllm/`: sparse-first inference engine with its own scheduler,
  model runner, cache managers, sparse controller, model definitions, and
  Triton kernels.
- `src/deltakv/`: HF/Transformers-side DeltaKV wrappers, compressor training,
  runtime parameter normalization, and baseline adapters.

## Sparse-vLLM Flow

The Sparse-vLLM inference path is:

1. `sparsevllm.LLM(model, **kwargs)`
2. `src/deltakv/configs/runtime_params.py` normalizes public runtime names.
3. `src/sparsevllm/config.py` validates engine config and method compatibility.
4. `src/sparsevllm/engine/cache_manager/base.py` selects a cache manager.
5. `Scheduler`, `ModelRunner`, `SparseController`, kernels, and the selected
   cache manager execute prefill and decode.

`src/sparsevllm/layers/attention.py` is intentionally generic. It writes K/V,
asks the sparse controller for the logical read view, and lets the cache manager
customize decode-time views through hooks such as `build_decode_view(...)`.

## HF DeltaKV Flow

The HF wrapper path is:

1. `deltakv.get_chat_api.get_generate_api(..., backend="hf")`
2. Runtime parameters are normalized for the HF backend.
3. `sparse_method` selects the HF model wrapper or baseline adapter.
4. The base model config or compressor checkpoint config is loaded.
5. DeltaKV cache implementations are selected by the wrapper config.
6. Generation runs through the repo's custom HF inference helpers.

Use the HF path when comparing against wrapper implementations or baseline
adapters. Use `backend="sparsevllm"` when measuring the sparse-first engine.

## Method Ownership

New Sparse-vLLM sparse methods should follow the cache-manager-first design:

- Method-specific runtime state belongs in
  `src/sparsevllm/engine/cache_manager/<method>.py`.
- Cross-layer observation or scheduling coordination belongs in
  `src/sparsevllm/engine/sparse_controller.py`.
- `src/sparsevllm/layers/attention.py` should only call generic hooks or shared
  kernels.
- Public runtime arguments should use canonical names documented in
  [runtime-parameter-semantics.md](../configuration/runtime-parameter-semantics.md).

When adding a first-class method, use the repo-local `$add-sparse-method` skill.
It encodes the expected file placement, cache-manager hooks, and validation
workflow.

## Scheduling Ownership

Prefill scheduling is part of the method contract. Default policies live in
`src/sparsevllm/method_registry.py`, `Config` resolves and validates the policy,
and `src/sparsevllm/engine/scheduler.py` implements the scheduling behavior.

The engine currently supports:

- `all_chunked`: all prefill requests are chunked and batched through the normal
  scheduler limits.
- `long_bs1full_short_batch`: a special policy for methods that need a complete
  long-prefill representation; long requests run full-prefill with batch size 1,
  while short requests remain chunked and batched.

Do not encode a method's prefill policy in benchmark scripts or one-off config
defaults. Add the method-to-policy mapping to the registry and update
`tests/test_prefill_schedule_policy.py`.

## Important Files

- `src/deltakv/configs/runtime_params.py`: public runtime parameter aliases and
  legacy-key rejection.
- `src/sparsevllm/config.py`: engine config defaults and validation.
- `src/sparsevllm/method_registry.py`: supported method names and prefill policy
  defaults.
- `src/sparsevllm/engine/cache_manager/base.py`: method-to-cache-manager
  routing and shared cache-manager hooks.
- `src/sparsevllm/engine/scheduler.py`: prefill/decode scheduling and admission.
- `src/sparsevllm/layers/attention.py`: generic K/V storage and attention
  compute path.
- `benchmark/`: LongBench, MathBench, SCBench, NIAH, and multimodal benchmark
  entrypoints.
