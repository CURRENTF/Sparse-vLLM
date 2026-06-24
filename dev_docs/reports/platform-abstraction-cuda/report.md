# Platform Abstraction CUDA Report

Date: 2026-06-23

Branch: `codex/platform-abstraction-cuda`

## Scope

This change implements the first CUDA-equivalent platform abstraction pass from
`dev_docs/plan/platform-abstraction-plan.md`. It keeps sparse method state owned
by cache managers and `SparseController`, and only moves device/runtime concerns
behind a lightweight platform boundary.

Implemented:

- Added `src/sparsevllm/platforms/` with a lazy `current_platform`, `Platform`
  interface, CUDA implementation, ROCm recognizer that fails fast until a real
  backend exists, and explicit CPU platform for import/unit-test paths.
- Routed `ModelRunner` device binding, torch distributed backend, TP barrier
  device ids, sample tensor transfer, device sync, and decode graph runner
  selection through `current_platform`.
- Added device-neutral config aliases while preserving legacy fields:
  `decode_graph`, `decode_graph_capture_sampling`,
  `decode_graph_capture_sizes`, `omnikv_decode_graph`, and
  `device_memory_utilization`.
- Added `CacheManager.platform` and `CacheManager.device`; migrated cache
  manager memory queries and normal runtime tensor allocation to those fields
  across standard, SnapKV/PyramidKV, QuEST, OmniKV via standard, DeltaKV,
  DeltaKV standalone, DeltaKV-SnapKV, and DeltaKV delta-quant managers.
- Routed `SparseController` temporary tensors through the cache manager device.
  The OmniKV fused op import is now lazy, so importing the controller no longer
  imports that Triton module.
- Updated CUDA graph static tensors to allocate on the cache manager device and
  synchronize through the platform. CUDA graph capture remains a CUDA-specific
  runner.
- Updated profiler synchronization to use the platform and added
  `SPARSEVLLM_SYNC_DEVICE=1` while preserving `CUDA_SYNC_SVLLM=1`.
- Updated DeltaKV compressor rebuild logic to place rebuilt modules on the cache
  manager device.

Not included in this first pass:

- Attention/op backend registry migration. `layers/attention.py` still owns the
  CUDA/Triton attention path.
- Full ROCm/NPU inference support. ROCm is detected through PyTorch HIP, but
  the current build raises immediately instead of inheriting CUDA capabilities.
- CPU inference. `SPARSEVLLM_PLATFORM=cpu` is for import/unit tests only.

## Validation

Static checks:

- Passed: `.venv/bin/python -m py_compile ...` for all touched Python files.
- Passed: `uv run --no-sync --with pytest python -m pytest tests/test_platforms.py tests/test_prefill_schedule_policy.py tests/test_tp_rpc.py`
  - Result: `32 passed`.
- Note: plain `uv run` attempted a project sync and failed building
  `flash-attn==2.8.3` under build isolation because `torch` is not declared as a
  build dependency by `flash-attn`. No global pip/conda path was used.

GPU availability before runs:

- `nvidia-smi` showed GPUs 0-7 idle, 0 MiB used, 0% utilization.
- CUDA tests used Qwen2.5 model files from
  `/data2/guquansheng/models/Qwen2.5-7B-Instruct-1M`.
- DeltaKV checkpoint smoke used
  `/data2/guquansheng/models/Qwen2.5-7B-Instruct-1M-Compressor`.

CUDA smoke artifacts were generated locally during validation, but raw
benchmark outputs are not committed. This report keeps only the summary and
reproduction commands.

Summary:

| Run | Methods | Result |
| --- | --- | --- |
| `smoke_sparse_families` | vanilla, streamingllm, snapkv, pyramidkv, omnikv, quest, deltakv-standalone, deltakv-snapkv, deltakv-delta-quant | 8 success, deltakv-delta-quant `model_failed` without required OmniKV chunk accel flag |
| `smoke_deltakv_delta_quant` | vanilla, deltakv-delta-quant | success with explicit `chunk_prefill_accel_omnikv=true` |
| `smoke_deltakv_triton_v4` | vanilla, deltakv-triton-v4 | success with compressor checkpoint |
| `smoke_decode_graph` | vanilla, omnikv with `decode_graph=true` and capture sampling | success |
| `smoke_tp2` | vanilla TP=2 | success |
| `smoke_tp2_snapkv_blocking` | snapkv TP=2 with `CUDA_LAUNCH_BLOCKING=1` | failed in existing Triton decode kernel |
| `main_compare_snapkv_tp2` | snapkv TP=2 on `origin/main` | same failure as this branch |

The SnapKV TP=2 failure is not introduced by this change. The same command on a
temporary `origin/main` worktree fails in `src/sparsevllm/triton_kernel/flash_decoding_stage2.py`
with `Triton Error [CUDA]: an illegal memory access was encountered`. This is a
pre-existing CUDA/Triton TP decode issue and remains a residual risk.

## Repro Commands

The most important commands used:

```bash
.venv/bin/python -m py_compile \
  src/sparsevllm/platforms/interface.py \
  src/sparsevllm/platforms/cuda.py \
  src/sparsevllm/platforms/rocm.py \
  src/sparsevllm/platforms/cpu.py \
  src/sparsevllm/platforms/device_runtime.py \
  src/sparsevllm/platforms/__init__.py \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/engine/cache_manager/standard.py \
  src/sparsevllm/engine/cache_manager/snapkv.py \
  src/sparsevllm/engine/cache_manager/quest.py \
  src/sparsevllm/engine/cache_manager/deltakv.py \
  src/sparsevllm/engine/cache_manager/deltakv_delta_quant.py \
  src/sparsevllm/engine/cache_manager/deltakv_snapkv.py \
  src/sparsevllm/engine/cache_manager/deltakv_standalone.py \
  src/sparsevllm/engine/model_runner.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/engine/decode_cuda_graph.py \
  src/sparsevllm/utils/profiler.py \
  src/sparsevllm/utils/loader.py \
  tests/test_platforms.py \
  tests/test_prefill_schedule_policy.py \
  tests/test_tp_rpc.py
```

```bash
uv run --no-sync --with pytest python -m pytest \
  tests/test_platforms.py \
  tests/test_prefill_schedule_policy.py \
  tests/test_tp_rpc.py
```

Example CUDA smoke:

```bash
CUDA_VISIBLE_DEVICES=0 SPARSEVLLM_PLATFORM=cuda \
.venv/bin/python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/guquansheng/models/Qwen2.5-7B-Instruct-1M \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv-standalone,deltakv-snapkv \
  --lengths 128 \
  --batch_sizes 1 \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.55,"engine_prefill_chunk_size":128,"max_num_batched_tokens":512,"max_num_seqs_in_batch":1,"max_decoding_seqs":1,"decode_keep_tokens":32,"prefill_keep_tokens":64,"sink_keep_tokens":4,"recent_keep_tokens":16,"full_attention_layers":"0","observation_layers":[0],"throughput_log_interval_s":0}'
```

## Notes

- The implementation intentionally does not move sparse method state into the
  platform layer. Method-specific metadata remains in cache managers, matching
  the repo-local `$add-sparse-method` guardrails.
- Platform discovery is lazy. Importing `sparsevllm.platforms` does not probe
  hardware until `current_platform` or `get_current_platform()` is accessed.
- Explicit platform selection uses `SPARSEVLLM_PLATFORM=cuda|rocm|cpu` or a
  selected `sparsevllm.platforms` entry point.
