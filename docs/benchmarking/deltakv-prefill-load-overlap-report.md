# DeltaKV Prefill Load Overlap Report

## 2026-07-05 detailed chunk-index analysis

- Status: completed from existing artifacts; no new exact sweep was launched because GPUs 4-7 were already occupied during this follow-up.
- Goal: measure whether one-layer chunk prefill attention compute can cover one-layer historical raw-KV H2D load, and show how the answer changes by chunk index.
- Script: `scripts/profiling/bench_prefill_load_overlap.py`
- Artifact root: `/data2/haojitai/outputs/Sparse-vLLM/prefill_load_overlap_20260705`
- Source files:
  - `refine_small_gpu5.csv`
  - `refine_small_chunks_long_n_gpu5.csv`
  - `gpu5_c8k_16k.csv`
  - `gpu6_c32k.csv`
  - `gpu7_c64k_128k.csv`
- Hardware: local H100 80GB. The original measurements used GPU5, GPU6, and GPU7. During this follow-up check, GPU4 had about 52.9 GiB allocated, GPU5 about 13.8 GiB, GPU6 about 21.5 GiB, and GPU7 about 40.4 GiB, so this report does not add new runs.
- Shape: Qwen2.5-7B GQA attention, `q_heads=28`, `kv_heads=4`, `head_dim=128`, `bf16`.
- Definition: `history n` is the number of historical tokens loaded before the current prefill chunk. For chunk index `i`, `history_n = (i - 1) * chunk_size`.
- Load timing: one layer of historical K+V copied from pinned CPU memory to GPU.
- Attention timing: one FlashAttention call with `q_len=chunk_size`, `kv_len=history_n + chunk_size`, `causal=True`.

Rows marked `exact` are directly present in the CSV artifacts. Rows marked `interp a-b` are linearly interpolated between adjacent measured history lengths `a` and `b`; they should be used for trend inspection, not as final exact benchmark values.

## Chunk-index table

| chunk | index | history n | source | load ms | attn ms | attn/load | cover? |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 512 | 1 | 0 | exact | 0.000 | 0.074 | inf | yes |
| 512 | 2 | 512 | exact | 0.030 | 0.074 | 2.48 | yes |
| 512 | 4 | 1536 | interp 1024-2048 | 0.068 | 0.100 | 1.47 | yes |
| 512 | 8 | 3584 | interp 2048-4096 | 0.144 | 0.149 | 1.04 | yes |
| 512 | 10 | 4608 | interp 4096-8192 | 0.181 | 0.175 | 0.96 | no |
| 512 | 14 | 6656 | interp 4096-8192 | 0.257 | 0.227 | 0.88 | no |
| 512 | final@900k | 900000 | exact | 34.55 | 23.17 | 0.67 | no |
| 1024 | 1 | 0 | exact | 0.000 | 0.089 | inf | yes |
| 1024 | 2 | 1024 | exact | 0.049 | 0.132 | 2.72 | yes |
| 1024 | 4 | 3072 | interp 2048-4096 | 0.125 | 0.223 | 1.79 | yes |
| 1024 | 8 | 7168 | interp 4096-8192 | 0.276 | 0.399 | 1.44 | yes |
| 1024 | 10 | 9216 | interp 8192-65536 | 0.353 | 0.486 | 1.38 | yes |
| 1024 | 14 | 13312 | interp 8192-65536 | 0.508 | 0.659 | 1.30 | yes |
| 1024 | final@900k | 900000 | exact | 34.51 | 38.05 | 1.10 | yes |
| 2048 | 1 | 0 | exact | 0.000 | 0.172 | inf | yes |
| 2048 | 2 | 2048 | exact | 0.087 | 0.358 | 4.10 | yes |
| 2048 | 4 | 6144 | interp 4096-8192 | 0.238 | 0.707 | 2.97 | yes |
| 2048 | 8 | 14336 | interp 8192-65536 | 0.543 | 1.389 | 2.56 | yes |
| 2048 | 10 | 18432 | interp 8192-65536 | 0.695 | 1.726 | 2.48 | yes |
| 2048 | 14 | 26624 | interp 8192-65536 | 0.999 | 2.401 | 2.40 | yes |
| 2048 | final@900k | 900000 | exact | 34.64 | 75.31 | 2.17 | yes |
| 4096 | 1 | 0 | exact | 0.000 | 0.470 | inf | yes |
| 4096 | 2 | 4096 | exact | 0.163 | 1.114 | 6.85 | yes |
| 4096 | 4 | 12288 | interp 8192-65536 | 0.466 | 2.312 | 4.96 | yes |
| 4096 | 8 | 28672 | interp 8192-65536 | 1.076 | 4.668 | 4.34 | yes |
| 4096 | 10 | 36864 | interp 8192-65536 | 1.381 | 5.846 | 4.23 | yes |
| 4096 | 14 | 53248 | interp 8192-65536 | 1.991 | 8.202 | 4.12 | yes |
| 4096 | final@900k | 900000 | exact | 34.74 | 134.68 | 3.88 | yes |
| 8192 | 1 | 0 | exact | 0.000 | 1.437 | inf | yes |
| 8192 | 2 | 8192 | exact | 0.314 | 3.898 | 12.41 | yes |
| 8192 | 4 | 24576 | interp 16384-32768 | 0.922 | 8.620 | 9.35 | yes |
| 8192 | 8 | 57344 | interp 32768-65536 | 2.143 | 18.175 | 8.48 | yes |
| 8192 | 10 | 73728 | interp 65536-131072 | 2.755 | 23.047 | 8.37 | yes |
| 8192 | 14 | 106496 | interp 65536-131072 | 3.979 | 32.945 | 8.28 | yes |
| 8192 | final@900k | 900000 | exact | 34.09 | 269.83 | 7.91 | yes |
| 16384 | 1 | 0 | exact | 0.000 | 5.091 | inf | yes |
| 16384 | 2 | 16384 | exact | 0.616 | 14.419 | 23.39 | yes |
| 16384 | 4 | 49152 | interp 32768-65536 | 1.832 | 34.729 | 18.96 | yes |
| 16384 | 8 | 114688 | interp 65536-131072 | 4.301 | 79.239 | 18.43 | yes |
| 16384 | 10 | 147456 | interp 131072-262144 | 5.538 | 100.410 | 18.13 | yes |
| 16384 | 14 | 212992 | interp 131072-262144 | 8.010 | 139.675 | 17.44 | yes |
| 16384 | final@900k | 900000 | exact | 34.72 | 594.47 | 17.12 | yes |
| 32768 | 1 | 0 | exact | 0.000 | 19.556 | inf | yes |
| 32768 | 2 | 32768 | exact | 1.333 | 61.613 | 46.22 | yes |
| 32768 | 4 | 98304 | interp 65536-131072 | 3.929 | 146.392 | 37.26 | yes |
| 32768 | 8 | 229376 | interp 131072-262144 | 8.948 | 317.394 | 35.47 | yes |
| 32768 | 10 | 294912 | interp 262144-524288 | 11.562 | 407.231 | 35.22 | yes |
| 32768 | 14 | 425984 | interp 262144-524288 | 16.977 | 593.274 | 34.95 | yes |
| 32768 | final@900k | 900000 | exact | 35.35 | 1219.37 | 34.50 | yes |
| 65536 | 1 | 0 | exact | 0.000 | 78.763 | inf | yes |
| 65536 | 2 | 65536 | exact | 2.433 | 250.825 | 103.11 | yes |
| 65536 | 4 | 196608 | interp 131072-262144 | 7.273 | 617.919 | 84.96 | yes |
| 65536 | 8 | 458752 | interp 262144-524288 | 16.974 | 1337.342 | 78.79 | yes |
| 65536 | 10 | 589824 | interp 524288-786432 | 21.919 | 1690.201 | 77.11 | yes |
| 65536 | 14 | 851968 | interp 786432-900000 | 31.675 | 2399.687 | 75.76 | yes |
| 65536 | final@900k | 900000 | exact | 33.29 | 2532.44 | 76.08 | yes |

## Interpretation

The first chunk and later chunks are materially different because the amount of historical KV to load grows with `history_n`. The attention cost also grows with `history_n`, but its slope depends strongly on `chunk_size`.

Small chunks are the danger zone:

- `chunk_size=512`: load begins to exceed attention around chunk 10 in this data, and at `history_n=900k` load is about 34.55 ms while attention is about 23.17 ms. H2D cannot be fully hidden.
- `chunk_size=1024`: the long-context endpoint is only marginal, about 38.05 ms attention vs 34.51 ms load. It can cover in this microbench, but leaves little slack for real implementation overhead.

Practical large chunks have substantial slack:

- `chunk_size=2048`: at `history_n=900k`, attention is about 2.17x load.
- `chunk_size=8192`: at `history_n=900k`, attention is about 7.91x load.
- `chunk_size=32768`: at chunk 2, attention is already about 46.22x load; at chunk 10 it is about 35.22x; at `history_n=900k` it is about 34.50x.
- `chunk_size=65536`: at chunk 2, attention is about 103.11x load; at chunk 10 it is about 77.11x; at `history_n=900k` it is about 76.08x.

Therefore the theoretical overlap argument is valid for the chunk sizes we actually care about (`32768` and `65536`). The observed end-to-end prefetch slowdown is not explained by raw H2D bandwidth. It is more likely caused by implementation-side serialization, such as synchronous CPU prefix reassembly, staging-copy allocation, stream synchronization, or extra Python/runtime overhead before the attention backend can consume the prefetched buffer.

## Caveats

- This is a single-layer microbenchmark. It does not include all vLLM scheduler overhead, DeltaKV final compression, Python hook overhead, allocator behavior, or full-model layer scheduling.
- Interpolated rows are for shape intuition. Exact validation for chunk index 10 and 14 at every chunk size should be run when GPUs 4-7 are idle.
- A real overlap implementation still needs proof from end-to-end profiling: H2D should be issued on a separate CUDA stream with pinned memory and `non_blocking=True`, and the consumer stream should only wait immediately before the attention backend needs the buffer.

## 2026-07-06 current prefetch-path diagnosis

- Status: completed.
- Goal: explain why `SPARSEVLLM_RAWKV_PREFETCH=1` was slower end-to-end even though raw H2D load is smaller than chunk attention compute.
- Script: `scripts/profiling/bench_rawkv_prefetch_path.py`
- Launcher: `scripts/tmp/run_rawkv_prefetch_path_profile_20260706.sh`
- Run root: `/data2/haojitai/outputs/Sparse-vLLM/rawkv_prefetch_path_20260706/full_gpu5_20260706_140334`
- GPU: `CUDA_VISIBLE_DEVICES=5`; GPU5 was idle before launch.
- Code: `f13a37a22d619decfaf95a1b38baa0a281b89bdd`; worktree had the new profiling script uncommitted.
- Shape: Qwen2.5-7B GQA raw KV, `kv_heads=4`, `head_dim=128`, `bf16`; storage chunk size `65536`.

The benchmark separates five costs:

- `cpu_reassembly_ms`: `RawKVOffloadBuffer.get_prefix_cpu()` on chunked storage. This allocates a new pinned CPU contiguous prefix and copies all historical CPU chunks into it.
- `current_prefetch_ms`: the historical `_deltakv_schedule_next_long_prefill_offload_prefetch()` pattern at the time of this run: call `get_prefix_cpu()`, allocate temporary GPU tensors, copy CPU prefix to GPU on a prefetch stream, then wait for measurement.
- `restore_prefix_ms`: direct `RawKVOffloadBuffer.restore_prefix()` from CPU chunks into GPU output tensors, without first constructing a contiguous CPU prefix.
- `ideal_h2d_ms`: copy from an already contiguous pinned CPU K+V prefix to GPU.
- `attn_ms`: one FlashAttention chunk call in the same harness.

| history n | chunk | CPU reassembly ms | current prefetch ms | restore_prefix ms | ideal H2D ms | attn ms | current/ideal | attn/current |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 32768 | 0.85 | 4.73 | 2.58 | 2.50 | 92.03 | 1.90 | 19.44 |
| 65536 | 65536 | 0.85 | 4.73 | 2.58 | 2.50 | 216.19 | 1.90 | 45.66 |
| 294912 | 32768 | 15.50 | 30.61 | 11.85 | 10.99 | 344.44 | 2.78 | 11.25 |
| 294912 | 65536 | 15.50 | 30.61 | 11.85 | 10.99 | 719.67 | 2.78 | 23.51 |
| 589824 | 32768 | 30.05 | 56.50 | 23.43 | 22.38 | 669.54 | 2.52 | 11.85 |
| 589824 | 65536 | 30.05 | 56.50 | 23.43 | 22.38 | 1363.04 | 2.52 | 24.13 |
| 900000 | 32768 | 43.33 | 87.86 | 34.25 | 33.38 | 1011.92 | 2.63 | 11.52 |
| 900000 | 65536 | 43.33 | 87.86 | 34.25 | 33.38 | 2039.90 | 2.63 | 23.22 |

Diagnosis:

The slowdown is not caused by raw PCIe/H2D bandwidth. At `history_n=900000`, direct chunked `restore_prefix()` is `34.25 ms`, essentially the same as ideal contiguous H2D at `33.38 ms`. The current prefetch path is `87.86 ms`, about `2.63x` ideal, because it first runs `get_prefix_cpu()` and spends `43.33 ms` constructing a new contiguous pinned CPU prefix. That CPU reassembly happens before the CUDA prefetch stream is issued, so it cannot overlap with current-layer FlashAttention.

The code path responsible is:

- Historical `src/sparsevllm/engine/cache_manager/deltakv_less_memory.py::_deltakv_schedule_next_long_prefill_offload_prefetch()`: called `raw_kv_offload_buffer.get_prefix_cpu()` before entering `torch.cuda.stream(stream)`.
- `src/sparsevllm/engine/cache_manager/raw_kv_offload.py::get_prefix_cpu()`: for chunked mode, allocates `k_out`/`v_out` pinned CPU tensors and copies every chunk into that contiguous prefix.

There is a second-order cost: the prefetched GPU tensors are temporary. When the layer is consumed, `before_prefill_layer_attention()` still copies `k_hist`/`v_hist` into `deltakv_prefill_staging_*` and, for sparse layers, reruns RoPE before attention. The prefetch therefore does not directly populate the final attention staging buffers.

Recommended fix direction:

Avoid `get_prefix_cpu()` in the prefetch path. Either copy CPU chunks directly into the destination GPU prefix tensor on the prefetch stream, or prefetch directly into the final staging cache slice that attention will read. The consumer should wait on a CUDA event only immediately before the final staging buffer is needed.

## 2026-07-06 direct-stage prefetch fix validation

- Status: completed.
- Code change: `RawKVOffloadBuffer.copy_prefix_to()` copies chunked CPU backing chunks directly into caller-provided destination tensors. DeltaKV long-prefill offload prefetch now schedules the nearest future layer after the current layer attention has finished, copies directly into the shared final staging slices on a CUDA stream, records an event, and the future layer only waits for that event before consuming the staging buffer.
- Safety constraint: direct staging uses the shared `deltakv_prefill_staging_kv_cache`, so only one future layer can be prefetched safely. The production scheduler always chooses the nearest next offload layer to avoid layer N+2 overwriting layer N+1's staged prefix.
- Profiling artifact: `/data2/haojitai/outputs/Sparse-vLLM/rawkv_prefetch_path_20260706/direct_stage_after_fix_gpu5.jsonl`
- End-to-end smoke artifact: `/data2/haojitai/outputs/Sparse-vLLM/rawkv_prefetch_path_20260706/e2e_smoke_direct_stage_gpu5_20260706_141531`

Single-point 900k validation on GPU5:

| history n | chunk | old current prefetch ms | direct-stage prefetch ms | ideal H2D ms | direct/ideal | attn/direct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900000 | 32768 | 206.04 | 33.53 | 33.31 | 1.01 | 30.16 |
| 900000 | 65536 | 206.04 | 33.53 | 33.31 | 1.01 | 60.84 |

The old current-prefetch number in this after-fix harness still runs the historical `get_prefix_cpu()` pattern for comparison. Its absolute value can vary with allocator and CPU-memory state, but the direct-stage path is stable: it is effectively identical to ideal pinned H2D and removes the CPU contiguous-prefix reassembly from the prefetch critical path.

End-to-end smoke:

```bash
CUDA_VISIBLE_DEVICES=5 \
SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS=1 \
SPARSEVLLM_RAWKV_PREFETCH=1 \
SPARSEVLLM_RAWKV_BUFFER_MODE=chunked \
PYTHONPATH=$PWD:$PWD/src \
TOKENIZERS_PARALLELISM=false \
/home/haojitai/miniconda3/envs/svllm/bin/python benchmark/microbench.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --methods deltakv \
  --lengths 2048 \
  --batch_sizes 1 \
  --output_len 1 \
  --hyper_params '{"deltakv_checkpoint_path":"/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor","gpu_memory_utilization":0.75,"engine_prefill_chunk_size":1024,"max_num_seqs_in_batch":1,"max_decoding_seqs":1,"decode_cuda_graph":false,"enforce_eager":true,"full_attention_layers":"0,2,4,11,16,22","enable_full_layer_kivi_quant":true,"full_layer_kv_quant_bits":4,"deltakv_latent_quant_bits":4,"deltakv_center_ratio":0.1}'
```

Result: `status=SUCCESS`, `ttft=0.431s`, `prefill_tp=4749.23 tok/s`, peak memory `15.69 GB`. This smoke forces the long-prefill offload path on a short 2048-token prompt by setting `SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS=1`; it validates the control flow but is not a long-context throughput result.

## 2026-07-06 DeltaKV 900k max-batch sweep

- Status: completed; max successful batch size is `3`, first failing batch size is `4`.
- Goal: measure the largest batch size that can run at `ctxlen=900000` after the direct-stage raw-KV prefetch fix.
- Launcher: `scripts/tmp/run_deltakv_900k_max_bs_sweep_20260706.sh`
- Run root: `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5`
- Summary: `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5/summary.json`
- Log: `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5/run.log`
- GPU: `CUDA_VISIBLE_DEVICES=5`; GPU5 was idle before launch. GPU4 had existing memory use, and GPU6/GPU7 were busy, so they were not used for this sweep.
- Code: branch `codex/deltakv-varlen-vram`, commit `f13a37a22d619decfaf95a1b38baa0a281b89bdd`, with uncommitted direct-stage prefetch and profiling changes.
- Model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- Compressor: `/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- Key runtime knobs at the time of this pre-merge artifact: `vllm_sparse_method=deltakv`, `length=900000`, `output_len=1`, `chunk_prefill_size=65536`, `max_num_batched_tokens=131072`, the historical third prefill-policy name used during development, `gpu_memory_utilization=0.9`, `SPARSEVLLM_RAWKV_BUFFER_MODE=chunked`, `SPARSEVLLM_RAWKV_PREFETCH=1`, full layers `0,2,4,11,16,22`, KIVI full-layer quant enabled, latent/full KV quant bits `4`. Current code uses public `prefill_schedule_policy=long_bs1full_short_batch` plus the cache-manager `requires_long_prefill_offload()` hook for the same long-context offload behavior.

| bs | status | TTFT s | prefill tok/s | benchmark mem GB | allocated tensors GiB | result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | SUCCESS | 633.26 | 1421.22 | 39.74 | 16.24 | `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5/bs1/result.jsonl` |
| 2 | SUCCESS | 633.29 | 1373.91 | 52.72 | 29.23 | `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5/bs2/result.jsonl` |
| 3 | SUCCESS | 633.36 | 1328.97 | 65.71 | 42.21 | `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5/bs3/result.jsonl` |
| 4 | FAILED | - | - | - | - | `/data2/haojitai/outputs/Sparse-vLLM/deltakv_900k_max_bs_direct_stage_20260706_gpu5/bs4/result.jsonl` |

The `bs=4` run passed cache allocation but failed during prefill in `DeltaKVCacheManager.before_prefill_layer_attention()` while applying RoPE to the staged K prefix:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.25 GiB. GPU 0 had 918.25 MiB free.
```

The failing line was `apply_rotary_emb(k_normed, cos, sin)`, whose implementation builds a concatenated post-rope tensor. This makes `bs=4` a runtime activation failure, not an admission-time cache-allocation failure.

## 2026-07-06 RawKV load path cleanup

- Status: implementation updated after the max-batch sweep.
- Goal: remove the remaining runtime load inefficiencies now that direct-stage prefetch is validated.
- Code paths:
  - `RawKVOffloadBuffer.copy_prefix_to()` is now the only production RawKV load primitive.
  - The legacy `RawKVOffloadBuffer.get_prefix_cpu()` API was removed from production code. The profiling script keeps a local legacy helper only to reproduce historical measurements.
  - The old miss fallback `restore_prefix() -> temporary GPU tensor -> staging copy` was removed from production code.
  - A prefetch miss now synchronously calls `copy_prefix_to()` into the final staging tensors, then continues with the existing sparse-layer RoPE step when needed.
- Runtime default: long-prefill raw-KV prefetch is enabled by default. `SPARSEVLLM_RAWKV_PREFETCH=0` remains as an explicit debug opt-out, but normal runs no longer need to set `SPARSEVLLM_RAWKV_PREFETCH=1`.
- Depth behavior: direct staging still uses one shared final staging buffer, so only the nearest next offload layer is prefetched safely.

Expected effect:

- Prefetch hit path: async CPU-chunk to final-staging copy plus one event wait.
- Prefetch miss path: synchronous CPU-chunk to final-staging copy, with no temporary GPU prefix allocation and no second staging copy.
- Layer-0 non-first chunks can still miss, but their miss path is now the same direct-stage copy primitive.

Smoke validation:

```bash
CUDA_VISIBLE_DEVICES=5 \
SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS=1 \
SPARSEVLLM_RAWKV_BUFFER_MODE=chunked \
PYTHONPATH=$PWD:$PWD/src \
TOKENIZERS_PARALLELISM=false \
/home/haojitai/miniconda3/envs/svllm/bin/python benchmark/microbench.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --methods deltakv \
  --lengths 2048 \
  --batch_sizes 1 \
  --output_len 1 \
  --hyper_params '{"deltakv_checkpoint_path":"/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor","gpu_memory_utilization":0.75,"engine_prefill_chunk_size":1024,"max_num_seqs_in_batch":1,"max_decoding_seqs":1,"decode_cuda_graph":false,"enforce_eager":true,"full_attention_layers":"0,2,4,11,16,22","enable_full_layer_kivi_quant":true,"full_layer_kv_quant_bits":4,"deltakv_latent_quant_bits":4,"deltakv_center_ratio":0.1}'
```

Artifact: `/data2/haojitai/outputs/Sparse-vLLM/rawkv_prefetch_path_20260706/direct_stage_miss_cleanup_smoke_gpu5_20260706_155330/run.log`

Result: `status=SUCCESS`, `ttft=0.19s`, `prefill_tp=10861.25 tok/s`, peak memory `15.69 GB`. The command intentionally does not set `SPARSEVLLM_RAWKV_PREFETCH=1`; it validates the default-enabled prefetch path and direct-stage miss cleanup on a short forced long-prefill offload smoke, not long-context throughput.

## 2026-07-06 policy-merge smoke

- Status: completed.
- Goal: validate that the public DeltaKV policy is now `long_bs1full_short_batch` and the ultra-long chunked path is selected through the cache-manager `requires_long_prefill_offload()` hook.
- GPU: `CUDA_VISIBLE_DEVICES=7`; GPU7 was idle before launch.
- Artifact: `/data2/haojitai/outputs/Sparse-vLLM/rawkv_prefetch_path_20260706/policy_merge_smoke_gpu7_20260706_161844/run.log`
- Command used `SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS=1`, `SPARSEVLLM_RAWKV_BUFFER_MODE=chunked`, and did not set the removed historical third policy name.

Result: `status=SUCCESS`, resolved config `prefill_schedule_policy='long_bs1full_short_batch'`, `ttft=0.18s`, `prefill_tp=11439.98 tok/s`, peak memory `15.69 GB`. This is a short forced-offload smoke, not a long-context throughput result.

## 2026-07-06 PyramidKV policy-merge smoke

- Status: completed after one failed attempt.
- Goal: validate that PyramidKV can use the same public `long_bs1full_short_batch` policy with the cache-manager `requires_long_prefill_offload()` hook, restoring historical RawKV into staging across prefill chunks and materializing the selected final KV on the last chunk.
- GPU: `CUDA_VISIBLE_DEVICES=7`; GPU7 was idle before launch.
- Successful artifact: `/data2/haojitai/outputs/Sparse-vLLM/pyramidkv_long_prefill_offload_smoke_20260706_164417_gpu7/run.log`
- Failed warmup artifact before the score-collection fix: `/data2/haojitai/outputs/Sparse-vLLM/pyramidkv_long_prefill_offload_smoke_20260706_164313_gpu7/run.log`
- Command used `SPARSEVLLM_LONG_PREFILL_OFFLOAD_MIN_TOKENS=1`, `SPARSEVLLM_RAWKV_BUFFER_MODE=chunked`, `--methods pyramidkv`, `--lengths 8192`, `--batch_sizes 1`, `--output_len 1`, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=false`, and `enforce_eager=true`.

Result: `status=SUCCESS`, resolved config `prefill_schedule_policy='long_bs1full_short_batch'`, `ttft=1.60s`, `prefill_tp=5129.17 tok/s`, peak memory `59.27 GB`. This is a short forced-offload smoke that exercises PyramidKV chunked staging and final materialization; it is not a long-context throughput result.
