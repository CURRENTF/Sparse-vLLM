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
