# DeltaKV RawKV Offload Experiments

## 2026-07-05 - Prefill Compute vs RawKV Load Microbench

- Status: completed.
- Goal: measure when one-layer chunk prefill attention is long enough to cover loading one layer of historical Qwen2.5-7B raw KV from pinned CPU memory.
- Script: `scripts/profiling/bench_prefill_load_overlap.py`
- Output root: `/data2/haojitai/outputs/Sparse-vLLM/prefill_load_overlap_20260705`
- Detailed chunk-index report: `docs/benchmarking/deltakv-prefill-load-overlap-report.md`
- Hardware: local H100 80GB; GPU5/6/7 were used. GPU4 had about 52 GB already allocated and was not used.
- Shape: Qwen2.5-7B GQA attention, `q_heads=28`, `kv_heads=4`, `head_dim=128`, `bf16`; load time is one layer of K+V history; attention time uses FlashAttention with `q_len=chunk_size`, `kv_len=n+chunk_size`, `causal=True`.

| Chunk size | At n=900k: load ms | At n=900k: attention ms | Attention / load | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 512 | 34.55 | 23.17 | 0.67x | Too small to cover full-prefix load at long context. |
| 1024 | 34.51 | 38.05 | 1.10x | Near the long-context crossover. |
| 2048 | 34.64 | 75.31 | 2.17x | Covers load. |
| 4096 | 34.74 | 134.68 | 3.88x | Covers load. |
| 8192 | 34.09 | 269.83 | 7.91x | Covers load comfortably. |
| 16384 | 34.72 | 594.47 | 17.12x | Covers load comfortably. |
| 32768 | 35.35 | 1219.37 | 34.50x | Covers load comfortably. |
| 65536 | 33.29 | 2532.44 | 76.08x | Covers load comfortably. |
| 131072 | 33.24 | 5300.60 | 159.49x | Covers load comfortably. |

Conclusion: for realistic chunk sizes (`>=2048`, and especially the current `32768`/`65536` settings), raw H2D load is much shorter than chunk attention compute on H100. The earlier `prefetch=1` slowdown is therefore not explained by PCIe bandwidth; it points to implementation overhead such as synchronous CPU-side prefix reassembly or extra staging copies before attention. Very small chunks around `512` are the exception: load can dominate at long contexts.

## 2026-07-04 - Qwen2.5-7B-1M 900k Prefill

- Status: the chunked CPU RawKV offload mode meets the active throughput target at 900k context while avoiding the original DeltaKV full-prefill OOM.
- Working dir: `/home/haojitai/projects/Sparse-vLLM-deltakv-varlen-vram`
- Code: `de85dd3923318bd446fd99095c435eed1d18805c` plus uncommitted RawKVOffloadBuffer/deferred-prefill changes.
- Model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- DeltaKV compressor: `/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- Common config: `batch_sizes=1`, `lengths=900000`, `output_len=1`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.9`, `max_num_seqs_in_batch=1`, `max_decoding_seqs=1`, `full_attn_layers=0,2,4,11,16,22`, `enable_full_layer_kivi_quant=true`, `full_layer_kv_quant_bits=4`, `kv_quant_bits=4`, `cluster_ratio=0.1`, `decode_cuda_graph=false`, `enforce_eager=true`.

| Run | Status | Chunk | Method | TTFT | Prefill tok/s | Peak memory | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_prefill_after_fix_900k_chunk32k_20260704_gpu5` | completed | 32768 | DeltaKV | 805.52s | 1117.29 | 40.30 GB | CPU RawKV offload; below target. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_prefill_after_fix_900k_chunk64k_20260704_gpu5` | completed | 65536 | DeltaKV | 710.06s | 1267.51 | 41.24 GB | Best completed DeltaKV run so far; 82.7% of vanilla. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_chunked_900k_chunk64k_20260704_gpu5` | completed | 65536 | DeltaKV | 639.29s | 1407.82 | 41.24 GB | `SPARSEVLLM_RAWKV_BUFFER_MODE=chunked`; 91.86% of vanilla. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_final_900k_chunk64k_20260704_gpu7` | completed | 65536 | DeltaKV | 637.62s | 1411.50 | 41.24 GB | Final post-quality-fix run; 92.10% of vanilla. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_chunk256k_900k_20260704_gpu5` | completed | 262144 | DeltaKV | 747.50s | 1204.03 | 48.06 GB | Larger chunk did not improve throughput. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_prefetch_d1_900k_chunk64k_20260704_gpu6` | completed | 65536 | DeltaKV | 857.10s | 1050.05 | 42.74 GB | CPU prefetch depth 1 made throughput worse; 68.51% of vanilla. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_prefetch_d2_900k_chunk64k_20260704_gpu5` | completed | 65536 | DeltaKV | 794.41s | 1132.92 | 44.34 GB | CPU prefetch depth 2 made throughput worse. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_postrope_900k_chunk64k_20260704_gpu5` | aborted | 65536 | DeltaKV | n/a | n/a | n/a | Sparse post-RoPE CPU store was slower than baseline and was stopped. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_directfull_900k_chunk64k_20260704_gpu5` | aborted | 65536 | DeltaKV | n/a | n/a | n/a | Full-layer direct restore did not finish faster than baseline and was stopped. |
| `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_prefill_after_fix_900k_chunk32k_20260704_gpu5` | completed | 32768 | Vanilla | 587.22s | 1532.65 | 72.99 GB | Vanilla comparison; 90% target is 1379.39 tok/s. |

Conclusion: chunked CPU RawKV storage is required for the 900k target. Contiguous pinned CPU backing store spends too much time on large per-layer allocations and writes; with chunked storage, DeltaKV reaches 1411.50 tok/s, above the 1379.39 tok/s target. Current CPU prefetch is not a useful optimization: depth 1 drops to 1050.05 tok/s and depth 2 drops to 1132.92 tok/s while increasing memory. The end-to-end run still shows long non-token intervals from deferred restore/finalize work, so future tuning should profile those phases instead of only increasing chunk size.

## 2026-07-04 - LongBench Quality Regression

- Baseline run: `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_offload_baseline_full_20260704_gpu5/sparsevllm_regression/deltakv_rawkv_offload_baseline_full_20260704_gpu5_quality`
- Failed intermediate run: `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_after_chunked_full_20260704_gpu7/sparsevllm_regression/deltakv_rawkv_after_chunked_full_20260704_gpu7_quality`
- Final run: `/data2/haojitai/outputs/Sparse-vLLM/deltakv_rawkv_after_chunked_fix_quality_20260704_gpu7/sparsevllm_regression/deltakv_rawkv_after_chunked_fix_quality_20260704_gpu7_quality`

| Run | Status | Vanilla | DeltaKV | Delta vs baseline DeltaKV | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline full quality | completed | 58.28 | 57.96 | 0.00 | Pre-change reference. |
| Intermediate chunked-deferred quality | failed quality gate | 58.28 | 9.38 | -48.58 | Deferred policy accidentally chunked ordinary LongBench prompts. |
| Final quality after deferred threshold fix | completed | 58.28 | 57.96 | 0.00 | Restores old full-prefill staging for prompts below 262144 tokens. |

Correctness conclusion: the raw-KV deferred path must be gated to truly long prompts. The default threshold is `SPARSEVLLM_DEFERRED_PREFILL_MIN_TOKENS=262144`; prompts below that keep the legacy DeltaKV full-prefill staging semantics, while 900k prompts still use chunked deferred prefill.
