# OmniKV Decode 128k BS4 Optimization - 2026-05-16

## Scope

Branch: `perf/omnikv-decode-128k-bs4`

Goal: optimize Sparse-VLLM OmniKV decode throughput at 128k context and batch
size 4 on local GPU 6, while keeping the HF side unchanged for logits
comparison.

Retained code changes:

- carry CPU-known `max_context_len` through cache-manager batch state to avoid
  per-layer decode `context_lens.max().item()` synchronization
- cap decode `max_context_len` by the actual packed active-slot width so
  method-specific packed views such as QuEST/DeltaKV are not oversized
- vectorize OmniKV top-k extraction for the non-DeltaKV path instead of
  per-batch `.item()` loops
- request unsorted top-k indices for OmniKV because the sparse view only needs
  the selected token set
- allow `build_omnikv_keep_and_slots()` to receive an explicit padded max
  sparse context length
- reuse decode stage1/stage2 workspace buffers within a decode step
- fast-path greedy sampling to avoid building the stochastic sampling path when
  every sequence uses `temperature=0`

Tested but rejected:

- decode score reduced to 2D atomic max: correct enough, but slower
- decode score reduced to per-KV-head max: correct enough, but slower
- sorted decode top-k indices for more chronological KV reads: no throughput
  gain and slightly larger logits drift
- bf16 decode score buffer: correct enough, but slower
- `torch.compile` whole model or decode-only wrapper: failed in Inductor with
  `XBLOCK too large. Maximum: 4096. Actual: 8192`
- sparse decode `BLOCK_SEQ=512`: slower than the existing 256-token block
- swapping the existing compiled RoPE and in-place SiLU paths to the repo's
  Triton kernels: decode logits stayed aligned, but both vanilla and OmniKV
  throughput regressed
- routing RMSNorm through Triton kernels: standalone microbenchmarks looked
  promising, but end-to-end 128k bs4 decode regressed and was reverted
- compiling the isolated Qwen2 MLP with `torch.compile(mode="reduce-overhead")`:
  microbenchmark was effectively neutral (`0.163 ms` eager vs `0.160 ms`
  compiled for BS4), so no code change was made

## Environment

- Host: local `guest-KR6288-X2-A0-R0-00`
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- GPU: `CUDA_VISIBLE_DEVICES=6`, NVIDIA H100 80GB HBM3
- Conda env: `svllm`
- Base model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- Compressor path used by logits script:
  `/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- Code base at start of this optimization branch: `86c9485`
- Output root:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4`

## Main Benchmark Config

Shared command shape:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --methods <vanilla|omnikv> \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"tensor_parallel_size":1,"max_num_seqs_in_batch":4,"max_decoding_seqs":4,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,4,7,14","chunk_prefill_accel_omnikv":true,"mlp_chunk_size":16384,"throughput_log_interval_s":0.0}'
```

Resolved OmniKV parameters:

- `prefill_schedule_policy=all_chunked`
- `num_top_tokens=4096`
- `num_top_tokens_in_prefill=4096`
- `num_sink_tokens=8`
- `num_recent_tokens=128`
- `full_attn_layers=[0,1,2,4,7,14]`
- `chunk_prefill_accel_omnikv=True`

## Results

| Run | Method | Context | BS | Decode tok/s | ITL ms | Notes | Log |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | vanilla | 128000 | 4 | 139.0 | 28.78 | before this branch's perf edits | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/baseline_20260516_034411/baseline_vanilla_omnikv_128k_bs4_out64.log` |
| baseline | omnikv | 128000 | 4 | 117.7 | 34.00 | before this branch's perf edits | same as above |
| after maxlen/topk | omnikv | 128000 | 4 | 139.45 | 28.68 | removed decode max-len sync and vectorized OmniKV top-k | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/after_vector_topk_maxlen_clean_20260516_040810/omnikv_128k_bs4_out64.log` |
| current full baseline | vanilla | 128000 | 4 | 151.71 | 26.37 | same branch after max-len optimization | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/vanilla_after_maxlen_20260516_041107/vanilla_128k_bs4_out64.log` |
| workspace reuse | omnikv | 128000 | 4 | 141.39 | 28.29 | retained workspace reuse | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/after_decode_workspace_reuse_20260516_042539/omnikv_128k_bs4_out64.log` |
| dense lower-bound probe | vanilla | 1024 | 4 | 153.71 | 26.02 | short context, attention cost near-minimal | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/dense_lower_bound_len1024_20260516_042718/vanilla_1k_bs4_out64.log` |
| greedy sampler re-run | vanilla | 128000 | 4 | 152.67 | 26.20 | greedy sampler fast path, fair combined run | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/after_greedy_sampler_20260516_043423/vanilla_omnikv_128k_bs4_out64.log` |
| greedy sampler re-run | omnikv | 128000 | 4 | 138.26 | 28.93 | same run as above | same as above |
| Triton RoPE/SiLU rejected | vanilla | 128000 | 4 | 141.60 | 28.25 | reverted after regression | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/after_triton_rope_silu_20260516_044122/vanilla_omnikv_128k_bs4_out64.log` |
| Triton RoPE/SiLU rejected | omnikv | 128000 | 4 | 133.43 | 29.98 | reverted after regression | same as above |
| final retained | vanilla | 128000 | 4 | 152.04 | 26.31 | retained max-len/top-k/workspace/greedy path | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/final_retained_20260516_045011/vanilla_omnikv_128k_bs4_out64.log` |
| final retained | omnikv | 128000 | 4 | 142.10 | 28.15 | final measured speedup `0.93x` vs vanilla | same as above |
| Triton RMSNorm rejected | vanilla | 128000 | 4 | 139.50 | 28.67 | reverted after regression | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/triton_rmsnorm_20260516_045852/vanilla_omnikv_128k_bs4_out64.log` |
| Triton RMSNorm rejected | omnikv | 128000 | 4 | 135.23 | 29.58 | reverted after regression | same as above |

The target based on the final full-attention baseline is
`152.04 * 2.5 = 380.10 tok/s`. The final retained OmniKV run reaches
`142.10 tok/s`, or `0.93x` of full attention. The short-context dense probe
reaches only `153.71 tok/s`, so the present eager per-layer execution path has
a dense-model launch/linear/lm-head floor far above the 2.5x target. Sparse
attention optimizations alone are therefore insufficient for this target.

## Logits Checks

HF side was not modified in this branch. The retained max-len/top-k
implementation was checked against HF OmniKV on the long case:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src conda run -n svllm \
  python scripts/debug/compare_logits_hf_sparsevllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --compressor_path /data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor \
  --output_dir /data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_vector_topk_20260516_040515 \
  --cases long \
  --methods omnikv \
  --cuda_visible_devices 6 \
  --master_port 29616 \
  --max_model_len 12000 \
  --long_tokens 9000 \
  --decode_keep_tokens 4096 \
  --prefill_keep_tokens 4096 \
  --sink_keep_tokens 8 \
  --recent_keep_tokens 128 \
  --full_attention_layers 0,1,2,4,7,14 \
  --engine_prefill_chunk_size 4096 \
  --chunk_prefill_accel_omnikv \
  --gpu_memory_utilization 0.9 \
  --mlp_chunk_size 16384
```

Result file:

- `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_vector_topk_20260516_040515/long_omnikv.json`

Decode result:

- `max_abs_diff=0.296875`
- `mean_abs_diff=0.030925391241908073`
- `argmax_match=true`
- top-k overlap: top-1 `1.0`, top-5 `1.0`, top-10 `0.9`, top-50 `1.0`

The final retained unsorted-topk path was also checked on the long case:

- Result file:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_topk_unsorted_20260516_044750/long_omnikv.json`
- Decode result: `max_abs_diff=0.28125`,
  `mean_abs_diff=0.030360523611307144`, `argmax_match=true`
- top-k overlap: top-1 `1.0`, top-5 `1.0`, top-10 `0.9`, top-50 `1.0`

The rejected Triton RoPE/SiLU variant was checked before reverting:

- Result file:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_triton_rope_silu_20260516_043920/long_omnikv.json`
- Decode result: `max_abs_diff=0.3125`,
  `mean_abs_diff=0.059485841542482376`, `argmax_match=true`
- Rejected because the corresponding 128k bs4 throughput run regressed to
  `133.43 tok/s` for OmniKV.

The rejected Triton RMSNorm variant was checked before reverting:

- Result file:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_triton_rmsnorm_20260516_045808/long_omnikv.json`
- Decode result: `max_abs_diff=0.5`,
  `mean_abs_diff=0.03933897614479065`, `argmax_match=true`
- Rejected because the corresponding 128k bs4 throughput run regressed to
  `135.23 tok/s` for OmniKV.

## Profiling Notes

High-level synchronized profile after max-len/top-k optimization:

- Log:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/profile_highlevel_after_maxlen_20260516_041236/omnikv_128k_bs4_out16_profile.log`
- `model_run_decode`: `32.8568 ms`
- `model_run_model_decode`: `30.9914 ms`
- `sparse_update_dynamic_indices`: `0.5803 ms` per observation call
- `cache_prepare_decode`: `0.8394 ms`

Earlier temporary per-layer profiling showed that decode time is dominated by
many small eager model kernels plus score-bearing observation attention, not by
Python view construction alone. The short-context dense probe confirms that the
current eager dense model path already sits around 26 ms/step at BS4.

## Verification

Commands run during this iteration:

```bash
conda run -n svllm python -m py_compile \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/engine/cache_manager/standard.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/layers/attention.py \
  src/sparsevllm/triton_kernel/omnikv_fused.py \
  src/sparsevllm/utils/context.py

git diff --check
```

Final validation pass:

```bash
conda run -n svllm python -m compileall -q src tests
conda run -n svllm python -m unittest discover -s tests -p 'test*.py'
```

Result:

- `compileall`: passed
- unittest discover: 70 tests, OK

Additional focused checks:

- `python -m py_compile src/sparsevllm/engine/sparse_controller.py`: passed
- `python -m py_compile src/sparsevllm/layers/attention.py`: passed
- `python -m unittest tests.test_mlp_chunking tests.test_sampler`: passed
- `git diff --check`: passed
