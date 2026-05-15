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
- add an explicit `omnikv_decode_cuda_graph` path for fixed-batch greedy OmniKV
  decode on TP=1:
  - engine warmup pre-captures `run_model + argmax` for `max_decoding_seqs`
  - decode steps update stable-address `input_ids`, `positions`,
    `slot_mapping`, `context_lens`, and `req_indices` buffers before graph
    replay
  - graph keepalive retains captured sparse-view tensors so later real prefill
    cannot invalidate graph addresses
  - ordinary HF/Sparse-VLLM logits comparison still uses the non-graph
    `run_model()` path

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
- using `torch.empty` instead of `torch.full(-1e20)` for OmniKV decode
  attention-score buffers: logits stayed aligned, but 128k bs4 decode
  regressed and was reverted
- aggressive OmniKV ablation with `full_attention_layers="0"`: reduced TTFT
  and kept decode argmax aligned in the smoke case, but logit drift increased
  and 128k bs4 decode still reached only `147.18 tok/s`
- fixed-view CUDA Graph replay of OmniKV `run_model()` after a completed 128k
  bs4 prefill: a feasibility probe reduced model-only decode time from
  `26.72 ms` to `9.80 ms`, but it was not a correct generation loop because
  token ids, positions, slot mappings, sparse views, cache state, and sampling
  metadata were captured from one static step
- graph decode `BLOCK_SEQ=512`: still slower than 256 on the 128k bs4 graph
  path (`335.67 tok/s` at out512), so the retained path keeps 256

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
  --output_len <64|512|2048|4096> \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"tensor_parallel_size":1,"max_num_seqs_in_batch":4,"max_decoding_seqs":4,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,4,7,14","chunk_prefill_accel_omnikv":true,"mlp_chunk_size":16384,"throughput_log_interval_s":0.0}'
```

For the retained graph run, OmniKV additionally sets:

```json
{"omnikv_decode_cuda_graph": true}
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
| empty decode score rejected | vanilla | 128000 | 4 | 148.08 | 27.01 | same-run baseline | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/decode_score_empty_20260516_050852/vanilla_omnikv_128k_bs4_out64.log` |
| empty decode score rejected | omnikv | 128000 | 4 | 124.99 | 32.00 | reverted after regression | same as above |
| aggressive full0 ablation | omnikv | 128000 | 4 | 147.18 | 27.18 | `full_attention_layers="0"`, not the paper/default path | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/aggressive_full0_20260516_051516/omnikv_128k_bs4_out64.log` |
| fixed-view graph probe | omnikv | 128000 | 4 | model-only 408.3 est. | 9.80 model ms | not a correct generation loop | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/cudagraph_feasibility_20260516_051814/summary.json` |
| graph no prewarm | omnikv | 128000 | 4 | 305.68 | 13.09 | out512; first decode includes warmup/capture | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/graph_128k_bs4_out512_20260516_053030/omnikv_128k_bs4_out512_graph.log` |
| graph no prewarm | omnikv | 128000 | 4 | 368.45 | 10.86 | out2048; capture amortized but still below target | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/graph_128k_bs4_out2048_20260516_053200/omnikv_128k_bs4_out2048_graph.log` |
| graph no prewarm | omnikv | 128000 | 4 | 375.55 | 10.65 | out4096; below same-length 2.5x target | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/graph_128k_bs4_out4096_20260516_053332/omnikv_128k_bs4_out4096_graph.log` |
| same-length baseline | vanilla | 128000 | 4 | 153.92 | 25.99 | out4096 full-attention baseline | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/vanilla_128k_bs4_out4096_20260516_053523/vanilla_128k_bs4_out4096.log` |
| retained graph prewarm | omnikv | 128000 | 4 | 395.72 | 10.11 | out4096; warmup pre-captures graph, formal decode is replay-only | `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/graph_prewarm_128k_bs4_out4096_20260516_054817/omnikv_128k_bs4_out4096_graph_prewarm.log` |

Same-length target for the final retained run:

- Vanilla full-attention baseline, out4096: `153.92 tok/s`
- Required `2.5x`: `153.92 * 2.5 = 384.80 tok/s`
- Retained OmniKV graph-prewarm run: `395.72 tok/s`
- Final speedup: `395.72 / 153.92 = 2.57x`

The earlier eager retained path reached only `142.10 tok/s` at out64, or
`0.93x` of full attention. The short-context dense probe reached only
`153.71 tok/s`, so sparse attention changes alone were insufficient; the
successful path is reducing per-token launch overhead with a fixed-batch CUDA
Graph while preserving OmniKV's paper/default layer routing.

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

The rejected empty decode-score buffer variant was checked before reverting:

- Result file:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_decode_score_empty_20260516_050807/long_omnikv.json`
- Decode result: `max_abs_diff=0.28125`,
  `mean_abs_diff=0.030360523611307144`, `argmax_match=true`
- Rejected because the corresponding 128k bs4 throughput run regressed to
  `124.99 tok/s` for OmniKV.

The aggressive `full_attention_layers="0"` ablation was checked as a
diagnostic, not as the retained/paper path:

- Result file:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_aggressive_full0_20260516_051430/long_omnikv.json`
- Decode result: `max_abs_diff=1.375`,
  `mean_abs_diff=0.21411097049713135`, `argmax_match=true`
- top-k overlap: top-1 `1.0`, top-5 `0.8`, top-10 `0.9`,
  top-50 `0.94`
- Throughput result: `147.18 tok/s`, still below the retained full-attention
  baseline `152.04 tok/s`.

After adding the graph-prewarm path, the non-graph logits comparison was rerun
against the unchanged HF OmniKV backend:

- Result file:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_graph_branch_20260516_055109/long_omnikv.json`
- Decode result: `max_abs_diff=0.28125`,
  `mean_abs_diff=0.030360523611307144`, `argmax_match=true`
- top-k overlap: top-1 `1.0`, top-5 `1.0`, top-10 `0.9`,
  top-50 `1.0`

Graph replay was also compared against the eager Sparse-VLLM OmniKV greedy
token path in separate processes:

- Compare result:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/graph_vs_eager_tokens_separate_20260516_055008/compare.json`
- Config: 2048-token prompts, bs4, `max_tokens=32`, `decode_keep_tokens=256`,
  `prefill_keep_tokens=256`, `sink_keep_tokens=8`, `recent_keep_tokens=64`,
  `full_attention_layers="0,1,2,4,7,14"`
- Result: exact token match between graph and eager Sparse-VLLM OmniKV

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

CUDA Graph feasibility probe:

- Output:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/cudagraph_feasibility_20260516_051814/summary.json`
- Config: OmniKV, 128k context, bs4, `output_len=3` setup to reach decode
  state, paper/default `full_attention_layers="0,1,2,4,7,14"`,
  `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`,
  `chunk_prefill_accel_omnikv=True`, `mlp_chunk_size=16384`
- Result: eager model-only decode `26.7199 ms`; fixed-view graph replay
  `9.7963 ms`; prefill completed in 32 chunked steps
- Interpretation: graph capture has enough launch-overhead headroom for the
  target, but the probe intentionally reused one captured decode view. A
  retained implementation must use stable-address decode metadata buffers and
  update their contents before graph replay so each generated token observes
  the correct ids, positions, cache slots, and sparse decode views.

## Verification

Commands run during this iteration:

```bash
conda run -n svllm python -m py_compile \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/engine/cache_manager/standard.py \
  src/sparsevllm/engine/llm_engine.py \
  src/sparsevllm/engine/model_runner.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/config.py \
  src/sparsevllm/layers/attention.py \
  src/sparsevllm/triton_kernel/omnikv_fused.py \
  src/sparsevllm/utils/profiler.py \
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
- unittest discover: 72 tests, OK

Additional focused checks:

- `python -m py_compile src/sparsevllm/engine/sparse_controller.py`: passed
- `python -m py_compile src/sparsevllm/layers/attention.py`: passed
- graph-vs-eager Sparse-VLLM token smoke:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/graph_vs_eager_tokens_separate_20260516_055008/compare.json`,
  `match=true`
- HF logits smoke:
  `/data2/haojitai/outputs/Sparse-vLLM/omnikv_decode_128k_bs4/logits_smoke_graph_branch_20260516_055109/long_omnikv.json`,
  decode `argmax_match=true`
- `python -m unittest tests.test_mlp_chunking tests.test_sampler`: passed
- `git diff --check`: passed
