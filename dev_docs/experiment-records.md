# Experiment Records

### 2026-05-16 20:50 CST - non-deltakv-128k-bs4-decode-cuda-graph

- Status: completed
- Goal: Benchmark overall Sparse-vLLM throughput at 128k context, batch size 4, with `decode_cuda_graph=true` for all current non-DeltaKV methods.
- Working dir: `<PROJECT_ROOT>`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods vanilla,streamingllm,snapkv,pyramidkv,quest,omnikv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":true,"decode_cuda_graph_capture_sizes":4,"decode_cuda_graph_capture_sampling":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0"}'
```

- Code: `perf/omnikv-decode-128k-bs4` / `774a71e`; worktree has relevant uncommitted CUDA graph support changes.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1.
- Data: synthetic prompt token ids, 4 requests, 128000 prompt tokens each, greedy decode for 128 output tokens each.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, Qwen2.5 7B, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=128, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `quest_token_budget=4096`, `full_attention_layers="0"`.
- Logs: `<OUTPUT_ROOT>/sparsevllm_128k_bs4_cuda_graph_non_deltakv_20260516_2050/bench.log`; run metadata `<OUTPUT_ROOT>/sparsevllm_128k_bs4_cuda_graph_non_deltakv_20260516_2050/run_metadata.txt`.
- Results: source `<OUTPUT_ROOT>/sparsevllm_128k_bs4_cuda_graph_non_deltakv_20260516_2050/bench.log`.

| Method | Len | BS | TTFT(s) | Prefill tok/s | Decode tok/s | ITL(ms) | AvgBS | Mem(GB) | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 128000 | 4 | 57.33 | 8931.5 | 219.2 | 18.25 | 4.0 | 72.39 | 1.00x |
| streamingllm | 128000 | 4 | 57.49 | 8905.8 | 143.6 | 27.86 | 4.0 | 72.65 | 0.66x |
| snapkv | 128000 | 4 | 57.69 | 8875.1 | 370.9 | 10.79 | 4.0 | 72.65 | 1.69x |
| pyramidkv | 128000 | 4 | FAILED | - | - | - | - | - | - |
| quest | 128000 | 4 | 59.24 | 8642.5 | 227.5 | 17.59 | 4.0 | 72.39 | 1.04x |
| omnikv | 128000 | 4 | 17.37 | 29477.5 | 528.5 | 7.57 | 4.0 | 72.39 | 2.41x |

- Notes: DeltaKV methods intentionally excluded because `decode_cuda_graph` currently fail-fasts for DeltaKV family. `pyramidkv` failed before generation: the auto layer-ratio allocation left the smallest layer with only 34297 free slots, below the 128000-token prompt admission requirement.

### 2026-05-16 21:32 CST - non-deltakv-128k-bs4-decode-cuda-graph-rerun

- Status: completed
- Goal: Rerun the same 128k context, batch size 4, non-DeltaKV `decode_cuda_graph=true` throughput benchmark on GPU 5 after verifying the GPU is idle.
- Working dir: `<PROJECT_ROOT>`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods vanilla,streamingllm,snapkv,pyramidkv,quest,omnikv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":true,"decode_cuda_graph_capture_sizes":4,"decode_cuda_graph_capture_sampling":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0"}'
```

- Code: `perf/omnikv-decode-128k-bs4` / `774a71e`; worktree has relevant uncommitted CUDA graph support changes.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1. Pre-run GPU 5 state: 4 MiB used, 0% utilization, no compute apps observed.
- Data: synthetic prompt token ids, 4 requests, 128000 prompt tokens each, greedy decode for 128 output tokens each.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, Qwen2.5 7B, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=128, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `quest_token_budget=4096`, `full_attention_layers="0"`.
- Logs: `<OUTPUT_ROOT>/sparsevllm_128k_bs4_cuda_graph_non_deltakv_rerun_20260516_2103/bench.log`; run metadata `<OUTPUT_ROOT>/sparsevllm_128k_bs4_cuda_graph_non_deltakv_rerun_20260516_2103/run_metadata.txt`.
- Results: source `<OUTPUT_ROOT>/sparsevllm_128k_bs4_cuda_graph_non_deltakv_rerun_20260516_2103/bench.log`.

| Method | Len | BS | TTFT(s) | Prefill tok/s | Decode tok/s | ITL(ms) | AvgBS | Mem(GB) | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 128000 | 4 | 57.17 | 8956.0 | 218.8 | 18.28 | 4.0 | 72.39 | 1.00x |
| streamingllm | 128000 | 4 | 57.48 | 8908.0 | 144.8 | 27.62 | 4.0 | 72.65 | 0.66x |
| snapkv | 128000 | 4 | 57.70 | 8873.7 | 366.7 | 10.91 | 4.0 | 72.65 | 1.68x |
| pyramidkv | 128000 | 4 | FAILED | - | - | - | - | - | - |
| quest | 128000 | 4 | 59.30 | 8634.1 | 226.9 | 17.63 | 4.0 | 72.39 | 1.04x |
| omnikv | 128000 | 4 | 17.37 | 29472.2 | 530.4 | 7.54 | 4.0 | 72.39 | 2.42x |

- Notes: Rerun results match the previous run within small variance. Pre-run GPU 5 was idle: 4 MiB used, 0% utilization, and no compute apps. During the run there was a temporary GPU-idle interval while `torch/_inductor` compile workers were active, then the benchmark resumed on GPU 5. `pyramidkv` failed for the same prompt-admission reason as the previous run: the smallest layer had only 34297 free slots, below the 128000-token prompt admission requirement.

### 2026-05-16 22:11 CST - pyramidkv-128k-bs4-full-prefill-staging-first-smoke

- Status: failed
- Goal: Verify PyramidKV with `long_bs1full_short_batch` plus shared full-prefill staging on 128k context, batch size 4, before adding chunked attention output projection.
- Working dir: `<PROJECT_ROOT>`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods pyramidkv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 32 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":true,"decode_cuda_graph_capture_sizes":4,"decode_cuda_graph_capture_sampling":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0"}'
```

- Code: `perf/omnikv-decode-128k-bs4` / `b0aa540d7812d152fb8fd59b87dcb00a78ffb44d`; worktree had 8 relevant changed paths.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1. Pre-run GPU 5 state: 4 MiB used, 0% utilization, no compute apps observed.
- Data: synthetic prompt token ids, 4 requests, 128000 prompt tokens each, greedy decode for 32 output tokens each.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, resolved as `model_type='qwen2'`, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=32, `prefill_schedule_policy='long_bs1full_short_batch'`, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `full_attention_layers="0"`.
- Logs: `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_staging_smoke_20260516_2211/bench.log`; run metadata `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_staging_smoke_20260516_2211/run_metadata.txt`.
- Results: failed after warmup with CUDA OOM in Qwen2 attention `o_proj` (`src/sparsevllm/layers/linear.py`, `F.linear`) while trying to allocate 876 MiB. PyramidKV admission and staging were active: log recorded `prefill_schedule_policy='long_bs1full_short_batch'` and `prefill_staging_slots=128132`.
- Notes: This failure isolated a full-prefill activation peak, not persistent KV admission. It motivated chunking the attention output projection in the Qwen2/Qwen3 attention modules.

### 2026-05-16 22:15 CST - pyramidkv-128k-bs4-full-prefill-staging-o-proj-chunked

- Status: completed
- Goal: Re-run PyramidKV 128k context, batch size 4 after chunking attention output projection, validating that shared full-prefill staging plus `long_bs1full_short_batch` completes.
- Working dir: `<PROJECT_ROOT>`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods pyramidkv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 32 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":true,"decode_cuda_graph_capture_sizes":4,"decode_cuda_graph_capture_sampling":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0"}'
```

- Code: `perf/omnikv-decode-128k-bs4` / `b0aa540d7812d152fb8fd59b87dcb00a78ffb44d`; worktree had 10 relevant changed paths.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1. Pre-run GPU 5 state: 4 MiB used, 0% utilization, no compute apps observed.
- Data: synthetic prompt token ids, 4 requests, 128000 prompt tokens each, greedy decode for 32 output tokens each.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, resolved as `model_type='qwen2'`, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=32, `prefill_schedule_policy='long_bs1full_short_batch'`, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `full_attention_layers="0"`.
- Logs: `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_staging_smoke_20260516_2215/bench.log`; run metadata `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_staging_smoke_20260516_2215/run_metadata.txt`.
- Results: source `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_staging_smoke_20260516_2215/bench.log`.

| Method | Len | BS | TTFT(s) | Prefill tok/s | Decode tok/s | ITL(ms) | AvgBS | Mem(GB) | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pyramidkv | 128000 | 4 | 21.49 | 6019.6 | 331.2 | 12.08 | 3.9 | 77.86 | 1.00x |

- Notes: Logs confirm the run used `prefill_schedule_policy='long_bs1full_short_batch'` and `prefill_staging_slots=128132`. Runtime throughput logs show long full-prefill steps with `prefill_tokens=128000` and decreasing `prf(L/S)=3/0`, `2/0`, `1/0`, which matches bs1 full-prefill scheduling for long prompts.

### 2026-05-16 22:48 CST - pyramidkv-128k-bs4-output128-full-prefill-staging

- Status: completed
- Goal: Re-run only PyramidKV at 128k context, batch size 4, `output_len=128` so its decode throughput can be compared against the previous same-output-length vanilla baseline without rerunning the full method table.
- Working dir: `<PROJECT_ROOT>`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods pyramidkv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":true,"decode_cuda_graph_capture_sizes":4,"decode_cuda_graph_capture_sampling":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0"}'
```

- Code: `perf/omnikv-decode-128k-bs4` / `50131ee934d68fe8ad2ac410fd92f372839e794e`; worktree clean at run start.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1. Pre-run GPU 5 state: 4 MiB used, 0% utilization, no compute apps observed.
- Data: synthetic prompt token ids, 4 requests, 128000 prompt tokens each, greedy decode for 128 output tokens each.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, resolved as `model_type='qwen2'`, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=128, `prefill_schedule_policy='long_bs1full_short_batch'`, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `full_attention_layers="0"`.
- Logs: `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_output128_20260516_2248/bench.log`; run metadata `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_output128_20260516_2248/run_metadata.txt`.
- Results: source `<OUTPUT_ROOT>/sparsevllm_pyramidkv_128k_bs4_output128_20260516_2248/bench.log`.

| Method | Len | BS | TTFT(s) | Prefill tok/s | Decode tok/s | ITL(ms) | AvgBS | Mem(GB) | Decode speedup vs previous vanilla |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pyramidkv | 128000 | 4 | 21.15 | 6046.2 | 339.5 | 11.78 | 4.0 | 77.86 | 1.55x |

- Notes: Previous same-output-length vanilla baseline is the 2026-05-16 21:32 rerun (`decode_tp=218.8 tok/s`). PyramidKV speedup is `339.52 / 218.8 = 1.5517x`. It is below the previous SnapKV decode throughput (`366.7 tok/s`, `0.93x` of SnapKV) and OmniKV decode throughput (`530.4 tok/s`, `0.64x` of OmniKV).

### 2026-05-16 23:00 CST - omnikv-128k-bs4-full-layers-0-1-2-4-7-14

- Status: completed
- Goal: Re-run only OmniKV at 128k context, batch size 4, `output_len=128` with paper-style full-attention layers `0,1,2,4,7,14`.
- Working dir: `<PROJECT_ROOT>`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --methods omnikv \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":true,"decode_cuda_graph_capture_sizes":4,"decode_cuda_graph_capture_sampling":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1,2,4,7,14"}'
```

- Code: `perf/omnikv-decode-128k-bs4` / `50131ee934d68fe8ad2ac410fd92f372839e794e`; worktree had one docs-only changed path at run start.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1. Pre-run GPU 5 state: 4 MiB used, 0% utilization, no compute apps observed.
- Data: synthetic prompt token ids, 4 requests, 128000 prompt tokens each, greedy decode for 128 output tokens each.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, resolved as `model_type='qwen2'`, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=128, `prefill_schedule_policy='all_chunked'`, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `full_attn_layers=[0,1,2,4,7,14]`, resolved `obs_layer_ids=[2,4,7,14]`.
- Logs: `<OUTPUT_ROOT>/sparsevllm_omnikv_128k_bs4_full_layers_0_1_2_4_7_14_20260516_2300/bench.log`; run metadata `<OUTPUT_ROOT>/sparsevllm_omnikv_128k_bs4_full_layers_0_1_2_4_7_14_20260516_2300/run_metadata.txt`.
- Results: source `<OUTPUT_ROOT>/sparsevllm_omnikv_128k_bs4_full_layers_0_1_2_4_7_14_20260516_2300/bench.log`.

| Method | Len | BS | TTFT(s) | Prefill tok/s | Decode tok/s | ITL(ms) | AvgBS | Mem(GB) | Decode speedup vs previous vanilla |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| omnikv | 128000 | 4 | 28.63 | 17886.9 | 394.3 | 10.14 | 4.0 | 72.61 | 1.80x |

- Notes: Previous same-output-length vanilla baseline is the 2026-05-16 21:32 rerun (`decode_tp=218.8 tok/s`). This OmniKV configuration is `394.29 / 218.8 = 1.8021x` vs vanilla, `1.0752x` vs SnapKV (`366.7 tok/s`), and `0.7434x` of the earlier one-full-layer OmniKV run (`530.4 tok/s`).

### 2026-05-16 23:05 CST - longbench-hotpotqa-sparse-method-correctness

- Status: completed for non-DeltaKV methods; DeltaKV representative run was intentionally stopped and excluded after user requested no DeltaKV test for this pass.
- Goal: Validate that each currently active non-DeltaKV sparse method can produce normal LongBench HotPotQA outputs and that scores are broadly aligned with the known Qwen2.5-7B-1M HotPotQA range.
- Working dir: `<PROJECT_ROOT>`
- Command template:

```bash
CUDA_VISIBLE_DEVICES=5 \
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_LONGBENCH_DATA_DIR=<DATA_ROOT>/LongBench \
PYTHONPATH=<PROJECT_ROOT>/src:${PYTHONPATH:-} \
conda run -n svllm python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-sparsevllm-${method}-hotpotqa \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --ws 1 --batch_size 1 --backend sparsevllm \
  --sparse_method ${method} \
  --task hotpotqa \
  --temperature 0 --top_p 1 --top_k 20 --thinking_mode off \
  --hyper_param '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0"}' \
  --output_root <OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/${method}
```

- OmniKV override: `full_attention_layers="0,1,2,4,7,14"`, resolving to `full_attn_layers=[0,1,2,4,7,14]` and `obs_layer_ids=[2,4,7,14]`.
- Code: `perf/omnikv-decode-128k-bs4` / `50131ee934d68fe8ad2ac410fd92f372839e794e`; worktree had docs-only changes at run start.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1.
- Data: `<DATA_ROOT>/LongBench/data/hotpotqa.jsonl`, 200 examples.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, Sparse-VLLM backend, greedy decode with `temperature=0`, `top_p=1`, `top_k=20`, `batch_size=1`, `thinking_mode=off`.
- Logs: `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/logs/`; status file `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/status.tsv`.
- Results: sources are per-method `result.json` and `hotpotqa.jsonl` under `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/`.

| Method | Status | Rows | Empty pred | HotPotQA score | Output dir |
|---|---:|---:|---:|---:|---|
| vanilla | 0 | 200 | 0 | 60.05 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/vanilla` |
| streamingllm | 0 | 200 | 0 | 42.17 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/streamingllm` |
| snapkv | 0 | 200 | 0 | 59.31 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/snapkv` |
| pyramidkv | 0 | 200 | 0 | 46.94 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/pyramidkv` |
| quest | 0 | 200 | 0 | 59.53 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/quest` |
| omnikv (`chunk_prefill_accel_omnikv=true`) | 0 | 200 | 0 | 54.75 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/omnikv` |
| omnikv (`chunk_prefill_accel_omnikv=false`) | 0 | 200 | 0 | 59.65 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_accel_ablation_20260516_2344/sparsevllm-omnikv-no-prefill-accel` |

- Reference anchors: previous full LongBench DeltaKV HotPotQA record in `dev_docs/code-change-history/prefill-schedule-policy-2026-05-16.md` is `59.69`; local archived `qwen25_7b_deltakv_cr30/result.json` has `hotpotqa=58.35`.
- Notes: All completed methods produced 200 JSONL rows with no empty predictions, so the output/eval path is healthy. The OmniKV correctness score should use `chunk_prefill_accel_omnikv=false`, which is the config default and gives `59.65`, aligned with vanilla, SnapKV, Quest, and the reference range. The `54.75` OmniKV result was caused by explicitly enabling the experimental OmniKV prefill acceleration in this LongBench command and should not be used as the correctness score. StreamingLLM and PyramidKV are lower on HotPotQA under these retention settings, but they completed normally and should be treated as quality differences rather than pipeline failures.

### 2026-05-16 23:40 CST - omnikv-hotpotqa-prefill-accel-ablation

- Status: completed
- Goal: Check whether the low Sparse-VLLM OmniKV HotPotQA score came from enabling OmniKV prefill acceleration, and compare against the HF OmniKV path.
- Working dir: `<PROJECT_ROOT>`
- Commands:

```bash
CUDA_VISIBLE_DEVICES=5 \
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_LONGBENCH_DATA_DIR=<DATA_ROOT>/LongBench \
PYTHONPATH=<PROJECT_ROOT>/src:${PYTHONPATH:-} \
conda run -n svllm python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-hf-omnikv-hotpotqa \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --ws 1 --batch_size 1 --backend hf \
  --sparse_method omnikv \
  --task hotpotqa \
  --temperature 0 --top_p 1 --top_k 20 --thinking_mode off \
  --hyper_param '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1,2,4,7,14"}' \
  --output_root <OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_hf_svllm_omnikv_compare_20260516_2340/hf-omnikv

CUDA_VISIBLE_DEVICES=5 \
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_LONGBENCH_DATA_DIR=<DATA_ROOT>/LongBench \
PYTHONPATH=<PROJECT_ROOT>/src:${PYTHONPATH:-} \
conda run -n svllm python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-sparsevllm-omnikv-no-prefill-accel-hotpotqa \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --ws 1 --batch_size 1 --backend sparsevllm \
  --sparse_method omnikv \
  --task hotpotqa \
  --temperature 0 --top_p 1 --top_k 20 --thinking_mode off \
  --hyper_param '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":false,"full_attention_layers":"0,1,2,4,7,14"}' \
  --output_root <OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_accel_ablation_20260516_2344/sparsevllm-omnikv-no-prefill-accel
```

- Code: `main` / `49920ae6d18c83a45d0c7d043c4a72ad6fca9f90`; worktree clean at run start.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPU `CUDA_VISIBLE_DEVICES=5` (`NVIDIA H100 80GB HBM3`), TP=1.
- Data: `<DATA_ROOT>/LongBench/data/hotpotqa.jsonl`, 200 examples.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, greedy decode with `temperature=0`, `top_p=1`, `top_k=20`, `batch_size=1`, `thinking_mode=off`.
- Results:

| Variant | Rows | Empty pred | HotPotQA score | Output dir |
|---|---:|---:|---:|---|
| HF OmniKV, prefill accel true | 200 | 0 | 59.23 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_hf_svllm_omnikv_compare_20260516_2340/hf-omnikv` |
| Sparse-VLLM OmniKV, prefill accel true | 200 | 0 | 54.75 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_correctness_sparsevllm_20260516_2305/omnikv` |
| Sparse-VLLM OmniKV, prefill accel false | 200 | 0 | 59.65 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_accel_ablation_20260516_2344/sparsevllm-omnikv-no-prefill-accel` |

- Notes: Sparse-VLLM OmniKV aligns with the expected HotPotQA range when `chunk_prefill_accel_omnikv=false`. Later alias-controlled follow-up on 2026-05-17 showed this section's HF row was not chunked as intended: HF ignored `engine_prefill_chunk_size` and retained its default near-full `chunk_prefill_size=100000000`. Use the 2026-05-17 chunk-alignment entry below for the apples-to-apples interpretation.

### 2026-05-17 00:17 CST - omnikv-hotpotqa-prefill-chunk-alignment

- Status: completed
- Goal: Determine whether the OmniKV HotPotQA score gap comes from a Sparse-VLLM implementation mismatch or from different prefill chunking budgets between HF and Sparse-VLLM.
- Working dir: `<PROJECT_ROOT>`
- Commands:

```bash
CUDA_VISIBLE_DEVICES=5 \
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_LONGBENCH_DATA_DIR=<DATA_ROOT>/LongBench \
PYTHONPATH=<PROJECT_ROOT>/src:${PYTHONPATH:-} \
conda run -n svllm python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-hf-omnikv-hf-prefill-chunk4096-hotpotqa \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --ws 1 --batch_size 1 --backend hf \
  --sparse_method omnikv \
  --task hotpotqa \
  --temperature 0 --top_p 1 --top_k 20 --thinking_mode off \
  --hyper_param '{"gpu_memory_utilization":0.9,"hf_prefill_chunk_size":4096,"max_num_batched_tokens":65536,"decode_cuda_graph":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1,2,4,7,14"}' \
  --output_root <OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_chunk_align_20260517_001700/hf-omnikv-hf-prefill-chunk4096

CUDA_VISIBLE_DEVICES=6 \
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_LONGBENCH_DATA_DIR=<DATA_ROOT>/LongBench \
PYTHONPATH=<PROJECT_ROOT>/src:${PYTHONPATH:-} \
conda run -n svllm python -u benchmark/long_bench/pred.py \
  --model qwen25-7b-svllm-omnikv-engine-prefill-chunk32768-hotpotqa \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --ws 1 --batch_size 1 --backend sparsevllm \
  --sparse_method omnikv \
  --task hotpotqa \
  --temperature 0 --top_p 1 --top_k 20 --thinking_mode off \
  --hyper_param '{"gpu_memory_utilization":0.65,"engine_prefill_chunk_size":32768,"max_num_batched_tokens":65536,"decode_cuda_graph":false,"sink_keep_tokens":8,"recent_keep_tokens":128,"decode_keep_tokens":4096,"prefill_keep_tokens":4096,"quest_token_budget":4096,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1,2,4,7,14"}' \
  --output_root <OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_chunk_align_20260517_001700/sparsevllm-omnikv-engine-prefill-chunk32768
```

- Code: `main` / `75b8f80`; worktree clean at run start.
- Environment: `guest-KR6288-X2-A0-R0-00`, conda env `svllm`, GPUs `CUDA_VISIBLE_DEVICES=5,6` (`NVIDIA H100 80GB HBM3`), TP=1.
- Data: `<DATA_ROOT>/LongBench/data/hotpotqa.jsonl`, 200 examples. Tokenized prompt lengths before chat template: max `17655`, so Sparse-VLLM `engine_prefill_chunk_size=32768` covers each HotPotQA prompt in one prefill step.
- Model: `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`, greedy decode with `temperature=0`, `top_p=1`, `top_k=20`, `batch_size=1`, `thinking_mode=off`.
- Logs: `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_chunk_align_20260517_001700/logs/`.
- Results:

| Variant | Resolved prefill chunk | Rows | HotPotQA score | Output dir |
|---|---:|---:|---:|---|
| HF OmniKV, `hf_prefill_chunk_size=4096`, prefill accel true | 4096 | 200 | 55.11 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_chunk_align_20260517_001700/hf-omnikv-hf-prefill-chunk4096` |
| Sparse-VLLM OmniKV, `engine_prefill_chunk_size=32768`, prefill accel true | 32768, full for this dataset | 200 | 59.65 | `<OUTPUT_ROOT>/benchmark/long_bench/hotpotqa_omnikv_prefill_chunk_align_20260517_001700/sparsevllm-omnikv-engine-prefill-chunk32768` |

- Notes: HF with the correct HF-side alias logs `hf_prefill_chunk_size` normalized to `chunk_prefill_size` and then drops to `55.11`, close to Sparse-VLLM's earlier `54.75` when `engine_prefill_chunk_size=4096`. Sparse-VLLM recovers to `59.65` when its prefill chunk budget is increased enough to avoid mid-prompt chunk-boundary compression. The quality gap is therefore caused by chunked OmniKV prefill acceleration semantics, not by `prefill_keep_tokens` or `topk(sorted=True)`.
