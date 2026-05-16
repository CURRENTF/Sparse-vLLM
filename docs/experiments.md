# Experiment Records

### 2026-05-16 20:50 CST - non-deltakv-128k-bs4-decode-cuda-graph

- Status: completed
- Goal: Benchmark overall Sparse-vLLM throughput at 128k context, batch size 4, with `decode_cuda_graph=true` for all current non-DeltaKV methods.
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
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
- Model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`, Qwen2.5 7B, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=128, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `quest_token_budget=4096`, `full_attention_layers="0"`.
- Logs: `/data2/haojitai/outputs/sparsevllm_128k_bs4_cuda_graph_non_deltakv_20260516_2050/bench.log`; run metadata `/data2/haojitai/outputs/sparsevllm_128k_bs4_cuda_graph_non_deltakv_20260516_2050/run_metadata.txt`.
- Results: source `/data2/haojitai/outputs/sparsevllm_128k_bs4_cuda_graph_non_deltakv_20260516_2050/bench.log`.

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
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 conda run -n svllm python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
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
- Model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`, Qwen2.5 7B, bf16 config, Sparse-VLLM backend.
- Hyperparameters: bs=4, length=128000, output_len=128, `engine_prefill_chunk_size=4096`, `decode_cuda_graph=true`, capture size 4, `decode_keep_tokens=4096`, `prefill_keep_tokens=4096`, `sink_keep_tokens=8`, `recent_keep_tokens=128`, `quest_token_budget=4096`, `full_attention_layers="0"`.
- Logs: `/data2/haojitai/outputs/sparsevllm_128k_bs4_cuda_graph_non_deltakv_rerun_20260516_2103/bench.log`; run metadata `/data2/haojitai/outputs/sparsevllm_128k_bs4_cuda_graph_non_deltakv_rerun_20260516_2103/run_metadata.txt`.
- Results: source `/data2/haojitai/outputs/sparsevllm_128k_bs4_cuda_graph_non_deltakv_rerun_20260516_2103/bench.log`.

| Method | Len | BS | TTFT(s) | Prefill tok/s | Decode tok/s | ITL(ms) | AvgBS | Mem(GB) | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 128000 | 4 | 57.17 | 8956.0 | 218.8 | 18.28 | 4.0 | 72.39 | 1.00x |
| streamingllm | 128000 | 4 | 57.48 | 8908.0 | 144.8 | 27.62 | 4.0 | 72.65 | 0.66x |
| snapkv | 128000 | 4 | 57.70 | 8873.7 | 366.7 | 10.91 | 4.0 | 72.65 | 1.68x |
| pyramidkv | 128000 | 4 | FAILED | - | - | - | - | - | - |
| quest | 128000 | 4 | 59.30 | 8634.1 | 226.9 | 17.63 | 4.0 | 72.39 | 1.04x |
| omnikv | 128000 | 4 | 17.37 | 29472.2 | 530.4 | 7.54 | 4.0 | 72.39 | 2.42x |

- Notes: Rerun results match the previous run within small variance. Pre-run GPU 5 was idle: 4 MiB used, 0% utilization, and no compute apps. During the run there was a temporary GPU-idle interval while `torch/_inductor` compile workers were active, then the benchmark resumed on GPU 5. `pyramidkv` failed for the same prompt-admission reason as the previous run: the smallest layer had only 34297 free slots, below the 128000-token prompt admission requirement.
