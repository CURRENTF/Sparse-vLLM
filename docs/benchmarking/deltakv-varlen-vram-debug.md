# DeltaKV Varlen VRAM Debug

This page records a targeted DeltaKV/Sparse-vLLM memory probe. It is not a
quality benchmark.

## 2026-07-04 GPU7 Tensor-Space Probe

- Working directory:
  `/home/haojitai/projects/Sparse-vLLM-deltakv-varlen-vram`
- Branch and commit: `codex/deltakv-varlen-vram`, `de85dd3`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`
- Compressor:
  `/data2/haojitai/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor`
- Synthetic input: repeated token id `100`
- Output root:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_varlen_vram_debug_20260704_141323`
- Results:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_varlen_vram_debug_20260704_141323/results.jsonl`
- Tensor-space table:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_varlen_vram_debug_20260704_141323/tensor_space_tables.md`
- Log:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_varlen_vram_debug_20260704_141323/run.log`
- Exit code: `0`

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD:$PWD/src TOKENIZERS_PARALLELISM=false \
  /home/haojitai/miniconda3/envs/svllm/bin/python \
  scripts/tmp/debug_deltakv_varlen_vram.py \
  --model-path /data2/haojitai/models/Qwen3-4B-Instruct-2507 \
  --compressor-path /data2/haojitai/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor \
  --output-jsonl "$RUN_ROOT/results.jsonl" \
  --short-len 1024 \
  --long-len 8192 \
  --output-len 4 \
  --engine-prefill-chunk-size 4096 \
  --gpu-memory-utilization 0.75 \
  --deltakv-latent-quant-bits 0 \
  --full-layer-kv-quant-bits 0 \
  --enable-full-layer-kivi-quant 1 \
  --max-steps 64
```

Summary:

| case | status | prompt | max_model_len | peak GiB | persistent GiB | savings |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| short_prompt_small_cap | success | 1024 | 1156 | 10.10 | 1.33 | -0.038 |
| short_prompt_large_cap | success | 1024 | 8360 | 13.50 | 1.81 | -0.262 |
| long_prompt_large_cap | success | 8192 | 8360 | 16.30 | 1.81 | -0.262 |

Persistent tensor-space totals:

| category | short small-cap MiB | short large-cap MiB | long large-cap MiB |
| --- | ---: | ---: | ---: |
| kv_or_latent | 1365.16 | 1846.46 | 1846.49 |
| slot_map | 1.12 | 5.53 | 5.63 |
| other | 0.04 | 0.03 | 0.04 |
| metadata | 0.00 | 0.00 | 0.09 |
| scale_min_metadata | 0.00 | 0.00 | 0.00 |

Interpretation:

With the same `max_model_len=8360`, the 1024-token and 8192-token prompts have
the same persistent DeltaKV tensor allocation: about `1.81` GiB. The peak GPU
memory changes because runtime prefill work changes, but the cache tensor space
does not shrink with actual prompt length. Reducing the capacity knob to
`max_model_len=1156` drops persistent allocation to about `1.33` GiB, so the
dominant allocation follows configured capacity rather than varlen input length.

## 2026-07-04 GPU7 Max-Model-Len 1e6 Tensor-Space Probe

- Status: completed
- Working directory:
  `/home/haojitai/projects/Sparse-vLLM-deltakv-varlen-vram`
- Branch and commit: `codex/deltakv-varlen-vram`, `de85dd3`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`
- Compressor:
  `/data2/haojitai/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor`
- Output root:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_max_model_len_1e6_tensor_probe_20260704_141849`
- Results:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_max_model_len_1e6_tensor_probe_20260704_141849/results.jsonl`
- Sorted tensor table:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_max_model_len_1e6_tensor_probe_20260704_141849/tensor_space_gb_sorted.md`
- Log:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_max_model_len_1e6_tensor_probe_20260704_141849/run.log`
- Exit code: `0`

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD:$PWD/src TOKENIZERS_PARALLELISM=false \
  /home/haojitai/miniconda3/envs/svllm/bin/python \
  scripts/tmp/debug_deltakv_varlen_vram.py \
  --model-path /data2/haojitai/models/Qwen3-4B-Instruct-2507 \
  --compressor-path /data2/haojitai/checkpoints/compressor/Qwen3-4B-Instruct-2507-Compressor \
  --output-jsonl "$RUN_ROOT/results.jsonl" \
  --single-max-model-len 1000000 \
  --init-only \
  --short-len 1024 \
  --output-len 4 \
  --engine-prefill-chunk-size 4096 \
  --gpu-memory-utilization 0.75 \
  --deltakv-latent-quant-bits 0 \
  --full-layer-kv-quant-bits 0 \
  --enable-full-layer-kivi-quant 1 \
  --max-steps 64
```

Summary:

| case | status | max_model_len | peak GiB | allocated GiB | observed_savings | tensor_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| max_model_len_1000000 | success | 1000000 | 60.04 | 50.22 | 0.524 | 33 |

Category totals:

| category | GiB | GB |
| --- | ---: | ---: |
| kv_or_latent | 49.770 | 53.440 |
| slot_map | 0.447 | 0.480 |
| metadata | 0.000 | 0.000 |
| other | 0.000 | 0.000 |
| scale_min_metadata | 0.000 | 0.000 |

Runtime adjusted `max_num_batched_tokens` from `1000000` to `109243` to avoid
OOM, but `max_model_len` remained `1000000`. The dominant tensors were
`deltakv_latent_cache` at `25.648` GiB, `deltakv_full_kv_cache` at `14.575`
GiB, and the max-model-len-sized prefill staging workspaces at `3.815` GiB and
`1.907` GiB.

## 2026-07-04 GPU7 Qwen2.5-7B-1M Max-Model-Len 1e6 Tensor-Space Probe

- Status: completed
- Working directory:
  `/home/haojitai/projects/Sparse-vLLM-deltakv-varlen-vram`
- Branch and commit: `codex/deltakv-varlen-vram`, `de85dd3`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Model: `/data2/haojitai/models/Qwen2.5-7B-Instruct-1M`
- Compressor:
  `/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor`
- Output root:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_qwen25_7b_1m_max_model_len_1e6_tensor_probe_20260704_142734`
- Results:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_qwen25_7b_1m_max_model_len_1e6_tensor_probe_20260704_142734/results.jsonl`
- Sorted tensor table:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_qwen25_7b_1m_max_model_len_1e6_tensor_probe_20260704_142734/tensor_space_gb_sorted.md`
- Log:
  `/data2/haojitai/outputs/Sparse-vLLM/deltakv_qwen25_7b_1m_max_model_len_1e6_tensor_probe_20260704_142734/run.log`
- Exit code: `0`

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD:$PWD/src TOKENIZERS_PARALLELISM=false \
  /home/haojitai/miniconda3/envs/svllm/bin/python \
  scripts/tmp/debug_deltakv_varlen_vram.py \
  --model-path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --compressor-path /data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor \
  --output-jsonl "$RUN_ROOT/results.jsonl" \
  --single-max-model-len 1000000 \
  --init-only \
  --short-len 1024 \
  --output-len 4 \
  --engine-prefill-chunk-size 4096 \
  --gpu-memory-utilization 0.75 \
  --deltakv-latent-quant-bits 0 \
  --full-layer-kv-quant-bits 0 \
  --enable-full-layer-kivi-quant 1 \
  --max-steps 64
```

Summary:

| case | status | max_model_len | peak GiB | allocated GiB | allocated GB | observed_savings | tensor_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| max_model_len_1000000 | success | 1000000 | 43.78 | 27.64 | 29.68 | 0.507 | 31 |

Category totals:

| category | GiB | GB |
| --- | ---: | ---: |
| kv_or_latent | 27.164 | 29.167 |
| slot_map | 0.473 | 0.508 |
| metadata | 0.000 | 0.000 |
| other | 0.000 | 0.000 |
| scale_min_metadata | 0.000 | 0.000 |

Runtime adjusted `max_num_batched_tokens` from `1000000` to `56098` to avoid
OOM, but `max_model_len` remained `1000000`. The checkpoint synced
`kv_compressed_size` from `128` to `256`. The dominant tensors were
`deltakv_latent_cache` at `13.518` GiB, `deltakv_full_kv_cache` at `8.198`
GiB, and `full_kv_cache` at `2.003` GiB.

## 2026-07-04 900k bs1 Vanilla-vs-DeltaKV OOM Diagnosis

- Status: analyzed from existing artifacts
- Vanilla success artifact:
  `/data2/haojitai/outputs/Sparse-vLLM/qwen25_7b_1m_deltakv_vs_vanilla_maxbs/qwen25_7b_1m_ctx64kplus_out256_gpu5_7_20260703/probes/vanilla/len900000/bs1/result.jsonl`
- DeltaKV failure artifact:
  `/data2/haojitai/outputs/Sparse-vLLM/qwen25_7b_1m_deltakv_vs_vanilla_maxbs/qwen25_7b_1m_ctx64kplus_out256_gpu5_7_20260703/probes/deltakv-less-memory-cudagraph/len900000/bs1/result.jsonl`
- Vanilla config used `prefill_schedule_policy='all_chunked'`; DeltaKV used
  `prefill_schedule_policy='long_bs1full_short_batch'`.
- Vanilla succeeded with `mem=71.56` GB and `kv_cache` shape
  `[2, 28, 1047924, 4, 128]`.
- DeltaKV failed during prefill attention at
  `src/sparsevllm/layers/attention_backend.py:33`, allocating
  `torch.empty_like(q)`.
- OOM request: `6.01` GiB.

Interpretation:

For Qwen2.5-7B-1M, one full 900k-token attention output tensor has size
`900000 * 28 * 128 * 2 = 6.01` GiB. Vanilla avoids that activation peak because
it chunks prefill. DeltaKV schedules long prompts as single-sequence full-prefill
steps, so the 900k-token `q` and output `o` tensors are materialized for the
whole prompt. The failure is therefore a full-prefill temporary activation
memory failure, not a persistent KV-cache admission failure.
