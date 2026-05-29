# Benchmarks

This page collects the supported benchmark entrypoints and the canonical ways
to run them.

## Throughput Benchmark

Use `scripts/benchmarks/bench_sparse_vllm.py` to measure TTFT, prefill
throughput, decode throughput, and GPU memory.

Notes:

- Prefer `--hyper_params` to pass Sparse-vLLM settings as a JSON object.
- `--hyper_params` accepts canonical runtime names such as `sparse_method`,
  `engine_prefill_chunk_size`, `decode_keep_tokens`, and `prefill_keep_tokens`.
- `--lengths` measures prompt length; the script sets
  `max_model_len = length + output_len + 100` internally.

Baseline:

```bash
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization": 0.9}'
```

## MathBench With Sparse-vLLM

These examples are convenient for quick GSM8K / AIME-style comparisons while
exercising the Sparse-vLLM engine directly. For dataset details, see
[`benchmark/math_bench/README.md`](../../benchmark/math_bench/README.md).

Full-attention baseline:

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-full \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "vanilla"}'
```

OmniKV:

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "chunk_prefill_accel_omnikv": false, "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024}'
```

DeltaKV requires a compressor trained for the same base model. Replace the
checkpoint path below with a matching compressor for the model you run.

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"engine_prefill_chunk_size": 512, "prefill_keep_tokens": 16384, "max_num_batched_tokens": 8192, "max_num_seqs_in_batch": 30, "sparse_method": "deltakv-triton-v4", "chunk_prefill_accel_omnikv": true, "full_attention_layers": "0,1,2,4,7,14", "decode_keep_tokens": 1024, "deltakv_checkpoint_path": "/root/autodl-fs/checkpoints/compressor/<MATCHING_COMPRESSOR_DIR>", "deltakv_latent_dim": 256}'
```

When `--backend sparsevllm`, method selection happens through `sparse_method`
and checkpoints through `deltakv_checkpoint_path`.

## LongBench With Sparse-vLLM

Use this path when you want LongBench results from the actual Sparse-vLLM
engine rather than the HF wrapper models.

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend sparsevllm \
  --task qasper,hotpotqa,multi_news \
  --hyper_param '{"engine_prefill_chunk_size": 4096, "sparse_method": "omnikv", "chunk_prefill_accel_omnikv": true, "prefill_keep_tokens": 4096, "decode_keep_tokens": 2048, "full_attention_layers": "0,1,2,4,7,14", "recent_keep_tokens": 128, "sink_keep_tokens": 8}'
```

For a full LongBench run, omit `--task`. To switch to DeltaKV, keep
`--backend sparsevllm` and set `sparse_method="deltakv"` or
`"deltakv-triton-v4"` plus a matching `deltakv_checkpoint_path`. For the
no-checkpoint direct residual ablation, set
`sparse_method="deltakv-delta-quant"` and omit `deltakv_checkpoint_path`.

## LongBench With HF Wrappers

Use the HF backend when you want to compare against the DeltaKV / SnapKV /
PyramidKV wrapper models implemented under `src/deltakv/`.

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "/root/autodl-fs/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor" \
  --hyper_param '{"hf_prefill_chunk_size": 2048000, "prefill_keep_tokens": 4096, "chunk_prefill_accel_omnikv": true, "decode_keep_tokens": 0.11, "full_attention_layers": "0,1,2,4,7,14", "recent_keep_tokens": 128, "sink_keep_tokens": 8, "use_compression": true, "use_cluster": true, "deltakv_center_ratio": 0.1}'
```

To compare other baselines, keep `--backend hf` and switch `--sparse_method`
and `--hyper_param`.

Example OmniKV HF params:

```json
{"hf_prefill_chunk_size":4096,"prefill_keep_tokens":4096,"decode_keep_tokens":2048,"full_attention_layers":"0,1,2,4,7,14","recent_keep_tokens":128,"sink_keep_tokens":8}
```

Example SnapKV HF params:

```json
{"decode_keep_tokens":0.2,"pool_kernel_size":7}
```

Example KVZip HF params:

```json
{"ratio":0.3,"level":"pair","kv_type":"evict","prefill_chunk_size":16000}
```

For `kvzip`, the vendored baseline lives in `baselines/kvzip/`. Build its CUDA
extension first:

```bash
cd baselines/kvzip/csrc
make
```

## Multimodal

LLaVA-OneVision benchmark notes live under
[`docs/benchmarking/multimodal/`](multimodal/).
