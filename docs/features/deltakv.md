# DeltaKV

DeltaKV compresses the KV cache for long-context inference. This repo contains
the Sparse-vLLM inference path, HF wrapper comparisons, compressor training
entrypoint, and benchmark integrations.

## Inference

Set `sparse_method` to one of:

- `"deltakv"`
- `"deltakv-triton"`, `"deltakv-triton-v2"`, `"deltakv-triton-v3"`, `"deltakv-triton-v4"`
- `"deltakv-delta-quant"` for the no-checkpoint direct residual quantization ablation

For compressor-backed DeltaKV inference, also pass
`deltakv_checkpoint_path="/path/to/local/trained_compressor_dir_or_file"`.
`deltakv-delta-quant` does not load or require a compressor checkpoint.

DeltaKV knobs you may need:

- `deltakv_checkpoint_path`: local path to trained compressor weights.
- `deltakv_latent_dim`: latent dimension of compressed KV.
- `deltakv_center_ratio`, `cluster_metric`: reference selection and clustering behavior.
- `deltakv_neighbor_count`: number of selected center/reference tokens used for reconstruction.
- `deltakv_latent_quant_bits`: `4` packs DeltaKV-style cached state as int4 where supported.

## Direct Residual Quantization Ablation

`deltakv-delta-quant` is a Sparse-vLLM-only ablation that reuses DeltaKV center
selection and sparse decode views, but stores the token-space residual directly:

```text
residual = KV_before_rope - mean(selected_center_KV_before_rope)
```

With `deltakv_latent_quant_bits=4`, that residual is packed as int4 plus
per-token scale/min metadata. With `deltakv_latent_quant_bits=0`, the residual
is stored in the model dtype. This path deliberately does not use learned
`compress_down` or `compress_up` modules.

Quick throughput smoke:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path /data2/haojitai/models/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv-delta-quant \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"prefill_keep_tokens":64,"deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_quant_bits":4,"deltakv_full_pool_reserve_ratio":0.2}'
```

## Train a Compressor

The main entrypoint is:

- Python: `python src/deltakv/train_compressor.py ...`
- CLI script after installation: `deltakv-train ...`

The training script expects a tokenized and packed dataset saved by Hugging
Face `datasets` with `load_from_disk`.

```bash
python src/deltakv/train_compressor.py \
  --model_name_or_path <PATH_TO_BASE_MODEL> \
  --dataset_path <PATH_TO_DATASET_ON_DISK> \
  --output_dir <PATH_TO_OUTPUT_CHECKPOINT_DIR> \
  --deltakv_latent_dim 512 \
  --compressor_token_group_size 1 \
  --deltakv_neighbor_count 4 \
  --layer_chunk_size 1 \
  --batch_size 1 \
  --warmup_ratio 0.02 \
  --max_steps 20000 \
  --learning_rate 2e-4 \
  --use_nonlinear_compressor True \
  --ref_mode avg \
  --collect_kv_before_rope True \
  --model_type cluster_e2e \
  --cluster_soft_assignment False \
  --compressor_down_type mlp_swiglu \
  --compressor_down_intermediate_size 3072 \
  --compressor_up_type linear \
  --compressor_linear_bias False
```

Common knobs:

- `--deltakv_latent_dim`: compressed KV latent width.
- `--compressor_token_group_size`: token grouping for non-cluster compressor references.
- `--deltakv_neighbor_count`: number of selected ref/center tokens for cluster DeltaKV.
- `--model_type`: `e2e`, `cluster_e2e`, `cluster_e2e_big`.
- `--collect_kv_before_rope`: whether to collect KV before RoPE.

## Evaluate on LongBench

`benchmark/long_bench/pred.py` runs LongBench prediction and writes JSONL
outputs under a local output directory.

```bash
python benchmark/long_bench/pred.py \
  --model all \
  --model_path <PATH_TO_BASE_MODEL> \
  --tokenizer_path <PATH_TO_TOKENIZER_OR_MODEL> \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path "<LOCAL_PATH_TO_TRAINED_COMPRESSOR_DIR>" \
  --hyper_param '{"hf_prefill_chunk_size": 2048000, "prefill_keep_tokens": 4096,
  "chunk_prefill_accel_omnikv": true, "decode_keep_tokens": 0.17, "full_attention_layers": "0,1,2,8,18",
  "recent_keep_tokens": 128, "sink_keep_tokens": 8, "use_compression": true, "use_cluster": true, "deltakv_center_ratio": 0.1}'
```

Notes:

- `--backend` supports `hf` and `sparsevllm`.
- `--hyper_param` accepts either a JSON string or a path to a JSON file.
- `full_attention_layers` is passed as a comma-separated string of layer indices.

## Checkpoints

- Public compressor checkpoints are listed in [Getting Started](../getting_started/README.md#deltakv-checkpoints).
- `deltakv_checkpoint_path` can point to a local directory or a single checkpoint file.
- The loader scans `*.safetensors` first, then `*.bin` and `*.pt`.
- Split-KV checkpoints (`k_compress_*` / `v_compress_*`) are not supported by the Sparse-vLLM loader.
