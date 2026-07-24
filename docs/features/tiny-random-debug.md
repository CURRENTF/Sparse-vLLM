# Tiny Random Debug Mode

Tiny random mode constructs a smaller model from the source checkpoint's
`config.json` and initializes deterministic fake weights without reading any
checkpoint tensor files. It is intended for model-development, tensor-parallel,
prefill/decode, and numerical-alignment debugging. Its outputs do not represent
model quality.

The implementation lives in the opt-in
`sparsevllm.debug.tiny_random` module. Normal inference does not import this
module. The only core integration points are model configuration and
startup-time weight initialization; scheduler, attention, cache, and model
forward hot paths are unchanged.

## Configuration

Enable the mode with these environment variables:

```bash
export SPARSEVLLM_TINY_RANDOM=1
export SPARSEVLLM_TINY_RANDOM_CONFIG="$PWD/configs/debug/qwen3_tiny_random.json"
export SPARSEVLLM_TINY_RANDOM_SEED=17
```

The JSON override file accepts only:

- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `vocab_size`
- `max_position_embeddings`

`num_hidden_layers`, `hidden_size`, and `intermediate_size` are required and
must not enlarge the source model. Invalid head dimensions and TP-incompatible
attention heads, KV heads, or vocabulary sizes fail before model construction.
The source model directory is still required for configuration and tokenizer
metadata, but its `.safetensors` files are not opened.

## Qwen3-8B two-GPU check

Use the project-root uv environment:

```bash
CUDA_VISIBLE_DEVICES=5,6 \
PYTHONPATH="$PWD:$PWD/src" \
.venv/bin/python scripts/debug/compare_logits_hf_sparsevllm.py \
  --model_path /data2/pretrain_models/Qwen3-8B \
  --cases short \
  --methods vanilla \
  --tiny_random_config "$PWD/configs/debug/qwen3_tiny_random.json" \
  --tiny_random_seed 17 \
  --tensor_parallel_size 2 \
  --max_model_len 2048 \
  --engine_prefill_chunk_size 256 \
  --max_num_batched_tokens 512 \
  --max_num_seqs_in_batch 1 \
  --max_decoding_seqs 1 \
  --gpu_memory_utilization 0.02 \
  --mlp_chunk_size 256 \
  --teacher_forced_decode_steps 2 \
  --output_dir /tmp/sparsevllm_tiny_random_qwen3_8b_tp2
```

The comparison script constructs the HF reference from the same reduced config
and seed, then compares teacher-forced prefill and decode logits. Tiny-random HF
comparison currently supports `vanilla` only.

## Limitations

- Quantized base-model weights are not supported.
- Qwen3.5 mixed attention is not supported yet.
- DeltaKV learned compressor weights are not supported.
- This mode does not test checkpoint loading or downstream task quality.
