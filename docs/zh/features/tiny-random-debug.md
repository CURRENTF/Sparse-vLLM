# Tiny Random 调试模式

Tiny random 模式根据源 checkpoint 的 `config.json` 构造更小的模型，并初始化确定性的伪造权重，不读取任何 checkpoint tensor 文件。它用于模型开发、TP、prefill/decode 和数值对齐调试，其输出不代表模型质量。

实现位于显式启用的 `sparsevllm.debug.tiny_random` 模块。普通推理不会导入该模块。核心集成点仅有模型配置和启动时权重初始化；scheduler、attention、cache 和模型 forward 热路径均不改变。

## 配置

使用以下环境变量启用该模式：

```bash
export SPARSEVLLM_TINY_RANDOM=1
export SPARSEVLLM_TINY_RANDOM_CONFIG="$PWD/configs/debug/qwen3_tiny_random.json"
export SPARSEVLLM_TINY_RANDOM_SEED=17
```

JSON override 文件仅接受：

- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `vocab_size`
- `max_position_embeddings`

`num_hidden_layers`、`hidden_size` 和 `intermediate_size` 是必填项，且不能扩大源模型。无效的 head dimension，以及与 TP 不兼容的 attention head、KV head 或 vocabulary size，会在模型构造前失败。仍需提供源模型目录以读取配置和 tokenizer metadata，但不会打开其中的 `.safetensors` 文件。

## Qwen3-8B 双 GPU 检查

使用项目根目录的 uv 环境：

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

对比脚本使用相同的缩减配置和 seed 构造 HF reference，然后比较 teacher-forced prefill 和 decode logits。Tiny-random HF 对比目前只支持 `vanilla`。

## 限制

- 不支持量化 base-model 权重。
- 暂不支持 Qwen3.5 mixed attention。
- 不支持 DeltaKV learned compressor 权重。
- 此模式不测试 checkpoint 加载或下游任务质量。
