# DeltaKV

DeltaKV 为长上下文推理压缩 KV cache。本仓库包含 Sparse-vLLM inference path、HF wrapper 对比、compressor 训练入口和 benchmark 集成。

## 推理

将 `sparse_method` 设置为以下值之一：

- `"deltakv"`
- `"deltakv-less-memory"` 和 `"deltakv-less-memory-cudagraph"` 等 legacy alias
  仍会被规范为 `"deltakv"`，以兼容旧配置。

进行 DeltaKV 推理时，还需传入
`deltakv_checkpoint_path="/path/to/local/trained_compressor_dir_or_file"`。
当前 Sparse-vLLM DeltaKV runtime 依赖 compressor；缺少 checkpoint 的情况仅用于构造测试，不能用于可报告的 benchmark run。

可能需要使用的 DeltaKV 参数：

- `deltakv_checkpoint_path`：已训练 compressor 权重的本地路径。
- `deltakv_latent_dim`：压缩 KV 的 latent dimension。
- `deltakv_center_ratio`、`cluster_metric`：reference selection 和 clustering 行为。
- `deltakv_neighbor_count`：重建时使用的 center/reference token 数量。
- `deltakv_latent_quant_bits`：在支持的路径上，设为 `4` 会以 int4 打包 DeltaKV cache state。

## 精简 Runtime

Sparse-vLLM DeltaKV runtime 使用 `src/sparsevllm/engine/cache_manager/` 下的 cache-manager 实现。它按照 `full_attention_layers` 保留 full layer，为 sparse layer 存储 compressor residual state，并在 decode 中使用 graph-stable metadata。

支持两类存储路径：BF16/FP16 full layer 加 BF16/FP16 compressor latent residual，或者 KIVI int4 full layer 加 int4 compressor residual。

快速吞吐量 smoke test：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"chunk_prefill_accel_omnikv":true,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"prefill_keep_tokens":64,"deltakv_checkpoint_path":"<COMPRESSOR_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

## 训练 Compressor

主入口为：

- Python：`python src/deltakv/train_compressor.py ...`
- 安装后的 CLI：`deltakv-train ...`

训练脚本要求输入由 Hugging Face `datasets` tokenization 和 packing 后、通过 `load_from_disk` 保存的数据集。

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

常用参数：

- `--deltakv_latent_dim`：压缩 KV 的 latent width。
- `--compressor_token_group_size`：non-cluster compressor reference 的 token grouping。
- `--deltakv_neighbor_count`：cluster DeltaKV 选择的 ref/center token 数量。
- `--model_type`：`e2e`、`cluster_e2e`、`cluster_e2e_big`。
- `--collect_kv_before_rope`：是否在 RoPE 之前收集 KV。

## 在 LongBench 上评估

`benchmark/long_bench/pred.py` 运行 LongBench prediction，并将 JSONL 输出写入本地 output directory。

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

说明：

- `--backend` 支持 `hf` 和 `sparsevllm`。
- `--hyper_param` 接受 JSON 字符串或 JSON 文件路径。
- `full_attention_layers` 以逗号分隔的 layer index 字符串传入。

## Checkpoint

- 公开 compressor checkpoint 列在[快速开始](../getting_started/README.md#deltakv-checkpoint)中。
- `deltakv_checkpoint_path` 可以指向本地目录或单个 checkpoint 文件。
- loader 优先扫描 `*.safetensors`，随后扫描 `*.bin` 和 `*.pt`。
- Sparse-vLLM loader 不支持 split-KV checkpoint（`k_compress_*` / `v_compress_*`）。
