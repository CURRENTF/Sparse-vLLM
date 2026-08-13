# 可复现性

使用本页的稳定 checklist 复现 Sparse-vLLM 实验。不要把本地 run ledger 放入仓库；面向仓库的结果需要证据时，应引用原始 run artifact path。

## 环境

README 包含当前安装命令。预期 baseline 为：

- Python 3.10。
- 使用 `requirements/locks/canonical-cu129-py310.txt` 中冻结的完整 runtime
  与 test 环境。
- 带匹配 CUDA wheel 的 PyTorch 2.11.0，以及 Triton 3.6.0。
- `flashinfer-python==0.6.15.post1`，以及 CUDA 12.9 build 的
  `flashinfer-jit-cache==0.6.15.post1`。
- `sglang-kernel==0.4.5` 和 `einops>=0.8.2` 是 runtime 依赖。
- 需要预编译 device binary 时，可从通用 FlashInfer wheel index 安装匹配的 `flashinfer-cubin`。
- `transformers==5.13.1`。
- 使用 `MAX_JOBS=8 pip install flash-attn --no-build-isolation` 安装 `flash-attn`。
- 安装 lock 后，在仓库根目录运行 `pip install --no-deps -e .`。训练、
  benchmark 和测试依赖均包含在主安装中。
- 记录选择的 operator provider 和 CUDA compute capability。FP8 provider 根据本地 device capability 选择，warmup 期间不会下载 Hub kernel。
- RMSNorm 默认使用 `SPARSEVLLM_RMSNORM_PROVIDER=auto`，在已安装时优先选择 FlashInfer。设为 `triton` 可强制使用本地 Triton kernel；设为 `flashinfer` 可明确要求 FlashInfer。

每次报告 benchmark 时，都应记录 CUDA 版本、GPU 类型和数量、visible GPU ID、branch、commit 以及相关未提交改动。

## 模型与 Checkpoint

Base model 与 DeltaKV compressor checkpoint 必须匹配。公开 compressor checkpoint 列在 README 的 [DeltaKV checkpoint 下载](README.md#deltakv-checkpoint)一节。

将下载后的本地目录作为 `deltakv_checkpoint_path` 传入。当前 loader 读取本地 `model.safetensors` 文件；不要假设所有位置都可以直接传入 Hugging Face repo ID。

## 数据路径

LongBench 和 MathBench 从环境变量读取数据根目录：

- `SPARSEVLLM_OUTPUT_DIR`：benchmark prediction 和 log 的 output root。
- `SPARSEVLLM_DATA_DIR`：通用 benchmark dataset root。
- `SPARSEVLLM_LONGBENCH_DATA_DIR`：包含 `data/*.jsonl` 的 LongBench root。
- `SCBENCH_LOCAL_DATA_DIR`：standard SCBench 文件的可选 local root。
- `SCBENCH_PREPROCESSED_ROOT`：包含 SCBench 预处理 `<task>.parquet` 文件的 root。

Benchmark 入口不假设 host-specific dataset path。缺少必需 data root 或文件时，命令应快速失败并打印必须设置的环境变量或 CLI flag。

如果命令使用 `<DATA_ROOT>`、`<MODEL_ROOT>` 或 `<OUTPUT_ROOT>` 等本地 placeholder，请按目标机器改写，并在 run record 中记录最终路径。

## 参数规则

使用规范 public 参数名：

- `sparse_method`
- `deltakv_checkpoint_path`
- `decode_keep_tokens`
- `sink_keep_tokens`
- `recent_keep_tokens`
- `full_attention_layers`
- `engine_prefill_chunk_size`

新命令不要使用 `chunk_prefill_size`、`vllm_sparse_method`、`model_cls`、`compressor_path`、`deltakv_path`、`num_top_tokens` 或 `seq_chunk_size` 等 legacy public key。完整 alias map 和原生行为参见[运行时参数语义](../configuration/runtime-parameter-semantics.md)。

Sparse-vLLM 要求显式 integer keep budget；ratio 必须在启动前换算为 token count。

## Smoke Check

运行长 benchmark 前先执行小规模命令：

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 1 \
  --methods vanilla \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.8,"engine_prefill_chunk_size":512}'
```

基于 compressor 的 DeltaKV Sparse-vLLM smoke test：

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_checkpoint_path":"<LOCAL_COMPRESSOR_CHECKPOINT>","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

确认 loader log 显示 compressor 权重已加载。只有明确设置 `allow_missing_deltakv_path` 的构造测试才能省略 checkpoint。

## Artifact 要求

对于报告的结果，保存或记录：

- 准确命令和 working directory。
- Runtime config 和规范 sparse 参数。
- 模型、tokenizer、checkpoint、精度、backend 和量化设置。
- Dataset path、split、sample count、filtering/truncation 和 seed。
- benchmark 支持时，保存 raw output、parsed output、per-sample record、aggregate metric 和 run info。
- Log path 和 result file path。
- 失败或结论不确定的 run 应保存 failure status 和关键 error line。

没有 source log 或 result artifact 时，不要报告 metric。
