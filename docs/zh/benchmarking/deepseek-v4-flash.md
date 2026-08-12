# DeepSeek V4 Flash 验证与性能数据

本文记录 DeepSeek V4 Flash 首版适配的可复现实验结果。所有 GPU 结果只使用物理 GPU 4–7 中执行前确认空闲的设备。

## 正确性

| 检查 | 结果 |
| --- | --- |
| Sliding attention，13-token prefill | 对 Transformers reference 最大绝对误差 `1.19e-7` |
| CSA，13-token prefill | 对 Transformers reference 最大绝对误差 `1.19e-7` |
| HCA，140-token prefill | 对 Transformers reference 最大绝对误差 `1.79e-7` |
| Sliding decode | 最大绝对误差 `5.96e-8` |
| HCA decode | 最大绝对误差 `7.45e-8` |
| 正式 dense FP8 GEMM | 输出 finite；对显式 dequant reference 最大绝对误差 `0.142578`、平均绝对误差 `0.03160` |
| 正式 MXFP4 expert fused MoE | 输出 finite；对显式 MXFP4 dequant reference 最大绝对误差 `0.0009765625`、平均绝对误差 `0.00014795` |
| CUDA Graph 静态 CSA cache | 物理 GPU 6 捕获成功并连续 replay |
| CUDA Graph 内 EP+MXFP4 MoE | 物理 GPU 6,7；all-gather/fused-MoE/reduce-scatter replay 对 eager 最大误差 `0` |
| 完整 checkpoint index | 72,317 tensor；0 mapping error、0 unexpected skip |
| 正式 EP=4 rank-0 分片加载 | 48 shard、9,455 weight、42.50 GiB tensor，完整校验通过 |
| DeepSeek V4 及相关算子/并行/加载测试 | `140 passed, 1 skipped` |
| rebase 后全仓 CPU 测试 | `1357 passed, 138 skipped, 205 subtests passed` |

全仓结果使用 `SPARSEVLLM_PLATFORM=cpu CUDA_VISIBLE_DEVICES=''`，skip 来自目标分支中需要 GPU 或可选依赖的环境测试。另使用真实 checkpoint 构造 `Config`，确认 runtime class 为 `DeepseekV4ForCausalLM`、cache class 为 `DeepseekV4CacheManager`、拓扑为 `DPA_EP`、cache 层数为 43，且正式 FP8/MXFP4 量化保持启用。

## 架构 rebase 回归

本实现已 rebase 到 `codex/qwen36-moe-sparse-methods` 的 `187c69b`，并按该分支的 `ModelSpec`、`RuntimeLayout`、`ParallelTopology` 和集中 checkpoint validator 重构。rebase 后完成上述专项测试、全仓 CPU 测试和真实 checkpoint 配置链验证。

下述正式 GPU 性能数据采集于本次架构 rebase 之前。rebase 收尾时物理 GPU 4–7 均在被其他任务使用，因此按照设备隔离要求没有抢占重跑；这些数字保留为同一模型实现的已有基线，不宣称为 rebase 后重新测得的数据。

## Tiny-random engine smoke

以下数字只用于证明真实 engine 的 model construction、prefill、decode 和多进程 DPA 路径可运行，不代表正式模型性能。

| GPU | DP/EP | Prompt | Batch | Output | Prefill | Decode | TTFT | ITL | 单卡显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 物理 GPU 6 | 1/1 | 32 | 1 | 2 | `1140.8 tok/s` | `39.3 tok/s` | `0.03 s` | `25.44 ms` | `3.41 GB` |
| 物理 GPU 6,7 | 2/2 | 32 | 3 | 2 | `1208.8 tok/s` | `57.8 tok/s` | `0.08 s` | `51.93 ms` | `3.41 GB` |

双卡用 batch 3 故意制造不均衡 owner 数量，以覆盖 rank 补齐和全局 logits 重排。

## 正式 rank 分片加载

在空闲物理 GPU 6 上构造 EP=4 的 rank 0 分片并读取全部 48 个 shard；该检查不执行缺失其他 EP rank 的 forward。

| 指标 | 结果 |
| --- | ---: |
| 模型构造 | `2.02 s` |
| 权重加载 | `39.17 s` |
| 本地 weight 数 | `9,455` |
| loader tensor bytes | `42.50 GiB` |
| 常驻 GPU allocation | `42.44 GiB` |
| 43 层 MXFP4 interleave | `0.98 s` |
| 峰值 GPU allocation | `43.24 GiB` |

## 正式模型推理与性能

### 环境和方法

- 设备：空闲物理 GPU 6,7，`NVIDIA H100 80GB HBM3`，driver `580.65.06`。
- 软件：PyTorch `2.11.0+cu130`、CUDA runtime `13.0`、Transformers `5.13.1`、FlashInfer `0.6.15.post1`。
- checkpoint：48/48 shard；`TP=1`、`DP=EP=2`；无 MTP、无可选稀疏方法。
- engine：`max_model_len=4`、batch 1、prompt 1 token、output 2 token、`gpu_memory_utilization=0.995`。
- graph：`decode_cuda_graph=true`，capture batch `[1]`、context `[4]`，sampling 不进入 graph。
- 性能口径：engine warmup 后先完成一次请求，再连续执行 5 次相同请求；下表为这 5 次的中位数。Prefill/decode step 时间均包含本进程调度、模型执行和采样开销。

可复现实验的关键 kwargs：

```python
llm = LLM(
    "/data1/gqs/models/DeepSeek-V4-Flash-0731",
    tensor_parallel_size=1,
    data_parallel_size=2,
    expert_parallel_size=2,
    decode_cuda_graph=True,
    decode_cuda_graph_capture_sizes=[1],
    decode_cuda_graph_context_sizes=[4],
    enforce_eager=False,
    max_model_len=4,
    max_num_batched_tokens=4,
    engine_prefill_chunk_size=4,
    max_num_seqs_in_batch=1,
    max_decoding_seqs=1,
    max_num_seqs_in_gpu=1,
    mlp_chunk_size=4,
    gpu_memory_utilization=0.995,
    weight_loading_workers=2,
)
```

### 结果

| 指标 | 结果 |
| --- | ---: |
| 完整 engine 初始化 | `35.43 s` |
| 两 rank 权重加载 | `24.53 s` / `24.86 s` |
| 每 rank loader tensor bytes | `76.77 GiB` |
| 首个实测请求 | `444.6 ms` |
| 5 次请求总时间 | `2.2189 s` |
| 请求 E2E P50 | `443.07 ms` |
| Prefill step P50 | `390.78 ms` / `2.56 tok/s` |
| Decode step P50 / ITL | `52.31 ms` / `19.12 tok/s` |
| rank 0 PyTorch 常驻 / 峰值 allocation | `76.78 / 76.81 GiB` |
| 生成 token | `[294, 201]` |
| 捕获 graph 数 | `1` |
| graph key | `method='', batch_size=1, context_capacity=4, is_long_text=False, capture_sampling=False` |

完整权重、cache、DPA all-gather、MXFP4 EP fused MoE、reduce-scatter、全局 logits 重排均经过真实 engine forward。后续 graph replay 成功返回且两 rank 正常退出；若任一 rank 的 graph 或 collective 序列不一致，该路径会同步报错或超时。

以上是双卡极限显存下的首版正确性基线，不代表推荐服务吞吐。`DP=EP=4` 的正常上下文性能尚未测量，因为本次验收期间物理 GPU 4,5 被其他任务持续占用，按照设备隔离要求未抢占；四卡数据应单独补测，不能和本表的短上下文结果直接比较。
