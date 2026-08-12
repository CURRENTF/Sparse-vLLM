# DeepSeek V4 Flash 运行时

本文说明 `/data1/gqs/models/DeepSeek-V4-Flash-0731` 在 Sparse-vLLM 中的首版实现。目标是正确运行 CUDA Graph decode 和 DeepSeek 风格 DPA+EP；首版不执行 checkpoint 中的 MTP，也不叠加 Sparse-vLLM 可选稀疏方法。

## 架构落点

实现已对齐 `codex/qwen36-moe-sparse-methods` 的模型、布局和并行抽象：

- `ModelSpec` 声明模型运行类、专用 cache manager、DPA+EP 模式，以及 tiny-random 的量化和 checkpoint 校验策略。
- `models/checkpoint.py` 集中校验正式 checkpoint 的架构、维度和 FP8/MXFP4 格式；模型实现不重复承担配置识别。
- `RuntimeLayout` 将 CSA/HCA 识别为持有 KV 状态的完整 attention 层，统一生成 43 层 cache 布局。
- `ParallelTopology.DPA_EP` 定义 `TP=1`、`DP=EP`、attention singleton group，以及重叠的 data/expert group；`ParallelContext` 只提供 collective。
- runtime compatibility 由 `(model_type, parallel_mode)` 注册，DeepSeek V4 首版只允许 vanilla 方法。
- `model_runner.py` 只保留执行期的 owner 分片、固定 shape 补齐、全局 logits 重排和 CUDA Graph 约束。

## 支持边界

- checkpoint 架构必须为 `DeepseekV4ForCausalLM`，dense 权重为 E4M3 FP8、动态激活量化、UE8M0 `128 x 128` scale，expert 权重为 MXFP4、K32 UE8M0 scale。
- 运行布局固定为 `TP=1`、`DP=EP`，expert 数必须能被 EP 整除。
- Hopper 首版要求 SM90、CUDA runtime 12.8 或更新版本，以及仓库环境中的 FlashInfer。
- 仅支持 `vllm_sparse_method=""`。MTP、prefix cache 和其他可选稀疏方法会在配置阶段显式拒绝。
- decode attention 按稳定的 `seq_id % DP` owner 分片；MoE 在重叠的全局 EP group 中执行。首版 prefill 在各 DPA rank 复制执行，以保证所有 rank 都有完整、可校验的初始 cache。

## 模型和算子

原生实现位于 `src/sparsevllm/models/deepseek_v4_native.py`：

- mHC attention/FFN connection 和 hyper head 保留 fp32 Sinkhorn 与归一化语义。
- Q/KV/O、shared expert 等 dense projection 复用 Sparse-vLLM 的 FP8 linear provider。
- routed expert 仅在所属 EP rank 分配。checkpoint 的 `w1/w3/w2` packed MXFP4 bit pattern 原样加载，再由 FlashInfer Hopper W4A16 fused MoE 完成 interleave 和执行。
- Router 支持前三层 hash routing 和其余层 `sqrt(softplus(x))` routing；SwiGLU clamp 固定使用 checkpoint 的 limit 10。
- MTP tensor 只被识别并跳过，不进入参数分配和 forward。

完整 safetensors index 包含 72,317 个 tensor。EP=4、rank 0 的全索引映射审计结果为 0 个未知跳过、0 个目标缺失；远端 expert 和 MTP 是仅有的有意跳过项。

## Cache 和 CUDA Graph

`DeepseekV4CacheManager` 按请求行号持有以下状态：

- 每层一个 128-token shared-KV 环形窗口；
- 21 个 CSA 层每 4 token 一个 512-dim compressed KV 和 128-dim index KV；
- 20 个 HCA 层每 128 token 一个 512-dim compressed KV；
- CSA 的前窗 Ca overlap，以及 CSA/HCA 当前压缩窗口的 KV/gate ring。

decode 只通过 tensor index 原地更新这些状态，不创建或修改 Python `DynamicCache`，因此同一组地址可被 CUDA Graph 捕获和 replay。长 prefill 只把最后一个不重复的 sliding window 写回 ring，避免 advanced indexing 对重复 ring column 的未定义覆盖顺序。

Cache 按 `max_model_len` 和 `max_num_seqs_in_gpu` 预分配。若估算值超过 `gpu_memory_utilization` 给出的预算，启动会直接报出所需和可用 GiB；不会静默缩小上下文或请求容量。

## DPA+EP decode

每个 decode step 的流程为：

1. 所有 rank 收到同一全局请求列表，并选出本 rank 稳定 owner 的请求。
2. 请求较少的 rank 使用非 owner cache 行补齐到相同 shape；补齐输出会被丢弃。
3. attention 在本地 singleton group 执行。
4. 每个 MoE 层 all-gather hidden state 和 token id，各 EP rank 只计算本地 expert，再 reduce-scatter 回 attention rank；shared expert 保持本地执行。
5. full-vocab logits 在 EP group 中聚合，rank 0 按原始请求顺序重排并统一采样。

固定补齐 shape 使所有 rank 选择相同的 CUDA Graph batch family，也使 graph 内 NCCL collective 的调用顺序一致。

## 推荐启动参数

正式四卡短上下文服务可从以下 engine kwargs 开始，再按目标上下文增加 `max_model_len`：

```json
{
  "tensor_parallel_size": 1,
  "data_parallel_size": 4,
  "expert_parallel_size": 4,
  "decode_cuda_graph": true,
  "enforce_eager": false,
  "max_model_len": 512,
  "max_num_seqs_in_batch": 4,
  "max_decoding_seqs": 4,
  "max_num_seqs_in_gpu": 4,
  "max_num_batched_tokens": 512,
  "engine_prefill_chunk_size": 512,
  "gpu_memory_utilization": 0.9
}
```

当前已在两张 H100 80GB 上以 `DP=EP=2` 完成完整 checkpoint 推理和 CUDA Graph replay。该布局每 rank 常驻约 76.78 GiB，只适合 `max_model_len=4` 的最小正确性与性能验收；正常上下文应使用 `DP=EP=4` 或更多 rank，不能依靠静默缩减 cache 容量。

开发阶段设置：

```bash
export SPARSEVLLM_TINY_RANDOM=1
export SPARSEVLLM_TINY_RANDOM_CONFIG="$PWD/configs/debug/deepseek_v4_flash_tiny_random.json"
export SPARSEVLLM_TINY_RANDOM_SEED=17
```

Tiny random 不读取 safetensors，且性能与生成质量不能代表正式模型。
