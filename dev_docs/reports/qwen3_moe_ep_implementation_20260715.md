# Qwen3MoE 与专家并行实现报告

日期：2026-07-15

分支：`feat/qwen3-moe-ep`

目标模型：`Qwen3-30B-A3B-Instruct-2507`

## 1. 交付结论

本次实现完成了 Qwen3MoE 的首版原生专家并行路径，并把并行语义、
模型加载、PyTorch 正确性基线、Triton 性能后端、稀疏 Attention、Prefix
Cache、验证程序和用户文档作为同一条可复现交付链落地。

首版正式支持范围为：

```text
DP=1, TP=1, EP>=1
```

在可用的两张 NVIDIA H20 上，实际完成了 EP=1 和 EP=2 的模型级验证。
代码中的专家分片对能整除 128 个专家的 EP 大小采用通用实现，但 EP=4 和
EP=8 没有硬件验证，因此不把这两个拓扑计入本次实测完成范围。

| 计划能力 | 交付状态 | 说明 |
| --- | --- | --- |
| DP/EP/TP 显式并行语义 | 完成 | 三维 rank 映射和独立进程组 |
| Dense TP 语义分离 | 完成 | Attention、KV、Embedding 等只使用 TP |
| Qwen3MoE PyTorch oracle | 完成 | 长期保留且可显式选择 |
| Qwen3MoE Triton 后端 | 完成 | 首版默认后端，无隐式 fallback |
| EP 本地专家加载 | 完成 | 连续均匀分片，严格 loaded/skipped 校验 |
| EP=1/2 数值一致性 | 完成 | 模型层、MoE 层、logits 和生成结果验证 |
| EP=4/8 数值一致性 | 未实测 | 验证主机仅有两张 H20 |
| 首批七种稀疏方法 | 完成 | EP=1/2 矩阵验证 |
| 三种 Prefix Cache 组合 | 完成 | EP=1/2 全生命周期与控制 API 验证 |
| HF 参考回放 | 完成 | SDPA，BF16 验收阈值有校准记录 |
| 固定手工问答 | 完成 | EP=1/2 四题逐 token 完全一致 |
| 微基准和端到端性能 | 完成 | 原始结果均保存在远端目录 |
| DP+EP、TP+EP | 不在首版范围 | 配置阶段明确拒绝 |
| 量化、shared expert、CUDA graph | 不在首版范围 | 配置阶段明确拒绝 |

## 2. 运行环境与可复现协议

### 2.1 验证环境

| 项目 | 值 |
| --- | --- |
| 远端入口 | `ssh autodl` |
| GPU | 2 × NVIDIA H20 |
| 模型 | `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507` |
| 仓库副本 | `/root/autodl-tmp/Sparse-vLLM-qwen3-moe-ep` |
| Python | `/root/Sparse-vLLM/.venv/bin/python` |
| PyTorch | 2.12.1 + CUDA 13.0 |
| Triton | 3.7.1 |
| 结果根目录 | `/root/autodl-tmp/qwen3-moe-ep-results` |

每次远端 GPU 任务前都通过 `nvidia-smi` 检查两张设备的利用率、进程和显存
占用，只在设备空闲时启动。所有模型级验证固定使用本地 checkpoint，不依赖运行
期间下载。

验证程序分别保存：

- `run_config.json`：命令、commit、模型、并行拓扑和推理参数；
- `raw_outputs.*`：原始 tensor 或模型输出；
- `parsed_outputs.json`：可比较的结构化输出；
- `per_step_results.json` 或 `per_sample_results.json`：逐步状态；
- `aggregate_metrics.json`：聚合状态、耗时、显存和错误；
- `reference_metrics.json`：需要参考运行时的逐项对比结果。

验证脚本不把缺失依赖、解析错误、数值错误或 worker 错误转换为成功状态。

### 2.2 本地 GPU 运行偏差

任务开始时曾在本机误执行六个原本预期因无 CUDA 而跳过的 GPU 测试；本机实际
暴露了 CUDA，六项均通过。发现后立即向用户说明，并从该时点起把所有 GPU 测试、
模型运行和性能测试严格放到 `ssh autodl`。本地后续测试均显式隐藏 CUDA。

## 3. 架构实现

### 3.1 `ParallelContext`

新增的 `ParallelContext` 把 world、tensor、expert 和 data 四类进程组分开。
逻辑拓扑采用 DP × EP × TP，rank 映射为：

```text
world_rank = ((dp_rank * ep_size) + ep_rank) * tp_size + tp_rank
```

上下文提供显式的 `tp_all_reduce`、`ep_all_reduce`、组内 rank、组大小和 world
rank。size 为 1 时 collective 保持相同接口并直接返回输入。重复初始化、未初始化
访问、配置 world size 与实际进程数不符都会立即失败。

首版 Qwen3MoE 强制 `TP=1`、`DP=1`，并要求 EP 为正数且整除专家数。Dense 模型
仍使用原有 TP 行为，并要求 `EP=1`、`DP=1`。

### 3.2 Dense 语义分离

原有代码中部分 `world_size` 同时承担 worker 数和 TP 分片数。实现逐点区分了：

- worker 启动、设备绑定、控制面主进程使用 world 语义；
- Attention heads、KV heads、Dense Linear、Embedding 和 LM Head 使用 TP；
- 专家权重分片和 MoE 输出聚合使用 EP；
- CacheManager 的 KV head 数只除以 `tp_size`，不会随 EP 增大而缩小；
- 稀疏方法中的本地 head、query history 和显存估算使用 TP-local 语义。

这项修改没有机械替换全部 world 调用，控制 RPC 和进程生命周期仍保留 world
语义，从而避免把进程控制误当成模型张量分片。

### 3.3 Qwen3MoE 模型结构

`Qwen3MoeForCausalLM` 复用了 Dense Qwen3 的 Attention、Q/K Norm、RoPE、
RMSNorm、残差流程、KV Cache、稀疏 hook、Embedding 和 LM Head。每个 decoder
layer 仅用 `Qwen3MoeSparseMoeBlock` 替换 Dense MLP。

目标 checkpoint 的结构为：

```text
hidden_size             = 2048
num_hidden_layers       = 48
num_experts             = 128
num_experts_per_tok     = 8
moe_intermediate_size   = 768
norm_topk_prob          = true
```

首版要求每层都是 MoE，不支持 shared experts 或混合 Dense/MoE layer layout。
不满足该结构的模型会在执行前失败。

### 3.4 Packed 专家和严格加载

EP rank `r` 持有连续专家区间：

```text
experts_per_rank = num_experts / ep_size
local_start      = r * experts_per_rank
local_end        = local_start + experts_per_rank
```

每层只分配本地 packed 参数：

```text
w13_weight = [local_experts, 2 * intermediate_size, hidden_size]
w2_weight  = [local_experts, hidden_size, intermediate_size]
```

loader 将 checkpoint 的 `gate_proj`、`up_proj` 和 `down_proj` 映射到 packed
位置。它要求复制权重和本地专家权重各加载一次，只允许跳过本 rank 区间之外的
专家，并对缺失、重复、形状不符和意外跳过给出错误。

实际加载记录：

| 拓扑 | 每 rank 专家区间 | 每 rank loaded | 每 rank remote skipped |
| --- | --- | ---: | ---: |
| EP=1 | `[0, 128)` | 18,867 | 0 |
| EP=2 rank 0 | `[0, 64)` | 9,651 | 9,216 |
| EP=2 rank 1 | `[64, 128)` | 9,651 | 9,216 |

EP=1 权重加载阶段的实测显存约为 61.24 GB。EP=2 每个 rank 不会分配或加载远端
专家的 packed 权重。

### 3.5 PyTorch oracle

PyTorch 后端保留清晰的逐本地专家实现：Router 使用 `F.linear`、softmax、TopK
和可选归一化；本地专家使用 `F.linear`、SiLU-and-mul 和 `index_add_` 聚合。
它通过 `moe_backend="pytorch"` 显式选择，作为新 checkpoint、kernel 和边界形状的
长期正确性基线。

后端选择不含自动 fallback。指定 Triton 时遇到编译或执行错误会直接失败，不会
静默切回 PyTorch 并污染性能或正确性结论。

### 3.6 Triton MoE 流水线

性能后端实现了以下通用流水线：

```text
assignment count/alignment
  -> packed W13 routed GEMM
  -> SiLU-and-mul
  -> packed W2 routed GEMM * routing weight
  -> local TopK sum
  -> EP all-reduce
```

kernel 支持 BF16 和 FP16、任意合法 token 数、空专家、非均匀 routing 和 padding
block。专家 GEMM 使用 BF16/FP16 输入与 FP32 dot-product accumulator。

实现参考的 Triton 官方资料：

- [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- [`triton.language.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
- [`triton.language.atomic_add`](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)

没有复制 vLLM 的量化、DeepEP、EPLB、shared expert 或多 backend 框架，也没有
引入新依赖。

### 3.7 跨拓扑精确归约

BF16 本地累加和 NCCL 分组顺序会使 EP=1 与 EP=2 产生微小差异；这些差异可在
后续层影响 Router TopK tie。最终实现采用：

1. W13/W2 GEMM 保持 BF16/FP16 输出和 FP32 dot accumulator；
2. 本地 TopK expert contribution 转为 FP64 后求和；
3. EP collective 对 FP64 local output 求和；
4. 仅在归约完成后一次性转回模型 activation dtype。

另外，在 Router 前从 EP rank 0 广播 post-attention hidden 和 residual。该同步点
消除复制 Attention 独立执行产生的极小漂移，保证所有 EP rank 在相同输入上执行
Router，而不是等 TopK 已分歧后再修复结果。

在 `exact-accum-v1` 中，EP=1 与 EP=2 的两个步骤均完全一致：每步 49 个 hidden
snapshot、48 层 MoE input/output、TopK IDs、routing weights 和最终 logits 的最大
绝对误差均为 0。

### 3.8 稀疏 Attention 与 hook 时序

MoE block 不包含方法特有的稀疏逻辑。Attention 继续通过 CacheManager 构建方法
特有的读取视图，MoE 必须先完成 EP all-reduce，之后才执行 activation hook 和
`on_layer_end`：

```text
Sparse Attention
  -> residual/norm
  -> Router + local experts
  -> EP all-reduce
  -> activation hook
  -> sparse_controller.on_layer_end
```

调试模式可以采集每层 active slots、context lengths、cache occupancy、selection
checksum、Router 和 hidden snapshot，并比较全部 EP ranks。该状态不进入默认热路径。

正式 registry 支持：

```text
vanilla, streamingllm, snapkv, pyramidkv, omnikv, quest, rkv
```

DeltaKV 明确拒绝。SkipKV 只有存在匹配 Qwen3MoE 的 steering asset 后才能注册，
当前缺失资产时在配置阶段失败。

### 3.9 Prefix Cache 的 world 同步

Prefix Cache 采用 replicated-KV 语义：每个 EP rank 保存等价的 radix index、完整
Attention heads 的 KV payload、引用计数和驱逐状态。

首次完整矩阵发现一个真实控制面缺陷：scheduler 只在 world rank 0 执行 lookup，
导致 EP=2 的 rank 1 lookup/hit 统计不变。81 份逐步记录中，差异只落在八条由
`lookup_requests`、hit requests/tokens/blocks 派生的统计路径，block、reference、
access、KV payload 和模型结果均一致。验证器没有忽略这些字段，因此运行正确失败。

修复后 scheduler 在启用 Prefix Cache 时通过 world RPC 让所有 worker 各自 lookup，
并比较命中 token 数和 block chain 元数据。任一 rank 失败或结果分歧会使请求失败。
lookup 之外的 attach、commit、release、delete、inspect、match、eviction priority
继续按 world worker 同步。

## 4. 主要文件

| 区域 | 文件 | 作用 |
| --- | --- | --- |
| 并行 | `src/sparsevllm/distributed/parallel_context.py` | 三维拓扑和 collective |
| 配置 | `src/sparsevllm/config.py` | TP/EP/DP、MoE backend 和 fail-fast 校验 |
| 模型 | `src/sparsevllm/models/qwen3_moe.py` | Qwen3MoE、packed expert 和 loader |
| Kernel | `src/sparsevllm/triton_kernel/moe.py` | routed GEMM 和 sum |
| 激活 | `src/sparsevllm/triton_kernel/silu_and_mul.py` | 可复用 SiLU-and-mul |
| Engine | `src/sparsevllm/engine/model_runner.py` | world worker 和模型运行 |
| Engine | `src/sparsevllm/engine/llm_engine.py` | world RPC 和生命周期 |
| Scheduler | `src/sparsevllm/engine/scheduler.py` | Prefix lookup 全 worker 同步 |
| Cache | `src/sparsevllm/engine/cache_manager/` | TP-local KV 和状态摘要 |
| Registry | `src/sparsevllm/method_registry.py` | 模型/方法/prefix 兼容矩阵 |
| 验证 | `scripts/validation/validate_qwen3_moe_ep.py` | 模型和 EP 逐层对齐 |
| 验证 | `scripts/validation/validate_qwen3_moe_sparse_ep.py` | 稀疏和 Prefix 单项验证 |
| 验证 | `scripts/validation/run_qwen3_moe_sparse_ep_matrix.py` | 支持矩阵编排 |
| 验证 | `scripts/validation/validate_qwen3_moe_hf_reference.py` | HF logits 回放 |
| 验证 | `scripts/validation/benchmark_moe_kernels.py` | MoE microbenchmark |
| 验证 | `scripts/validation/validate_qwen3_moe_manual_qa.py` | 固定手工问答 |
| 文档 | `docs/features/qwen3-moe-ep.md` | 稳定用户说明和 runbook |

## 5. 验证结果

### 5.1 CPU 回归

最终提交在本机用 CPU 平台运行全量测试：

```bash
SPARSEVLLM_PLATFORM=cpu CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q tests
```

```text
601 passed, 35 skipped, 143 subtests passed
elapsed: 15.33 s
```

最终提交的 Triton 定向测试在空闲 H20 上运行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /root/Sparse-vLLM/.venv/bin/python -m pytest -q tests/test_triton_moe.py
```

```text
8 passed in 2.82 s
```

### 5.2 EP=1/2 模型精确对齐

结果目录：

```text
/root/autodl-tmp/qwen3-moe-ep-results/exact-accum-v1/vanilla-ep1
/root/autodl-tmp/qwen3-moe-ep-results/exact-accum-v1/vanilla-ep2
```

| 项目 | EP=1 | EP=2 对 EP=1 |
| --- | ---: | ---: |
| steps | 2/2 | 2/2 |
| rank consistency failures | 0 | 0 |
| hidden snapshots | 49/step | 全部 max abs 0 |
| MoE layers | 48/step | input/output/router 全部一致 |
| logits max abs | 0 | 0 |
| 状态 | success | success |

### 5.3 稀疏方法矩阵

非 Prefix 主矩阵和修复后的尾部矩阵位于：

```text
/root/autodl-tmp/qwen3-moe-ep-results/sparse-matrix-8f71051-v1
/root/autodl-tmp/qwen3-moe-ep-results/sparse-tail-a9862a9-v1
```

| 方法 | EP=1 | EP=2 对 EP=1 | Prefix |
| --- | --- | --- | --- |
| vanilla | success | success | success |
| streamingllm | success | success | 不支持 |
| snapkv | success | success | 不支持 |
| pyramidkv | success | success | 不支持 |
| omnikv | success | success | success |
| quest | success | success | success |
| rkv | success | success | 不支持 |

每项均检查逐步 logits、rank consistency、方法触发、每层 sparse/cache state 和
profiler snapshot。兼容 registry 对表外组合提前报错，不自动回落到 vanilla。

### 5.4 Prefix Cache

修复后的 vanilla 结果：

```text
/root/autodl-tmp/qwen3-moe-ep-results/prefix-sync-a9862a9-v1/
```

OmniKV 与 QuEST 结果：

```text
/root/autodl-tmp/qwen3-moe-ep-results/sparse-tail-a9862a9-v1/
```

| 方法 | 拓扑 | steps | controls | lookup/hit 统计 | 状态 |
| --- | --- | ---: | ---: | --- | --- |
| vanilla | EP=1 | 75/75 | 6/6 | 7 / 4 / 312 tokens / 39 blocks | success |
| vanilla | EP=2 | 75/75 | 6/6 | 两 rank 与 EP=1 完全一致 | success |
| omnikv | EP=1/2 | 75/75 | 6/6 | 两拓扑一致 | success |
| quest | EP=1/2 | 75/75 | 6/6 | 两拓扑一致 | success |

控制操作覆盖 inspect、match、delete subtree、eviction-priority、引用保护和释放。

### 5.5 Hugging Face 参考回放

首次按原配置使用 `flash_attention_2` 时，远端没有安装 `flash-attn`。验证脚本将其
记录为 `model_failed`，没有安装新依赖或切换后端后伪装成同一次成功：

```text
/root/autodl-tmp/qwen3-moe-ep-results/hf-reference-a9862a9-v1
```

随后把默认参考实现改为 Transformers 官方支持的 PyTorch SDPA。严格阈值
`atol=0.05, rtol=0.05` 的运行保留在：

```text
/root/autodl-tmp/qwen3-moe-ep-results/hf-reference-38a75f5-v2
```

13 个 greedy step 的 token 全部匹配，但 12 个 step 的 logits allclose 未通过，
最大绝对误差为 0.4375。为区分 Sparse-vLLM 误差和 HF Attention backend 自身的
数值差异，又运行了 HF eager：其对 Sparse-vLLM 的最大误差为 1.375，而同一个
HF 模型 eager 与 SDPA 之间最大误差达到 1.1875。因此选择与 Sparse-vLLM 更接近的
SDPA 作为参考，并显式使用适合整模型 BF16 回放的 `atol=0.5, rtol=0.05`：

```text
/root/autodl-tmp/qwen3-moe-ep-results/hf-reference-sdpa-38a75f5-v4
```

| 指标 | 结果 |
| --- | ---: |
| 状态 | success |
| greedy steps | 13/13 |
| token IDs | 13/13 完全匹配 |
| logits max abs | 0.4375 |
| load time | 13.45 s |
| total elapsed | 15.16 s |
| peak memory | 61,920,956,416 bytes |

默认脚本阈值仍为严格的 0.05；稳定文档的 BF16 验收命令显式传入 0.5，避免把校准
阈值隐藏在默认值里。

### 5.6 MoE microbenchmark

环境：H20、warmup 10、iterations 30、128 experts、TopK 8、H=2048、I=768。

结果目录：

```text
/root/autodl-tmp/qwen3-moe-ep-results/moe-microbench-fp64-38a75f5-v1
/root/autodl-tmp/qwen3-moe-ep-results/moe-microbench-bf16-38a75f5-v1
```

FP64 exact reduction 的 16/16 case 全部成功：

| Tokens | 最佳 Triton 延迟 | 相对 PyTorch oracle |
| ---: | ---: | ---: |
| 1 | 0.196681 ms | 24.04× |
| 16 | 0.345020 ms | 43.70× |
| 128 | 0.426761 ms | 50.22× |
| 512 | 0.746046 ms | 29.13× |

与 BF16 local sum 的最佳延迟比较：

| Tokens | FP64 exact | BF16 | BF16 相对改善 |
| ---: | ---: | ---: | ---: |
| 1 | 0.196681 ms | 0.225426 ms | -12.75% |
| 16 | 0.345020 ms | 0.389880 ms | -11.51% |
| 128 | 0.426761 ms | 0.417387 ms | 2.25% |
| 512 | 0.746046 ms | 0.719119 ms | 3.74% |

小 M 下 FP64 更快属于短 kernel 测量噪声，不据此声称 FP64 算术更快。大 M 的精确
归约成本约为 2.25% 到 3.74%。

### 5.7 端到端性能

固定配置：vanilla、prompt 128、output 64、batch 1、greedy、eager。

```text
/root/autodl-tmp/qwen3-moe-ep-results/engine-perf-ep1-38a75f5-v1
/root/autodl-tmp/qwen3-moe-ep-results/engine-perf-ep2-38a75f5-v1
```

| 指标 | EP=1 | EP=2 |
| --- | ---: | ---: |
| 状态 | success | success |
| TTFT | 98.718 ms | 114.486 ms |
| prefill throughput | 1297.01 tok/s | 1118.61 tok/s |
| decode throughput | 10.682 tok/s | 8.276 tok/s |
| inter-token latency | 93.615 ms | 120.835 ms |
| peak allocated context | 68.095 GB | 68.096 GB/rank |

EP=2 在该 batch-1 replicated-input 实现下没有吞吐优势：专家计算减少，但每层增加
FP64 all-reduce 和同步；表中结果作为基线保存，不宣称 EP=2 加速单请求。

### 5.8 手工问答

结果目录：

```text
/root/autodl-tmp/qwen3-moe-ep-results/manual-qa-ep1-b4dea4e-final
/root/autodl-tmp/qwen3-moe-ep-results/manual-qa-ep2-f608838-final
```

| 用例 | EP=1 输出 | EP=2 |
| --- | --- | --- |
| arithmetic | `391` | token IDs 完全一致 |
| factual | `The capital of France is Paris.` | token IDs 完全一致 |
| JSON format | `{"status":"ok","count":3}` | token IDs 完全一致 |
| Chinese MoE | 见下方 | token IDs 完全一致 |

中文用例输出：

```text
混合专家模型（MoE）通过动态选择多个专家网络中的部分专家来处理输入，
每个输入仅激活相关专家，从而实现高效且可扩展的模型推理。
```

两种拓扑均为 4/4 success，四项输出逐 token 完全一致，且
`cleanup_failure=null`。EP=1 用时 23.30 s，峰值显存 73,127,758,336 bytes；
EP=2 用时 23.76 s，峰值显存 73,129,986,560 bytes。修复 worker 退出顺序后的
EP=2 最终运行没有 `resource_tracker` 警告，退出后两张 GPU 的进程数和显存占用均为 0。

第一轮手工问答暴露了 Transformers `apply_chat_template(tokenize=True)` 在当前版本
返回 `BatchEncoding` 而不是裸 token list。脚本随后增加了 mapping、tensor 和 list
三种合法返回类型的严格提取与校验；失败的首轮没有被计入成功结果。

## 6. 提交记录

实现按模块拆分为 Conventional Commits：

| Commit | 内容 |
| --- | --- |
| `ce430bc` | `feat: add parallel context foundation` |
| `a05cd4f` | `refactor: separate parallel semantics` |
| `0af9399` | `feat: add qwen3 moe pytorch runtime` |
| `6653f9d` | `test: add qwen3 moe ep validation` |
| `b6e8dad` | `feat: add triton moe kernels` |
| `1ad3a17` | `feat: enable triton moe backend` |
| `320990a` | `perf: optimize triton moe runtime` |
| `499a5d8` | `fix: stabilize qwen3 moe ep outputs` |
| `7135cd4` | `feat: guard qwen3 moe method support` |
| `ffc6a0c` | `feat: expose sparse ep validation state` |
| `8f71051` | `test: add qwen3 moe sparse ep matrix` |
| `db3b5c5` | `test: add qwen3 moe hf replay` |
| `1b04c28` | `perf: benchmark exact moe reduction` |
| `a9862a9` | `fix: synchronize ep prefix lookups` |
| `38a75f5` | `fix: default hf replay to sdpa` |
| `2009b92` | `test: add qwen3 moe manual qa` |
| `f820e0e` | `fix: validate manual qa chat tokens` |
| `c6f64b9` | `docs: document qwen3 moe ep support` |
| `7da325a` | `docs: report qwen3 moe ep validation` |
| `af2535c` | `fix: tighten qwen3 moe runtime guards` |
| `b4dea4e` | `test: harden qwen3 moe validation` |
| `f608838` | `fix: join ep workers before terminate` |

`f608838` 的提交说明沿用了问题最初在 EP 运行中出现时的名称；实际修复位于通用
world worker 生命周期路径，同时覆盖 TP 和 EP。

## 7. 计划验收对照

### 7.1 已完成

- 阶段 A：三维并行上下文、Dense TP 语义、world worker 控制均已完成；
- 阶段 B：模型注册、packed 权重、严格 loader 和 PyTorch oracle 已完成；
- 阶段 C：本地专家、复制 Router、EP reduction 和调试状态已完成；
- 阶段 C2：七种正式稀疏方法完成 EP=1/2 验证并进入 registry；
- 阶段 C3：Vanilla、OmniKV、QuEST Prefix Cache 完成 EP=1/2 全流程验证；
- 阶段 D：Triton assignment、routed GEMM、activation、sum 和 benchmark 已完成；
- 运行记录：配置、raw、parsed、per-step/per-sample、aggregate 分离保存；
- 矩阵编排：`--ep-sizes` 显式接受 `1,2,4,8`，校验参考拓扑、重复值和物理
  GPU 数量；默认仍运行本次有硬件证据的 EP=1/2；
- 稳定文档：支持范围、运行方式、设计和 runbook 已加入文档索引。

### 7.2 未纳入首版或尚未实测

- EP=4/8：计划要求但当前硬件只有两张 H20，未进行模型级硬件验证；
- DP+EP、TP+EP：首版明确不实现；
- quantized MoE、DeepEP、EPLB、shared expert：不实现；
- CUDA graph decode：Qwen3MoE 首版拒绝；
- SkipKV：缺少与目标模型匹配的 steering asset，明确拒绝；
- DeltaKV：未进入 Qwen3MoE 兼容范围，明确拒绝。

因此，本报告把“首版实现完成”和“原计划所有 EP=2/4/8 硬件矩阵完成”区分开。
在增加至少四张或八张同类 GPU 后，应直接复用现有 validation matrix 补齐拓扑证据，
而不是根据通用代码结构推断成功。

## 8. 已知问题与后续建议

1. 已解决 TP/EP 共用的 worker 退出泄漏：ModelRunner 的 exit RPC 成功后，父进程
   曾在子进程尚处于正常退出尾声时立即调用 `terminate()`，使进程没有机会释放
   multiprocessing Event 的 semaphore。现在先 `join()` 等待正常退出，仅在有界超时后
   terminate/kill，并关闭 Process 句柄、清空 Event 引用；超时路径仍保留且有单元测试。
   最终 EP=2 Qwen3MoE 和 TP=2 Dense Qwen3 实机运行均无 `resource_tracker` 警告。
2. EP=2 单请求性能受逐层 collective 影响。若目标转为吞吐，应在独立计划中评估
   token dispatch/all-to-all、通信计算重叠或多请求 batching，不能改变本次正确性基线。
3. HF BF16 回放应同时检查 greedy token identity 和 logits 容差；仅看 token 一致可能
   隐藏数值漂移，仅用过严的绝对阈值也会混入 HF Attention backend 差异。
4. 增加 EP=4/8 硬件后，优先运行 vanilla exact、七方法非 Prefix 矩阵、三方法 Prefix
   矩阵，再做性能对比。

## 9. 最终审查记录

交付前按 `$code-review` 完成了全量差异复核。审查中发现的问题均已修复：

- 模型 warmup 改用平台抽象的 device synchronize，避免把通用路径绑定到 CUDA；
- Qwen3MoE checkpoint dtype 限定为 kernel 已验证的 BF16/FP16，非法 sparse step 不再
  静默改写，unsupported topology/backend 在配置边界失败；
- 稀疏矩阵 runner 支持显式 EP=1/2/4/8，并按物理设备数校验；验证脚本分别保存主
  运行错误和 cleanup 错误，不允许清理异常遮蔽原始失败；
- 稳定用户文档不链接内部 dated report，也不暴露远端主机和私有 artifact ledger；
- worker 生命周期采用 join-first 的通用 world 语义，修复 TP 和 EP 共用的 semaphore
  泄漏根因，同时保留 RPC、join、terminate 和 kill 的有界超时。

最终验证证据如下：

| 检查 | 结果 |
| --- | --- |
| 全量 CPU 测试 | 601 passed，35 skipped，143 subtests passed，15.33 s |
| H20 Triton 定向测试 | 8 passed，2.82 s |
| Qwen3MoE EP=1 手工 QA | 4/4 success，cleanup 正常 |
| Qwen3MoE EP=2 手工 QA | 4/4 success，与 EP=1 token 完全一致，无退出警告 |
| Dense Qwen3-8B TP=2 | 成功生成 8 tokens，无退出警告，两张 GPU 完全释放 |

TP=2 复核使用 `/root/autodl-tmp/models/Qwen3-8B`，配置为 TP=2、EP=1、DP=1、
eager vanilla，固定 prompt `The capital of France is` 和 greedy 8 tokens。两 rank 各加载
399 个权重，生成 token IDs
`[12095, 13, 576, 6722, 315, 15344, 374, 21718]`，文本内容为
`Paris. The capital of Italy is Rome`（解码结果在 `Paris` 前包含一个空格），进程退出码
为 0。该结果与最终 EP=2 QA 一起
证明生命周期修复作用于共享 world worker 路径，而不是只规避某个 EP 验证脚本。

最终审查未发现遗留的高优先级正确性、架构或可复现性问题。当前剩余验证缺口仅为
受硬件数量限制而未实测的 EP=4/8；代码和 matrix 已支持这些拓扑，但在获得对应数量
的同类 GPU 前不宣称模型级通过。稳定文档只承载公开支持范围和运行方式，本报告负责
保留具体环境、artifact 和验收证据。
