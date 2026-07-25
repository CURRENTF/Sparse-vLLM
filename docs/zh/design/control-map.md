# Sparse-vLLM 控制图

本页展示 Sparse-vLLM runtime 的所有权和 control flow。它不是 benchmark result，不应作为方法声明的证据。请用它判断改动应放在哪里、哪些文档可信，以及报告结果前应运行哪些检查。

## 文档导航

- 选择文档位置时，从 `docs/zh/README.md` 开始。
- 稳定的 runbook 和 contract 位于 `docs/zh/` 下对应主题目录。
- 不要在仓库文档中保存本地 run ledger。面向仓库的声明需要证据时，应使用 run artifact 本身。
- `docs/zh/configuration/runtime-parameter-semantics.md` 是规范参数 contract。添加新的 public run config 前应保持它同步。

## 一句话模型

Sparse-vLLM 是 sparse-first inference engine：`Scheduler` 决定运行什么，`ModelRunner` 执行，`Attention` 调用 generic hook，`CacheManager` 实现负责方法特定的 cache state、allocation、view、reconstruction 和 graph-stable metadata。

对于 SnapKV、PyramidKV、R-KV 和 SkipKV 的 chain prefix cache，
`ChainCacheIndex` 只负责逻辑生命周期，`ChainCacheCoordinator` 规划状态转换，
cache manager 保留全部 payload/metadata，`RuntimeState` 执行回收。OpenAI
dispatcher 在构造 stream 前确认 admission；smart-router affinity 来自并行的
只读 worker probe，而不是 router 自己维护的 chain map。Rank 0 仅为保持文本
continuation 的准确 BPE identity 而保留紧凑逻辑 token ID；该历史有容量上限，
且不属于物理 KV payload。

## Runtime 流程

```mermaid
flowchart TD
    A["LLM(..., **kwargs)"] --> B["normalize_runtime_params(..., backend='sparsevllm')"]
    B --> C["Config.__post_init__: validate model, method, graph, budgets"]
    C --> D["ModelRunner: load model, create CacheManager, SparseController"]
    C --> E["Scheduler: waiting/decode queues and admission"]
    E --> F["LLMEngine.step"]
    F --> G["Scheduler.schedule"]
    G --> H["ModelRunner.run"]
    H --> I["CacheManager.prepare_step"]
    I --> J["SparseController.prepare_forward"]
    J --> K["model layers"]
    K --> L["Attention.forward"]
    L --> M["CacheManager store/view/reconstruct hooks"]
    L --> N["SparseController read view and layer-end selection"]
    H --> O["Sampler"]
    H --> P["SparseController.post_forward"]
    P --> Q["CacheManager eviction/compression hooks"]
    F --> R["Scheduler.postprocess and finished/free slots"]
```

## 目录所有权

| 路径 | 角色 | 所有权规则 |
| --- | --- | --- |
| `src/sparsevllm/config.py` | Runtime dataclass、validation、graph constraint、方法规范化 default。 | Public 参数行为必须同步到 `docs/zh/configuration/runtime-parameter-semantics.md`。 |
| `src/sparsevllm/method_registry.py` | 稀疏方法 alias 和默认 prefill policy。 | 新 method string 和 policy default 从这里开始。 |
| `src/sparsevllm/engine/llm_engine.py` | Public engine lifecycle、tokenizer、scheduler loop、warmup、吞吐量 logging。 | 不应增加方法特定 runtime 逻辑。 |
| `src/sparsevllm/engine/scheduler.py` | Prefill/decode batching、长短请求分离、prompt admission、preemption。 | 使用 cache-manager budget hook，不了解方法内部实现。 |
| `src/sparsevllm/engine/model_runner.py` | 模型加载、TP RPC、CUDA Graph runner、prepare/run/sample orchestration。 | 负责执行机制，不负责 token-selection policy。 |
| `src/sparsevllm/engine/cache_manager/base.py` | Cache-manager interface 和方法 routing。 | 方法特定的 persistent state 属于该 interface 之后。 |
| `src/sparsevllm/engine/cache_manager/*.py` | 各稀疏方法的 physical/logical KV state。 | 稀疏方法实现的主要位置。 |
| `src/sparsevllm/engine/sparse_controller.py` | 跨 layer attention-score 收集、动态 token selection、post-forward compression trigger。 | Persistent method metadata 应保存在 cache manager，而不是这里。 |
| `src/sparsevllm/layers/attention.py` | 通用 KV store、attention kernel dispatch 和 hook 调用。 | 必要时添加 generic hook；避免方法特定 branch。 |
| `src/sparsevllm/triton_kernel/` | Kernel 实现。 | shape/dtype 假设无效时，kernel wrapper 应快速失败。 |
| `src/deltakv/` | HF wrapper、compressor 训练、HF 侧 cache 和模型集成。 | 用于 HF parity 和 compressor 行为，不负责 Sparse-vLLM engine。 |
| `benchmark/` 和 `scripts/` | 评估、调试、分析和吞吐量脚本。 | 分别保存 raw output、parsed output、per-sample status、aggregate metric 和 run info。 |

## 方法类别

| 类别 | Sparse-vLLM 方法名 | 核心行为 | 主要文件 |
| --- | --- | --- | --- |
| Dense | `vanilla` / `""` | 完整 KV cache，无 sparse selection。 | `standard.py`、通用 attention path |
| Streaming window | `streamingllm`, `attention-sink`, `attention_sink` | 物理淘汰，只保留 sink 加 recent token。 | `streamingllm.py`、`standard.py` 风格机制 |
| SnapKV / PyramidKV | `snapkv`, `pyramidkv` | 基于 score 的 keep selection 后执行 physical eviction；PyramidKV 改变 per-layer budget。 | `snapkv.py`, `sparse_controller.py` |
| OmniKV | `omnikv` | 根据 observation-layer score 构建 logical mask/view。应使用 `scripts/analysis/select_omnikv_full_layers.py` 针对模型校准 full layer，再通过 `full_attention_layers` 传入。 | `omnikv.py`, `sparse_controller.py`, `omnikv_fused.py` |
| QuEST | `quest` | Query-aware decode page/chunk selection。 | `quest.py` |
| DeltaKV | `deltakv` | 基于 compressor 的 hybrid cache：sparse full/reference pool 加 compressed latent state。 | `deltakv.py`, `deltakv_kernels.py` |
| DeltaKV | `deltakv` 加 legacy `deltakv-less-memory*` alias | 基于 compressor 的精简 DeltaKV runtime，使用 graph-stable decode metadata。 | `deltakv_runtime.py`, `deltakv_less_memory*.py`, `deltakv_kernels.py` |

## 状态所有权 Contract

- `Sequence` 持有 request-local counter：prompt length、prefilled length、当前 chunk size、generated token 数和 finished status。
- `Scheduler` 持有 queue membership 和 admission decision。它必须向 cache manager 查询 cost、budget 和 full-prefill routing。
- `CacheManager` 持有 physical slot、row map、full/sparse pool、compressed length、graph-stable metadata、临时 reconstruction slot 和方法 allocation 算术。
- `SparseController` 持有 per-step、per-layer sparse state 和选中 index 的跨 layer 传播。它不应持有 long-lived cache metadata。
- `Attention` 不持有 policy 或 persistent state。它存储当前 K/V、请求 read view、运行 generic prefill/decode kernel，并调用 layer-end hook。
- CUDA Graph runner 持有 graph capture/replay 机制；cache manager 提供 graph-stable buffer 和 plan reference。

## 当前最难控制的位置

| 文件 | 难点 | 处理方式 |
| --- | --- | --- |
| `src/sparsevllm/engine/cache_manager/deltakv_less_memory.py` | 体量很大的 direct-residual/full-layer-KIVI/static-graph 实现。 | 将其视为多个逻辑区域：allocation、prefill staging、full-layer KIVI、sparse raw/ref view、static decode plan、reconstruction/writeback。围绕改动区域添加测试。 |
| `src/sparsevllm/engine/cache_manager/deltakv.py` | Compressor-backed V4 path 同时包含 clustering、latent storage、full pool、staging、reconstruction 和 graph hook。 | 避免外观性修改。只在有针对性的 HF-vs-Sparse 或 kernel 对比支持下改动。 |
| `src/sparsevllm/engine/sparse_controller.py` | OmniKV、DeltaKV、SnapKV、PyramidKV 的跨 layer policy、score dtype 和 debug capture 在此汇合。 | 不要加入新的 persistent state；这里只添加 orchestration 或 score/selection 逻辑。 |
| `src/sparsevllm/layers/attention.py` | 文件不大，但所有方法都经过它，因此影响范围很大。 | 保持 method-agnostic。优先增加 cache-manager hook，而不是在此添加 branch。 |

## 改动护栏

修改 Sparse-vLLM runtime 代码前：

1. 确认 backend：HF、Sparse-vLLM 或两者。
2. 确认 method family 和 graph mode：eager、decode graph、prefill graph 或两者。
3. 确认 state owner。跨 step 持续存在的状态属于 cache manager。
4. 新配置不要使用 legacy public runtime name。使用 `sparse_method`、`deltakv_checkpoint_path`、`decode_keep_tokens`、`prefill_keep_tokens`、`sink_keep_tokens`、`recent_keep_tokens`、`full_attention_layers`、`hf_prefill_chunk_size` 和 `engine_prefill_chunk_size`。
5. Sparse-vLLM keep budget 是 token count，不是 ratio。
6. 所有 fallback 都必须显式且有文档记录。不要静默忽略错误 config、缺失 checkpoint、缺失 dataset、parse failure 或 metric failure。
7. 新增或重构稀疏方法时，保持 cache-manager-first：`attention.py` 通用、方法状态不放在 `utils/`，方法 default 注册到 `src/sparsevllm/method_registry.py`。

## 最小本地检查

以下低成本检查不需要模型权重，但需要 `README.md` 中的项目 runtime 环境或等价依赖（`torch`、`triton`、`transformers` 等）：

```bash
PYTHONPATH=$PWD:$PWD/src python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/method_registry.py \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/engine/scheduler.py \
  src/sparsevllm/engine/model_runner.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/layers/attention.py

PYTHONPATH=$PWD:$PWD/src python -m unittest \
  tests.test_runtime_param_normalization \
  tests.test_prefill_schedule_policy \
  tests.test_sampler

git diff --check
```

DeltaKV kernel/cache 改动的 CUDA 检查：

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=$PWD:$PWD/src python -m unittest \
  tests.test_deltakv_less_memory_kernel
```

声明 HF/Sparse parity 前的正确性检查：

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=$PWD:$PWD/src python \
  scripts/debug/compare_logits_hf_sparsevllm.py \
  --model_path <MODEL_DIR> \
  --compressor_path <COMPRESSOR_DIR> \
  --cases longbench \
  --methods deltakv \
  --sparse_method deltakv \
  --hf_sparse_method delta_compressed_quant_kivi_full_fp8_ref \
  --longbench_task hotpotqa \
  --longbench_sample_idx 0 \
  --teacher_forced_decode_steps 3 \
  --output_dir <OUTPUT_DIR> \
  <explicit sparse/runtime params>
```

正确性检查后的吞吐量检查：

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=$PWD:$PWD/src python \
  scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_DIR> \
  --methods <method> \
  --lengths <prompt_tokens> \
  --batch_sizes <batch_size> \
  --output_len <tokens> \
  --hyper_params '<canonical JSON params>'
```

## 清理候选项

以下任务用于恢复控制边界，不是紧急 correctness fix：

1. 在 cache-manager 创建过程针对 DeltaKV variant 修改 `config.vllm_sparse_method` 之前，增加 immutable `requested_sparse_method` 或 run-info 字段，使 log 和 artifact 更易解释。
2. 明确 cache manager 中的 RoPE 所有权。Qwen3 theta/dtype 修复说明 cache manager 需要清晰持有 RoPE 或相关 position module。
3. 让仓库文档专注于稳定 contract 和 runbook，不要添加本地 experiment ledger。
4. 在功能改动触及具体区域前，不要拆分巨大的 DeltaKV cache manager。拆分时，应分别保留 allocation、staging、graph metadata 和 reconstruction 测试。
5. 方法 alias、graph support 或 public 参数变化时，保持 `docs/zh/configuration/runtime-parameter-semantics.md` 同步。
