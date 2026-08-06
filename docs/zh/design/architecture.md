# 架构

Sparse-vLLM 分为两个 runtime family：

- `src/sparsevllm/`：sparse-first inference engine，包含自己的 scheduler、model runner、cache manager、sparse controller、模型定义和 Triton kernel。
- `src/deltakv/`：HF/Transformers 侧 DeltaKV wrapper、compressor 训练、runtime 参数规范化和 baseline adapter。

## Sparse-vLLM 流程

Sparse-vLLM inference path：

1. `sparsevllm.LLM(model, **kwargs)`
2. `src/deltakv/configs/runtime_params.py` 规范 public runtime 参数名。
3. `src/sparsevllm/config.py` 验证 engine config 和方法兼容性。
4. `src/sparsevllm/engine/cache_manager/base.py` 选择 cache manager。
5. `Scheduler`、`ModelRunner`、`SparseController`、kernel 和选中的 cache manager 执行 prefill 与 decode。

`src/sparsevllm/layers/attention.py` 有意保持通用。它写入 K/V，向 sparse controller 请求 logical read view，并允许 cache manager 通过 `build_decode_view(...)` 等 hook 自定义 decode-time view。

## HF DeltaKV 流程

HF wrapper path：

1. `deltakv.get_chat_api.get_generate_api(..., backend="hf")`
2. 针对 HF backend 规范化 runtime 参数。
3. `sparse_method` 选择 HF model wrapper 或 baseline adapter。
4. 加载 base model config 或 compressor checkpoint config。
5. wrapper config 选择 DeltaKV cache 实现。
6. generation 通过仓库自定义 HF inference helper 运行。

对比 wrapper 实现或 baseline adapter 时使用 HF path；测量 sparse-first engine 时使用 `backend="sparsevllm"`。

## 方法所有权

新增 Sparse-vLLM 稀疏方法应遵循 cache-manager-first 设计：

- 方法特定的 runtime state 属于 `src/sparsevllm/engine/cache_manager/<method>.py`。
- 跨 layer observation 或 scheduling coordination 属于 `src/sparsevllm/engine/sparse_controller.py`。
- `src/sparsevllm/layers/attention.py` 只应调用 generic hook 或 shared kernel。
- Public runtime 参数应使用[运行时参数语义](../configuration/runtime-parameter-semantics.md)中记录的规范名称。

新增一等方法时，请使用仓库内的 `$add-sparse-method` skill。该 skill 编码了预期的文件位置、cache-manager hook 和验证流程。

## Scheduling 所有权

Prefill scheduling 是方法 contract 的一部分。默认 policy 位于 `src/sparsevllm/method_registry.py`；`Config` 负责解析和验证 policy；`src/sparsevllm/engine/scheduler.py` 实现 scheduling 行为。

engine 当前支持：

- `all_chunked`：所有 prefill request 都通过常规 scheduler limit 进行 chunking 和 batching，每个 sequence 每步最多调度 `chunk_prefill_size` 个 token。
- `long_bs1full_short_batch`：在附加受支持的 prefix 后，residual token 数不超过 `long_prefill_offload_threshold` 的 request 使用 atomic full prefill，并且可以互相 batch；更大的 residual 单独运行 chunked RawKV offload prefill，每个 chunk 不超过 `chunk_prefill_size`。

DeltaKV 和 PyramidKV 通过 `prefill_execution_mode()` 实现 long branch。threshold 默认是 `65536` token（64K）。未设置 `chunk_prefill_size` 时，`Config` 默认令其等于 threshold；显式设置时必须满足 `0 < chunk_prefill_size <= long_prefill_offload_threshold`。必要时，`Config` 会把 `max_num_batched_tokens` 提高到 threshold，使边界大小的 full prefill 保持 atomic。PyramidKV 根据 chain prefix attach 后的 residual 应用该边界。DeltaKV 不提供 prefix caching；如果 attached-prefix prefill 到达其 cache manager，会快速失败，因为其 compressed row state 没有 prefix residency contract。

不要在 benchmark script 或一次性 config default 中编码某个方法的 prefill policy。应把 method-to-policy mapping 添加到 registry，并更新 `tests/test_prefill_schedule_policy.py`。

## 重要文件

- `src/deltakv/configs/runtime_params.py`：public runtime 参数 alias 和 legacy key rejection。
- `src/sparsevllm/config.py`：engine config default 和 validation。
- `src/sparsevllm/method_registry.py`：支持的方法名和 prefill policy default。
- `src/sparsevllm/engine/cache_manager/base.py`：method-to-cache-manager routing 和 shared cache-manager hook。
- `src/sparsevllm/engine/scheduler.py`：prefill/decode scheduling 和 admission。
- `src/sparsevllm/layers/attention.py`：通用 K/V storage 和 attention compute path。
- `benchmark/`：LongBench、MathBench、SCBench、NIAH 和多模态 benchmark 入口。
