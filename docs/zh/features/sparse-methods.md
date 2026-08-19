# 核心稀疏方法

Sparse-vLLM 围绕 cache-manager-first sparse runtime 构建。engine 支持 physical eviction、logical masking 和 hybrid compression，而不强迫 `attention.py` 持有方法特定状态。

## 支持的方法

将 `sparse_method` 设置为下列方法名之一。

| 方法 | 类别 | 说明 | 主要 Runtime 参数 |
| --- | --- | --- | --- |
| `vanilla` | Dense baseline | Full attention baseline，用于验证正确性并测量非稀疏 engine path。 | 仅使用通用 engine 参数。 |
| `streamingllm` | Physical eviction | StreamingLLM 风格的固定 sink 加 recent-window cache。保留 prefix/tail 策略之外的 token 会从 active KV cache 中被物理淘汰。 | `sink_keep_tokens`, `recent_keep_tokens` |
| `attention-sink` | Physical eviction | attention-sink alias policy，使用相同的 sink-token 和 recent-window 保留模型。适合将 sink-window 行为与其他 physical eviction 方法对比。 | `sink_keep_tokens`, `recent_keep_tokens` |
| `snapkv` | Physical eviction | SnapKV 风格的 token selection 在 prefill 后保留紧凑的重要历史 token 集合，只物理保留选中的 KV position，以减小 cache footprint。 | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `h2o` | Physical eviction | H2O 在与物理 row 对齐的 token vector 中累积归一化 token importance。Prefill 每个 chunk 都评分并物理淘汰；decode 每个 token 都评分，通常在 row 达到 `h2o_decode_budget + h2o_decode_eviction_interval` 时批量选择、compact 和释放，再回到 decode budget。若某次 decode 用尽最后一个空闲 KV slot，所有已超过 budget 的 active decode row 都会提前 compact，包括本轮未调度的 row，以保证下一步可继续；idle chain row 不参与这类压力淘汰。budget 与 interval 之和必须能被 64 整除，以满足 scored decode kernel 的对齐要求。每次淘汰保留 heavy hitter 与 recent 后缀，最后一个 prefill chunk 也收缩到 decode budget。Prefill 默认使用归一化 attention mass；实验 raw-QK 模式先对 observation window 的 max-logit token vector 做一次归一化再累计。decode 先在 query head 间对 raw QK logit 做 max reduction，再按 token 归一化。Sparse-vLLM v1 在 KV head 之间共享一套 token 选择。 | `h2o_decode_budget`, `h2o_decode_eviction_interval`（默认 128）, `h2o_prefill_budget`, `h2o_recent_ratio`, `h2o_prefill_score_window`, `sparse_prefill_score_mode` |
| `pyramidkv` | Physical eviction | PyramidKV 风格、依赖 layer 的 KV 保留方式。它在 layer 之间分配 sparse budget，并物理存储选中的 context token。 | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `omnikv` | Logical masking | OmniKV 保留 physical cache，但为选定 layer 构建 sparse attention view。适用于不改写 cache storage、同时降低 attention 计算量的场景。 | `full_attention_layers`, `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens` |
| `quest` | Query-aware page selection | QuEST 根据 decode query 选择 token page。prefill 保持 dense，decode 通过 page/chunk budget 执行 sparse selection。 | `quest_chunk_size`, `quest_skip_layers`, `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens` |
| `deltakv` | Hybrid compression | 依赖 compressor 的精简 DeltaKV runtime。旧配置中的 `deltakv-less-memory*` 名称会规范到此方法，但实际 benchmark run 仍需要匹配的 compressor checkpoint。 | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

Sparse-vLLM 在内部将该值存为 `vllm_sparse_method`，但 public command 和 `LLM(...)` kwarg 应使用 `sparse_method`。

SnapKV、PyramidKV 和 H2O 的 `sparse_prefill_score_mode` 默认值为
`probability`，保持归一化 softmax probability 的评分定义。实验选项
`tilelang_raw_qk` 改为按 observation query 与 query head 上的最大 raw QK
logit 排序；它要求 BF16、head dimension 128、FP32 score storage、
TileLang 0.1.9、apache-tvm-ffi 0.1.10 和 SM90。不支持的 raw 模式配置会明确失败，不会静默
回退到另一种评分定义。在 H2O raw 模式中，max-logit token vector 会先做
一次归一化，再按 observation query 数加权后写入累计 importance vector。
仅在该 raw 模式下，`h2o_prefill_score_window=0` 表示完整当前 chunk；
probability 模式仍要求窗口位于 `[1, 128]`。

## Prefill Scheduling Policy

Prefill scheduling 是方法 contract 的一部分，由 registry 管理。唯一事实来源是 `src/sparsevllm/method_registry.py`；benchmark script 和用户配置不应重新定义方法语义。

| Policy | Runtime 语义 | 当前默认方法 |
| --- | --- | --- |
| `all_chunked` | 每个 prefill request 都受 `chunk_prefill_size` 和 scheduler 常规 batch 限制约束；忽略 `long_prefill_offload_threshold`。 | `vanilla`, `streamingllm`, `attention-sink`, `snapkv`, `h2o`, `quest`, `omnikv` |
| `long_bs1full_short_batch` | 在附加受支持的 prefix 后，residual 不超过 `long_prefill_offload_threshold` 时使用 atomic full prefill，并且可以互相 batch；更大的 residual 被隔离，并使用不超过 `chunk_prefill_size` 的 RawKV offload chunk。 | `pyramidkv` 和 DeltaKV family 方法 |

DeltaKV family 方法和 PyramidKV 只对外提供 `long_bs1full_short_batch` policy。threshold 默认是 `65536` token（64K）。未设置 `engine_prefill_chunk_size` 时，它默认等于 threshold；显式值必须为正数且不大于 threshold。必要时，`Config` 会提高 `max_num_batched_tokens`，使一个 threshold 大小的 full prefill 能够原子容纳。PyramidKV 根据 chain prefix attach 后的 residual 进行分类。DeltaKV 不支持 prefix caching，并会在修改 compressed 或 quantized row metadata 前拒绝 attached-prefix prefill。

启用 full-layer KIVI 时，DeltaKV 的 decode 常驻 raw 尾部池与 `max_model_len` 大小的 prefill staging buffer 是两块独立容量。多个 short prefill 通过互不重叠的 request range 共享 staging buffer；常驻 raw 尾部的 slot 数不是 prefill batch 上限。

## Prefix Cache 模式

`enable_prefix_caching=true` 支持两种有意分离的布局。
`prefix_cache_mode=auto` 为 vanilla/OmniKV/QuEST 选择 radix，为
SnapKV/H2O/PyramidKV/R-KV/SkipKV 选择线性 chain。也可以显式请求 `radix`
或 `chain`，但不兼容的方法/模式组合会快速失败。

Chain 布局跨 turn 保留同一个驻留 `seq_id`，且永不分支。调用方发送完整逻辑
上下文和服务端返回的 `chain_id`；服务端验证 processed boundary 后只转发新增
suffix。方法 KV 与 metadata 仍由 cache manager 持有。Idle chain 采用严格
LRU 回收，active writer 保持 pinned。Rank 0 使用紧凑 32-bit storage 保存
processed logical token ID，以便文本 continuation 保持驻留的 BPE tokenization。
该 CPU 历史受 `max_model_len * max_num_seqs_in_gpu` 限制，并随 chain 一起回收。

`Config` 会把 `None`、空字符串和 `auto` 解析为 registry default。与方法默认值不一致的显式 policy 会快速失败，避免实验静默改变 scheduler 语义。任何 policy override 都应视为显式 ablation，并随 benchmark result 一起记录。

## Runtime 所有权

- 方法特定的 runtime state 属于 `src/sparsevllm/engine/cache_manager/`。
- 跨 layer observation 或 scheduling coordination 属于 `src/sparsevllm/engine/sparse_controller.py`。
- `src/sparsevllm/layers/attention.py` 应保持通用，只调用 shared hook。
- 新的一等方法必须在 `src/sparsevllm/method_registry.py` 中注册默认 prefill policy，并在 `tests/test_prefill_schedule_policy.py` 中覆盖。

## Query-Aware 参数

`quest` runtime 参数：

- `quest_chunk_size`：QuEST page/chunk 的 token 数量；
- `sink_keep_tokens`、`decode_keep_tokens`、`recent_keep_tokens`：QuEST 在 config 构造期间将三者相加，一次性得到 decode token budget；
- `quest_skip_layers`：在 decode 中保持前 N 个 layer 为 dense。

`quest_token_budget` 已不再是 runtime input。传入该参数会快速失败；请删除它，改为配置上述三个通用 keep-token 字段。
