# 核心稀疏方法

Sparse-vLLM 围绕 cache-manager-first sparse runtime 构建。engine 支持 physical eviction、logical masking 和 hybrid compression，而不强迫 `attention.py` 持有方法特定状态。

## 支持的方法

将 `sparse_method` 设置为下列方法名之一。

| 方法 | 类别 | 说明 | 主要 Runtime 参数 |
| --- | --- | --- | --- |
| `vanilla` | Dense baseline | Full attention baseline，用于验证正确性并测量非稀疏 engine path。 | 仅使用通用 engine 参数。 |
| `streamingllm` | Physical eviction | StreamingLLM 风格的固定 sink 加 recent-window cache。保留 prefix/tail 策略之外的 token 会从 active KV cache 中被物理淘汰。 | `sink_keep_tokens`, `recent_keep_tokens` |
| `attention-sink` | Physical eviction | attention-sink alias policy，使用相同的 sink-token 和 recent-window 保留模型。适合将 sink-window 行为与其他 physical eviction 方法对比。 | `sink_keep_tokens`, `recent_keep_tokens` |
| `snapkv` | Physical eviction | SnapKV 风格的 token selection 使用 prompt 末尾的 observation window，在生成前选出并保留紧凑的重要 prompt KV。当前与论文对齐的 decode 路径不再评分，也不会再次执行 SnapKV selection，只追加生成 token。 | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `h2o` | Physical eviction | 中间 prefill chunk 可压缩到 `h2o_prefill_budget`，最终 prompt 压缩到 `h2o_decode_budget`。默认 decode 不评分或驱逐，物理 row 随生成 token 增长；开启 `h2o_decode_eviction` 后逐步累计概率分数并周期驱逐。 | `h2o_decode_eviction`, `h2o_decode_budget`, `h2o_decode_eviction_interval`, `h2o_prefill_budget`, `h2o_recent_ratio`, `h2o_prefill_score_window`, `sparse_prefill_score_mode` |
| `pyramidkv` | Physical eviction | PyramidKV 风格、依赖 layer 的 KV 保留方式。它在 layer 之间分配 sparse budget，并物理存储选中的 context token。 | `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `sparse_prefill_score_mode` |
| `omnikv` | Logical masking，可选 offload | 跨层共享 token 选择；可将稀疏层完整历史保存在 pinned CPU 内存，decode 精确取回当前选择的 KV。 | `full_attention_layers`, `decode_keep_tokens`, `sink_keep_tokens`, `recent_keep_tokens`, `enable_omnikv_offload` |
| `quest` | Query-aware page selection | QuEST 根据持久化的 page min/max summary 选择 token page，prefill 保持 dense。显式 KV 模型在 key 坐标中评分；GLM-4.7-Flash 使用匹配的 absorbed decode query 对融合 MLA latent/RoPE cache 评分，同时 compute payload 继续保持 latent。 | `quest_chunk_size`, `quest_skip_layers`, `sink_keep_tokens`, `decode_keep_tokens`, `recent_keep_tokens` |
| `deltakv` | Hybrid compression | 依赖 compressor 的精简 DeltaKV runtime。旧配置中的 `deltakv-less-memory*` 名称会规范到此方法，但实际 benchmark run 仍需要匹配的 compressor checkpoint。 | `deltakv_checkpoint_path`, `deltakv_latent_dim`, `deltakv_center_ratio`, `deltakv_neighbor_count`, `deltakv_latent_quant_bits`, `full_layer_kv_quant_bits` |

Sparse-vLLM 在 public command、`LLM(...)`、runtime config 与内部消费者中统一使用 `sparse_method`。


## OmniKV KV offload

在 `LLM(...)` 或运行配置中设置
`sparse_method="omnikv", enable_omnikv_offload=True`。开关默认 `False`，
用于其他 sparse method 会报错。继续使用模型原来的 full-layer profile 和 token 预算。

支持 CUDA uniform FP16/BF16 显式 KV，以及已有 BF16 MLA 的 512 维 latent
与 64 维 RoPE 布局。全注意力层的完整 KV 留在 GPU；稀疏层的完整历史保存到
pinned CPU 内存，GPU 保留有界的、按请求独立的 LRU 缓存池。
Top-K 选择保持精确：命中项直接复用，只回读未命中的历史；搬运按稀疏层逐层提前执行。
MLA 保持压缩表示。沿用模型已有 TP、EP、TP+EP 语义，各 rank 独立保存 backing。

历史 KV 预分配以 `max_model_len × max_num_seqs_in_gpu` 为上限，
开启 prefix caching 也不扩大该上限；GPU 和主机内存预算可能进一步降低容量。
独立的 prefix offload 存储也计入主机预算，并考虑 pinned 分配的大小取整。

可用 `omnikv_offload_cache_tokens` 设置每个稀疏层、每个请求的 GPU 缓存容量。
默认 `None` 将 sink/keep/recent 总预算向上取整到 2 的幂，并以 `max_model_len`
为上限。显式正数必须覆盖选择预算；设为 `0` 则关闭 LRU，每步完整回读 selected
历史。更大的缓存池会用更多显存换取更少的回读；缓存池和索引元数据均计入容量预算。

在模型已有兼容范围内，可以同时开启 prefix caching 和 decode CUDA Graph。
Qwen3-MoE 当前不支持 OmniKV prefix caching，开启 active offload 也不改变这个限制。
命中请求共享历史，
suffix prefill 使用完整前缀，新生成的 suffix 保持私有。
`enable_prefix_cache_offload` 是独立开关，用于备份和降级闲置前缀块，
包括全注意力层 KV。开启 active OmniKV offload 时该组合也支持 MLA；
需要设置正数 `prefix_cache_host_size_gb`，并足以容纳配置的前缀块容量。

该选项用于释放 KV 显存容量，但会消耗大量 PCIe/host 内存带宽，固定 batch
的 decode 延迟可能上升。全注意力层 KV 和一份完整历史 prefill buffer
仍随上下文增长。Chunked prefill 从主机恢复此前的稀疏层历史，当前 chunk
直接使用 GPU 上的 KV，同时保留向主机写穿；历史重载仍可能增加 TTFT。
多路 CPU 主机应尽量让 pinned 内存位于对应 GPU 的本地 NUMA 节点，
远端内存和其他任务的主机内存流量可能降低搬运吞吐。
对延迟敏感的部署，应先用 BenchProbe 匹配实际模型、上下文和并发进行测量。

Prefill 加速由 `prefill_sparse_method` 独立选择。当前支持两种方法：
`h2o_prefill` 用于中间 chunk 的物理 KV 压缩，`flashprefill_v2` 用于稀疏化
prefill attention 计算。它们是同一条轴上的备选项，可以分别与兼容的 cache/decode
方法组合。H2O prefill/decode 组合矩阵及“省略”和“显式空字符串”的兼容规则见
[runtime 参数语义](../configuration/runtime-parameter-semantics.md#prefill-sparsity)。

> [!NOTE]
> 两种 score-free decode contract 的论文来源不同。[SnapKV 论文](https://arxiv.org/abs/2404.14469)
> 使用 prompt 末尾的 observation window 选择 prompt KV；增加 decode-time
> 重新评分和淘汰属于 Sparse-vLLM 增强。[H2O 论文](https://arxiv.org/abs/2306.14048)
> 则定义了跨连续 decode step 的动态保留策略。Sparse-vLLM 对中间 chunk 的 H2O
> 压缩是自己提出的 prefill 扩展。最终 prompt 压缩虽然发生在 final-prefill boundary，
> 但它准备的是生成阶段消费的短 cache，因此属于 decode contract。可选的在线评分更新
> 向原始 H2O 算法靠近；周期性的 batch 淘汰和有限 prefill observation window
> 仍属于系统或算法变体。

SnapKV 的 `sparse_prefill_score_mode` 默认为 `logits`，PyramidKV 默认为
`probability`。H2O（含独立的 `h2o_prefill`）默认使用
`sparse_prefill_score_mode="logits"` 和 `h2o_prefill_score_window=128`。
显式指定的 score mode 和 window 会覆盖默认值，但 `h2o_decode_eviction=True`
会强制使用 `sparse_prefill_score_mode="probability"`。

H2O 默认采用有限 query window 的近似评分。如需使用完整 chunk 的归一化
attention mass 评分，请显式设置 `sparse_prefill_score_mode="probability"`
和 `h2o_prefill_score_window=0`。此时每个 KV layer 独立地对完整当前 query
chunk 的归一化 attention probability 求和，并在 prefill chunk 之间累计
attention mass。H2O probability 模式会输出性能警告，因为即使复用 attention
LSE，仍需额外计算 QK 评分。两种模式都要求每个 H2O KV layer 独立保存评分；
decode score 收集与周期淘汰默认关闭，可通过 `h2o_decode_eviction=True` 开启。

该开关要求 `sparse_method="h2o"`。开启后每个 decode step 累计归一化 attention
mass，物理 row 达到 `h2o_decode_budget + h2o_decode_eviction_interval` 时保留
heavy hitters 和 recent tokens，压缩回 decode budget；显存容量压力可使超预算的
active row 提前驱逐。即使显式设置 `logits`，也会强制改为 `probability` 并输出警告。
`h2o_prefill_score_window` 保持用户设置，允许非零值；沿用概率模式的 `[0, 128]`
范围。MLA latent 模型使用显式近似：对跨 head 取 max 的 decode logits 计算
`softmax(scale * RAW_QK_REDUCED)`，再累计和驱逐。这不等价于每个 head 先归一化
再归约，尚未与原 H2O 完全对齐；启用时每个进程仅警告一次。MLA prefill 评分和
默认的 score-free decode 路径不受影响。

## Prefill Scheduling Policy

Prefill scheduling 是方法 contract 的一部分，由 registry 管理。唯一事实来源是 `src/sparsevllm/method_registry.py`；benchmark script 和用户配置不应重新定义方法语义。

| Policy | Runtime 语义 | 当前默认方法 |
| --- | --- | --- |
| `all_chunked` | 每个 prefill request 都受 `engine_prefill_chunk_size` 和 scheduler 常规 batch 限制约束；忽略 `long_prefill_offload_threshold`。 | `vanilla`, `streamingllm`, `attention-sink`, `snapkv`, `h2o`, `quest`, `omnikv` |
| `long_bs1full_short_batch` | 在附加受支持的 prefix 后，residual 不超过 `long_prefill_offload_threshold` 时使用 atomic full prefill，并且可以互相 batch；更大的 residual 被隔离，并使用不超过 `engine_prefill_chunk_size` 的 RawKV offload chunk。 | `pyramidkv` 和 DeltaKV family 方法 |

DeltaKV family 方法和 PyramidKV 只对外提供 `long_bs1full_short_batch` policy。threshold 默认是 `65536` token（64K）。未设置 `engine_prefill_chunk_size` 时，它默认等于 threshold；显式值必须为正数且不大于 threshold。必要时，`Config` 会提高 `max_num_batched_tokens`，使一个 threshold 大小的 full prefill 能够原子容纳。PyramidKV 根据 chain prefix attach 后的 residual 进行分类。DeltaKV 不支持 prefix caching，并会在修改 compressed 或 quantized row metadata 前拒绝 attached-prefix prefill。

启用 full-layer KIVI 时，DeltaKV 的 decode 常驻 raw 尾部池与 `max_model_len` 大小的 prefill staging buffer 是两块独立容量。多个 short prefill 通过互不重叠的 request range 共享 staging buffer；常驻 raw 尾部的 slot 数不是 prefill batch 上限。

## Prefix Cache 模式

`enable_prefix_caching=true` 支持两种有意分离的布局。
`prefix_cache_mode=auto` 为 vanilla/OmniKV/QuEST 选择 radix，为
SnapKV/H2O/PyramidKV/R-KV/SkipKV 选择线性 chain。也可以显式请求 `radix`
或 `chain`，但不兼容的方法/模式组合会快速失败。
GLM-4.7-Flash QuEST 是 storage-specific 例外：当前 latent QuEST 路径不支持
Prefix Cache 或 Prefix offload，配置会明确拒绝这两种组合。
已有 vanilla/OmniKV radix tree 可通过
[Prefix cache 修剪](prefix-cache-pruning.md)中的 SnapKV 或 KVzip 打分维护接口
进行物理压紧；QuEST tree 会明确拒绝修剪。

Chain 布局跨 turn 保留同一个驻留 `seq_id`，且永不分支。调用方发送完整逻辑
上下文和服务端返回的 `chain_id`；服务端验证 processed boundary 后只转发新增
suffix。方法 KV 与 metadata 仍由 cache manager 持有。Idle chain 采用严格
LRU 回收，active writer 保持 pinned。Rank 0 使用紧凑 32-bit storage 保存
processed logical token ID，以便文本 continuation 保持驻留的 BPE tokenization。
该 CPU 历史受 `max_model_len * max_num_seqs_in_gpu` 限制，并随 chain 一起回收。

`Config` 会把 `None`、空字符串和 `auto` 解析为 registry default。与方法默认值不一致的显式 policy 会快速失败，避免实验静默改变 scheduler 语义。任何 policy override 都应视为显式 ablation，并随 benchmark result 一起记录。

## Runtime 所有权

- 持久物理缓存和跟随 Prefix Cache 的元数据属于
  `src/sparsevllm/engine/cache_manager/`。
- 当前步骤的逐层逻辑状态、打分、跨层选择以及压缩/淘汰触发属于
  `src/sparsevllm/engine/sparse_methods/` 下的
  `SparseMethodRuntime`。
- `src/sparsevllm/engine/sparse_controller.py` 是稳定、与方法无关的统一入口，
  不得包含方法名热路径分支。
- `src/sparsevllm/layers/attention.py` 应保持通用，只调用 shared hook。
- 新的一等方法必须在 `src/sparsevllm/method_registry.py` 中注册默认 prefill policy，并在 `tests/test_prefill_schedule_policy.py` 中覆盖。

完整的接口、职责划分、Prefix Cache、CUDA Graph 和扩展规则参见
[稀疏方法运行时架构](../design/sparse-method-runtime.md)。

## Query-Aware 参数

`quest` runtime 参数：

- `quest_chunk_size`：QuEST page/chunk 的 token 数量；
- `sink_keep_tokens`、`decode_keep_tokens`、`recent_keep_tokens`：QuEST 在 config 构造期间将三者相加，一次性得到 decode token budget；
- `quest_skip_layers`：在 decode 中保持前 N 个 layer 为 dense。

`quest_token_budget` 已不再是 runtime input。传入该参数会快速失败；请删除它，改为配置上述三个通用 keep-token 字段。
