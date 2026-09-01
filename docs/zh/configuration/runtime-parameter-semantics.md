# 运行时参数语义

Sparse-vLLM 只有一个推理后端：`src/sparsevllm/` 下的原生引擎。
`LLM(...)`、`Config`、JSON 配置、benchmark manifest 与内部代码使用完全
相同的 runtime 参数名；引擎不再维护 public-to-internal alias 层。

## Public 参数名

命令、JSON 配置和 benchmark manifest 应使用语义化 public 名称：

| 规范名称 | 含义 |
| --- | --- |
| `sparse_method` | 稀疏方法选择器。 |
| `prefill_sparse_method` | 与 cache/decode `sparse_method` 正交的 prefill attention 算法选择器。 |
| `deltakv_checkpoint_path` | DeltaKV compressor checkpoint 路径。 |
| `engine_prefill_chunk_size` | Prefill 最大调度 chunk。 |
| `sink_keep_tokens` | Sink token 预算。 |
| `recent_keep_tokens` | Recent token 预算。 |
| `full_attention_layers` | `auto`（默认）、逗号分隔的字符串或 full-layer index 列表。`auto` 按方法和模型名精确匹配 profile；catalog 条目可以由 OmniKV 和 DeltaKV 共享，也可以限定到单一方法。 |
| `deltakv_neighbor_count` | DeltaKV reference neighbor 数量。 |
| `deltakv_center_ratio` | DeltaKV reference center 比例。 |
| `deltakv_latent_dim` | Compressor latent 宽度。 |
| `deltakv_latent_quant_bits` | Latent state 量化位数。 |
| `deltakv_latent_quant_group_size` | Latent 量化 group size。 |
| `gpu_memory_utilization` | 引擎可使用的 GPU 显存比例。 |
| `decode_graph` | 启用 decode CUDA Graph。 |

`full_attention_layers=auto` 会对模型路径或仓库名的最后一段做不区分大小写的
精确匹配，并识别 `models--org--model/snapshots/...` 形式的 Hugging Face cache
路径；不会使用子串模糊匹配。未登记的 OmniKV 或 DeltaKV 模型会明确提示先
校准。当 catalog 条目同时列出两个方法时，两者有意共享相同的 full-layer
anchor；显式传入的层列表仍会覆盖 `auto`。

`sparse_method`、`deltakv_checkpoint_path`、`engine_prefill_chunk_size`、
`sink_keep_tokens`、`recent_keep_tokens`、`full_attention_layers`、
`deltakv_neighbor_count`、`deltakv_center_ratio`、`deltakv_latent_dim`、
`deltakv_latent_quant_bits`、`deltakv_latent_quant_group_size`、`device_memory_utilization` 和
`decode_graph*` 等旧 alias 不再接受。未知名称会在 engine boundary 直接失败，
不会被重写。

## Token 预算

原生 Sparse-vLLM 的 token 预算必须是显式整数。像
`decode_keep_tokens=0.17` 这样的 ratio 值依赖目标 context length，因此会被
拒绝；启动前应先转换为 token 数。

`quest_token_budget` 不是 public 参数。QuEST 的总选择预算由
`sink_keep_tokens`、`decode_keep_tokens` 和 `recent_keep_tokens` 推导。

## Prefill 调度

Prefill policy 的唯一事实来源是 `src/sparsevllm/method_registry.py`：

- `all_chunked` 使用 `engine_prefill_chunk_size` 进行常规 chunked batching。
- `long_bs1full_short_batch` 对符合条件的 request 执行 atomic full prefill，
  并使用 `long_prefill_offload_threshold` 划分长请求。

不要在 benchmark script 中复制 method policy。运行报告应记录解析后的
method、policy、chunk size、context length、batch size 和 checkpoint 路径。

## Prefill 稀疏

`prefill_sparse_method` 独立选择 prefill attention 算法，不替代
`sparse_method`。`flashprefill_v2` 支持 `vanilla`、`omnikv`、`quest`、
`snapkv` 和 `h2o`，但仅限 explicit-KV MHA 模型；MLA latent 模型会在配置阶段拒绝
该 prefill 方法。这些组合也支持各方法已经支持的 prefix-cache mode。H2O 默认解析为
`prefill_sparse_method="h2o_prefill"`；显式选择 `flashprefill_v2` 只改变 prefill
attention 计算。H2O 仍然通过 method-owned posthoc scorer 收集分数，并执行原有的
prefill KV 压缩。CacheManager 拥有这套物理生命周期，prepared prefill Provider
只消费其 view，不检查 cache method 名称。已验证的
kernel 契约和必须校准的参数见
[FlashPrefill V2](../features/flashprefill-v2.md)。

## DeltaKV

可报告的 DeltaKV 推理必须使用兼容 compressor checkpoint。对于已登记模型，
默认的 `full_attention_layers=auto` 会与 OmniKV 共用模型 profile：

```python
from sparsevllm import LLM

llm = LLM(
    "/path/to/Qwen3-4B-Instruct-2507",
    sparse_method="deltakv",
    deltakv_checkpoint_path="/path/to/compressor",
    full_attention_layers="auto",
    decode_keep_tokens=2048,
    recent_keep_tokens=128,
    sink_keep_tokens=8,
    engine_prefill_chunk_size=16384,
)
```

原生实现、cache metadata、loader 和 kernel 都位于 `src/sparsevllm/`。
Compressor 训练由 [CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV)
独立维护。

## Benchmark adapter

文本 benchmark 共用 `benchmark/model_adapters/sparsevllm.py`。它接收相同的
public 参数，构造原生 engine，并为 LongBench、MathBench、NIAH 和 RULER core
提供轻量 generation callable。SCBench 使用原生 `sparsevllm` attention
type，不存在 `--backend hf` 选项。
