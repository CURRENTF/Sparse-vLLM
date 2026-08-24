# 运行时参数语义

Sparse-vLLM 只有一个推理后端：`src/sparsevllm/` 下的原生引擎。
`LLM(...)`、`Config`、JSON 配置、benchmark manifest 与内部代码使用完全
相同的 runtime 参数名；引擎不再维护 public-to-internal alias 层。

## Public 参数名

命令、JSON 配置和 benchmark manifest 应使用语义化 public 名称：

| 规范名称 | 含义 |
| --- | --- |
| `sparse_method` | 稀疏方法选择器。 |
| `deltakv_checkpoint_path` | DeltaKV compressor checkpoint 路径。 |
| `engine_prefill_chunk_size` | Prefill 最大调度 chunk。 |
| `sink_keep_tokens` | Sink token 预算。 |
| `recent_keep_tokens` | Recent token 预算。 |
| `full_attention_layers` | 逗号分隔的 full-layer index。 |
| `deltakv_neighbor_count` | DeltaKV reference neighbor 数量。 |
| `deltakv_center_ratio` | DeltaKV reference center 比例。 |
| `deltakv_latent_dim` | Compressor latent 宽度。 |
| `deltakv_latent_quant_bits` | Latent state 量化位数。 |
| `deltakv_latent_quant_group_size` | Latent 量化 group size。 |
| `gpu_memory_utilization` | 引擎可使用的 GPU 显存比例。 |
| `decode_graph` | 启用 decode CUDA Graph。 |

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

## DeltaKV

可报告的 DeltaKV 推理必须使用兼容 compressor checkpoint：

```python
from sparsevllm import LLM

llm = LLM(
    "/path/to/model",
    sparse_method="deltakv",
    deltakv_checkpoint_path="/path/to/compressor",
    full_attention_layers="0,1,3,9,13,16,21,28",
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
public 参数，构造原生 engine，并为 LongBench、MathBench、NIAH 和 RULER-VT
提供轻量 generation callable。SCBench 使用原生 `sparsevllm` attention
type，不存在 `--backend hf` 选项。
