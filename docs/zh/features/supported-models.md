# 支持的模型

本页汇总 Sparse-vLLM 支持的模型、精度、并行方式和稀疏方法组合。

`精度`指 checkpoint 的权重格式。`TP`、`DP` 和 `EP` 分别表示 tensor parallelism、data parallelism 和 expert parallelism。勾号表示支持该模式；数字限制表示对应的并行规模必须使用该值。

## 模型与并行支持

| 模型 | `model_type` | 精度 | TP | DP | EP |
| --- | --- | --- | :---: | :---: | :---: |
| Qwen2.5 | `qwen2` | BF16 / FP16 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3 | `qwen3` | BF16 / FP16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / 块级 FP8 | 仅支持 1 | 仅支持 1 | ✅ |
| Qwen3.5 / Qwen3.6 | `qwen3_5` | 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 | ✅ | 仅支持 1 | 仅支持 1 |
| MiniMax M2.7 | `minimax_m2` | 块级 FP8，非量化权重使用 BF16 | 仅支持 1 | 仅支持 1 | ✅ |

TP 规模限制为 1 到 8，并且 checkpoint 维度（包括 attention head 数和 vocabulary 大小）必须能被所选 TP 规模整除。Qwen3MoE 的 EP 规模必须整除 `num_experts`；MiniMax M2.7 的 EP 规模必须整除 `num_local_experts`。

块级 FP8 要求使用 E4M3 权重、动态激活量化以及 `128 x 128` 的权重块大小。Qwen3.5 和 Qwen3.6 的配置在内部统一规范为 `model_type=qwen3_5`。

## 稀疏方法支持

| 模型 | Vanilla | StreamingLLM | SnapKV | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| Qwen3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | 需要压缩器² |
| Qwen3MoE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / Qwen3.6 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | 匹配的 checkpoint³ |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV 仅支持已发布 steering vector 的模型：
`DeepSeek-R1-Distill-Qwen-7B`、`DeepSeek-R1-Distill-Qwen-14B` 和
`DeepSeek-R1-Distill-Llama-8B`。

² DeltaKV 需要针对 base model 训练的 compressor checkpoint。

³ Qwen3.5 和 Qwen3.6 需要 mixed-attention runtime 可接受的匹配 DeltaKV checkpoint。

`—` 表示当前不支持该组合。
