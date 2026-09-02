# 支持的模型

本页汇总 Sparse-vLLM 支持的模型、精度、并行方式和稀疏方法组合。

`精度`指 checkpoint 的权重格式。`TP`、`DP` 和 `EP` 分别表示 tensor parallelism、data parallelism 和 expert parallelism。勾号表示支持该模式；数字限制表示对应的并行规模必须使用该值。

## 模型与并行支持

| 模型 | `model_type` | 精度 | TP | DP | EP |
| --- | --- | --- | :---: | :---: | :---: |
| Qwen2.5 | `qwen2` | BF16 / FP16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3 Dense | `qwen3` | BF16 / FP16 / 块级 FP8 | ✅（FP8：1/2/4/8） | 仅支持 1 | 仅支持 1 |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / 块级 FP8 | ✅ | 仅支持 1 | ✅ |
| Qwen3.5 / 3.6 / 3.8 | `qwen3_5` | BF16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3.6 MoE | `qwen3_5_moe` | BF16 / 块级 FP8 | ✅ | 仅支持 1 | ✅ |
| GLM-4.7-Flash | `glm4_moe_lite` | BF16 | ✅ | 仅支持 1 | 1 / 2 / 4 |
| Gemma 4 Dense / MoE | `gemma4` | BF16 / FP16 | ✅ | 仅支持 1 | ✅（仅 MoE） |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| MiniMax M2.7 | `minimax_m2` | 块级 FP8，非量化权重使用 BF16 | ✅ | 仅支持 1 | ✅ |

TP 规模限制为 1 到 8，并且 checkpoint 维度（包括 attention head 数和 vocabulary
大小）必须能被所选 TP 规模整除。

MoE 模型可能在内部组合 tensor parallelism 和 expert parallelism；不合法的 TP/EP
组合会在配置阶段被拒绝。

块级 FP8 要求使用 E4M3 权重、动态激活量化以及 `128 x 128` 的权重块大小。
Llama、Qwen2 和 Qwen3 Dense FP8 checkpoint 还要求每个 TP-local dense
projection 维度按 128 对齐，且非量化参数保持 BF16。

## 稀疏方法支持

| 模型 | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| Qwen3 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | 需要压缩器² |
| Qwen3MoE | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / 3.6 / 3.8 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | 匹配的 checkpoint³ |
| Qwen3.6 MoE | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| GLM-4.7-Flash | ✅⁵ | ✅⁵ | ✅⁵ | 实验性⁴⁵ | — | ✅⁵ | 实验性⁵ | ✅⁵ | — | — |
| Gemma 4 Dense / MoE | ✅ | ✅⁶ | — | — | — | ✅ | — | — | — | — |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV 仅支持已发布 steering vector 的模型：
`DeepSeek-R1-Distill-Qwen-7B`、`DeepSeek-R1-Distill-Qwen-14B` 和
`DeepSeek-R1-Distill-Llama-8B`。

² DeltaKV 需要针对 base model 训练的 compressor checkpoint。

³ Qwen3.5、Qwen3.6 和 Qwen3.8 需要 mixed-attention runtime 可接受的匹配 DeltaKV checkpoint。

⁴ H2O 的 tensor-parallel 执行可能产生与 TP=1 不同的稀疏选择。各模型原有的
TP、EP、DP 限制仍然适用。

⁵ GLM 要求 `DP=1`。QuEST 支持仍为实验性。

⁶ 带共享 KV 层的 Gemma 4 checkpoint 不支持逐层 StreamingLLM eviction；
Vanilla 和 OmniKV 仍受支持。

## 原生多模态支持

设置 `enable_multimodal=True` 后，可通过 OpenAI 兼容的 Chat 和 Responses API
传入受支持的图片和视频。不受支持的媒体会返回错误。

| 模型家族 | 图片 | 视频 | 音频 |
| --- | :---: | :---: | :---: |
| Qwen3.5 / 3.6 / 3.8 Dense 与 Qwen3.5 / 3.6 MoE | ✅ | ✅ | — |
| Gemma 4 Dense 与 MoE | ✅ | ✅ | — |

`—` 表示当前不支持该组合。
