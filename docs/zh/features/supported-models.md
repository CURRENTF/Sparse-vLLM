# 支持的模型

本页汇总 Sparse-vLLM 支持的模型、精度、并行方式和稀疏方法组合。

`精度`指 checkpoint 的权重格式。`TP`、`DP` 和 `EP` 分别表示 tensor parallelism、data parallelism 和 expert parallelism。勾号表示支持该模式；数字限制表示对应的并行规模必须使用该值。

## 模型与并行支持

| 模型 | `model_type` | 精度 | TP | DP | EP |
| --- | --- | --- | :---: | :---: | :---: |
| Qwen2.5 | `qwen2` | BF16 / FP16 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3 Dense | `qwen3` | BF16 / FP16 / 块级 FP8 | ✅（FP8：1/2/4/8） | 仅支持 1 | 仅支持 1 |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / 块级 FP8 | ✅（TP > 1 时模型 dtype 仅支持 BF16） | 仅支持 1 | ✅ |
| Qwen3.5 / Qwen3.6 | `qwen3_5` | BF16 / 块级 FP8 | ✅ | 仅支持 1 | 仅支持 1 |
| Qwen3.6 MoE | `qwen3_5_moe` | BF16 / 块级 FP8 | ✅ | 仅支持 1 | ✅ |
| GLM-4.7-Flash | `glm4_moe_lite` | BF16 | 1 / 2 / 4（仅 H100）⁵ | 仅支持 1 | 1 / 2 / 4⁵ |
| Gemma 4 Dense / MoE | `gemma4` | BF16 / FP16 | ✅ | 仅支持 1 | ✅（仅 MoE） |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 | ✅ | 仅支持 1 | 仅支持 1 |
| MiniMax M2.7 | `minimax_m2` | 块级 FP8，非量化权重使用 BF16 | ✅ | 仅支持 1 | ✅ |

TP 规模限制为 1 到 8，并且 checkpoint 维度（包括 attention head 数和 vocabulary 大小）必须能被所选 TP 规模整除。Qwen3MoE 的 EP 规模必须整除 `num_experts`；MiniMax M2.7 的 EP 规模必须整除 `num_local_experts`。

当外层 TP 规模 `T` 大于 1 时，Qwen3MoE 和 MiniMax M2.7 使用混合并行布局；
GLM-4.7-Flash 在 `T > 1` 且 `E > 1` 时使用相同布局：attention TP 为 `T`、
MoE EP 为 `E`、MoE TP 为 `T / E`，distributed world size 为 `T`。该布局
要求 `DP=1` 且 `T % E == 0`；专家数量必须能被 `E` 整除，MoE intermediate
dimension 必须能被 `T / E` 整除。Qwen3MoE 的外层 TP 要求模型 dtype 为
BF16；FP16 Qwen3MoE checkpoint 仅支持 `TP=1`。当 `TP=1` 时，原有 EP
布局的 world size 为 `E`。

块级 FP8 要求使用 E4M3 权重、动态激活量化以及 `128 x 128` 的权重块大小。
Qwen3.5/Qwen3.6 Dense 配置在内部统一规范为 `model_type=qwen3_5`；Qwen3.6
MoE 使用 `model_type=qwen3_5_moe`。

GLM-4.7-Flash 在 NVIDIA H100 80GB HBM3 上使用 BF16 latent MLA，要求
`DP=1` 且 `enforce_eager=True`。已验证的 `(TP, EP)` 布局为 `(1,1)`、
`(2,1)`、`(4,1)`、`(1,2)`、`(1,4)`、`(2,2)`、`(4,2)` 和 `(4,4)`。
在全部八种布局中，vanilla、StreamingLLM、SnapKV、H2O、OmniKV 和 R-KV
均支持 decode CUDA Graph 与 Prefix Cache 的联合组合。Prefix Cache 对
vanilla 和 OmniKV 使用 radix 模式，对 StreamingLLM、SnapKV、H2O 和 R-KV
使用 chain 模式。Prefix offload、量化和其他稀疏方法仍不支持。loader
会有意跳过 checkpoint 中的 MTP 层。

## 稀疏方法支持

| 模型 | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| Qwen3 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | 需要压缩器² |
| Qwen3MoE | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / Qwen3.6 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | 匹配的 checkpoint³ |
| Qwen3.6 MoE | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| GLM-4.7-Flash | ✅⁵ | ✅⁵ | ✅⁵ | 实验性⁴⁵ | — | ✅⁵ | — | ✅⁵ | — | — |
| Gemma 4 Dense / MoE | ✅ | ✅⁶ | — | — | — | ✅ | — | — | — | — |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | 指定 checkpoint¹ | 需要 compressor² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | 实验性⁴ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV 仅支持已发布 steering vector 的模型：
`DeepSeek-R1-Distill-Qwen-7B`、`DeepSeek-R1-Distill-Qwen-14B` 和
`DeepSeek-R1-Distill-Llama-8B`。

² DeltaKV 需要针对 base model 训练的 compressor checkpoint。

³ Qwen3.5 和 Qwen3.6 需要 mixed-attention runtime 可接受的匹配 DeltaKV checkpoint。

⁴ H2O 通过 TP-local 稀疏选择支持 tensor parallel：每个 rank 使用本地
attention head 或 KV head 独立计算分数并保留 token，不跨 rank 聚合 sparse
index。因此其算法行为不保证与 TP=1 或全局 head 选择等价；各模型原有的
TP、EP、DP 限制仍然适用。

⁵ GLM 支持限定为上文列出的八种 `(TP, EP)` 布局，且要求 `DP=1`。在
`TP>1` 时，基于 head 评分的稀疏方法使用 TP-local selection，不跨 rank
聚合 sparse index，因此其选择语义不保证与 `TP=1` 相同。

⁶ 带共享 KV 层的 Gemma 4 checkpoint 不支持逐层 StreamingLLM eviction；
Vanilla 和 OmniKV 仍受支持。

## 原生多模态支持

设置 `enable_multimodal=True` 后，可使用 checkpoint 自带的媒体塔。
Sparse-vLLM 接受 OpenAI 兼容的 Chat 与 Responses content part，并使用
checkpoint 自身的 processor 和 chat template；不受支持的媒体会在接纳阶段
明确报错。

| 模型家族 | 图片 | 视频 | 音频 |
| --- | :---: | :---: | :---: |
| Qwen3.5 / Qwen3.6 Dense 与 MoE | ✅ | ✅ | — |
| Gemma 4 Dense 与 MoE | ✅ | ✅ | — |

`—` 表示当前不支持该组合。
