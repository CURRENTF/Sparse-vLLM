# Supported Models

This page summarizes the model, precision, parallelism, and sparse-method
combinations supported by Sparse-vLLM.

`Precision` describes the checkpoint weight format. `TP`, `DP`, and `EP`
stand for tensor, data, and expert parallelism. A check mark means that the
mode is supported; a numeric restriction means that the corresponding
parallel size must use that value.

## Model and Parallelism Support

| Model | `model_type` | Precision | TP | DP | EP |
| --- | --- | --- | :---: | :---: | :---: |
| Qwen2.5 | `qwen2` | BF16 / FP16 / block FP8 | ✅ | 1 only | 1 only |
| Qwen3 Dense | `qwen3` | BF16 / FP16 / block FP8 | ✅ (FP8: 1/2/4/8) | 1 only | 1 only |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / block FP8 | ✅ | 1 only | ✅ |
| Qwen3.5 / 3.6 / 3.8 | `qwen3_5` | BF16 / block FP8 | ✅ | 1 only | 1 only |
| Qwen3.6 MoE | `qwen3_5_moe` | BF16 / block FP8 | ✅ | 1 only | ✅ |
| GLM-4.7-Flash | `glm4_moe_lite` | BF16 | ✅ | 1 only | 1 / 2 / 4 |
| Gemma 4 Dense / MoE | `gemma4` | BF16 / FP16 | ✅ | 1 only | ✅ (MoE only) |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 / block FP8 | ✅ | 1 only | 1 only |
| MiniMax M2.7 | `minimax_m2` | block FP8 with BF16 non-quantized weights | ✅ | 1 only | ✅ |

TP is limited to sizes 1 through 8 and requires the checkpoint dimensions,
including the attention heads and vocabulary size, to be divisible by the
selected TP size.

MoE models may combine tensor and expert parallelism internally; invalid TP/EP
combinations are rejected during configuration.

Block FP8 support requires E4M3 weights, dynamic activation quantization, and
a `128 x 128` weight block size. Llama, Qwen2, and Qwen3 dense FP8 checkpoints
also require every TP-local dense projection dimension to be 128-aligned and
keep non-quantized parameters in BF16.

## Sparse Method Support

| Model | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoints¹ | Compressor required² |
| Qwen3 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Compressor required² |
| Qwen3MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / 3.6 / 3.8 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Matched checkpoint³ |
| Qwen3.6 MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| GLM-4.7-Flash | ✅⁵ | ✅⁵ | ✅⁵ | Experimental⁴⁵ | — | ✅⁵ | Experimental⁵ | ✅⁵ | — | — |
| Gemma 4 Dense / MoE | ✅ | ✅⁶ | — | — | — | ✅ | — | — | — | — |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoint¹ | Compressor required² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV is limited to the released steering-vector models:
`DeepSeek-R1-Distill-Qwen-7B`, `DeepSeek-R1-Distill-Qwen-14B`, and
`DeepSeek-R1-Distill-Llama-8B`.

² DeltaKV requires a compressor checkpoint trained for the base model.

³ Qwen3.5, Qwen3.6, and Qwen3.8 require a matching DeltaKV checkpoint accepted
by the mixed-attention runtime.

⁴ H2O tensor-parallel execution may produce different sparse selections from
TP=1. Model-specific TP, EP, and DP restrictions still apply.

⁵ GLM requires `DP=1`. QuEST support is experimental.

⁶ Gemma 4 checkpoints with shared KV layers reject per-layer StreamingLLM
eviction. Vanilla and OmniKV remain supported.

## Native Multimodal Support

Set `enable_multimodal=True` to accept supported image and video inputs through
the OpenAI-compatible Chat and Responses APIs. Unsupported media return an error.

| Model family | Image | Video | Audio |
| --- | :---: | :---: | :---: |
| Qwen3.5 / 3.6 / 3.8 Dense and Qwen3.5 / 3.6 MoE | ✅ | ✅ | — |
| Gemma 4 Dense and MoE | ✅ | ✅ | — |

`—` means that the combination is not currently supported.
