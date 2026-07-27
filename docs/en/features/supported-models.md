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
| Qwen2.5 | `qwen2` | BF16 / FP16 | ✅ | 1 only | 1 only |
| Qwen3 Dense | `qwen3` | BF16 / FP16 / block FP8 | ✅ (FP8: 1/2/4/8) | 1 only | 1 only |
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / block FP8 | 1 only | 1 only | ✅ |
| Qwen3.5 / Qwen3.6 | `qwen3_5` | block FP8 | ✅ | 1 only | 1 only |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 | ✅ | 1 only | 1 only |
| MiniMax M2.7 | `minimax_m2` | block FP8 with BF16 non-quantized weights | 1 only | 1 only | ✅ |

TP is limited to sizes 1 through 8 and requires the checkpoint dimensions,
including the attention heads and vocabulary size, to be divisible by the
selected TP size. Qwen3MoE EP must divide `num_experts`; MiniMax M2.7 EP must
divide `num_local_experts`.

Block FP8 support requires E4M3 weights, dynamic activation quantization, and
a `128 x 128` weight block size. Qwen3.5 and Qwen3.6 configurations are
normalized internally to `model_type=qwen3_5`.

## Sparse Method Support

| Model | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoints¹ | Compressor required² |
| Qwen3 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Compressor required² |
| Qwen3MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / Qwen3.6 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Matched checkpoint³ |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoint¹ | Compressor required² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV is limited to the released steering-vector models:
`DeepSeek-R1-Distill-Qwen-7B`, `DeepSeek-R1-Distill-Qwen-14B`, and
`DeepSeek-R1-Distill-Llama-8B`.

² DeltaKV requires a compressor checkpoint trained for the base model.

³ Qwen3.5 and Qwen3.6 require a matching DeltaKV checkpoint accepted by the
mixed-attention runtime.

⁴ H2O supports tensor parallelism with TP-local sparse selection: each rank
scores and retains tokens using its local heads or KV heads, without cross-rank
sparse-index aggregation. This is not guaranteed to be equivalent to TP=1 or
global-head selection. Model-specific TP, EP, and DP restrictions still apply.

`—` means that the combination is not currently supported.
