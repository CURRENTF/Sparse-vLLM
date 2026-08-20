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
| Qwen3MoE | `qwen3_moe` | BF16 / FP16 / block FP8 | ✅ (TP > 1: BF16 model dtype only) | 1 only | ✅ |
| Qwen3.5 / 3.6 / 3.8 | `qwen3_5` | BF16 / block FP8 | ✅ | 1 only | 1 only |
| Qwen3.6 MoE | `qwen3_5_moe` | BF16 / block FP8 | ✅ | 1 only | ✅ |
| GLM-4.7-Flash | `glm4_moe_lite` | BF16 | 1 / 2 / 4 (H100 only)⁵ | 1 only | 1 / 2 / 4⁵ |
| Gemma 4 Dense / MoE | `gemma4` | BF16 / FP16 | ✅ | 1 only | ✅ (MoE only) |
| Llama 3 / 3.1 | `llama` | BF16 / FP16 | ✅ | 1 only | 1 only |
| MiniMax M2.7 | `minimax_m2` | block FP8 with BF16 non-quantized weights | ✅ | 1 only | ✅ |

TP is limited to sizes 1 through 8 and requires the checkpoint dimensions,
including the attention heads and vocabulary size, to be divisible by the
selected TP size.

Qwen3MoE and MiniMax M2.7 use a hybrid layout when the outer TP size `T` is
greater than 1. GLM-4.7-Flash uses the same layout when both `T > 1` and
`E > 1`: attention TP is `T`, MoE EP is `E`, MoE TP is `T / E`, and the
distributed world size is `T`. This layout requires `DP=1` and `T % E == 0`.
The expert count must be divisible by `E`, and the MoE intermediate dimension
must be divisible by `T / E`. Qwen3MoE outer TP requires a BF16 model dtype;
FP16 Qwen3MoE checkpoints are limited to `TP=1`. When `TP=1`, the existing EP
layout uses world size `E`.

On H100 80GB, unquantized BF16 Qwen3-30B-A3B with EP1 uses the
`sgl_triton_hybrid` MoE provider for TP1 and TP2. The provider runs the ported
SGL fused-MoE kernel for profiled token buckets (64 tokens or more for both
TP1 and TP2) and the generic Triton kernel below those
thresholds. Other shapes and topologies keep their existing providers. Runtime
operator statistics report the kernel used by each branch.

Block FP8 support requires E4M3 weights, dynamic activation quantization, and
a `128 x 128` weight block size. Qwen3.5, Qwen3.6, and Qwen3.8 dense
checkpoints share the `qwen3_5` runtime architecture and therefore the same
precision, parallelism, sparse-method, and multimodal support. Qwen3.6 MoE
uses `model_type=qwen3_5_moe`.

GLM-4.7-Flash uses BF16 latent MLA on NVIDIA H100 80GB HBM3 and requires
`DP=1`. The validated `(TP, EP)` layouts are
`(1,1)`, `(2,1)`, `(4,1)`, `(1,2)`, `(1,4)`, `(2,2)`, `(4,2)`, and `(4,4)`.
Across all eight layouts, vanilla, StreamingLLM, SnapKV, H2O, OmniKV, and R-KV
support decode CUDA Graph and prefix caching together. Prefix caching uses
radix mode for vanilla and OmniKV, and chain mode for StreamingLLM, SnapKV,
H2O, and R-KV. Prefix offload, quantization, and the other sparse methods
remain unsupported. The loader intentionally skips the checkpoint's MTP
layer.

## Sparse Method Support

| Model | Vanilla | StreamingLLM | SnapKV | H2O | PyramidKV | OmniKV | QuEST | R-KV | SkipKV | DeltaKV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoints¹ | Compressor required² |
| Qwen3 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Compressor required² |
| Qwen3MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| Qwen3.5 / 3.6 / 3.8 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | Matched checkpoint³ |
| Qwen3.6 MoE | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |
| GLM-4.7-Flash | ✅⁵ | ✅⁵ | ✅⁵ | Experimental⁴⁵ | — | ✅⁵ | — | ✅⁵ | — | — |
| Gemma 4 Dense / MoE | ✅ | ✅⁶ | — | — | — | ✅ | — | — | — | — |
| Llama 3 / 3.1 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | Selected checkpoint¹ | Compressor required² |
| MiniMax M2.7 | ✅ | ✅ | ✅ | Experimental⁴ | ✅ | ✅ | ✅ | ✅ | — | — |

¹ SkipKV is limited to the released steering-vector models:
`DeepSeek-R1-Distill-Qwen-7B`, `DeepSeek-R1-Distill-Qwen-14B`, and
`DeepSeek-R1-Distill-Llama-8B`.

² DeltaKV requires a compressor checkpoint trained for the base model.

³ Qwen3.5, Qwen3.6, and Qwen3.8 require a matching DeltaKV checkpoint accepted
by the mixed-attention runtime.

⁴ H2O supports tensor parallelism with TP-local sparse selection: each rank
scores and retains tokens using its local heads or KV heads, without cross-rank
sparse-index aggregation. This is not guaranteed to be equivalent to TP=1 or
global-head selection. Model-specific TP, EP, and DP restrictions still apply.

⁵ GLM support is limited to the eight `(TP, EP)` layouts listed above with
`DP=1`. At `TP>1`, head-scored sparse methods use TP-local selection without
cross-rank sparse-index aggregation, so their selection semantics are not
guaranteed to match `TP=1`.

⁶ Gemma 4 checkpoints with shared KV layers reject per-layer StreamingLLM
eviction. Vanilla and OmniKV remain supported.

## Native Multimodal Support

Set `enable_multimodal=True` to use a checkpoint's native media towers.
Sparse-vLLM accepts OpenAI-compatible Chat and Responses content parts and
uses the checkpoint processor and chat template. Unsupported media fail
explicitly during admission.

| Model family | Image | Video | Audio |
| --- | :---: | :---: | :---: |
| Qwen3.5 / 3.6 / 3.8 Dense and Qwen3.5 / 3.6 MoE | ✅ | ✅ | — |
| Gemma 4 Dense and MoE | ✅ | ✅ | — |

`—` means that the combination is not currently supported.
