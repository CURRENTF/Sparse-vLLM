# GLM per-tensor FP8 checkpoint adapter

The safetensors reader pairs each E4M3 projection with its scalar `weight_scale`.
It rejects non-scalar, non-positive, non-finite and ambiguous duplicate scales.
After slicing the weight for TP, it broadcasts the scalar into the existing
128-by-128 runtime scale layout. The FP8 weight bytes and their represented
values are unchanged. This is weight-layout normalization, not requantization.
For tensor-scaled checkpoints, Linear binds SGL's public channel-scaled CUTLASS
GEMM with dynamic per-token activation quantization. The provider expands each
logical projection's weight scale into output channels once after loading;
weights are not requantized when projections have different scales. Block-scaled
checkpoints retain their existing block-wise W8A8 providers.

Routed tensor-scaled experts use the existing Triton routed-GEMM pipeline with
per-token activation quantization and weight/input scales applied in the
epilogue, outside the K loop. The loader verifies constant logical-projection
scales before publishing weights. The block grid is only loader storage for
this contract; `MoeOpSpec.block_shape=None` distinguishes it from block W8A8.

The reference is vLLM `58ad1f3b8973b23943107b51230d594050b42ec3`,
`layers/quantization/fp8.py` and `layers/fused_moe/oracle/fp8.py`:
tensor and block checkpoints have distinct quantization keys; dynamic tensor
Linear can use per-token CUTLASS. vLLM's FlashInfer MoE tensor path requires
static activation scales and cannot directly serve this dynamic checkpoint.
This adapter is not bitwise vLLM parity: it retains distinct logical weight
scales instead of merging/requantizing them and uses per-token activations
for the routed experts as well as Linear.

This checkpoint omits the optional `generation_config.json`. The engine takes
EOS metadata from the model config when that local file is absent, preserving
all stop IDs. An existing invalid generation config still raises its parsing
error; it is never treated as an absent file.

GLM's merged Q/KV-A projection preserves separate checkpoint scales. All merged
boundaries must be block aligned; the final projection may have a partial block.
Dense MLPs, shared experts and routed experts use quantized layers and providers.
BF16 shared-expert fusion profiles remain restricted to BF16 weights. A separate
H100 TP1/EP1 tensor-FP8 profile packs the shared expert for graph decode batches
1–4 (Torch 2.11, Triton 3.6, CUDA 13.0). Prefill and larger decode batches retain
separate shared projections over views of the packed weights; scale layout and
activation ordering remain provider-owned. No second copy of expert weights is
created and graph replay never re-packs weights or changes providers.

MLA `kv_b_proj` also owns a prepared BF16 copy for the absorbed query and value
BMMs. This copy is derived once during weight loading, before graph capture;
the prefill projection retains FP8 weights. Quantized Q-B projection executes
through its bound provider followed by the same rotary transform as prefill.
There is no per-token weight dequantization or provider reselection.

`tests/test_glm_fp8_loading.py` covers scalar normalization, failed loads,
TP slicing, merged projections, actual GLM dense/shared/routed weight loading,
and CUDA projection/MoE numerical equivalence with graph replay. CPU checks
only establish loader behavior. Real generation and matched BF16/FP8 efficiency
must be measured on idle permitted devices using the canonical efficiency
probe and its request-level metric contract.
