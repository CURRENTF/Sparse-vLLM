# GLM MLA TileLang kernel

The decode kernel is adapted from
`examples/deepseek_mla/example_mla_decode_paged.py` in Tile-AI/TileLang commit
`c7fabc4cc65e480b88b7606eb1bc9c340dbd8c8c` under the MIT license.

Local changes implement the GLM-4.7-Flash TP1/TP2/TP4 decode contracts used by
Sparse-vLLM:

- BF16 query and cache tensors;
- direct strided 20/10/5 TP-local queries, zero-padded inside complete MMA tiles;
- page-size-one indirect cache slots;
- explicit caller-owned output and split-KV workspaces;
- optional fused FP32 raw-QK score reduced by max over the real local heads
  before applying the attention softmax scale;
- indirect request rows and safe `-1` padding outside each context;
- CUDA Graph-compatible execution.

The module only defines kernels. Provider selection, workspace ownership,
dependency checks, launch-config selection, and fallback policy belong under
`sparsevllm.operators`.

The production adapter binds the validated GLM TP1/TP2/TP4 H100 BF16 contract.
It chooses an offline-calibrated split, head-tile size, and score reduction
mode from the static batch/context table. TileLang tensors must be contiguous,
and the reduced score/context capacity must be a multiple of the kernel's
64-token tile. Unsupported score dtype, layout, or capacity stays on the
existing Triton provider through an explicit pre-launch shape dispatch.
