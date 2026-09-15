# DeepSeek V4 external output projection

The inverse-rotary grouped projection operator binds its provider before
checkpoint loading. The external provider consumes BF16 attention heads,
applies inverse adjacent-pair RoPE and power-of-two block FP8 quantization,
then executes DeepGEMM's public `fp8_einsum` with the original FP8 checkpoint
weights. It retains FP32 block scales rather than expanding weights to BF16.
The composed provider preserves the previous rotary and BF16 grouped-GEMM path
when the optional dependencies or supported tensor contract are unavailable.
Execution failures never trigger provider switching.

## Dependency contract

- vLLM: `>=0.29,<0.30`, original installed source, Apache-2.0. The adapter uses
  `_fused_inv_rope_fp8_quant_per_head` from
  `models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`.
  This is an internal symbol, not the public engine API. The adapter checks its
  signature and executes only the unchanged original function definition in a
  private module with Triton globals. It neither copies the numerical source
  into this repository nor imports the vLLM engine or changes `sys.path`.
- DeepGEMM: version 2.6.1, MIT, upstream revision
  `8b1392b978f5a03c828dd1711090d7fb50958b8a`, as pinned by vLLM 0.29.0.
  Its pybind extension must be built against the active Torch and Python.
  A binary from another vLLM environment does not provide a stable Torch ABI.
- This adapter implements the SM90 FP32, MN-major aligned scale layout,
  D512 heads, D64 rotary dimensions and 128-aligned output features.
  SM100's packed scale representation requires a separate adapter contract.

Build the optional package in a separate source checkout, using the same
interpreter as Sparse-vLLM. Set `SPARSEVLLM_PYTHON` to that interpreter's
absolute path and `DEEPGEMM_SOURCE` to the desired checkout directory:

```bash
git clone https://github.com/deepseek-ai/DeepGEMM.git "$DEEPGEMM_SOURCE"
git -C "$DEEPGEMM_SOURCE" checkout 8b1392b978f5a03c828dd1711090d7fb50958b8a
git -C "$DEEPGEMM_SOURCE" submodule update --init --recursive
cd "$DEEPGEMM_SOURCE"
MAX_JOBS=2 DG_FORCE_BUILD=1 TORCH_CUDA_ARCH_LIST=9.0a \
  "$SPARSEVLLM_PYTHON" setup.py bdist_wheel
uv pip install --python "$SPARSEVLLM_PYTHON" --no-deps dist/deep_gemm-*.whl
```

The build also requires setuptools, wheel, a compatible CUDA toolkit and a C++
compiler. Configure `SPARSEVLLM_VLLM_MOE_LIBRARY` as described in the model
installation guide, retaining the original wheel's Python source and metadata.
No build, download or environment modification occurs during model startup.
An absent optional package permits the composed provider; an installed but
incompatible dependency produces an actionable startup failure.

## Ownership and validation

The provider owns FP8 weight layout, block scales, cosine tables and temporary
activation layouts. Cosine tables are shared by device, context capacity and
frequency values; weak references allow the last model owner to release them.
The model's `wo_a` checkpoint hook remains unchanged. Cache-manager physical
storage and SparseController selection responsibilities are unaffected.

Negative positions are padding: safe positions are prepared before table
access, and the final projection is cleared for padded rows even when the
attention input is nonzero. Context bounds are checked on device. Graph replay
reads live input and position tensors without selecting a new provider or
changing its launch plan based on position values.

`tests/test_inverse_rotary_grouped_linear.py` checks checkpoint scale layout,
an independent mathematical FP8 oracle, graph replay with changing positions
and padding, empty batches, the composed path and optional dependency failures.
Run it on an idle SM90 GPU with the dependencies above. Numerical comparisons
follow vLLM's fused FP32 rotation before FP8 quantization; they do not require
equality with the former intermediate BF16 rotation.

Performance measurements must include safe-position preparation, padding
clearing, temporary allocations and projection. Report those measurements
separately from end-to-end throughput, TTFT and TPOT. A microbenchmark does not
establish full-model quality or performance parity.
