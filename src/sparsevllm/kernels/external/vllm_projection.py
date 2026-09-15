"""Pinned vLLM inverse-RoPE quantizer and native-Torch DeepGEMM binding."""

import ast
from functools import lru_cache
import importlib
import importlib.util
import inspect
import sys
import types

import torch

from sparsevllm.kernels.external.support import ExternalKernelContractError
from sparsevllm.kernels.external.vllm_support import vllm_library


@lru_cache(maxsize=1)
def inverse_projection_ops():
    feature = "inverse rotary FP8 grouped projection"
    if importlib.util.find_spec("deep_gemm") is None:
        return None
    library = vllm_library(feature, "_moe_C_stable_libtorch.abi3.so")
    if library is None:
        return None
    try:
        import triton
        import triton.language as tl

        gemm = importlib.import_module("deep_gemm")
        if gemm.__version__ != "2.6.1":
            raise ValueError(f"requires DeepGEMM 2.6.1 built against the active Torch, found {gemm.__version__}")
        einsum = gemm.fp8_einsum
        source = library.parent / "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        symbol = "_fused_inv_rope_fp8_quant_per_head"
        node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol]
        # This internal upstream function only needs Triton. Execute its original
        # definition, including source locations, without vLLM's engine imports.
        # No upstream namespaces or numerical source are copied or rewritten.
        name = "sparsevllm.kernels.external._vllm_inverse_quant"
        module = types.ModuleType(name)
        module.__file__ = str(source)
        module.__dict__.update(triton=triton, tl=tl)
        sys.modules[name] = module
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), module.__dict__)
            kernel = getattr(module, symbol)
            if tuple(inspect.signature(kernel.fn).parameters) != (
                "o_ptr", "positions_ptr", "cos_sin_cache_ptr", "fp8_ptr", "scale_ptr", "num_tokens",
                "heads_per_group", "o_stride_token", "o_stride_head", "cache_stride_pos",
                "fp8_stride_group", "fp8_stride_token", "scale_stride_group", "scale_stride_k",
                "fp8_max", "eps", "QUANT_GROUP_SIZE", "CHUNKS_PER_HEAD", "ROPE_START", "HALF_ROPE",
                "TMA_ALIGNED_SCALES", "USE_GDC", "launch_pdl",
            ):
                raise ValueError("unsupported vLLM inverse quantizer interface")
        except BaseException:
            del sys.modules[name]
            raise
        gemm.set_pdl(True)
    except (OSError, ImportError, AttributeError, TypeError, ValueError, RuntimeError, SyntaxError) as error:
        raise ExternalKernelContractError("vllm/deep_gemm", feature, str(error)) from error
    return kernel, einsum


def quantize_inverse_rotary(kernel, x, positions, cos_sin_cache, groups):
    """SM90 FP32 scale layout; caller supplies safe positions and BF16 D512 heads."""
    rows, heads, dim = x.shape
    inner = heads // groups * dim
    padded = (rows + 3) // 4 * 4
    values = torch.empty((groups, rows, inner), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((groups, inner // 128, padded), device=x.device, dtype=torch.float32)
    scales = scales.transpose(1, 2)[:, :rows]
    if rows:
        kernel[(padded, heads)](
            x, positions, cos_sin_cache, values, scales, rows,
            heads_per_group=heads // groups, o_stride_token=x.stride(0), o_stride_head=x.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0), fp8_stride_group=values.stride(0),
            fp8_stride_token=values.stride(1), scale_stride_group=scales.stride(0),
            scale_stride_k=scales.stride(2), fp8_max=448., eps=1e-10,
            QUANT_GROUP_SIZE=128, CHUNKS_PER_HEAD=4, ROPE_START=64, HALF_ROPE=32,
            TMA_ALIGNED_SCALES=False, USE_GDC=False, launch_pdl=False, num_stages=1, num_warps=1,
        )
    return values.transpose(0, 1), scales.transpose(0, 1)
