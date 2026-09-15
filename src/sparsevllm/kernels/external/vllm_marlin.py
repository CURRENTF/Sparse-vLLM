"""vLLM 0.29 stable-ABI Marlin operations and provider-owned weight packing.

Scale permutation follows vLLM's marlin_utils.py and marlin_utils_fp4.py:
SPDX-License-Identifier: Apache-2.0
Copyright contributors to the vLLM project.
"""

from functools import lru_cache

import torch

from sparsevllm.kernels.external.vllm_moe import clipped_swiglu_op
from sparsevllm.kernels.external.vllm_support import _load_op

# vLLM ScalarType.float_(2, 1, finite_values_only=True, nan_repr=NONE).
MXFP4_TYPE_ID = 2 | (1 << 8) | (1 << 16) | (1 << 49)


@lru_cache(maxsize=1)
def marlin_ops():
    repack = _load_op(
        "MXFP4 Marlin repack", "_C_stable_libtorch.abi3.so", "_C", "gptq_marlin_repack",
        ("b_q_weight", "perm", "size_k", "size_n", "num_bits", "is_a_8bit"),
    )
    if repack is None:
        return None
    align = _load_op(
        "Marlin expert alignment", "_moe_C_stable_libtorch.abi3.so", "_moe_C", "moe_align_block_size",
        ("topk_ids", "num_experts", "block_size", "sorted_token_ids", "experts_ids", "num_tokens_post_pad", "maybe_expert_map"),
    )
    gemm = _load_op(
        "MXFP4 Marlin GEMM", "_moe_C_stable_libtorch.abi3.so", "_moe_C", "moe_wna16_marlin_gemm",
        ("a", "c_or_none", "b_q_weight", "b_bias_or_none", "b_scales", "a_scales", "global_scale",
         "b_zeros_or_none", "g_idx_or_none", "perm_or_none", "workspace", "sorted_token_ids",
         "expert_ids", "num_tokens_past_padded", "topk_weights", "moe_block_size", "top_k",
         "mul_topk_weights", "b_type_id", "size_m", "size_n", "size_k", "is_full_k",
         "use_atomic_add", "use_fp32_reduce", "is_zp_float", "thread_k", "thread_n", "blocks_per_sm"),
    )
    return repack, align, gemm, clipped_swiglu_op()


def pack_mxfp4_projection(repack, weight, scale, empty_permutation):
    n, packed_k = weight.shape
    packed = repack(weight.view(torch.int32).T.contiguous(), empty_permutation,
                    packed_k * 2, n, 4, False)
    scale_permutation = [i + 8 * j for i in range(8) for j in range(8)]
    scales = scale.T.contiguous().reshape(-1, 64)[:, scale_permutation].reshape(-1, n)
    scales = scales.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, n).contiguous()
    return packed, scales.view(torch.float8_e8m0fnu)
