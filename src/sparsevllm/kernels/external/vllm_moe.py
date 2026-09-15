"""Optional stable-ABI vLLM MoE kernels, without importing the vLLM engine."""

from functools import lru_cache

import torch

from sparsevllm.kernels.external.vllm_support import _load_op


@lru_cache(maxsize=1)
def sqrt_softplus_op():
    return _load_op(
        "stable-ABI sqrt-softplus router", "_moe_C_stable_libtorch.abi3.so",
        "_moe_C", "topk_softplus_sqrt",
        ("topk_weights", "topk_indices", "token_expert_indices", "gating_output",
         "renormalize", "routed_scaling_factor", "bias", "input_ids", "tid2eid", "is_padding"),
    )


@lru_cache(maxsize=1)
def clipped_swiglu_op():
    return _load_op(
        "stable-ABI clipped SwiGLU", "_C_stable_libtorch.abi3.so",
        "_C", "silu_and_mul_with_clamp", ("result", "input", "limit", "alpha", "beta"),
    )


@lru_cache(maxsize=1)
def hash_inputs_op():
    def prepare(input_ids, vocab_size, index_dtype):
        return input_ids.to(index_dtype), (input_ids < 0) | (input_ids >= vocab_size)

    return torch.compile(prepare, fullgraph=True, dynamic=True)
