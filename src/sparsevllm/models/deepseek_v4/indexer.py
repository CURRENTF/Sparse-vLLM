"""Native index projections and compression; logical selection belongs to runtime."""

from torch import nn

from sparsevllm.distributed import get_parallel_context
from sparsevllm.layers.linear import ColumnParallelLinear
from sparsevllm.models.deepseek_v4.compression import DeepseekV4Compressor
from sparsevllm.operators.native_rope import NativeRotarySpec, prepare_native_rotary
from sparsevllm.operators.rotated_mxfp4 import RotatedMxfp4Spec, prepare_rotated_mxfp4


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config, *, quantization, max_num_tokens, max_num_requests):
        super().__init__()
        if (not quantization.enabled or quantization.scale_fmt != "ue8m0"
                or quantization.activation_dtype != "bfloat16"
                or quantization.max_num_tokens is None or max_num_tokens > quantization.max_num_tokens):
            raise ValueError("Native index projections require BF16/UE8M0 FP8 with prepared token capacity.")
        tp = get_parallel_context().attn_tp_size
        if config.index_n_heads % tp:
            raise ValueError("Index head count must be divisible by attention TP size.")
        self.num_heads, self.head_dim = config.index_n_heads // tp, config.index_head_dim
        self.weight_scale = config.index_head_dim ** -.5 * config.index_n_heads ** -.5
        self.wq_b = ColumnParallelLinear(config.q_lora_rank, config.index_n_heads * config.index_head_dim,
                                        quantization=quantization)
        self.weights_proj = ColumnParallelLinear(config.hidden_size, config.index_n_heads).bfloat16()
        self.compressor = DeepseekV4Compressor(config, ratio=4, head_dim=self.head_dim, rotate=True,
                                              max_num_tokens=max_num_tokens, max_num_requests=max_num_requests)
        device_index = self.wq_b.weight.device.index
        self.rope = prepare_native_rotary(NativeRotarySpec(self.head_dim, config.qk_rope_head_dim),
                                          device_index=device_index)
        self.rotation = prepare_rotated_mxfp4(RotatedMxfp4Spec(self.head_dim), device_index=device_index)

    def project(self, x, normalized_query_latent, positions, inv_freq):
        query = self.wq_b(normalized_query_latent).view(-1, self.num_heads, self.head_dim)
        self.rope.run(query, positions, inv_freq, out=query)
        query = self.rotation.run(query)
        weights = self.weights_proj(x) * self.weight_scale
        return query, weights
