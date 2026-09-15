"""Native compressor parameters; persistent carry belongs to the cache owner."""

import torch
from torch import nn

from sparsevllm.layers.linear import ReplicatedLinear
from sparsevllm.operators.compression import CompressionOpSpec, prepare_compression
from sparsevllm.operators.float32_linear import Float32LinearSpec, prepare_float32_linear
from sparsevllm.operators.native_rope import NativeRotarySpec, prepare_native_rotary
from sparsevllm.operators.rotated_mxfp4 import RotatedMxfp4Spec, prepare_rotated_mxfp4


class DeepseekV4Compressor(nn.Module):
    def __init__(self, config, *, ratio, head_dim, rotate, max_num_tokens, max_num_requests):
        super().__init__()
        width = head_dim * (1 + (ratio == 4))
        # The checkpoint stores these projections in BF16. The native model
        # explicitly promotes both weights and input before the FP32 GEMMs.
        self.wkv = ReplicatedLinear(config.hidden_size, width).float().requires_grad_(False)
        self.wgate = ReplicatedLinear(config.hidden_size, width).float().requires_grad_(False)
        self.ape = nn.Parameter(torch.empty(ratio, width, dtype=torch.float32), requires_grad=False)
        self.norm_weight = nn.Parameter(torch.empty(head_dim, dtype=torch.float32), requires_grad=False)
        device_index = self.wkv.weight.device.index
        self.projection = prepare_float32_linear(
            Float32LinearSpec(config.hidden_size, width, max_num_tokens, num_projections=2),
            device_index=device_index,
        )
        self.pool = prepare_compression(CompressionOpSpec(ratio, head_dim, max_num_tokens, max_num_requests),
                                        device_index=device_index)
        self.transform = prepare_native_rotary(
            NativeRotarySpec(head_dim, config.qk_rope_head_dim, "weighted_fp32", config.rms_norm_eps,
                             quantize_nope=not rotate), device_index=device_index,
        )
        self.rotation = prepare_rotated_mxfp4(RotatedMxfp4Spec(head_dim), device_index=device_index) if rotate else None

    def forward(self, x, view, inv_freq):
        kv, gate = self.projection.run_many(x, (self.wkv.weight, self.wgate.weight))
        values, positions = self.pool.run(kv, gate, self.ape, view)
        shaped = values[:, None]
        self.transform.run(shaped, positions, inv_freq, weight=self.norm_weight, out=shaped)
        if self.rotation is not None:
            values = self.rotation.run(values)
        return values, positions
