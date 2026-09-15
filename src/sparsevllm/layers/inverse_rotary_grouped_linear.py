"""Checkpoint projection consuming attention heads and their rotary positions."""

from torch import nn

from sparsevllm.operators.inverse_rotary_grouped_linear import (
    InverseRotaryGroupedLinearSpec, prepare_inverse_rotary_grouped_linear,
)


class InverseRotaryGroupedLinear(nn.Module):
    """Inverse rotary then grouped projection; the provider may consume x in place."""

    def __init__(self, groups, heads, head_dim, output_features, *, inv_freq,
                 max_num_tokens, max_positions, device_index):
        super().__init__()
        spec = InverseRotaryGroupedLinearSpec(
            groups, heads, head_dim, output_features, max_num_tokens, max_positions,
            tuple(inv_freq.float().cpu().tolist()),
        )
        self.provider = prepare_inverse_rotary_grouped_linear(spec, device_index=device_index)
        weight, scale = self.provider.allocate_weights()
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)
        self._weight_loaded = False

    def load_quantized_weight(self, weight, scale):
        self.provider.load_fp8_weights(self.weight, self.weight_scale, weight, scale)
        self._weight_loaded = True

    def forward(self, x, positions):
        return self.provider.run(x, positions, self.weight, self.weight_scale)
