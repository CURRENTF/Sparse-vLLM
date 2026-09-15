"""Model-facing grouped projection with provider-owned BF16 weight layout."""

import torch
from torch import nn

from sparsevllm.operators.grouped_linear import GroupedLinearSpec, prepare_grouped_linear


class GroupedLinear(nn.Module):
    def __init__(self, groups, input_features, output_features, *, max_num_tokens):
        super().__init__()
        self.provider = prepare_grouped_linear(GroupedLinearSpec(groups, input_features, output_features, max_num_tokens),
                                               device_index=torch.cuda.current_device())
        self.register_buffer("weight", self.provider.allocate_weights())
        self._weight_loaded = False

    def load_quantized_weight(self, weight, scale):
        self.provider.load_fp8_weights(self.weight, weight, scale)
        self._weight_loaded = True

    def forward(self, x):
        return self.provider.run(x, self.weight)
