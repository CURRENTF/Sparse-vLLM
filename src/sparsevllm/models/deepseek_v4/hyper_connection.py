"""Native block residual mixing with FP32 checkpoint parameters."""

import torch
from torch import nn

from sparsevllm.operators.hyper_connection import (
    HyperConnectionHeadSpec, HyperConnectionSpec, prepare_hyper_connection,
    prepare_hyper_connection_head,
)


class DeepseekV4HyperConnection(nn.Module):
    def __init__(self, config, *, max_num_tokens):
        super().__init__()
        streams = config.hc_mult
        mix_dim = (2 + streams) * streams
        self.fn = nn.Parameter(torch.empty(mix_dim, streams * config.hidden_size, dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.base = nn.Parameter(torch.empty(mix_dim, dtype=torch.float32), requires_grad=False)
        self.op = prepare_hyper_connection(HyperConnectionSpec(
            config.hidden_size, streams, max_num_tokens, config.rms_norm_eps,
            config.hc_eps, config.hc_sinkhorn_iters,
        ), device_index=self.fn.device.index)

    def pre(self, residual):
        return self.op.pre(residual, self.fn, self.scale, self.base)

    def post(self, x, residual, post, comb):
        return self.op.post(x, residual, post, comb)


class DeepseekV4HyperConnectionHead(nn.Module):
    def __init__(self, config, *, max_num_tokens):
        super().__init__()
        self.fn = nn.Parameter(torch.empty(config.hc_mult, config.hc_mult * config.hidden_size,
                                          dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32), requires_grad=False)
        self.op = prepare_hyper_connection_head(HyperConnectionHeadSpec(
            config.hidden_size, config.hc_mult, max_num_tokens, config.rms_norm_eps, config.hc_eps,
        ), device_index=self.fn.device.index)

    def forward(self, residual):
        return self.op.run(residual, self.fn, self.scale, self.base)
