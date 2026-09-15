"""Native attention and FFN sublayers with four-stream residual mixing."""

from torch import nn

from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.models.deepseek_v4.attention import DeepseekV4Attention
from sparsevllm.models.deepseek_v4.hyper_connection import DeepseekV4HyperConnection
from sparsevllm.models.deepseek_v4.moe import DeepseekV4Moe


class DeepseekV4Block(nn.Module):
    def __init__(self, config, layer_id, *, quantization, max_model_len, max_num_tokens,
                 max_num_requests, mlp_chunk_size, cuda_graph, parallel_collectives=None):
        super().__init__()
        self.attn = DeepseekV4Attention(
            config, layer_id, quantization=quantization, max_model_len=max_model_len,
            max_num_tokens=max_num_tokens, max_num_requests=max_num_requests,
        )
        self.ffn = DeepseekV4Moe(
            config, layer_id, quantization=quantization, mlp_chunk_size=mlp_chunk_size,
            cuda_graph=cuda_graph, parallel_collectives=parallel_collectives,
        )
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps).bfloat16()
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps).bfloat16()
        self.hc_attn = DeepseekV4HyperConnection(config, max_num_tokens=max_num_tokens)
        self.hc_ffn = DeepseekV4HyperConnection(config, max_num_tokens=max_num_tokens)

    def forward(self, residual, input_ids, cache, batch, index_selection=None):
        x, post, comb = self.hc_attn.pre(residual)
        x = self.attn(self.attn_norm(x), cache, batch, index_selection)
        residual = self.hc_attn.post(x, residual, post, comb)
        x, post, comb = self.hc_ffn.pre(residual)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return self.hc_ffn.post(x, residual, post, comb)
