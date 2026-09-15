"""Full native decoder using cache-owned views and runtime-owned selection."""

from dataclasses import replace

import torch
from torch import nn
import torch.nn.functional as F

from sparsevllm.distributed import get_parallel_context
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.models.deepseek_v4.block import DeepseekV4Block
from sparsevllm.models.deepseek_v4.checkpoint import DeepseekV4Checkpoint
from sparsevllm.models.deepseek_v4.hyper_connection import DeepseekV4HyperConnectionHead
from sparsevllm.utils.context import get_context


class DeepseekV4Model(nn.Module):
    def __init__(self, config, *, quantization, max_model_len, max_num_tokens,
                 max_num_requests, mlp_chunk_size, cuda_graph, parallel_collectives=None):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(DeepseekV4Block(
            config, layer, quantization=quantization, max_model_len=max_model_len,
            max_num_tokens=max_num_tokens, max_num_requests=max_num_requests,
            mlp_chunk_size=mlp_chunk_size, cuda_graph=cuda_graph, parallel_collectives=parallel_collectives,
        ) for layer in range(config.num_hidden_layers))
        self.hc_head = DeepseekV4HyperConnectionHead(config, max_num_tokens=max_num_tokens)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps).bfloat16()

    def forward(self, input_ids, *, executions, index_selection):
        if len(executions) != len(self.layers):
            raise ValueError("Native decoder requires one physical cache and batch view per layer.")
        # Negative IDs denote graph padding. Hash routing still receives the
        # original sentinel; embedding only needs a valid, unobserved row.
        residual = self.embed(input_ids.clamp_min(0))[:, None].repeat(1, self.hc_mult, 1)
        context = get_context()
        for layer_id, (layer, execution) in enumerate(zip(self.layers, executions)):
            context.now_layer_idx = layer_id
            residual = layer(residual, input_ids, execution.storage, execution.batch, index_selection)
        return self.norm(self.hc_head(residual))


class DeepseekV4ForCausalLM(DeepseekV4Checkpoint, nn.Module):
    @staticmethod
    def build_runtime_kwargs(config, *, engine_config, parallel_context, collective_runtime, max_decode_tokens, **_):
        capacity = max(engine_config.max_num_batched_tokens, max_decode_tokens)
        return dict(
            quantization=config.quantization_config, max_model_len=engine_config.max_model_len,
            max_num_tokens=capacity, max_num_requests=max(engine_config.max_num_seqs_in_batch, max_decode_tokens),
            mlp_chunk_size=engine_config.mlp_chunk_size, cuda_graph=engine_config.decode_graph,
            parallel_collectives=collective_runtime.request_moe_collectives(
                attention_max_rows=max_decode_tokens, moe_max_rows=max_decode_tokens * parallel_context.attn_dp_size,
                max_local_tokens=engine_config.max_num_batched_tokens, hidden_size=config.hidden_size,
                dtype=torch.bfloat16, backend=engine_config.moe_backend, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, moe_reduction_dtype=torch.float32,
            ),
        )

    def __init__(self, config, *, quantization, max_model_len, max_num_tokens,
                 max_num_requests, mlp_chunk_size, cuda_graph, parallel_collectives=None):
        super().__init__()
        parallel = get_parallel_context()
        if parallel.attn_tp_size != 1:
            raise ValueError("Native sparse attention currently requires attention TP=1; use DP x EP.")
        if config.tie_word_embeddings:
            raise ValueError("Native checkpoint requires independent embedding and output weights.")
        if max_num_tokens <= 0 or max_num_requests <= 0 or max_model_len <= 0:
            raise ValueError("Native model execution capacities must be positive.")
        self.config = config
        quantization = replace(quantization, max_num_tokens=max(max_num_tokens, mlp_chunk_size))
        self.model = DeepseekV4Model(
            config, quantization=quantization, max_model_len=max_model_len, max_num_tokens=max_num_tokens,
            max_num_requests=max_num_requests, mlp_chunk_size=mlp_chunk_size, cuda_graph=cuda_graph,
            parallel_collectives=parallel_collectives,
        )
        # The checkpoint head is BF16, but the reference promotes its weights
        # and normalized activations before the final vocabulary projection.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        self.requires_grad_(False)

    def forward(self, input_ids, positions):
        context = get_context()
        return self.model(input_ids, executions=context.cache_manager.native_attention_executions(),
                          index_selection=context.sparse_controller)

    def forward_idle_experts(self, hidden_states):
        input_ids = torch.empty(0, device=hidden_states.device, dtype=torch.int64)
        for layer in self.model.layers:
            layer.ffn(hidden_states, input_ids)

    def decode_attention_providers(self):
        return tuple(layer.attn.attention for layer in self.model.layers)

    @torch.inference_mode()
    def warmup_moe(self, num_tokens=1):
        hidden = torch.zeros((num_tokens, self.config.hidden_size),
                             device=self.lm_head.weight.device, dtype=torch.bfloat16)
        ids = torch.zeros(num_tokens, device=hidden.device, dtype=torch.int64)
        for layer in self.model.layers:
            layer.ffn(hidden, ids)

    def compute_logits(self, hidden_states, *, cu_seqlens=None):
        context = get_context()
        if cu_seqlens is None and context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
        if cu_seqlens is not None:
            hidden_states = hidden_states[cu_seqlens[1:] - 1]
        return F.linear(hidden_states.float(), self.lm_head.weight)
