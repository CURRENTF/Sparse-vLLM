"""Native shared-KV attention over cache/runtime-owned physical views."""

import torch
from torch import nn

from sparsevllm.distributed import get_parallel_context
from sparsevllm.layers.inverse_rotary_grouped_linear import InverseRotaryGroupedLinear
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from sparsevllm.layers.rotary_embedding import _compute_rope_parameters
from sparsevllm.models.deepseek_v4.compression import DeepseekV4Compressor
from sparsevllm.models.deepseek_v4.indexer import DeepseekV4Indexer
from sparsevllm.operators.indexed_shared_kv_attention import IndexedSharedKVAttentionSpec, prepare_indexed_shared_kv_attention
from sparsevllm.operators.native_rope import NativeRotarySpec, prepare_native_rotary


class DeepseekV4Attention(nn.Module):
    def __init__(self, config, layer_id, *, quantization, max_model_len, max_num_tokens, max_num_requests,
                 cache_dtype=torch.bfloat16):
        super().__init__()
        if (not quantization.enabled or quantization.scale_fmt != "ue8m0"
                or quantization.max_num_tokens is None or max_num_tokens > quantization.max_num_tokens):
            raise ValueError("Native attention requires UE8M0 FP8 and a prepared token capacity.")
        self.parallel = get_parallel_context()
        tp = self.parallel.attn_tp_size
        if config.num_attention_heads % tp or config.o_groups % tp:
            raise ValueError("Native attention heads and output groups must divide attention TP.")
        self.num_heads, self.num_groups = config.num_attention_heads // tp, config.o_groups // tp
        self.head_dim = config.head_dim
        self.ratio = config.compress_ratios[layer_id]
        self.wq_a = ReplicatedLinear(config.hidden_size, config.q_lora_rank, quantization=quantization)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps).bfloat16()
        self.wq_b = ColumnParallelLinear(config.q_lora_rank, config.num_attention_heads * config.head_dim, quantization=quantization)
        self.wkv = ReplicatedLinear(config.hidden_size, config.head_dim, quantization=quantization)
        self.kv_norm_weight = nn.Parameter(torch.empty(config.head_dim, dtype=torch.float32), requires_grad=False)
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False)
        self.wo_b = RowParallelLinear(config.o_groups * config.o_lora_rank, config.hidden_size,
                                      quantization=quantization, reduce_results=False)
        self.compressor = self.indexer = None
        if self.ratio:
            self.compressor = DeepseekV4Compressor(config, ratio=self.ratio, head_dim=config.head_dim, rotate=False,
                                                  max_num_tokens=max_num_tokens, max_num_requests=max_num_requests)
        if self.ratio == 4:
            self.indexer = DeepseekV4Indexer(config, quantization=quantization, max_num_tokens=max_num_tokens,
                                            max_num_requests=max_num_requests)
        rope_scaling = tuple(sorted(dict(config.rope_scaling, attention_factor=1.).items())) if self.ratio else None
        theta = config.compress_rope_theta if self.ratio else config.rope_theta
        inv_freq, _ = _compute_rope_parameters(config.qk_rope_head_dim, theta, rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        device_index = self.wq_a.weight.device.index
        self.query_transform = prepare_native_rotary(
            NativeRotarySpec(config.head_dim, config.qk_rope_head_dim, "query_bf16", config.rms_norm_eps),
            device_index=device_index,
        )
        self.key_transform = prepare_native_rotary(
            NativeRotarySpec(config.head_dim, config.qk_rope_head_dim, "weighted_fp32", config.rms_norm_eps, quantize_nope=True),
            device_index=device_index,
        )
        self.wo_a = InverseRotaryGroupedLinear(
            self.num_groups, self.num_heads, config.head_dim, config.o_lora_rank, inv_freq=inv_freq,
            max_num_tokens=max_num_tokens, max_positions=max_model_len, device_index=device_index,
        )
        compressed_capacity = 512 if self.ratio == 4 else max(1, max_model_len // self.ratio) if self.ratio else 0
        self.attention = prepare_indexed_shared_kv_attention(IndexedSharedKVAttentionSpec(
            self.num_heads, config.head_dim, config.sliding_window + compressed_capacity,
            config.head_dim ** -.5, max_num_tokens,
            cache_dtype=cache_dtype,
        ), device_index=device_index)

    def forward(self, x, cache, batch, index_selection=None):
        cache.prepare(batch)
        positions = batch.window.positions
        qr = self.q_norm(self.wq_a(x))
        query = self.wq_b(qr).view(len(x), self.num_heads, self.head_dim)
        self.query_transform.run(query, positions, self.inv_freq, out=query)
        kv = self.wkv(x)[:, None]
        self.key_transform.run(kv, positions, self.inv_freq, weight=self.kv_norm_weight, out=kv)
        cache.store_window(batch, kv)
        selected = None
        if self.compressor is not None:
            values, compressed_positions = self.compressor(x, batch.compression, self.inv_freq)
            cache.store_compressed(batch, values, compressed_positions)
        if self.indexer is not None:
            values, compressed_positions = self.indexer.compressor(x, batch.index_compression, self.inv_freq)
            cache.store_index(batch, values, compressed_positions)
            index_query, weights = self.indexer.project(x, qr, positions, self.inv_freq)
            selected = index_selection.select_compressed_index(index_query, weights, batch.index_view, out=batch.selected_compressed)
        view = cache.attention_view(batch, selected)
        output = self.attention.run(query, view, self.attn_sink)
        cache.finish_window(batch, kv)
        result = self.wo_b(self.wo_a(output, positions).flatten(1))
        if self.parallel.attn_tp_size > 1:
            result = self.parallel.attn_tp.all_reduce(result.float()).bfloat16()
        return result
