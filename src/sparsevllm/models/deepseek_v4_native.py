from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sparsevllm.distributed import get_parallel_context
from sparsevllm.layers.linear import ReplicatedLinear
from sparsevllm.operators.moe import MoeOpSpec, model_activation_dtype, resolve_moe_provider

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention as ReferenceDeepseekV4Attention,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from sparsevllm.utils.context import get_context


class DeepseekV4GroupedFp8Linear(ReplicatedLinear):
    """Eight independent grouped projections stored in one checkpoint tensor."""

    def __init__(self, config) -> None:
        self.num_groups = int(config.o_groups)
        self.group_input_size = int(config.num_attention_heads * config.head_dim) // self.num_groups
        self.group_output_size = int(config.o_lora_rank)
        super().__init__(
            self.group_input_size,
            self.num_groups * self.group_output_size,
            quantization=config.quantization_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.num_groups, self.group_input_size):
            raise ValueError(
                "DeepSeek V4 grouped projection expects "
                f"[..., {self.num_groups}, {self.group_input_size}], got {tuple(x.shape)}."
            )
        if not self.quantized:
            weight = self.weight.view(
                self.num_groups, self.group_output_size, self.group_input_size
            )
            return torch.einsum("...gi,goi->...go", x, weight)
        outputs = []
        scale_rows = self.group_output_size // 128
        for group in range(self.num_groups):
            row_start = group * self.group_output_size
            scale_start = group * scale_rows
            outputs.append(
                self.quant_provider(
                    x[..., group, :],
                    self.weight[row_start : row_start + self.group_output_size],
                    self.weight_scale_inv[scale_start : scale_start + scale_rows],
                )
            )
        return torch.stack(outputs, dim=-2)


class DeepseekV4HyperConnection(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.eps = float(config.hc_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * int(config.hidden_size)))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))
        self.rms_eps = float(config.rms_norm_eps)

    def forward(self, streams: torch.Tensor):
        hc = self.hc_mult
        flat = streams.flatten(start_dim=-2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
            [hc, hc, hc * hc], dim=-1
        )
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind()
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(
            comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale
            + comb_b.view(hc, hc),
            dim=-1,
        ) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=-2).to(streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.eps = float(config.hc_eps)
        self.rms_eps = float(config.rms_norm_eps)
        self.fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * int(config.hidden_size))
        )
        self.base = nn.Parameter(torch.empty(self.hc_mult))
        self.scale = nn.Parameter(torch.empty(1))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(start_dim=-2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        weights = torch.sigmoid(
            F.linear(flat, self.fn.float()) * self.scale.float() + self.base.float()
        ) + self.eps
        return (weights.unsqueeze(-1) * streams).sum(dim=-2).to(streams.dtype)


class DeepseekV4Router(nn.Module):
    def __init__(self, config, *, hash_routing: bool) -> None:
        super().__init__()
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_size = int(config.hidden_size)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.hash_routing = bool(hash_routing)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        if self.hash_routing:
            self.register_buffer(
                "tid2eid",
                torch.empty(int(config.vocab_size), self.top_k, dtype=torch.long),
            )
            self.register_buffer("bias", None)
        else:
            self.register_buffer("tid2eid", None)
            self.register_buffer("bias", torch.empty(self.num_experts))

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        scores = F.softplus(F.linear(hidden_states, self.weight)).sqrt()
        if self.hash_routing:
            topk_ids = self.tid2eid[input_ids].long()
        else:
            topk_ids = torch.topk(
                scores + self.bias,
                self.top_k,
                dim=-1,
                sorted=False,
            ).indices
        topk_weights = scores.gather(-1, topk_ids)
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
        return topk_weights * self.routed_scaling_factor, topk_ids


class DeepseekV4SharedExperts(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.moe_intermediate_size) * int(config.n_shared_experts)
        quantization = config.quantization_config
        self.limit = float(config.swiglu_limit)
        self.w1 = ReplicatedLinear(hidden_size, intermediate_size, quantization=quantization)
        self.w2 = ReplicatedLinear(intermediate_size, hidden_size, quantization=quantization)
        self.w3 = ReplicatedLinear(hidden_size, intermediate_size, quantization=quantization)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.w1(hidden_states).clamp(max=self.limit)
        up = self.w3(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.w2(F.silu(gate) * up)


class DeepseekV4PackedExperts(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        parallel = get_parallel_context()
        self.ep_rank = int(parallel.ep_rank)
        self.ep_size = int(parallel.ep_size)
        self.num_experts = int(config.n_routed_experts)
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.moe_intermediate_size)
        self.op_spec = MoeOpSpec(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=int(config.num_experts_per_tok),
            activation_dtype=model_activation_dtype(config),
            weight_dtype=torch.uint8,
            block_shape=(1, 32),
            ep_size=self.ep_size,
            cuda_graph=bool(getattr(config, "decode_cuda_graph", False)),
            scale_dtype=torch.uint8,
            activation_limit=float(config.swiglu_limit),
        )
        self.provider = resolve_moe_provider(self.op_spec)
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "w13_scale_inv",
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size // 32,
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "w2_scale_inv",
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size // 32,
                dtype=torch.uint8,
            ),
        )
        self._loaded: set[tuple[int, str]] = set()
        self._prepared = False

    def is_local_expert(self, expert_id: int) -> bool:
        return self.local_expert_start <= int(expert_id) < self.local_expert_end

    def load_expert_weight(
        self,
        expert_id: int,
        projection: str,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> None:
        expert_id = int(expert_id)
        if not self.is_local_expert(expert_id):
            raise ValueError(f"Expert {expert_id} is not owned by this EP rank.")
        key = (expert_id, projection)
        if key in self._loaded:
            raise ValueError(f"Duplicate DeepSeek V4 expert tensor {key}.")
        self.provider.load_expert_projection(
            self.op_spec,
            local_expert_id=expert_id - self.local_expert_start,
            projection=projection,
            loaded_weight=weight,
            loaded_scale=scale,
            w13_weight=self.w13_weight.data,
            w2_weight=self.w2_weight.data,
            w13_scale_inv=self.w13_scale_inv,
            w2_scale_inv=self.w2_scale_inv,
        )
        self._loaded.add(key)

    def validate_loaded_weights(self) -> None:
        expected = {
            (expert_id, projection)
            for expert_id in range(self.local_expert_start, self.local_expert_end)
            for projection in ("gate", "down", "up")
        }
        missing = sorted(expected - self._loaded)
        if missing:
            raise ValueError(f"Missing local DeepSeek V4 expert tensors: {missing[:8]}.")

    def prepare_for_inference(self) -> None:
        if self._prepared:
            return
        w13, w2, s13, s2 = self.provider.prepare_weights(
            self.w13_weight,
            self.w2_weight,
            self.w13_scale_inv,
            self.w2_scale_inv,
        )
        self.w13_weight.data = w13
        self.w2_weight.data = w2
        self.w13_scale_inv = s13
        self.w2_scale_inv = s2
        self._prepared = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self._prepared:
            raise RuntimeError("DeepSeek V4 FP4 experts were not prepared after loading.")
        return self.provider.run(
            self.op_spec,
            hidden_states,
            topk_ids,
            topk_weights,
            self.w13_weight,
            self.w2_weight,
            self.w13_scale_inv,
            self.w2_scale_inv,
            local_expert_start=self.local_expert_start,
            ep_rank=self.ep_rank,
        )


class DeepseekV4Moe(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.parallel = get_parallel_context()
        self.gate = DeepseekV4Router(
            config, hash_routing=config.mlp_layer_types[int(layer_idx)] == "hash_moe"
        )
        self.experts = DeepseekV4PackedExperts(config)
        self.shared_experts = DeepseekV4SharedExperts(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        output_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, output_shape[-1])
        input_ids = input_ids.reshape(-1)
        shared = self.shared_experts(hidden_states)
        global_hidden = self.parallel.ep_all_gather_into_tensor(hidden_states)
        global_input_ids = self.parallel.ep_all_gather_into_tensor(input_ids)
        topk_weights, topk_ids = self.gate(global_hidden, global_input_ids)
        routed = self.experts(global_hidden, topk_ids, topk_weights)
        routed = self.parallel.ep_reduce_scatter_tensor(routed)
        return (routed + shared).view(output_shape)


class DeepseekV4Attention(ReferenceDeepseekV4Attention):
    """Reference DSV4 attention math backed by Sparse-vLLM FP8 operators."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        quantization = config.quantization_config
        self.q_a_proj = ReplicatedLinear(
            int(config.hidden_size), int(config.q_lora_rank), quantization=quantization
        )
        self.q_b_proj = ReplicatedLinear(
            int(config.q_lora_rank),
            int(config.num_attention_heads * config.head_dim),
            quantization=quantization,
        )
        self.kv_proj = ReplicatedLinear(
            int(config.hidden_size), int(config.head_dim), quantization=quantization
        )
        self.o_a_proj = DeepseekV4GroupedFp8Linear(config)
        self.o_b_proj = ReplicatedLinear(
            int(config.o_groups * config.o_lora_rank),
            int(config.hidden_size),
            quantization=quantization,
        )
        if self.compressor is not None and hasattr(self.compressor, "indexer"):
            self.compressor.indexer.q_b_proj = ReplicatedLinear(
                int(config.q_lora_rank),
                int(config.index_n_heads * config.index_head_dim),
                quantization=quantization,
            )

    @staticmethod
    def _store_rows(cache: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor, values: torch.Tensor) -> None:
        cache[rows.long(), columns.long()] = values

    def _compress_step(
        self,
        module,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
        *,
        indexer: bool = False,
    ) -> None:
        manager = get_context().cache_manager
        layer_idx = int(self.layer_idx)
        is_csa = self.layer_type == "compressed_sparse_attention"
        ratio = int(module.compress_rate)
        head_dim = int(module.head_dim)
        slot = manager.csa_slot(layer_idx) if is_csa else manager.hca_slot(layer_idx)
        projected_kv = module.kv_proj(hidden_states)
        projected_gate = module.gate_proj(hidden_states)

        if is_csa:
            prefix = "index" if indexer else "csa"
            ring_kv = getattr(manager, f"{prefix}_ring_kv")[slot]
            ring_gate = getattr(manager, f"{prefix}_ring_gate")[slot]
            overlap_kv = getattr(manager, f"{prefix}_overlap_kv")[slot]
            overlap_gate = getattr(manager, f"{prefix}_overlap_gate")[slot]
            output_cache = manager.csa_index[slot] if indexer else manager.csa_kv[slot]
        else:
            ring_kv = manager.hca_ring_kv[slot]
            ring_gate = manager.hca_ring_gate[slot]
            output_cache = manager.hca_kv[slot]

        for token_idx in range(hidden_states.shape[1]):
            position = positions[:, token_idx]
            ring_column = torch.remainder(position, ratio).long()
            kv_token = projected_kv[:, token_idx]
            gate_token = projected_gate[:, token_idx] + module.position_bias[ring_column]
            self._store_rows(ring_kv, rows, ring_column, kv_token)
            self._store_rows(ring_gate, rows, ring_column, gate_token)
            current_kv = ring_kv[rows.long()]
            current_gate = ring_gate[rows.long()]

            if is_csa:
                combined_kv = torch.cat(
                    [overlap_kv[rows.long()], current_kv[..., head_dim:]], dim=1
                )
                combined_gate = torch.cat(
                    [overlap_gate[rows.long()], current_gate[..., head_dim:]], dim=1
                )
                next_overlap_kv = current_kv[..., :head_dim]
                next_overlap_gate = current_gate[..., :head_dim]
            else:
                combined_kv = current_kv
                combined_gate = current_gate

            compressed = module.kv_norm(
                (
                    combined_kv
                    * combined_gate.softmax(dim=1, dtype=torch.float32).to(combined_kv.dtype)
                ).sum(dim=1)
            )
            window_position = (position - ratio + 1).clamp_min(0).unsqueeze(1)
            cos, sin = module.rotary_emb(
                compressed.unsqueeze(1),
                position_ids=window_position,
                layer_type=module.rope_layer_type,
            )
            compressed = apply_rotary_pos_emb(
                compressed[:, None, None, :], cos, sin
            )[:, 0, 0]
            closes_window = torch.remainder(position + 1, ratio).eq(0)
            entry = torch.div(position, ratio, rounding_mode="floor").clamp(
                max=output_cache.shape[1] - 1
            )
            old = output_cache[rows.long(), entry.long()]
            self._store_rows(
                output_cache,
                rows,
                entry,
                torch.where(closes_window[:, None], compressed, old),
            )
            if is_csa:
                old_kv = overlap_kv[rows.long()]
                old_gate = overlap_gate[rows.long()]
                overlap_kv[rows.long()] = torch.where(
                    closes_window[:, None, None], next_overlap_kv, old_kv
                )
                overlap_gate[rows.long()] = torch.where(
                    closes_window[:, None, None], next_overlap_gate, old_gate
                )

    def _compressed_kv(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        manager = get_context().cache_manager
        batch, seq_len, _ = hidden_states.shape
        if self.compressor is None:
            return hidden_states.new_empty((batch, seq_len, 0, self.head_dim)), torch.empty(
                (batch, seq_len, 0), dtype=torch.bool, device=hidden_states.device
            )

        self._compress_step(self.compressor, hidden_states, positions, rows)
        ratio = int(self.compressor.compress_rate)
        capacity = manager.compressed_capacity(ratio, positions)
        entry_ids = torch.arange(capacity, device=hidden_states.device)
        threshold = torch.div(positions + 1, ratio, rounding_mode="floor")

        if self.layer_type == "heavily_compressed_attention":
            cache = manager.hca_kv[manager.hca_slot(self.layer_idx), rows.long(), :capacity]
            valid = entry_ids.view(1, 1, -1) < threshold.unsqueeze(-1)
            return cache.unsqueeze(1).expand(-1, seq_len, -1, -1), valid

        indexer = self.compressor.indexer
        self._compress_step(indexer, hidden_states, positions, rows, indexer=True)
        slot = manager.csa_slot(self.layer_idx)
        index_cache = manager.csa_index[slot, rows.long(), :capacity]
        outer_cache = manager.csa_kv[slot, rows.long(), :capacity]
        if capacity == 0:
            return outer_cache.unsqueeze(1).expand(-1, seq_len, -1, -1), torch.empty(
                (batch, seq_len, 0), dtype=torch.bool, device=hidden_states.device
            )

        cos, sin = indexer.rotary_emb(
            hidden_states, position_ids=positions, layer_type=indexer.rope_layer_type
        )
        index_q = indexer.q_b_proj(q_residual).view(
            batch, seq_len, indexer.num_heads, indexer.head_dim
        )
        index_q = apply_rotary_pos_emb(index_q.transpose(1, 2), cos, sin).transpose(1, 2)
        scores = torch.einsum("bshd,btd->bsht", index_q.float(), index_cache.float())
        scores = F.relu(scores) * indexer.scorer.softmax_scale
        weights = indexer.scorer.weights_proj(hidden_states).float() * indexer.scorer.weights_scaling
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        valid_entries = entry_ids.view(1, 1, -1) < threshold.unsqueeze(-1)
        scores = scores.masked_fill(~valid_entries, float("-inf"))
        top_k = min(int(indexer.index_topk), capacity)
        indices = scores.topk(top_k, dim=-1).indices
        valid = indices < threshold.unsqueeze(-1)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        selected = torch.gather(
            outer_cache.unsqueeze(1).expand(-1, seq_len, -1, -1), 2, gather_index
        )
        return selected, valid

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, past_key_values
        context = get_context()
        manager = context.cache_manager
        state = manager.get_layer_batch_states(self.layer_idx)
        rows = kwargs.pop("cache_rows", state.req_indices).long()
        if kwargs:
            raise TypeError(f"Unexpected DeepSeek V4 attention arguments: {sorted(kwargs)}")
        batch, seq_len, _ = hidden_states.shape
        if rows.numel() != batch:
            raise RuntimeError(
                f"DeepSeek V4 attention row count mismatch: rows={rows.numel()} batch={batch}."
            )

        cos, sin = position_embeddings[self.rope_layer_type]
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(self.q_b_norm(q), cos, sin)
        kv = self.kv_proj(hidden_states).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        kv = apply_rotary_pos_emb(self.kv_norm(kv), cos, sin)[:, 0]

        prefix_len = self.sliding_window - 1
        prefix_offsets = torch.arange(
            prefix_len, 0, -1, device=hidden_states.device
        )
        prefix_positions = position_ids[:, :1] - prefix_offsets
        prefix_valid = prefix_positions >= 0
        prefix_columns = torch.remainder(prefix_positions, self.sliding_window).long()
        raw_layer = manager.raw_kv[self.layer_idx]
        prefix = raw_layer[rows[:, None], prefix_columns]
        raw = torch.cat([prefix, kv], dim=1)
        raw_positions = torch.cat([prefix_positions, position_ids], dim=1)
        raw_valid = raw_positions.unsqueeze(1) <= position_ids.unsqueeze(-1)
        raw_valid &= raw_positions.unsqueeze(1) >= position_ids.unsqueeze(-1) - self.sliding_window + 1
        raw_valid[:, :, :prefix_len] &= prefix_valid.unsqueeze(1)

        compressed, compressed_valid = self._compressed_kv(
            hidden_states, q_residual, position_ids, rows
        )
        raw_per_query = raw.unsqueeze(1).expand(-1, seq_len, -1, -1)
        all_kv = torch.cat([raw_per_query, compressed], dim=2)
        valid = torch.cat([raw_valid, compressed_valid], dim=-1)
        scores = torch.einsum("bhsd,bskd->bhsk", q, all_kv) * self.scaling
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        sinks = self.sinks.view(1, -1, 1, 1).expand(batch, -1, seq_len, -1)
        logits = torch.cat([scores, sinks], dim=-1)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(logits, dim=-1)[..., :-1].to(all_kv.dtype)
        output = torch.einsum("bhsk,bskd->bhsd", probs, all_kv)
        output = apply_rotary_pos_emb(output, cos, -sin).transpose(1, 2)

        columns = torch.remainder(position_ids, self.sliding_window).long()
        # A long prefill revisits ring columns. Advanced assignment with
        # duplicate indices has undefined winner ordering on both CPU and CUDA,
        # so persist only the final, column-unique window.
        store_start = max(0, seq_len - self.sliding_window)
        raw_layer[rows[:, None], columns[:, store_start:]] = kv[:, store_start:]
        grouped = output.reshape(batch, seq_len, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), None


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.attn = DeepseekV4Attention(config, layer_idx)
        self.ffn = DeepseekV4Moe(config, layer_idx)
        self.attn_norm = DeepseekV4RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        self.ffn_norm = DeepseekV4RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        self.hc_attn = DeepseekV4HyperConnection(config)
        self.hc_ffn = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor,
        past_key_values,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        post, comb, collapsed = self.hc_attn(hidden_states)
        attention_output, _ = self.attn(
            self.attn_norm(collapsed),
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_rows=cache_rows,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(
            -2
        ) + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states)
        post, comb, collapsed = self.hc_ffn(hidden_states)
        ffn_output = self.ffn(self.ffn_norm(collapsed), input_ids)
        return post.to(dtype).unsqueeze(-1) * ffn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


class DeepseekV4Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.layers = nn.ModuleList(
            DeepseekV4DecoderLayer(config, layer_idx)
            for layer_idx in range(int(config.num_hidden_layers))
        )
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = DeepseekV4RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or position_ids.shape != input_ids.shape:
            raise ValueError(
                "Native DeepSeek V4 expects [batch, sequence] input and position tensors, "
                f"got {tuple(input_ids.shape)} and {tuple(position_ids.shape)}."
            )
        inputs_embeds = self.embed(input_ids)
        position_embeddings = {
            "main": self.rotary_emb(
                inputs_embeds, position_ids=position_ids, layer_type="main"
            ),
            "compress": self.rotary_emb(
                inputs_embeds, position_ids=position_ids, layer_type="compress"
            ),
        }
        hidden_states = inputs_embeds.unsqueeze(2).expand(
            -1, -1, int(self.config.hc_mult), -1
        ).contiguous()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_rows=cache_rows,
            )
        return self.norm(self.hc_head(hidden_states))
