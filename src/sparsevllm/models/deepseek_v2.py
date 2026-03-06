from typing import Any

import torch
from torch import nn
import torch.distributed as dist

from sparsevllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.models.deepseek_v32 import (
    DeepSeekV32Args,
    DeepSeekV32MLA,
    DeepSeekV32MLP,
    DeepSeekV32MoE,
    _getattr,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from sparsevllm.triton_kernel.deepseek_v2_prefill import context_attention_fwd_with_v
from sparsevllm.utils.context import get_context


def _args_from_hf_config(hf_config: Any, *, dsa_topk: int | None = None) -> DeepSeekV32Args:
    model_type = str(_getattr(hf_config, "model_type", "") or "")
    if model_type != "deepseek_v2":
        raise ValueError(f"DeepSeekV2ForCausalLM expects model_type='deepseek_v2', got {model_type!r}.")

    q_lora_rank_raw = _getattr(hf_config, "q_lora_rank", 0)
    q_lora_rank = int(q_lora_rank_raw or 0)
    if q_lora_rank != 0:
        raise ValueError(f"DeepSeek-V2 path expects q_lora_rank == 0, got {q_lora_rank}.")

    n_heads = int(_getattr(hf_config, "num_attention_heads"))
    qk_nope_head_dim = int(_getattr(hf_config, "qk_nope_head_dim"))
    qk_rope_head_dim = int(_getattr(hf_config, "qk_rope_head_dim"))
    moe_intermediate_size = int(_getattr(hf_config, "moe_intermediate_size"))
    n_shared_experts = int(_getattr(hf_config, "n_shared_experts", 0))

    return DeepSeekV32Args(
        dim=int(_getattr(hf_config, "hidden_size")),
        n_layers=int(_getattr(hf_config, "num_hidden_layers")),
        n_heads=n_heads,
        vocab_size=int(_getattr(hf_config, "vocab_size")),
        q_lora_rank=0,
        kv_lora_rank=int(_getattr(hf_config, "kv_lora_rank")),
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=int(_getattr(hf_config, "v_head_dim")),
        index_n_heads=int(_getattr(hf_config, "index_n_heads", n_heads)),
        index_head_dim=int(_getattr(hf_config, "index_head_dim", qk_nope_head_dim + qk_rope_head_dim)),
        index_topk=int(dsa_topk if dsa_topk is not None else _getattr(hf_config, "index_topk", 0)),
        rope_theta=float(_getattr(hf_config, "rope_theta", 10000.0)),
        rope_scaling=_getattr(hf_config, "rope_scaling", None),
        rotary_emb_interleaved=bool(_getattr(hf_config, "rotary_emb_interleaved", True)),
        intermediate_size=int(_getattr(hf_config, "intermediate_size")),
        moe_intermediate_size=moe_intermediate_size,
        n_dense_layers=int(_getattr(hf_config, "n_dense_layers", _getattr(hf_config, "first_k_dense_replace", 0))),
        n_routed_experts=int(_getattr(hf_config, "n_routed_experts", 0)),
        n_shared_experts=n_shared_experts,
        shared_experts_intermediate_size=int(moe_intermediate_size * n_shared_experts),
        num_experts_per_tok=int(_getattr(hf_config, "num_experts_per_tok", 0)),
        routed_scaling_factor=float(_getattr(hf_config, "routed_scaling_factor", 1.0)),
        scoring_func=str(_getattr(hf_config, "scoring_func", "softmax")).lower(),
        aux_loss_alpha=float(_getattr(hf_config, "aux_loss_alpha", 0.0)),
        seq_aux=bool(_getattr(hf_config, "seq_aux", False)),
        norm_topk_prob=bool(_getattr(hf_config, "norm_topk_prob", False)),
        rms_norm_eps=float(_getattr(hf_config, "rms_norm_eps", 1e-6)),
        use_dsa=False,
    )


class DeepSeekV2ForCausalLM(nn.Module):
    def __init__(self, hf_config: Any, *, dsa_topk: int | None = None, use_flash_mla: bool = True):
        super().__init__()
        self.hf_model_type = str(_getattr(hf_config, "model_type", "") or "")
        self.args = _args_from_hf_config(hf_config, dsa_topk=dsa_topk)
        if dist.get_world_size() != 1:
            raise NotImplementedError("DeepSeekV2ForCausalLM currently supports tensor_parallel_size=1 only.")

        self.tok_embeddings = VocabParallelEmbedding(self.args.vocab_size, self.args.dim)
        self.layers = nn.ModuleList(
            [DeepSeekV2DecoderLayer(i, self.args, use_flash_mla=use_flash_mla) for i in range(self.args.n_layers)]
        )
        self.norm = RMSNorm(self.args.dim, eps=self.args.rms_norm_eps)
        self.output = ParallelLMHead(self.args.vocab_size, self.args.dim)

        max_pos = int(_getattr(hf_config, "max_position_embeddings", 4096))
        freqs_cis = precompute_freqs_cis(
            dim=self.args.qk_rope_head_dim,
            max_position_embeddings=max_pos,
            base=self.args.rope_theta,
            rope_scaling=self.args.rope_scaling,
            device=torch.device("cuda"),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.tok_embeddings(input_ids)
        residual = None
        context = get_context()

        for i, layer in enumerate(self.layers):
            context.now_layer_idx = i
            hidden_states, residual = layer(positions, hidden_states, residual, self.freqs_cis)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(hidden_states)


class DeepSeekV2MLA(DeepSeekV32MLA):
    def _attn_chunked_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_all: torch.Tensor,
        pe_all: torch.Tensor,
        prompt_cache_len: int,
    ) -> torch.Tensor:
        wkv_b = self._wkv_b_per_head()
        w_k = wkv_b[:, : self.qk_nope_head_dim, :]
        w_v = wkv_b[:, -self.v_head_dim :, :]

        k_nope = torch.einsum("sc,hdc->shd", kv_all, w_k).contiguous()
        k_rope = pe_all.unsqueeze(1).contiguous()
        v = torch.einsum("sc,hdc->shd", kv_all, w_v).contiguous()

        if q_nope.is_cuda:
            out_v = torch.empty(
                (q_nope.shape[0], self.n_heads, self.v_head_dim),
                dtype=v.dtype,
                device=q_nope.device,
            )
            b_start_loc = torch.zeros((1,), dtype=torch.int32, device=q_nope.device)
            b_kv_start_loc = torch.zeros((1,), dtype=torch.int32, device=q_nope.device)
            b_seq_len = torch.tensor([kv_all.shape[0]], dtype=torch.int32, device=q_nope.device)
            b_prompt_cache_len = torch.tensor([prompt_cache_len], dtype=torch.int32, device=q_nope.device)
            context_attention_fwd_with_v(
                q_nope.contiguous(),
                q_pe.contiguous(),
                k_nope,
                k_rope,
                v,
                out_v,
                b_start_loc,
                b_kv_start_loc,
                b_seq_len,
                b_prompt_cache_len,
                max_input_len=int(q_nope.shape[0]),
                softmax_scale=self.softmax_scale,
            )
        else:
            logits_nope = torch.einsum("thd,shd->ths", q_nope.float(), k_nope.float())
            logits_pe = torch.einsum("thd,sd->ths", q_pe.float(), pe_all.float())
            logits = (logits_nope + logits_pe) * self.softmax_scale
            q_positions = torch.arange(q_nope.shape[0], device=q_nope.device, dtype=torch.long) + int(prompt_cache_len)
            k_positions = torch.arange(kv_all.shape[0], device=kv_all.device, dtype=torch.long)
            causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            attn = torch.softmax(logits.masked_fill(~causal_mask[:, None, :], float("-inf")), dim=-1, dtype=torch.float32)
            out_v = torch.einsum("ths,shd->thd", attn.to(v.dtype), v)

        return self.wo(out_v.flatten(1, -1))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, freqs_cis_table: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        cache_manager = ctx.cache_manager
        layer_idx = int(ctx.now_layer_idx)
        batch_states = cache_manager.get_layer_batch_states(layer_idx)

        if ctx.is_prefill:
            if ctx.cu_seqlens_q is None or ctx.cu_seqlens_q.numel() <= 1:
                return torch.empty_like(hidden_states)
            cu = ctx.cu_seqlens_q.to(torch.int64)
            kv_cache_layer, pe_cache_layer = cache_manager.get_layer_mla_cache(layer_idx)
            slot_table = cache_manager.get_layer_buffer_req_to_token_slots(layer_idx)
            outputs: list[torch.Tensor] = []
            for i in range(int(cu.numel() - 1)):
                start = int(cu[i].item())
                end = int(cu[i + 1].item())
                x = hidden_states[start:end]
                pos = positions[start:end].to(torch.int64)
                if x.numel() == 0:
                    continue

                start_pos = int(pos[0].item())
                end_pos = int(batch_states.context_lens[i].item())
                row_idx = int(batch_states.req_indices[i].item())

                _, q_nope, q_pe = self._project_q(x)

                kv = self.wkv_a(x)
                kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv = self.kv_norm(kv)

                freqs_cis = freqs_cis_table.index_select(0, pos)
                q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=self.interleaved)
                k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=self.interleaved)

                slots = batch_states.slot_mapping[start:end].to(torch.int64)
                kv_cache_layer.index_copy_(0, slots, kv)
                pe_cache_layer.index_copy_(0, slots, k_pe)

                if start_pos == 0 and end_pos == kv.shape[0]:
                    kv_all = kv
                    pe_all = k_pe
                else:
                    seq_slots = slot_table[row_idx, :end_pos].to(torch.int64)
                    kv_all = kv_cache_layer.index_select(0, seq_slots)
                    pe_all = pe_cache_layer.index_select(0, seq_slots)

                outputs.append(self._attn_chunked_prefill(q_nope, q_pe, kv_all, pe_all, prompt_cache_len=start_pos))

            if not outputs:
                return torch.empty_like(hidden_states)
            return torch.cat(outputs, dim=0)

        return super().forward(positions, hidden_states, freqs_cis_table)


class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV32Args, *, use_flash_mla: bool):
        super().__init__()
        self.attn = DeepSeekV2MLA(args, use_flash_mla=use_flash_mla)
        if layer_id < args.n_dense_layers:
            self.ffn = DeepSeekV32MLP(args.dim, args.intermediate_size)
        else:
            self.ffn = DeepSeekV32MoE(args)

        self.attn_norm = RMSNorm(args.dim, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        freqs_cis_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.attn_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.attn_norm(hidden_states, residual)
        hidden_states = self.attn(positions, hidden_states, freqs_cis_table)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.ffn(hidden_states)
        return hidden_states, residual
