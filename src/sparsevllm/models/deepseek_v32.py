import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from sparsevllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from sparsevllm.utils.context import get_context
from sparsevllm.utils.flash_mla import try_get_flash_mla, flash_mla_sparse_attn


def _getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform on the last dimension (power-of-2 size).

    This is a pure-torch fallback to avoid relying on `fast_hadamard_transform`.
    It matches the unnormalized Walsh-Hadamard transform commonly used in DeepSeek code.
    """
    n = int(x.shape[-1])
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard transform expects power-of-2 dim, got {n}.")
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y[..., :h]
        b = y[..., h : 2 * h]
        y = torch.cat([a + b, a - b], dim=-1)
        h *= 2
    return y.reshape_as(x)


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """DeepSeek "rotate activation": half-swap then Hadamard on each half."""
    x = x.float()
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x = torch.cat([x2, x1], dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat([_hadamard_transform(x1.contiguous()), _hadamard_transform(x2.contiguous())], dim=-1)
    return x.to(torch.bfloat16)


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: int, high: int, half_dim: int, device: torch.device) -> torch.Tensor:
    if low == high:
        high += 1e-3
    t = torch.arange(half_dim, device=device, dtype=torch.float32)
    m = (t - float(low)) / float(high - low)
    return m.clamp_(0.0, 1.0)


def _yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1.0:
        return 1.0
    return float(0.1 * mscale * math.log(scale) + 1.0)


def precompute_freqs_cis(
    *,
    dim: int,
    max_position_embeddings: int,
    base: float,
    rope_scaling: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor:
    """Precompute complex RoPE frequencies with optional YARN scaling."""
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    mscale = 1.0

    if rope_scaling is not None:
        rope_type = str(rope_scaling.get("type", "")).lower().strip()
        if rope_type != "yarn":
            raise NotImplementedError(f"Unsupported rope_scaling.type={rope_scaling.get('type')}")

        scale = float(rope_scaling.get("factor", 1.0))
        beta_fast = float(rope_scaling.get("beta_fast", 32.0))
        beta_slow = float(rope_scaling.get("beta_slow", 1.0))
        orig_max = int(rope_scaling.get("original_max_position_embeddings", max_position_embeddings))

        inv_freq_extrap = inv_freq
        inv_freq_interp = inv_freq / scale

        low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, half_dim, device)
        inv_freq = inv_freq_interp * (1.0 - inv_freq_mask) + inv_freq_extrap * inv_freq_mask

        mscale_val = float(rope_scaling.get("mscale", 1.0))
        mscale_all_dim_val = float(rope_scaling.get("mscale_all_dim", 1.0))
        mscale = _yarn_get_mscale(scale, mscale_val) / _yarn_get_mscale(scale, mscale_all_dim_val)

    t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_pos, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) * float(mscale)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Apply RoPE using complex multiplication.

    Assumption: sequence dimension is the first dimension of `x`, matching `freqs_cis.shape[0]`.
    """
    orig_dtype = x.dtype
    x = x.float()
    if interleaved:
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        freqs = freqs_cis
        # Broadcast freqs over any extra dims after sequence.
        while freqs.ndim < x_complex.ndim:
            freqs = freqs.unsqueeze(1)
        out = x_complex * freqs
        out = torch.view_as_real(out).reshape_as(x)
    else:
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], 2, -1).transpose(-2, -1))
        freqs = freqs_cis
        while freqs.ndim < x_complex.ndim:
            freqs = freqs.unsqueeze(1)
        out = x_complex * freqs
        out = torch.view_as_real(out).transpose(-2, -1).reshape_as(x)
    return out.to(orig_dtype)


@dataclass(frozen=True)
class DeepSeekV32Args:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    rotary_emb_interleaved: bool
    intermediate_size: int
    moe_intermediate_size: int
    n_dense_layers: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    scoring_func: str
    aux_loss_alpha: float
    seq_aux: bool
    norm_topk_prob: bool
    rms_norm_eps: float


def _args_from_hf_config(hf_config: Any, *, dsa_topk: int | None = None) -> DeepSeekV32Args:
    dim = int(_getattr(hf_config, "hidden_size"))
    n_layers = int(_getattr(hf_config, "num_hidden_layers"))
    n_heads = int(_getattr(hf_config, "num_attention_heads"))
    vocab_size = int(_getattr(hf_config, "vocab_size"))

    q_lora_rank = int(_getattr(hf_config, "q_lora_rank"))
    kv_lora_rank = int(_getattr(hf_config, "kv_lora_rank"))
    qk_nope_head_dim = int(_getattr(hf_config, "qk_nope_head_dim"))
    qk_rope_head_dim = int(_getattr(hf_config, "qk_rope_head_dim"))
    v_head_dim = int(_getattr(hf_config, "v_head_dim"))

    index_n_heads = int(_getattr(hf_config, "index_n_heads"))
    index_head_dim = int(_getattr(hf_config, "index_head_dim"))
    index_topk = int(dsa_topk if dsa_topk is not None else _getattr(hf_config, "index_topk"))

    rope_theta = float(_getattr(hf_config, "rope_theta", 10000.0))
    rope_scaling = _getattr(hf_config, "rope_scaling", None)
    rotary_emb_interleaved = bool(_getattr(hf_config, "rotary_emb_interleaved", False))

    intermediate_size = int(_getattr(hf_config, "intermediate_size"))
    moe_intermediate_size = int(_getattr(hf_config, "moe_intermediate_size"))
    n_dense_layers = int(_getattr(hf_config, "n_dense_layers", 0))
    n_routed_experts = int(_getattr(hf_config, "n_routed_experts", 0))
    n_shared_experts = int(_getattr(hf_config, "n_shared_experts", 0))
    num_experts_per_tok = int(_getattr(hf_config, "num_experts_per_tok", 0))
    routed_scaling_factor = float(_getattr(hf_config, "routed_scaling_factor", 1.0))
    scoring_func = str(_getattr(hf_config, "scoring_func", "softmax")).lower()
    aux_loss_alpha = float(_getattr(hf_config, "aux_loss_alpha", 0.0))
    seq_aux = bool(_getattr(hf_config, "seq_aux", False))
    norm_topk_prob = bool(_getattr(hf_config, "norm_topk_prob", True))
    rms_norm_eps = float(_getattr(hf_config, "rms_norm_eps", 1e-6))

    return DeepSeekV32Args(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        rotary_emb_interleaved=rotary_emb_interleaved,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        n_dense_layers=n_dense_layers,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        scoring_func=scoring_func,
        aux_loss_alpha=aux_loss_alpha,
        seq_aux=seq_aux,
        norm_topk_prob=norm_topk_prob,
        rms_norm_eps=rms_norm_eps,
    )


class DeepSeekV32MLP(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, intermediate, bias=False)
        self.w3 = ColumnParallelLinear(dim, intermediate, bias=False)
        self.w2 = RowParallelLinear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekV32Gate(nn.Module):
    def __init__(self, args: DeepSeekV32Args):
        super().__init__()
        self.topk = int(args.num_experts_per_tok)
        self.n_routed_experts = int(args.n_routed_experts)
        self.routed_scaling_factor = float(args.routed_scaling_factor)
        self.scoring_func = str(args.scoring_func)
        self.alpha = float(args.aux_loss_alpha)
        self.seq_aux = bool(args.seq_aux)
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.gating_dim = int(args.dim)

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x, self.weight)
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            raise NotImplementedError(f"Unknown scoring_func={self.scoring_func}")

        if self.topk > 1:
            topk_weight, topk_idx = scores.topk(self.topk, dim=-1, sorted=False)
        else:
            topk_weight, topk_idx = scores.max(dim=-1, keepdim=True)

        if self.scoring_func == "sigmoid" or (self.topk > 1 and self.norm_topk_prob):
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class DeepSeekV32MoE(nn.Module):
    def __init__(self, args: DeepSeekV32Args):
        super().__init__()
        self.dim = int(args.dim)
        self.n_routed_experts = int(args.n_routed_experts)
        self.n_shared_experts = int(args.n_shared_experts)
        self.gate = DeepSeekV32Gate(args)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        if self.world_size != 1:
            raise NotImplementedError("DeepSeekV32MoE currently supports tensor_parallel_size=1 only.")

        self.experts = nn.ModuleList(
            [DeepSeekV32MLP(args.dim, args.moe_intermediate_size) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = DeepSeekV32MLP(args.dim, args.intermediate_size * args.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        x = x.view(-1, self.dim)

        topk_idx, topk_weight = self.gate(x)
        y = torch.zeros_like(x)

        # Dispatch: accumulate contributions expert-by-expert (inference-first; correctness over speed).
        for expert_id in range(self.n_routed_experts):
            token_ids, which = (topk_idx == expert_id).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            out = self.experts[expert_id](x[token_ids])
            y[token_ids] += out * topk_weight[token_ids, which].unsqueeze(-1)

        y = y + self.shared_experts(x)
        y = y.view(*orig_shape)
        return y


class DeepSeekV32Indexer(nn.Module):
    """Torch fallback Indexer for DeepSeek DSA (no DeepGEMM / fp8 kernel)."""

    def __init__(self, args: DeepSeekV32Args):
        super().__init__()
        self.n_heads = int(args.index_n_heads)
        self.head_dim = int(args.index_head_dim)
        self.topk = int(args.index_topk)
        self.rope_head_dim = int(args.qk_rope_head_dim)
        self.interleaved = False

        self.q_norm = RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(args.dim, self.n_heads, bias=False, dtype=torch.float32)

        self.softmax_scale = float(self.head_dim ** -0.5)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, freqs_cis: torch.Tensor, causal: bool) -> torch.Tensor:
        # x: (T, dim), qr: (T, q_lora_rank), freqs_cis: (T, rope_dim/2)
        t = int(x.shape[0])
        q = self.wq_b(self.q_norm(qr)).view(t, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=self.interleaved)
        q = torch.cat([q_pe, q_nope], dim=-1)
        q = _rotate_activation(q)

        weights = self.weights_proj(x.float()) * (self.n_heads ** -0.5)
        weights = weights * self.softmax_scale

        k = self.wk(x).view(t, self.n_heads, self.head_dim)
        k = self.k_norm(k)
        k_pe, k_nope = k.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=self.interleaved)
        k = torch.cat([k_pe, k_nope], dim=-1)
        k = _rotate_activation(k)

        # index_score[t, s] = sum_h w[t,h] * relu(dot(q[t,h], k[s,h]))
        dots = torch.einsum("thd,shd->ths", q.float(), k.float())
        dots = torch.relu(dots)
        dots = dots * weights[:, :, None].to(dots.dtype)
        index_score = dots.sum(dim=1)  # (T, T)

        if causal:
            # Disallow attending to future keys.
            mask = torch.full((t, t), float("-inf"), device=x.device, dtype=index_score.dtype)
            mask = torch.triu(mask, diagonal=1)
            index_score = index_score + mask

        k_top = min(self.topk, t)
        topk_idx = torch.topk(index_score, k=k_top, dim=-1).indices  # (T, k_top)
        if causal:
            # `topk` still returns indices for -inf entries; mark them invalid explicitly.
            row_pos = torch.arange(t, device=topk_idx.device, dtype=topk_idx.dtype)
            topk_idx = torch.where(topk_idx <= row_pos[:, None], topk_idx, -1)
        if k_top < self.topk:
            pad = torch.full((t, self.topk - k_top), -1, device=topk_idx.device, dtype=topk_idx.dtype)
            topk_idx = torch.cat([topk_idx, pad], dim=-1)
        return topk_idx


class DeepSeekV32MLA(nn.Module):
    def __init__(self, args: DeepSeekV32Args, *, use_flash_mla: bool):
        super().__init__()
        tp_size = dist.get_world_size()
        if tp_size != 1:
            raise NotImplementedError("DeepSeekV32MLA currently supports tensor_parallel_size=1 only.")

        self.args = args
        self.use_flash_mla = bool(use_flash_mla)

        self.n_heads = int(args.n_heads)
        self.qk_nope_head_dim = int(args.qk_nope_head_dim)
        self.qk_rope_head_dim = int(args.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(args.v_head_dim)
        self.kv_lora_rank = int(args.kv_lora_rank)

        self.wq_a = ColumnParallelLinear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(args.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = ColumnParallelLinear(args.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, args.dim, bias=False)

        self.softmax_scale = float(self.qk_head_dim ** -0.5)
        self.interleaved = bool(args.rotary_emb_interleaved)

        self.indexer = DeepSeekV32Indexer(args)

    def _wkv_b_per_head(self) -> torch.Tensor:
        # (n_heads * (qk_nope+v), kv_lora_rank) -> (n_heads, qk_nope+v, kv_lora_rank)
        w = self.wkv_b.weight
        return w.view(self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)

    def _attn_sparse_prefill(
        self,
        q_nope: torch.Tensor,  # (T, H, qk_nope)
        q_pe: torch.Tensor,  # (T, H, rope)
        kv: torch.Tensor,  # (T, kv_lora)
        k_pe: torch.Tensor,  # (T, rope)
        topk_idx: torch.Tensor,  # (T, topk)
    ) -> torch.Tensor:
        wkv_b = self._wkv_b_per_head()
        w_k = wkv_b[:, : self.qk_nope_head_dim, :]  # (H, qk_nope, kv_lora)
        w_v = wkv_b[:, -self.v_head_dim :, :]  # (H, v, kv_lora)

        q_nope_proj = torch.einsum("thd,hdc->thc", q_nope, w_k)  # (T, H, kv_lora)
        q_key = torch.cat([q_nope_proj, q_pe], dim=-1).to(torch.bfloat16)  # (T, H, 576)
        kv_key = torch.cat([kv, k_pe], dim=-1).to(torch.bfloat16).unsqueeze(1)  # (T, 1, 576)

        # FlashMLA sparse prefill (optional).
        if self.use_flash_mla and try_get_flash_mla() is not None:
            indices = topk_idx.to(torch.int32).unsqueeze(1)  # (T, 1, topk)
            out_latent = flash_mla_sparse_attn(
                q_key,
                kv_key,
                indices,
                sm_scale=self.softmax_scale,
                d_v=self.kv_lora_rank,
                topk_length=int((topk_idx >= 0).sum(dim=-1).max().item()),
            )  # (T, H, kv_lora)
        else:
            # Torch fallback: compute sparse attention in latent space using gathered top-k KV.
            t, h = int(q_key.shape[0]), int(q_key.shape[1])
            out_latent = torch.empty((t, h, self.kv_lora_rank), device=q_key.device, dtype=torch.bfloat16)
            for qi in range(t):
                idx = topk_idx[qi]
                idx = idx[idx >= 0].to(torch.long)
                if idx.numel() == 0:
                    out_latent[qi].zero_()
                    continue
                kv_sel = kv.index_select(0, idx)  # (K, kv_lora)
                pe_sel = k_pe.index_select(0, idx)  # (K, rope)

                logits_nope = q_nope_proj[qi].matmul(kv_sel.t())  # (H, K)
                logits_pe = q_pe[qi].matmul(pe_sel.t())  # (H, K)
                logits = (logits_nope + logits_pe) * self.softmax_scale
                attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)
                out_latent[qi] = attn.matmul(kv_sel)  # (H, kv_lora)

        out_v = torch.einsum("thc,hdc->thd", out_latent, w_v)  # (T, H, v)
        return self.wo(out_v.flatten(1, -1))

    def _attn_dense_decode(
        self,
        q_nope: torch.Tensor,  # (1, H, qk_nope)
        q_pe: torch.Tensor,  # (1, H, rope)
        kv_all: torch.Tensor,  # (S, kv_lora)
        pe_all: torch.Tensor,  # (S, rope)
    ) -> torch.Tensor:
        wkv_b = self._wkv_b_per_head()
        w_k = wkv_b[:, : self.qk_nope_head_dim, :]
        w_v = wkv_b[:, -self.v_head_dim :, :]

        q_nope_proj = torch.einsum("bhd,hdc->bhc", q_nope, w_k)  # (1, H, kv_lora)
        logits_nope = q_nope_proj[0].matmul(kv_all.t())  # (H, S)
        logits_pe = q_pe[0].matmul(pe_all.t())  # (H, S)
        logits = (logits_nope + logits_pe) * self.softmax_scale
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)  # (H, S)
        out_latent = attn.matmul(kv_all)  # (H, kv_lora)
        out_v = torch.einsum("hc,hdc->hd", out_latent, w_v)  # (H, v)
        return self.wo(out_v.flatten(0).unsqueeze(0))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, freqs_cis_table: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        cache_manager = ctx.cache_manager
        layer_idx = int(ctx.now_layer_idx)
        batch_states = cache_manager.get_layer_batch_states(layer_idx)

        if ctx.is_prefill:
            if ctx.cu_seqlens_q is None or ctx.cu_seqlens_q.numel() <= 1:
                return torch.empty_like(hidden_states)
            cu = ctx.cu_seqlens_q.to(torch.int64)
            outputs: list[torch.Tensor] = []
            for i in range(int(cu.numel() - 1)):
                start = int(cu[i].item())
                end = int(cu[i + 1].item())
                x = hidden_states[start:end]
                pos = positions[start:end].to(torch.int64)
                if x.numel() == 0:
                    continue
                start_pos = int(pos[0].item())
                if start_pos != 0:
                    raise RuntimeError(
                        "DeepSeek-V3.2 DSA path currently requires non-chunked prefill. "
                        "Set `chunk_prefill_size` >= prompt length so prefill runs in a single chunk."
                    )

                # Compute Q/KV for this sequence.
                qr = self.wq_a(x)
                q = self.wq_b(self.q_norm(qr)).view(-1, self.n_heads, self.qk_head_dim)
                q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

                kv = self.wkv_a(x)
                kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv = self.kv_norm(kv)

                freqs_cis = freqs_cis_table.index_select(0, pos)  # (T, rope/2)
                q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=self.interleaved)
                k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=self.interleaved)

                # Persist KV to cache for future decode.
                slots = batch_states.slot_mapping[start:end].to(torch.int64)
                kv_cache_layer, pe_cache_layer = cache_manager.get_layer_mla_cache(layer_idx)
                kv_cache_layer.index_copy_(0, slots, kv)
                pe_cache_layer.index_copy_(0, slots, k_pe)

                # Top-k indices (DSA) + sparse attention prefill.
                topk_idx = self.indexer(x, qr, freqs_cis, causal=True)
                out = self._attn_sparse_prefill(q_nope, q_pe, kv, k_pe, topk_idx)
                outputs.append(out)

            if not outputs:
                return torch.empty_like(hidden_states)
            return torch.cat(outputs, dim=0)

        # Decode path (dense; uses persisted KV cache).
        bsz = int(hidden_states.shape[0])
        outputs = torch.empty_like(hidden_states)
        kv_cache_layer, pe_cache_layer = cache_manager.get_layer_mla_cache(layer_idx)
        slot_table = cache_manager.get_layer_buffer_req_to_token_slots(layer_idx)

        for i in range(bsz):
            x = hidden_states[i : i + 1]
            pos = positions[i : i + 1].to(torch.int64)
            end_pos = int(batch_states.context_lens[i].item())
            row_idx = int(batch_states.req_indices[i].item())

            # Compute Q/KV for the new token and update cache.
            qr = self.wq_a(x)
            q = self.wq_b(self.q_norm(qr)).view(1, self.n_heads, self.qk_head_dim)
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            kv = self.wkv_a(x)
            kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv = self.kv_norm(kv)

            freqs_cis = freqs_cis_table.index_select(0, pos)  # (1, rope/2)
            q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=self.interleaved)
            k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=self.interleaved)

            slot = batch_states.slot_mapping[i : i + 1].to(torch.int64)
            kv_cache_layer.index_copy_(0, slot, kv)
            pe_cache_layer.index_copy_(0, slot, k_pe)

            # Gather full KV for this sequence and run dense decode.
            seq_slots = slot_table[row_idx, :end_pos].to(torch.int64)
            kv_all = kv_cache_layer.index_select(0, seq_slots)
            pe_all = pe_cache_layer.index_select(0, seq_slots)
            outputs[i : i + 1] = self._attn_dense_decode(q_nope, q_pe, kv_all, pe_all)

        return outputs


class DeepSeekV32DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV32Args, *, use_flash_mla: bool):
        super().__init__()
        self.attn = DeepSeekV32MLA(args, use_flash_mla=use_flash_mla)
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


class DeepSeekV32ForCausalLM(nn.Module):
    def __init__(self, hf_config: Any, *, dsa_topk: int | None = None, use_flash_mla: bool = True):
        super().__init__()
        self.args = _args_from_hf_config(hf_config, dsa_topk=dsa_topk)
        if dist.get_world_size() != 1:
            raise NotImplementedError("DeepSeekV32ForCausalLM currently supports tensor_parallel_size=1 only.")

        self.tok_embeddings = VocabParallelEmbedding(self.args.vocab_size, self.args.dim)
        self.layers = nn.ModuleList(
            [DeepSeekV32DecoderLayer(i, self.args, use_flash_mla=use_flash_mla) for i in range(self.args.n_layers)]
        )
        self.norm = RMSNorm(self.args.dim, eps=self.args.rms_norm_eps)
        self.output = ParallelLMHead(self.args.vocab_size, self.args.dim)

        # Precompute rotary freqs (YARN).
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
