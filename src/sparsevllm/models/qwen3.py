import os
import torch
from torch import nn
from transformers import Qwen3Config
from sparsevllm.distributed import get_parallel_context
from sparsevllm.utils.log import logger
from sparsevllm.utils.context import get_context

from sparsevllm.layers.activation import SiluAndMul
from sparsevllm.layers.attention import Attention
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from sparsevllm.layers.rotary_embedding import get_rope
from sparsevllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def _get_rope_theta(config: Qwen3Config) -> float:
    if hasattr(config, "rope_theta"):
        return config.rope_theta
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    return 10000


def _get_rope_scaling(config: Qwen3Config):
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return None

    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        is_default_rope = rope_type in (None, "default")
        allowed_default_keys = {"rope_type", "type", "rope_theta"}
        if is_default_rope and set(rope_scaling).issubset(allowed_default_keys):
            return None

    raise NotImplementedError(f"Unsupported Qwen3 rope_scaling={rope_scaling!r}.")


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        proj_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        tp_size = get_parallel_context().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.proj_chunk_size = int(proj_chunk_size)
        if self.proj_chunk_size <= 0:
            raise ValueError(f"proj_chunk_size must be > 0, got {proj_chunk_size}.")

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _o_proj_chunked(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.proj_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            out.copy_(self.o_proj(x))
            return out
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self.o_proj(x[start:end]))
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        cache_manager = context.cache_manager
        layer_idx = context.now_layer_idx
        # DeltaKV compressors are trained on raw K/V before QK norm and RoPE.
        pre_rope_k = k
        pre_rope_v = v
        cache_manager.save_raw_kv_if_needed(layer_idx, pre_rope_k, pre_rope_v)
        debug_layers = os.getenv("SPARSEVLLM_DEBUG_CAPTURE_PRE_ROPE_LAYERS")
        if debug_layers:
            wanted = {int(part) for part in debug_layers.split(",") if part.strip()}
            if int(layer_idx) in wanted:
                self.debug_last_pre_rope_positions = positions.detach().clone()
                self.debug_last_pre_rope_k = pre_rope_k.detach().clone()
                self.debug_last_pre_rope_v = pre_rope_v.detach().clone()
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        cache_manager.save_rope_kv_if_needed(layer_idx, k, v)
        debug_qk_layers = os.getenv("SPARSEVLLM_DEBUG_QK_LAYERS")
        debug_qk_capture = False
        if debug_qk_layers:
            wanted = {int(part) for part in debug_qk_layers.split(",") if part.strip()}
            debug_qk_capture = int(context.now_layer_idx) in wanted
            if debug_qk_capture:
                self.debug_last_k_raw = pre_rope_k.detach().clone()
                self.debug_last_q_postrope = q.detach().clone()
                self.debug_last_k_postrope = k.detach().clone()
                self.debug_last_v = v.detach().clone()
                self.debug_last_qk_positions = positions.detach().clone()
        o = self.attn(q, k, v)
        projected = self._o_proj_chunked(o.flatten(1, -1), hidden_states)
        if debug_qk_capture:
            self.debug_last_attn_output = o.detach().clone()
            self.debug_last_o_proj_output = projected.detach().clone()
        return projected


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mlp_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
        self.mlp_chunk_size = int(mlp_chunk_size)
        if self.mlp_chunk_size <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {mlp_chunk_size}.")

    def _forward_chunk(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

    def forward(self, x):
        chunk_size = int(self.mlp_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            return self._forward_chunk(x)

        out = torch.empty_like(x)
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self._forward_chunk(x[start:end]))
        return out


class Qwen3DecoderLayerBase(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            head_dim=config.head_dim,
            rope_theta=_get_rope_theta(config),
            rope_scaling=_get_rope_scaling(config),
            proj_chunk_size=getattr(config, "mlp_chunk_size", 16384),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            mlp_chunk_size=getattr(config, "mlp_chunk_size", 16384),
        )


class Qwen3ModelBase(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        layer_cls: type[nn.Module],
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([layer_cls(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 稀疏策略控制器，由 ModelRunner 动态注入
        self.sparse_controller = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        context = get_context()
        debug_layers_env = os.getenv("SPARSEVLLM_DEBUG_HIDDEN_LAYERS")
        debug_layers = None
        if debug_layers_env:
            debug_layers = {int(part) for part in debug_layers_env.split(",") if part.strip()}
            self.debug_last_hidden_states = {
                -1: hidden_states[-1:].detach().clone(),
            }
        
        for i, layer in enumerate(self.layers):
            context.now_layer_idx = i
            hidden_states, residual = layer(positions, hidden_states, residual)
            if self.sparse_controller is not None:
                hidden_states, residual = self.sparse_controller.apply_activation_hook(
                    i,
                    hidden_states,
                    residual,
                    context,
                )
            if debug_layers is not None and i in debug_layers:
                layer_output = hidden_states if residual is None else hidden_states + residual
                self.debug_last_hidden_states[int(i)] = layer_output[-1:].detach().clone()
            
            # 回调控制器执行稀疏逻辑
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(i, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_layers is not None:
            self.debug_last_hidden_states[self.config.num_hidden_layers] = hidden_states[-1:].detach().clone()
        return hidden_states


class Qwen3Model(Qwen3ModelBase):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config, Qwen3DecoderLayer)


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
