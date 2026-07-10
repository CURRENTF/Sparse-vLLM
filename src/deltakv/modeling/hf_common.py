from __future__ import annotations

import inspect
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from deltakv.configs.model_config_cls import parse_full_attn_layers
from deltakv.modeling.cache_factory import create_hf_sparse_cache, is_hf_sparse_cache_instance, set_deltakv_cache_impl
from deltakv.modeling.compressor import create_compressor, reshape_and_apply_qk_norm
from deltakv.modeling.token_select import omnikv_token_selection


_MASK_ARGUMENTS = {
    "config",
    "inputs_embeds",
    "input_embeds",
    "attention_mask",
    "cache_position",
    "past_key_values",
    "position_ids",
}


def _mask_signature(mask_factory) -> tuple[str, frozenset[str]]:
    parameters = inspect.signature(mask_factory).parameters
    embed_arg = None
    for name in ("inputs_embeds", "input_embeds"):
        if name in parameters:
            embed_arg = name
            break
    if embed_arg is None:
        raise TypeError(
            f"{mask_factory.__name__} must accept either 'inputs_embeds' or 'input_embeds'."
        )
    unsupported_required = [
        name
        for name, parameter in parameters.items()
        if name not in _MASK_ARGUMENTS
        and parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if unsupported_required:
        raise TypeError(
            f"{mask_factory.__name__} has unsupported required arguments: {unsupported_required}."
        )
    return embed_arg, frozenset(parameters)


def _call_mask_factory(mask_factory, signature, inputs_embeds, mask_kwargs):
    embed_arg, accepted_args = signature
    kwargs = {name: value for name, value in mask_kwargs.items() if name in accepted_args}
    kwargs[embed_arg] = inputs_embeds
    return mask_factory(**kwargs)


def assert_hf_bs1(input_shape: tuple[int, int], attention_mask: Optional[torch.Tensor]) -> None:
    if input_shape[0] != 1:
        raise NotImplementedError("HF DeltaKV supports batch_size=1 only; use Sparse-vLLM for batched inference.")
    if attention_mask is not None and bool((attention_mask == 0).any()):
        raise NotImplementedError("HF DeltaKV does not support padded inputs; use Sparse-vLLM for batched/padded inference.")


def _causal_mask_from_positions(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if query_positions.dim() == 1:
        query_positions = query_positions.unsqueeze(0)
    if key_positions.dim() == 1:
        key_positions = key_positions.unsqueeze(0)
    blocked = key_positions[:, None, None, :] > query_positions[:, None, :, None]
    mask = torch.zeros(blocked.shape, device=key_positions.device, dtype=dtype)
    return mask.masked_fill(blocked, torch.finfo(dtype).min)


def apply_single_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotate_half) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (states * cos) + (rotate_half(states) * sin)


def _top_tokens(config, obs_index: Optional[int], is_prefill: bool):
    num_top_tokens = config.num_top_tokens_in_prefill if is_prefill else config.num_top_tokens
    if isinstance(num_top_tokens, (list, tuple)):
        return num_top_tokens[obs_index]
    if isinstance(num_top_tokens, str) and "," in num_top_tokens:
        return [float(x.strip()) for x in num_top_tokens.split(",")][obs_index]
    return num_top_tokens


def _variant_class(name: str, base_cls: type, cache_impl: str):
    class Variant(base_cls):
        def __init__(self, config):
            set_deltakv_cache_impl(config, cache_impl)
            super().__init__(config)

    Variant.__name__ = name
    Variant.__qualname__ = name
    return Variant


def _decoder_hidden_states(layer_outputs):
    if isinstance(layer_outputs, tuple):
        return layer_outputs[0]
    return layer_outputs


def build_inference_classes(
    *,
    prefix: str,
    config_cls: type,
    attention_cls: type,
    layer_cls: type,
    model_cls: type,
    lm_cls: type,
    rotate_half,
    eager_attention_forward,
    all_attention_functions,
    create_causal_mask,
    create_sliding_window_causal_mask=None,
    use_qk_norm: bool = False,
    pass_sliding_window: bool = True,
):
    causal_mask_signature = _mask_signature(create_causal_mask)
    sliding_mask_signature = (
        _mask_signature(create_sliding_window_causal_mask)
        if create_sliding_window_causal_mask is not None
        else None
    )

    class AttnKVCompress(attention_cls):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            full_layers = parse_full_attn_layers(config.full_attn_layers)
            config.full_attn_layers = full_layers
            self.is_full_layer = layer_idx in full_layers
            self.is_obs_layer = bool(full_layers) and layer_idx in full_layers and (layer_idx + 1) not in full_layers
            self.obs_index = (
                sorted(idx for idx in full_layers if (idx + 1) not in full_layers).index(layer_idx)
                if self.is_obs_layer
                else None
            )
            self.compress_down = create_compressor(is_down=True, config=config)
            self.compress_up = create_compressor(is_down=False, config=config)
            self.config = config
            self.layer_idx = layer_idx

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value=None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            visual_mask = kwargs.pop("deltakv_visual_token_mask", None)
            input_shape = hidden_states.shape[:-1]
            bs, q_len, _ = hidden_states.shape
            assert_hf_bs1((bs, q_len), None)

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            if not self.config.collect_kv_before_rope:
                raise NotImplementedError("DeltaKV HF inference expects collect_kv_before_rope=True.")

            hidden_shape = (bs, q_len, -1, self.head_dim)
            cur_cos, cur_sin = position_embeddings
            use_full_layer_kivi_postrope_cache = bool(
                self.is_full_layer
                and past_key_value is not None
                and hasattr(past_key_value, "_full_layer_kivi_enabled")
                and past_key_value._full_layer_kivi_enabled()
            )
            debug_qk_layers = os.getenv("DELTAKV_DEBUG_QK_LAYERS")
            debug_qk_capture = False
            if debug_qk_layers:
                wanted = {int(part) for part in debug_qk_layers.split(",") if part.strip()}
                debug_qk_capture = int(self.layer_idx) in wanted
            if use_full_layer_kivi_postrope_cache:
                if debug_qk_capture:
                    self.debug_last_k_raw = (
                        key_states.view(bs, q_len, self.config.num_key_value_heads, self.head_dim)
                        .transpose(1, 2)
                        .detach()
                        .clone()
                    )
                if use_qk_norm:
                    query_states, key_states = reshape_and_apply_qk_norm(
                        self,
                        query_states,
                        key_states,
                        hidden_shape,
                        (bs, q_len, self.config.num_key_value_heads, self.head_dim),
                    )
                else:
                    query_states = query_states.view(
                        bs,
                        q_len,
                        self.config.num_attention_heads,
                        self.head_dim,
                    ).transpose(1, 2)
                    key_states = key_states.view(
                        bs,
                        q_len,
                        self.config.num_key_value_heads,
                        self.head_dim,
                    ).transpose(1, 2)
                if debug_qk_capture:
                    self.debug_last_k_norm = key_states.detach().clone()
                query_states = apply_single_rope(query_states, cur_cos, cur_sin, rotate_half)
                key_states = apply_single_rope(key_states, cur_cos, cur_sin, rotate_half)
                if debug_qk_capture:
                    self.debug_last_k_full_kivi_postrope_input = key_states.detach().clone()
                key_states = key_states.transpose(1, 2).reshape(bs, q_len, -1).contiguous()

            key_states, value_states, full_idx = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position, "deltakv_visual_token_mask": visual_mask},
                compressor_down=self.compress_down,
                compressor_up=self.compress_up,
            )

            if debug_qk_capture and not use_full_layer_kivi_postrope_cache:
                self.debug_last_k_raw = (
                    key_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim)
                    .transpose(1, 2)
                    .detach()
                    .clone()
                )

            if use_full_layer_kivi_postrope_cache:
                key_states = key_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            else:
                if use_qk_norm:
                    query_states, key_states = reshape_and_apply_qk_norm(
                        self,
                        query_states,
                        key_states,
                        hidden_shape,
                        (bs, -1, self.config.num_key_value_heads, self.head_dim),
                    )
                else:
                    query_states = query_states.view(bs, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            if debug_qk_capture and not use_full_layer_kivi_postrope_cache:
                self.debug_last_k_norm = key_states.detach().clone()

            if not use_full_layer_kivi_postrope_cache:
                query_states = apply_single_rope(query_states, cur_cos, cur_sin, rotate_half)
                safe_full_idx = full_idx.clamp(min=0, max=past_key_value.cos.shape[1] - 1)
                k_cos = past_key_value.cos.gather(1, safe_full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
                k_sin = past_key_value.sin.gather(1, safe_full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
                key_states = apply_single_rope(key_states, k_cos, k_sin, rotate_half)

            if debug_qk_capture:
                self.debug_last_q_postrope = query_states.detach().clone()
                self.debug_last_k_postrope = key_states.detach().clone()
                self.debug_last_v = value_states.detach().clone()
                self.debug_last_qk_positions = full_idx.detach().clone()

            if attention_mask is not None:
                attention_mask = _causal_mask_from_positions(
                    cache_position,
                    full_idx,
                    dtype=query_states.dtype,
                )

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[self.config._attn_implementation]
            extra = dict(kwargs)
            if pass_sliding_window and hasattr(self, "sliding_window"):
                extra["sliding_window"] = self.sliding_window
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **extra,
            )

            if self.is_obs_layer:
                compressed_len = past_key_value.get_observable_compressed_length(q_len)
            else:
                compressed_len = past_key_value.get_compressed_length(self.layer_idx)
            do_obs = (
                bool(self.config.deltakv_use_omnikv_selection)
                and self.is_obs_layer
                and compressed_len > 0
                and (self.config.chunk_prefill_accel_omnikv or q_len == 1)
            )
            if do_obs:
                candidate_key = key_states[:, :, self.config.num_sink_tokens : self.config.num_sink_tokens + compressed_len, :]
                top_token_idx, token_scores = omnikv_token_selection(
                    self,
                    query_states,
                    candidate_key,
                    self.scaling,
                    _top_tokens(self.config, self.obs_index, q_len > 1),
                    pool_kernel_size=self.config.pool_kernel_size,
                    last_token_scores=past_key_value.token_scores.get(self.layer_idx),
                    score_method=self.config.omnikv_score_method,
                )
                past_key_value.token_scores[self.layer_idx] = token_scores
                past_key_value.top_token_idx[self.layer_idx] = top_token_idx
            if os.getenv("DEBUG") and self.layer_idx == 0:
                print(f"[DeltaKV HF] key_states={tuple(key_states.shape)} do_obs={do_obs} q_len={q_len}", flush=True)
            projected = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
            if debug_qk_capture:
                self.debug_last_attn_output = attn_output.detach().clone()
                self.debug_last_o_proj_output = projected.detach().clone()
            return projected, attn_weights

    class LayerKVCompress(layer_cls):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.self_attn = AttnKVCompress(config=config, layer_idx=layer_idx)

    class ModelKVCompress(model_cls):
        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.ModuleList([LayerKVCompress(config, i) for i in range(config.num_hidden_layers)])
            self.config = config
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            deltakv_visual_token_mask: Optional[torch.Tensor] = None,
            **flash_attn_kwargs,
        ):
            return_full_hidden = bool(flash_attn_kwargs.pop("deltakv_return_full_hidden", False))
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                assert_hf_bs1(tuple(input_ids.shape), attention_mask)
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                assert_hf_bs1(tuple(inputs_embeds.shape[:2]), attention_mask)
            if self.gradient_checkpointing and self.training and use_cache:
                raise RuntimeError("HF DeltaKV inference model should not be used for training with gradient checkpointing.")
            if not is_hf_sparse_cache_instance(past_key_values, self.config):
                raise TypeError(f"{prefix}ModelKVCompress expects an HF sparse cache created by create_hf_sparse_cache().")
            if cache_position is None:
                past_seen = past_key_values.get_seq_length()
                cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            mask_kwargs = {
                "config": self.config,
                "attention_mask": None,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": _call_mask_factory(
                    create_causal_mask, causal_mask_signature, inputs_embeds, mask_kwargs
                )
            }
            if create_sliding_window_causal_mask is not None and getattr(self, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = _call_mask_factory(
                    create_sliding_window_causal_mask,
                    sliding_mask_signature,
                    inputs_embeds,
                    mask_kwargs,
                )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            cos, sin = position_embeddings
            past_key_values.cos = cos if past_key_values.cos is None else torch.cat([past_key_values.cos, cos], dim=1)
            past_key_values.sin = sin if past_key_values.sin is None else torch.cat([past_key_values.sin, sin], dim=1)

            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                attention_type = getattr(decoder_layer, "attention_type", "full_attention")
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    deltakv_visual_token_mask=deltakv_visual_token_mask,
                    **flash_attn_kwargs,
                )
                hidden_states = _decoder_hidden_states(layer_outputs)
                if output_attentions:
                    if not isinstance(layer_outputs, tuple):
                        raise ValueError("output_attentions=True requires decoder layers to return attention weights.")
                    all_self_attns += (layer_outputs[1],)
            hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states if return_full_hidden else hidden_states[:, -1:],
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

    class KVCompress(lm_cls):
        def __init__(self, config):
            super().__init__(config)
            self.model = ModelKVCompress(config)
            self.config = config
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep=0,
            **kwargs,
        ):
            del inputs_embeds, labels, position_ids, cache_position
            if input_ids is None:
                raise ValueError(f"{prefix}KVCompress expects input_ids.")
            assert_hf_bs1(tuple(input_ids.shape), attention_mask)
            use_cache = True if use_cache is None else use_cache
            if not use_cache:
                raise ValueError("DeltaKV inference model must use cache.")
            if not is_hf_sparse_cache_instance(past_key_values, self.config):
                past_key_values = create_hf_sparse_cache(self.config)
            outputs = None
            for chunk_ids in input_ids.split(max(1, int(self.config.chunk_prefill_size)), dim=-1):
                outputs = lm_cls.forward(
                    self,
                    chunk_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )
                past_key_values = outputs.past_key_values
            return outputs

    for cls, suffix in [
        (AttnKVCompress, "AttnKVCompress"),
        (LayerKVCompress, "LayerKVCompress"),
        (ModelKVCompress, "ModelKVCompress"),
        (KVCompress, "KVCompress"),
    ]:
        cls.__name__ = f"{prefix}{suffix}"
        cls.__qualname__ = f"{prefix}{suffix}"

    return AttnKVCompress, LayerKVCompress, ModelKVCompress, KVCompress, _variant_class


def build_cluster_training_classes(
    *,
    prefix: str,
    attention_cls: type,
    layer_cls: type,
    model_cls: type,
    lm_cls: type,
    apply_rotary_pos_emb,
    eager_attention_forward,
    all_attention_functions,
    create_causal_mask,
    create_sliding_window_causal_mask=None,
    use_qk_norm: bool = False,
    pass_sliding_window: bool = True,
):
    causal_mask_signature = _mask_signature(create_causal_mask)
    sliding_mask_signature = (
        _mask_signature(create_sliding_window_causal_mask)
        if create_sliding_window_causal_mask is not None
        else None
    )
    run_mode = {"value": "comp"}

    class AttnKVClusterCompress(attention_cls):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            if getattr(config, "split_kv", False):
                self.k_compress_down = create_compressor(is_down=True, config=config)
                self.k_compress_up = create_compressor(is_down=False, config=config)
                self.v_compress_down = create_compressor(is_down=True, config=config)
                self.v_compress_up = create_compressor(is_down=False, config=config)
            else:
                self.compress_down = create_compressor(is_down=True, config=config)
                self.compress_up = create_compressor(is_down=False, config=config)
            self.cluster_metric = config.cluster_metric
            self.cluster_on_kv = config.cluster_on_kv
            self.buffer_raw_kv = None
            self.buffer_recon_kv = None
            self.buffer_comp_kv = None
            self.buffer_ideal_res = None

        def _scores(self, a, b):
            if self.cluster_metric == "l2":
                return -torch.cdist(a, b)
            if self.cluster_metric == "dot":
                return torch.matmul(a, b.transpose(-1, -2))
            if self.cluster_metric == "cosine":
                return torch.matmul(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).transpose(-1, -2))
            raise ValueError(f"Unknown cluster_metric: {self.cluster_metric}")

        def _refs(self, scores, prototypes):
            bs, seq_len, _ = scores.shape
            token_dim = prototypes.shape[-1]
            k = min(self.config.get_cluster_neighbor_count(), prototypes.shape[1])
            topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)
            indices = topk_indices.reshape(bs, -1)[:, :, None].expand(-1, -1, token_dim)
            gathered = prototypes.gather(dim=1, index=indices).view(bs, seq_len, k, token_dim)
            if self.config.cluster_soft_assignment:
                weights = F.softmax(topk_scores / self.config.cluster_temp, dim=-1).to(prototypes.dtype)
                return (gathered * weights.unsqueeze(-1)).sum(dim=2)
            return gathered.mean(dim=2)

        def _plan(self, states, kv_flat):
            sink_size = 16
            if states.shape[1] <= sink_size:
                raise ValueError(f"cluster_e2e_big training requires seq_len > {sink_size}; got {states.shape[1]}.")
            step = max(1, int(1 / self.config.cluster_ratio))
            sink = torch.arange(0, sink_size, device=states.device)
            rem = torch.arange(sink_size, states.shape[1], step, device=states.device)
            centers = torch.cat([sink, rem])
            rows = torch.arange(states.shape[1] - sink_size, device=states.device).view(-1, 1) + sink_size
            mask = centers.view(1, -1) <= rows
            return sink_size, centers, mask

        def comp_then_reconstruct(self, key_states, value_states):
            bs, seq_len, k_dim = key_states.shape
            kv_flat = torch.cat([key_states, value_states], dim=-1)
            sink_size, centers, mask = self._plan(key_states, kv_flat)
            if not getattr(self.config, "split_kv", False):
                kv_sink = kv_flat[:, :sink_size]
                kv_rem = kv_flat[:, sink_size:]
                feat = kv_flat if self.cluster_on_kv else key_states
                scores = self._scores(feat[:, sink_size:], feat[:, centers]).masked_fill(~mask.unsqueeze(0), float("-inf"))
                refs = self._refs(scores, kv_flat[:, centers])
                comp = self.compress_down(kv_rem) - self.compress_down(refs)
                self.buffer_comp_kv = comp
                self.buffer_ideal_res = kv_rem - refs
                recon_rem = (self.compress_up(comp) + refs).to(kv_sink.dtype)
                recon = torch.cat([kv_sink, recon_rem], dim=1)
                self.buffer_recon_kv = recon
                return (*torch.split(recon, k_dim, dim=-1), F.mse_loss(recon, self.buffer_raw_kv))

            k_sink, v_sink = key_states[:, :sink_size], value_states[:, :sink_size]
            k_rem, v_rem = key_states[:, sink_size:], value_states[:, sink_size:]
            k_scores = self._scores(k_rem, key_states[:, centers]).masked_fill(~mask.unsqueeze(0), float("-inf"))
            v_feat = value_states if self.cluster_on_kv else key_states
            v_centers = value_states[:, centers] if self.cluster_on_kv else key_states[:, centers]
            v_scores = self._scores(v_feat[:, sink_size:], v_centers).masked_fill(~mask.unsqueeze(0), float("-inf"))
            ref_k = self._refs(k_scores, key_states[:, centers])
            ref_v = self._refs(v_scores, value_states[:, centers])
            comp_k = self.k_compress_down(k_rem) - self.k_compress_down(ref_k)
            comp_v = self.v_compress_down(v_rem) - self.v_compress_down(ref_v)
            self.buffer_comp_kv = torch.cat([comp_k, comp_v], dim=-1)
            self.buffer_ideal_res = torch.cat([k_rem - ref_k, v_rem - ref_v], dim=-1)
            recon_k = torch.cat([k_sink, (self.k_compress_up(comp_k) + ref_k).to(k_sink.dtype)], dim=1)
            recon_v = torch.cat([v_sink, (self.v_compress_up(comp_v) + ref_v).to(v_sink.dtype)], dim=1)
            recon = torch.cat([recon_k, recon_v], dim=-1)
            self.buffer_recon_kv = recon
            return recon_k, recon_v, F.mse_loss(recon, self.buffer_raw_kv)

        def _shape_qkv(self, query_states, key_states, value_states, hidden_shape):
            if use_qk_norm:
                query_states, key_states = reshape_and_apply_qk_norm(self, query_states, key_states, hidden_shape)
            else:
                query_states = query_states.view(hidden_shape).transpose(1, 2)
                key_states = key_states.view(hidden_shape).transpose(1, 2)
            value_states = value_states.view(hidden_shape).transpose(1, 2)
            return query_states, key_states, value_states

        def raw_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            self.buffer_raw_kv = torch.cat([key_states, value_states], dim=-1)
            query_states, key_states, value_states = self._shape_qkv(query_states, key_states, value_states, hidden_shape)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            return self._attention_out(input_shape, query_states, key_states, value_states, attention_mask, kwargs, hidden_states.device, raw=True)

        def _attention_out(self, input_shape, query_states, key_states, value_states, attention_mask, kwargs, device, raw=False):
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[self.config._attn_implementation]
            extra = dict(kwargs)
            if pass_sliding_window and hasattr(self, "sliding_window"):
                extra["sliding_window"] = self.sliding_window
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **extra,
            )
            mse = torch.tensor(0.0, device=device) if raw else None
            return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), attn_weights, mse

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
            if run_mode["value"] == "raw":
                return self.raw_forward(hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            if not self.config.collect_kv_before_rope:
                raise NotImplementedError("cluster_e2e_big requires collect_kv_before_rope=True.")
            key_states, value_states, mse = self.comp_then_reconstruct(key_states, value_states)
            query_states, key_states, value_states = self._shape_qkv(query_states, key_states, value_states, hidden_shape)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            out, weights, _ = self._attention_out(input_shape, query_states, key_states, value_states, attention_mask, kwargs, hidden_states.device)
            return out, weights, mse

    class LayerKVClusterCompress(layer_cls):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.self_attn = AttnKVClusterCompress(config=config, layer_idx=layer_idx)

        def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, mse_loss = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)
            return (hidden_states, self_attn_weights, mse_loss) if output_attentions else (hidden_states, mse_loss)

    class ModelKVClusterCompress(model_cls):
        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.ModuleList([LayerKVClusterCompress(config, i) for i in range(config.num_hidden_layers)])
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, cache_position=None, **flash_attn_kwargs):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if self.gradient_checkpointing and self.training and use_cache:
                use_cache = False
            if not isinstance(past_key_values, (type(None), DynamicCache)):
                raise ValueError("cluster_e2e_big training expects a DynamicCache or None.")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()
            if cache_position is None:
                past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            masks = {
                "full_attention": _call_mask_factory(
                    create_causal_mask, causal_mask_signature, inputs_embeds, mask_kwargs
                )
            }
            if create_sliding_window_causal_mask is not None and getattr(self, "has_sliding_layers", False):
                masks["sliding_attention"] = _call_mask_factory(
                    create_sliding_window_causal_mask,
                    sliding_mask_signature,
                    inputs_embeds,
                    mask_kwargs,
                )
            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            mse_loss = torch.tensor(0.0, device=inputs_embeds.device)
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                attention_type = getattr(decoder_layer, "attention_type", "full_attention")
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=masks[attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
                hidden_states = layer_outputs[0]
                mse_loss = mse_loss + layer_outputs[-1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
            hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ), mse_loss

    class KVClusterCompress(lm_cls):
        def __init__(self, config):
            super().__init__(config)
            self.model = ModelKVClusterCompress(config)
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, cache_position=None, logits_to_keep=0, **kwargs):
            run_mode["value"] = "raw"
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    **kwargs,
                )
            run_mode["value"] = "comp"
            outputs, mse_loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            self._last_ntp_loss = loss.detach() if loss is not None else None
            self._last_mse_loss = mse_loss.detach()
            total_loss = (loss + mse_loss) if loss is not None else mse_loss
            if loss is not None and os.getenv("MSE_DETACH"):
                total_loss = loss + mse_loss.detach()
            elif loss is not None and os.getenv("NTP_DETACH"):
                total_loss = loss.detach() + mse_loss
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    for cls, suffix in [
        (AttnKVClusterCompress, "AttnKVClusterCompress"),
        (LayerKVClusterCompress, "LayerKVClusterCompress"),
        (ModelKVClusterCompress, "ModelKVClusterCompress"),
        (KVClusterCompress, "KVClusterCompress"),
    ]:
        cls.__name__ = f"{prefix}{suffix}"
        cls.__qualname__ = f"{prefix}{suffix}"

    AttnKVClusterCompress._deltakv_run_mode = run_mode
    ModelKVClusterCompress._deltakv_run_mode = run_mode
    KVClusterCompress._deltakv_run_mode = run_mode

    return AttnKVClusterCompress, LayerKVClusterCompress, ModelKVClusterCompress, KVClusterCompress
