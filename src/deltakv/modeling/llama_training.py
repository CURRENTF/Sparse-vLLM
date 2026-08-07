from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    create_causal_mask,
    eager_attention_forward,
)

from deltakv.modeling.hf_common import build_cluster_training_classes


(
    LlamaAttnKVClusterCompress,
    LlamaLayerKVClusterCompress,
    LlamaModelKVClusterCompress,
    LlamaKVClusterCompress,
) = build_cluster_training_classes(
    prefix="Llama",
    attention_cls=LlamaAttention,
    layer_cls=LlamaDecoderLayer,
    model_cls=LlamaModel,
    lm_cls=LlamaForCausalLM,
    apply_rotary_pos_emb=apply_rotary_pos_emb,
    eager_attention_forward=eager_attention_forward,
    all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    create_causal_mask=create_causal_mask,
    use_qk_norm=False,
    pass_sliding_window=False,
)

KVModelCompress = LlamaKVClusterCompress

__all__ = [
    "LlamaAttnKVClusterCompress",
    "LlamaLayerKVClusterCompress",
    "LlamaModelKVClusterCompress",
    "LlamaKVClusterCompress",
    "KVModelCompress",
]
