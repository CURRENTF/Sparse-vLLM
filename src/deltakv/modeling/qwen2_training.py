from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)

from deltakv.modeling.hf_common import build_cluster_training_classes


(
    Qwen2AttnKVClusterCompress,
    Qwen2LayerKVClusterCompress,
    Qwen2ModelKVClusterCompress,
    Qwen2KVClusterCompress,
) = build_cluster_training_classes(
    prefix="Qwen2",
    attention_cls=Qwen2Attention,
    layer_cls=Qwen2DecoderLayer,
    model_cls=Qwen2Model,
    lm_cls=Qwen2ForCausalLM,
    apply_rotary_pos_emb=apply_rotary_pos_emb,
    eager_attention_forward=eager_attention_forward,
    all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    create_causal_mask=create_causal_mask,
    create_sliding_window_causal_mask=create_sliding_window_causal_mask,
    use_qk_norm=False,
    pass_sliding_window=True,
)

__all__ = [
    "Qwen2AttnKVClusterCompress",
    "Qwen2LayerKVClusterCompress",
    "Qwen2ModelKVClusterCompress",
    "Qwen2KVClusterCompress",
]
