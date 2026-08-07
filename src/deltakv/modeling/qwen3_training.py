from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)

from deltakv.modeling.hf_common import build_cluster_training_classes


(
    Qwen3AttnKVClusterCompress,
    Qwen3LayerKVClusterCompress,
    Qwen3ModelKVClusterCompress,
    Qwen3KVClusterCompress,
) = build_cluster_training_classes(
    prefix="Qwen3",
    attention_cls=Qwen3Attention,
    layer_cls=Qwen3DecoderLayer,
    model_cls=Qwen3Model,
    lm_cls=Qwen3ForCausalLM,
    apply_rotary_pos_emb=apply_rotary_pos_emb,
    eager_attention_forward=eager_attention_forward,
    all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    create_causal_mask=create_causal_mask,
    create_sliding_window_causal_mask=create_sliding_window_causal_mask,
    use_qk_norm=True,
    pass_sliding_window=True,
)

__all__ = [
    "Qwen3AttnKVClusterCompress",
    "Qwen3LayerKVClusterCompress",
    "Qwen3ModelKVClusterCompress",
    "Qwen3KVClusterCompress",
]
