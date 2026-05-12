from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
    rotate_half,
)

from deltakv.configs.model_config_cls import KVQwen3Config
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
)
from deltakv.modeling.hf_common import build_inference_classes


(
    Qwen3AttnKVCompress,
    Qwen3LayerKVCompress,
    Qwen3ModelKVCompress,
    Qwen3KVCompress,
    _variant_class,
) = build_inference_classes(
    prefix="Qwen3",
    config_cls=KVQwen3Config,
    attention_cls=Qwen3Attention,
    layer_cls=Qwen3DecoderLayer,
    model_cls=Qwen3Model,
    lm_cls=Qwen3ForCausalLM,
    rotate_half=rotate_half,
    eager_attention_forward=eager_attention_forward,
    all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    create_causal_mask=create_causal_mask,
    create_sliding_window_causal_mask=create_sliding_window_causal_mask,
    use_qk_norm=True,
    pass_sliding_window=True,
)

Qwen3DeltaCompressedLatentWFull = _variant_class(
    "Qwen3DeltaCompressedLatentWFull",
    Qwen3KVCompress,
    DELTA_COMPRESSED_LATENT_W_FULL,
)
Qwen3DeltaOriginWoFull = _variant_class("Qwen3DeltaOriginWoFull", Qwen3KVCompress, DELTA_ORIGIN_WO_FULL)
Qwen3DeltaOriginWFull = _variant_class("Qwen3DeltaOriginWFull", Qwen3KVCompress, DELTA_ORIGIN_W_FULL)

__all__ = [
    "Qwen3AttnKVCompress",
    "Qwen3LayerKVCompress",
    "Qwen3ModelKVCompress",
    "Qwen3KVCompress",
    "Qwen3DeltaCompressedLatentWFull",
    "Qwen3DeltaOriginWoFull",
    "Qwen3DeltaOriginWFull",
]
