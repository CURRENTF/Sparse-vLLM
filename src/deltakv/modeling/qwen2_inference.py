from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
    rotate_half,
)

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
)
from deltakv.modeling.hf_common import build_inference_classes


(
    Qwen2AttnKVCompress,
    Qwen2LayerKVCompress,
    Qwen2ModelKVCompress,
    Qwen2KVCompress,
    _variant_class,
) = build_inference_classes(
    prefix="Qwen2",
    config_cls=KVQwen2Config,
    attention_cls=Qwen2Attention,
    layer_cls=Qwen2DecoderLayer,
    model_cls=Qwen2Model,
    lm_cls=Qwen2ForCausalLM,
    rotate_half=rotate_half,
    eager_attention_forward=eager_attention_forward,
    all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    create_causal_mask=create_causal_mask,
    create_sliding_window_causal_mask=create_sliding_window_causal_mask,
    use_qk_norm=False,
    pass_sliding_window=True,
)

Qwen2DeltaCompressedLatentWFull = _variant_class(
    "Qwen2DeltaCompressedLatentWFull",
    Qwen2KVCompress,
    DELTA_COMPRESSED_LATENT_W_FULL,
)
Qwen2DeltaOriginWoFull = _variant_class("Qwen2DeltaOriginWoFull", Qwen2KVCompress, DELTA_ORIGIN_WO_FULL)
Qwen2DeltaOriginWFull = _variant_class("Qwen2DeltaOriginWFull", Qwen2KVCompress, DELTA_ORIGIN_W_FULL)

__all__ = [
    "Qwen2AttnKVCompress",
    "Qwen2LayerKVCompress",
    "Qwen2ModelKVCompress",
    "Qwen2KVCompress",
    "Qwen2DeltaCompressedLatentWFull",
    "Qwen2DeltaOriginWoFull",
    "Qwen2DeltaOriginWFull",
]
