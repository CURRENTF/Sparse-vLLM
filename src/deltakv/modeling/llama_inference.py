from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    create_causal_mask,
    eager_attention_forward,
    rotate_half,
)

from deltakv.configs.model_config_cls import KVLlamaConfig
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
)
from deltakv.modeling.hf_common import build_inference_classes


(
    LlamaAttnKVCompress,
    LlamaLayerKVCompress,
    LlamaModelKVCompress,
    LlamaKVCompress,
    _variant_class,
) = build_inference_classes(
    prefix="Llama",
    config_cls=KVLlamaConfig,
    attention_cls=LlamaAttention,
    layer_cls=LlamaDecoderLayer,
    model_cls=LlamaModel,
    lm_cls=LlamaForCausalLM,
    rotate_half=rotate_half,
    eager_attention_forward=eager_attention_forward,
    all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    create_causal_mask=create_causal_mask,
    use_qk_norm=False,
    pass_sliding_window=False,
)

LlamaDeltaCompressedLatentWFull = _variant_class(
    "LlamaDeltaCompressedLatentWFull",
    LlamaKVCompress,
    DELTA_COMPRESSED_LATENT_W_FULL,
)
LlamaDeltaOriginWoFull = _variant_class("LlamaDeltaOriginWoFull", LlamaKVCompress, DELTA_ORIGIN_WO_FULL)
LlamaDeltaOriginWFull = _variant_class("LlamaDeltaOriginWFull", LlamaKVCompress, DELTA_ORIGIN_W_FULL)

__all__ = [
    "LlamaAttnKVCompress",
    "LlamaLayerKVCompress",
    "LlamaModelKVCompress",
    "LlamaKVCompress",
    "LlamaDeltaCompressedLatentWFull",
    "LlamaDeltaOriginWoFull",
    "LlamaDeltaOriginWFull",
]
