from deltakv.configs.model_config_cls import KVLlamaConfig
from deltakv.modeling.cache_factory import ALL_ORIGIN_RESIDUAL_QUANT_CACHE, set_deltakv_cache_impl
from deltakv.modeling.llama.llama_with_compress_inference import LlamaKVCompress


class LlamaAllOriginResidualQuant(LlamaKVCompress):
    """Compatibility wrapper for the unified Llama DeltaKV HF model."""

    def __init__(self, config: KVLlamaConfig):
        set_deltakv_cache_impl(config, ALL_ORIGIN_RESIDUAL_QUANT_CACHE)
        assert config.use_cluster, "AllOriginResidualQuant only supports use_cluster=True"
        super().__init__(config)
