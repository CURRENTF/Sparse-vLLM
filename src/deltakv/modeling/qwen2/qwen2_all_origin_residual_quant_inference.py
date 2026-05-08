from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.cache_factory import ALL_ORIGIN_RESIDUAL_QUANT_CACHE, set_deltakv_cache_impl
from deltakv.modeling.qwen2.qwen2_with_compress_inference import Qwen2KVCompress


class Qwen2AllOriginResidualQuant(Qwen2KVCompress):
    """Compatibility wrapper for the unified Qwen2 DeltaKV HF model."""

    def __init__(self, config: KVQwen2Config):
        set_deltakv_cache_impl(config, ALL_ORIGIN_RESIDUAL_QUANT_CACHE)
        assert config.use_cluster, "AllOriginResidualQuant only supports use_cluster=True"
        super().__init__(config)
