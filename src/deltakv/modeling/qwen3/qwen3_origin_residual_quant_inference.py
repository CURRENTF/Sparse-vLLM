from deltakv.configs.model_config_cls import KVQwen3Config
from deltakv.modeling.cache_factory import ORIGIN_RESIDUAL_QUANT_CACHE, set_deltakv_cache_impl
from deltakv.modeling.qwen3.qwen3_with_compress_inference import Qwen3KVCompress


class Qwen3OriginResidualQuant(Qwen3KVCompress):
    """Compatibility wrapper for the unified Qwen3 DeltaKV HF model."""

    def __init__(self, config: KVQwen3Config):
        set_deltakv_cache_impl(config, ORIGIN_RESIDUAL_QUANT_CACHE)
        super().__init__(config)
