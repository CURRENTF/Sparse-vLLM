import unittest

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.cache_factory import (
    ALL_ORIGIN_RESIDUAL_QUANT_CACHE,
    ORIGIN_RESIDUAL_QUANT_CACHE,
    STANDARD_CACHE,
    create_deltakv_cache,
    get_deltakv_cache_impl,
    set_deltakv_cache_impl,
)
from deltakv.modeling.kv_cache import ClusterCompressedKVCache, CompressedKVCache
from deltakv.modeling.origin_residual_quant_cache import (
    OriginResidualQuantClusterCompressedKVCache,
    OriginResidualQuantCompressedKVCache,
)
from deltakv.modeling.all_origin_residual_quant_cache import AllOriginResidualQuantClusterCompressedKVCache
from deltakv.modeling.llama.llama_all_origin_residual_quant_inference import LlamaAllOriginResidualQuant
from deltakv.modeling.llama.llama_origin_residual_quant_inference import LlamaOriginResidualQuant
from deltakv.modeling.qwen2.qwen2_all_origin_residual_quant_inference import Qwen2AllOriginResidualQuant
from deltakv.modeling.qwen2.qwen2_origin_residual_quant_inference import Qwen2OriginResidualQuant
from deltakv.modeling.qwen3.qwen3_all_origin_residual_quant_inference import Qwen3AllOriginResidualQuant
from deltakv.modeling.qwen3.qwen3_origin_residual_quant_inference import Qwen3OriginResidualQuant


class HfDeltaKVCacheFactoryTest(unittest.TestCase):
    def test_standard_cache_impl_matches_use_cluster(self):
        clustered = KVQwen2Config(use_cluster=True)
        self.assertEqual(get_deltakv_cache_impl(clustered), STANDARD_CACHE)
        self.assertIsInstance(create_deltakv_cache(clustered), ClusterCompressedKVCache)

        non_clustered = KVQwen2Config(use_cluster=False)
        self.assertIsInstance(create_deltakv_cache(non_clustered), CompressedKVCache)

    def test_origin_residual_quant_cache_impl_matches_use_cluster(self):
        clustered = KVQwen2Config(use_cluster=True)
        set_deltakv_cache_impl(clustered, ORIGIN_RESIDUAL_QUANT_CACHE)
        self.assertIsInstance(create_deltakv_cache(clustered), OriginResidualQuantClusterCompressedKVCache)

        non_clustered = KVQwen2Config(use_cluster=False)
        set_deltakv_cache_impl(non_clustered, ORIGIN_RESIDUAL_QUANT_CACHE)
        self.assertIsInstance(create_deltakv_cache(non_clustered), OriginResidualQuantCompressedKVCache)

    def test_all_origin_residual_quant_requires_cluster(self):
        cfg = KVQwen2Config(use_cluster=True)
        set_deltakv_cache_impl(cfg, ALL_ORIGIN_RESIDUAL_QUANT_CACHE)
        self.assertIsInstance(create_deltakv_cache(cfg), AllOriginResidualQuantClusterCompressedKVCache)

        cfg = KVQwen2Config(use_cluster=False)
        set_deltakv_cache_impl(cfg, ALL_ORIGIN_RESIDUAL_QUANT_CACHE)
        with self.assertRaisesRegex(ValueError, "requires use_cluster"):
            create_deltakv_cache(cfg)

    def test_origin_model_wrappers_reuse_base_forward(self):
        wrappers = [
            Qwen2OriginResidualQuant,
            Qwen2AllOriginResidualQuant,
            Qwen3OriginResidualQuant,
            Qwen3AllOriginResidualQuant,
            LlamaOriginResidualQuant,
            LlamaAllOriginResidualQuant,
        ]
        for wrapper_cls in wrappers:
            with self.subTest(wrapper_cls=wrapper_cls.__name__):
                self.assertNotIn("forward", wrapper_cls.__dict__)


if __name__ == "__main__":
    unittest.main()
