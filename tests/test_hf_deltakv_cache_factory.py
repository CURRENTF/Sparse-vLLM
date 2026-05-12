import unittest

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_COMPRESSED_LATENT_WO_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
    create_deltakv_cache,
    get_deltakv_cache_impl,
    set_deltakv_cache_impl,
)
from deltakv.modeling.cache_pipeline import (
    DeltaCompressedLatentWFullCache,
    DeltaCompressedLatentWoFullCache,
    DeltaOriginWFullCache,
    DeltaOriginWoFullCache,
)


class HfDeltaKVCacheFactoryTest(unittest.TestCase):
    def _config(self):
        return KVQwen2Config(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=16,
            intermediate_size=32,
            vocab_size=64,
            use_cluster=True,
            use_compression=False,
            full_attn_layers="0",
        )

    def test_default_cache_impl_is_delta_compressed_latent_without_full(self):
        cfg = self._config()
        self.assertEqual(get_deltakv_cache_impl(cfg), DELTA_COMPRESSED_LATENT_WO_FULL)
        self.assertIsInstance(create_deltakv_cache(cfg), DeltaCompressedLatentWoFullCache)

    def test_all_current_cache_impls_construct_expected_classes(self):
        cases = {
            DELTA_COMPRESSED_LATENT_WO_FULL: DeltaCompressedLatentWoFullCache,
            DELTA_COMPRESSED_LATENT_W_FULL: DeltaCompressedLatentWFullCache,
            DELTA_ORIGIN_WO_FULL: DeltaOriginWoFullCache,
            DELTA_ORIGIN_W_FULL: DeltaOriginWFullCache,
        }
        for impl, expected_cls in cases.items():
            with self.subTest(impl=impl):
                cfg = self._config()
                set_deltakv_cache_impl(cfg, impl)
                self.assertEqual(get_deltakv_cache_impl(cfg), impl)
                self.assertIsInstance(create_deltakv_cache(cfg), expected_cls)

    def test_removed_cache_impl_names_are_not_accepted(self):
        for impl in ("standard", "full", "origin", "all_origin"):
            with self.subTest(impl=impl):
                cfg = self._config()
                with self.assertRaisesRegex(ValueError, "Unknown deltakv_cache_impl"):
                    set_deltakv_cache_impl(cfg, impl)

    def test_hf_cache_factory_requires_cluster_path(self):
        cfg = self._config()
        cfg.use_cluster = False
        with self.assertRaisesRegex(ValueError, "cluster-only"):
            create_deltakv_cache(cfg)


if __name__ == "__main__":
    unittest.main()
