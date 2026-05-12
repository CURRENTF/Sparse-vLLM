import unittest

import torch

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.kv_cache import ClusterCompressedKVCache, CompressedKVCache


class VisualUniformPruningTest(unittest.TestCase):
    def test_visual_uniform_pruning_keeps_text_positions_in_buffer(self):
        config = KVQwen2Config(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=16,
            intermediate_size=32,
            vocab_size=64,
        )
        config.set_infer_args(
            use_compression=False,
            use_cluster=False,
            deltakv_latent_quant_bits=0,
            sink_keep_tokens=1,
            recent_keep_tokens=2,
            full_attention_layers="",
            visual_token_prune_only=True,
            visual_token_keep_ratio=0.5,
        )
        cache = CompressedKVCache(config)

        key = torch.randn(1, 6, 8)
        value = torch.randn(1, 6, 8)
        visual_mask = torch.tensor([[False, True, True, True, False, False]])
        cache_position = torch.arange(6)

        _, _, full_idx = cache.update(
            key,
            value,
            0,
            {
                "cache_position": cache_position,
                "deltakv_visual_token_mask": visual_mask,
            },
        )

        self.assertEqual(cache.comp_pos_cache[0].tolist(), [[1, 2]])
        self.assertEqual(cache.buffer_pos_cache[0].tolist(), [[4, 5]])
        self.assertEqual(full_idx.tolist(), [[0, 1, 2, 3, 4, 5]])

    def test_cluster_visual_pruning_without_compressor_has_valid_mask(self):
        config = KVQwen2Config(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=16,
            intermediate_size=32,
            vocab_size=64,
        )
        config.set_infer_args(
            use_compression=False,
            use_cluster=True,
            deltakv_latent_quant_bits=0,
            sink_keep_tokens=1,
            recent_keep_tokens=1,
            full_attention_layers="0",
            visual_token_prune_only=True,
            visual_token_keep_ratio=0.5,
            deltakv_neighbor_count=1,
            deltakv_center_ratio=0.5,
        )
        cache = ClusterCompressedKVCache(config)

        key = torch.randn(1, 6, 8)
        value = torch.randn(1, 6, 8)
        visual_mask = torch.tensor([[False, True, True, True, False, False]])

        cache.update(
            key,
            value,
            1,
            {
                "cache_position": torch.arange(6),
                "deltakv_visual_token_mask": visual_mask,
            },
        )

        self.assertEqual(cache.comp_pos_cache[1].tolist(), [[1, 2]])
        self.assertEqual(cache.buffer_pos_cache[1].tolist(), [[4, 5]])


if __name__ == "__main__":
    unittest.main()
