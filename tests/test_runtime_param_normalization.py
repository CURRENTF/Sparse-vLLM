import unittest

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.configs.runtime_params import normalize_runtime_params


class RuntimeParamNormalizationTest(unittest.TestCase):
    def test_sparsevllm_normalizes_canonical_runtime_params(self):
        normalized = normalize_runtime_params(
            {
                "sparse_method": "deltakv-triton-v4",
                "deltakv_checkpoint_path": "/tmp/compressor",
                "decode_keep_tokens": 2048,
                "prefill_keep_tokens": 4096,
                "sink_keep_tokens": 8,
                "recent_keep_tokens": 128,
                "full_attention_layers": "0,1,2,8,18",
                "observation_layers": [2, 8, 18],
                "deltakv_neighbor_count": 4,
                "deltakv_center_ratio": 0.1,
                "deltakv_latent_dim": 256,
                "deltakv_latent_quant_bits": 0,
                "engine_prefill_chunk_size": 512,
            },
            backend="sparsevllm",
        )

        self.assertIsNone(normalized.model_cls)
        self.assertIsNone(normalized.compressor_path)
        self.assertEqual(
            normalized.infer_config,
            {
                "vllm_sparse_method": "deltakv-triton-v4",
                "deltakv_path": "/tmp/compressor",
                "num_top_tokens": 2048,
                "num_top_tokens_in_prefill": 4096,
                "num_sink_tokens": 8,
                "num_recent_tokens": 128,
                "full_attn_layers": "0,1,2,8,18",
                "obs_layer_ids": [2, 8, 18],
                "deltakv_k_neighbors": 4,
                "cluster_ratio": 0.1,
                "kv_compressed_size": 256,
                "kv_quant_bits": 0,
                "chunk_prefill_size": 512,
            },
        )

    def test_hf_normalizes_canonical_runtime_params(self):
        normalized = normalize_runtime_params(
            {
                "sparse_method": "deltakv",
                "deltakv_checkpoint_path": "/tmp/compressor",
                "decode_keep_tokens": 0.17,
                "prefill_keep_tokens": 4096,
                "sink_keep_tokens": 8,
                "recent_keep_tokens": 128,
                "full_attention_layers": "0,1,2,8,18",
                "deltakv_neighbor_count": 4,
                "deltakv_center_ratio": 0.1,
                "deltakv_latent_dim": 256,
                "hf_prefill_chunk_size": 32768,
            },
            backend="hf",
        )

        self.assertEqual(normalized.model_cls, "deltakv")
        self.assertEqual(normalized.compressor_path, "/tmp/compressor")
        self.assertEqual(
            normalized.infer_config,
            {
                "num_top_tokens": 0.17,
                "num_top_tokens_in_prefill": 4096,
                "num_sink_tokens": 8,
                "num_recent_tokens": 128,
                "full_attn_layers": "0,1,2,8,18",
                "k_neighbors": 4,
                "cluster_ratio": 0.1,
                "kv_compressed_size": 256,
                "chunk_prefill_size": 32768,
            },
        )

    def test_conflicting_alias_values_raise(self):
        with self.assertRaisesRegex(ValueError, "decode_keep_tokens"):
            normalize_runtime_params(
                {"decode_keep_tokens": 2048, "num_top_tokens": 4096},
                backend="sparsevllm",
            )

    def test_sparsevllm_vanilla_alias_maps_to_empty_method(self):
        normalized = normalize_runtime_params({"sparse_method": "vanilla"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "")

    def test_sparsevllm_rejects_ratio_style_keep_budgets(self):
        with self.assertRaisesRegex(ValueError, "explicit token count"):
            normalize_runtime_params({"decode_keep_tokens": 0.17}, backend="sparsevllm")

    def test_hf_config_accepts_canonical_aliases(self):
        cfg = KVQwen2Config()
        cfg.set_infer_args(
            decode_keep_tokens=0.25,
            prefill_keep_tokens=4096,
            recent_keep_tokens=128,
            sink_keep_tokens=8,
            full_attention_layers="0,1,2",
            deltakv_neighbor_count=3,
            hf_prefill_chunk_size=32768,
            visual_token_prune_only=True,
            visual_token_keep_ratio=0.1,
        )

        self.assertEqual(cfg.num_top_tokens, 0.25)
        self.assertEqual(cfg.num_top_tokens_in_prefill, 4096)
        self.assertEqual(cfg.num_recent_tokens, 128)
        self.assertEqual(cfg.tail_token_size, 128)
        self.assertEqual(cfg.num_sink_tokens, 8)
        self.assertEqual(cfg.full_attn_layers, [0, 1, 2])
        self.assertEqual(cfg.k_neighbors, 3)
        self.assertEqual(cfg.chunk_prefill_size, 32768)
        self.assertTrue(cfg.visual_token_prune_only)
        self.assertEqual(cfg.visual_token_keep_ratio, 0.1)
        self.assertTrue(cfg.deltakv_visual_compress_only)
        self.assertEqual(cfg.deltakv_visual_keep_ratio, 0.1)

    def test_hf_config_normalizes_legacy_visual_prune_aliases(self):
        cfg = KVQwen2Config()
        cfg.set_infer_args(
            deltakv_visual_compress_only=True,
            deltakv_visual_keep_ratio=0.25,
        )

        self.assertTrue(cfg.visual_token_prune_only)
        self.assertEqual(cfg.visual_token_keep_ratio, 0.25)
        self.assertTrue(cfg.deltakv_visual_compress_only)
        self.assertEqual(cfg.deltakv_visual_keep_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
