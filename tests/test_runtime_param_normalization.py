import unittest

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.configs.runtime_params import normalize_runtime_params


class RuntimeParamNormalizationTest(unittest.TestCase):
    def test_sparsevllm_normalizes_canonical_runtime_params(self):
        normalized = normalize_runtime_params(
            {
                "sparse_method": "deltakv",
                "deltakv_checkpoint_path": "/tmp/compressor",
                "decode_keep_tokens": 2048,
                "sink_keep_tokens": 8,
                "recent_keep_tokens": 128,
                "full_attention_layers": "0,1,2,8,18",
                "deltakv_neighbor_count": 4,
                "deltakv_center_ratio": 0.1,
                "deltakv_latent_dim": 256,
                "deltakv_latent_quant_bits": 0,
                "engine_prefill_chunk_size": 512,
                "prefill_schedule_policy": "auto",
            },
            backend="sparsevllm",
        )

        self.assertIsNone(normalized.hf_model_cls)
        self.assertIsNone(normalized.hf_deltakv_checkpoint_path)
        self.assertEqual(
            normalized.infer_config,
            {
                "vllm_sparse_method": "deltakv",
                "deltakv_path": "/tmp/compressor",
                "decode_keep_tokens": 2048,
                "num_sink_tokens": 8,
                "num_recent_tokens": 128,
                "full_attn_layers": "0,1,2,8,18",
                "deltakv_k_neighbors": 4,
                "cluster_ratio": 0.1,
                "kv_compressed_size": 256,
                "kv_quant_bits": 0,
                "chunk_prefill_size": 512,
                "prefill_schedule_policy": "auto",
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
                "deltakv_latent_quant_group_size": 32,
                "hf_prefill_chunk_size": 32768,
            },
            backend="hf",
        )

        self.assertEqual(normalized.hf_model_cls, "deltakv")
        self.assertEqual(normalized.hf_deltakv_checkpoint_path, "/tmp/compressor")
        self.assertEqual(
            normalized.infer_config,
            {
                "num_top_tokens": 0.17,
                "num_top_tokens_in_prefill": 4096,
                "num_sink_tokens": 8,
                "num_recent_tokens": 128,
                "full_attn_layers": "0,1,2,8,18",
                "deltakv_neighbor_count": 4,
                "cluster_ratio": 0.1,
                "kv_compressed_size": 256,
                "kv_quant_group_size": 32,
                "chunk_prefill_size": 32768,
            },
        )

    def test_legacy_runtime_names_raise(self):
        for key in (
            "model_cls",
            "vllm_sparse_method",
            "compressor_path",
            "deltakv_path",
            "num_top_tokens",
            "chunk_prefill_size",
            "seq_chunk_size",
            "compressor_token_group_size",
            "ref_mode",
            "k_neighbors",
            "deltakv_visual_compress_only",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Legacy runtime parameter"):
                    normalize_runtime_params({key: "x"}, backend="sparsevllm")

    def test_sparsevllm_vanilla_alias_maps_to_empty_method(self):
        normalized = normalize_runtime_params({"sparse_method": "vanilla"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "")

    def test_sparsevllm_rkv_alias_maps_to_canonical_method(self):
        normalized = normalize_runtime_params({"sparse_method": "r-kv"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "rkv")

        normalized = normalize_runtime_params({"sparse_method": "skip-kv"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "skipkv")

    def test_hf_less_memory_routes_to_deltakv_model(self):
        normalized = normalize_runtime_params({"sparse_method": "deltakv-less-memory"}, backend="hf")
        self.assertEqual(normalized.hf_model_cls, "deltakv")

    def test_hf_kivi_alias_routes_to_hf_kivi_model(self):
        for sparse_method in ("hf_kivi", "kivi_hf"):
            with self.subTest(sparse_method=sparse_method):
                normalized = normalize_runtime_params({"sparse_method": sparse_method}, backend="hf")
                self.assertEqual(normalized.hf_model_cls, "hf_kivi")

    def test_sparsevllm_rejects_ratio_style_keep_budgets(self):
        with self.assertRaisesRegex(ValueError, "explicit token count"):
            normalize_runtime_params({"decode_keep_tokens": 0.17}, backend="sparsevllm")

    def test_sparsevllm_observation_layer_keys_are_unknown(self):
        from sparsevllm import LLM

        for key in ("observation_layers", "obs_layer_ids"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Unknown Sparse-vLLM config keys"):
                    LLM("/tmp/unused-model", **{key: [0]})

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
            deltakv_latent_quant_bits=2,
            full_layer_kv_quant_bits=4,
            full_layer_cluster_ratio=0.1,
            full_layer_stride_alpha=0.0,
            visual_token_prune_only=True,
            visual_token_keep_ratio=0.1,
        )

        self.assertEqual(cfg.num_top_tokens, 0.25)
        self.assertEqual(cfg.num_top_tokens_in_prefill, 4096)
        self.assertEqual(cfg.num_recent_tokens, 128)
        self.assertEqual(cfg.tail_token_size, 128)
        self.assertEqual(cfg.num_sink_tokens, 8)
        self.assertEqual(cfg.full_attn_layers, [0, 1, 2])
        self.assertEqual(cfg.deltakv_neighbor_count, 3)
        self.assertEqual(cfg.chunk_prefill_size, 32768)
        self.assertEqual(cfg.kv_quant_bits, 2)
        self.assertEqual(cfg.full_layer_kv_quant_bits, 4)
        self.assertEqual(cfg.full_layer_cluster_ratio, 0.1)
        self.assertEqual(cfg.full_layer_stride_alpha, 0.0)
        self.assertTrue(cfg.visual_token_prune_only)
        self.assertEqual(cfg.visual_token_keep_ratio, 0.1)

    def test_hf_config_rejects_legacy_visual_prune_aliases(self):
        cfg = KVQwen2Config()
        with self.assertRaisesRegex(ValueError, "Legacy runtime parameter"):
            cfg.set_infer_args(
                deltakv_visual_compress_only=True,
                deltakv_visual_keep_ratio=0.25,
            )


if __name__ == "__main__":
    unittest.main()
