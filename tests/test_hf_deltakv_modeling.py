import unittest
from types import SimpleNamespace

import torch

from deltakv.configs.model_config_cls import KVLlamaConfig, KVQwen2Config, KVQwen3Config
from deltakv.modeling.compressor import reshape_and_apply_qk_norm
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
    create_deltakv_cache,
)
from deltakv.modeling.cache_pipeline import HF_SPARSE_CACHE_OMNIKV, OmniKVRawCache
from deltakv.modeling.llama_inference import (
    LlamaDeltaCompressedLatentWFull,
    LlamaDeltaOriginWFull,
    LlamaDeltaOriginWoFull,
    LlamaKVCompress,
)
from deltakv.modeling.qwen2_inference import (
    Qwen2DeltaCompressedLatentWFull,
    Qwen2DeltaOriginWFull,
    Qwen2DeltaOriginWoFull,
    Qwen2KVCompress,
)
from deltakv.modeling.qwen3_inference import (
    Qwen3DeltaCompressedLatentWFull,
    Qwen3DeltaOriginWFull,
    Qwen3DeltaOriginWoFull,
    Qwen3KVCompress,
)
from deltakv.modeling.token_select import omnikv_token_selection


def _tiny_config(config_cls):
    cfg = config_cls(
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_sink_tokens=1,
        num_recent_tokens=8,
        full_attn_layers="0,1",
        use_cluster=True,
        use_compression=False,
        chunk_prefill_size=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    return cfg


class HfDeltaKVModelingTest(unittest.TestCase):
    MODEL_CASES = (
        ("qwen2", Qwen2KVCompress, KVQwen2Config),
        ("qwen3", Qwen3KVCompress, KVQwen3Config),
        ("llama", LlamaKVCompress, KVLlamaConfig),
    )
    OMNIKV_MODEL_CASES = (
        ("qwen2", Qwen2KVCompress, KVQwen2Config),
        ("llama", LlamaKVCompress, KVLlamaConfig),
    )

    def test_hf_deltakv_rejects_batched_inputs(self):
        for name, model_cls, config_cls in self.MODEL_CASES:
            with self.subTest(model=name):
                model = model_cls(_tiny_config(config_cls)).eval()
                with self.assertRaisesRegex(NotImplementedError, "batch_size=1"):
                    model(input_ids=torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long), use_cache=True)

    def test_hf_deltakv_rejects_padded_inputs(self):
        for name, model_cls, config_cls in self.MODEL_CASES:
            with self.subTest(model=name):
                model = model_cls(_tiny_config(config_cls)).eval()
                with self.assertRaisesRegex(NotImplementedError, "padded"):
                    model(
                        input_ids=torch.tensor([[0, 5, 6]], dtype=torch.long),
                        attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
                        use_cache=True,
                    )

    def test_hf_deltakv_accepts_bs1_unpadded_inputs(self):
        for name, model_cls, config_cls in self.MODEL_CASES:
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = model_cls(_tiny_config(config_cls)).eval()
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long), use_cache=True)
                self.assertEqual(out.logits.shape[:2], (1, 1))

    def test_hf_deltakv_short_prompt_smaller_than_sink_budget(self):
        for name, model_cls, config_cls in self.MODEL_CASES:
            with self.subTest(model=name):
                torch.manual_seed(0)
                cfg = _tiny_config(config_cls)
                cfg.num_sink_tokens = 8
                model = model_cls(cfg).eval()
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long), use_cache=True)
                self.assertEqual(out.logits.shape[:2], (1, 1))
                cache = out.past_key_values
                self.assertEqual(cache.sink_filled_count[0], 3)
                key_view, _, pos_view = cache._view(0, compressor_up=None, k_dim=8)
                self.assertEqual(key_view.shape[1], 3)
                self.assertEqual(pos_view.shape[1], 3)

    def test_hf_omnikv_accepts_non_cluster_raw_cache(self):
        for name, model_cls, config_cls in self.OMNIKV_MODEL_CASES:
            with self.subTest(model=name):
                torch.manual_seed(0)
                cfg = _tiny_config(config_cls)
                cfg.full_attn_layers = [0]
                cfg.num_sink_tokens = 2
                cfg.num_recent_tokens = 2
                cfg.tail_token_size = 2
                cfg.use_cluster = False
                cfg.use_compression = False
                cfg.kv_quant_bits = 0
                cfg.hf_sparse_cache_impl = HF_SPARSE_CACHE_OMNIKV
                model = model_cls(cfg).eval()
                with torch.no_grad():
                    prefill = model(input_ids=torch.tensor([[5, 6, 7, 8, 9, 10]], dtype=torch.long), use_cache=True)
                    decode = model(
                        input_ids=torch.tensor([[11]], dtype=torch.long),
                        past_key_values=prefill.past_key_values,
                        use_cache=True,
                    )
                self.assertEqual(decode.logits.shape[:2], (1, 1))
                self.assertIsInstance(decode.past_key_values, OmniKVRawCache)
                self.assertGreaterEqual(decode.past_key_values.get_compressed_length(0), 2)

    def test_hf_omnikv_raw_cache_keeps_exact_history_before_recent_tail(self):
        cfg = _tiny_config(KVQwen2Config)
        cfg.num_sink_tokens = 2
        cfg.num_recent_tokens = 3
        cfg.tail_token_size = 3
        cfg.full_attn_layers = [0]
        cfg.use_cluster = False
        cfg.use_compression = False
        cfg.hf_sparse_cache_impl = HF_SPARSE_CACHE_OMNIKV
        cache = OmniKVRawCache(cfg)

        first_key = torch.arange(10 * 8, dtype=torch.float32).view(1, 10, 8)
        first_value = first_key + 1000
        second_key = torch.arange(4 * 8, dtype=torch.float32).view(1, 4, 8) + 10_000
        second_value = second_key + 1000

        cache.update(first_key, first_value, 0, {"cache_position": torch.arange(10)})
        key_view, _, pos_view = cache.update(
            second_key,
            second_value,
            0,
            {"cache_position": torch.arange(10, 14)},
        )

        self.assertEqual(cache.get_observable_compressed_length(current_q_len=4), 5)
        self.assertEqual(cache.get_compressed_length(0), 9)
        self.assertEqual(cache.buffer_key_cache[0].shape[1], 3)
        self.assertEqual(key_view.shape[1], 14)
        self.assertEqual(pos_view.tolist(), [list(range(14))])

    def test_omnikv_selection_uses_raw_qk_logits_for_decode(self):
        module = SimpleNamespace(num_key_value_groups=1)
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]])
        _, scores = omnikv_token_selection(
            module,
            query,
            key,
            scaling=0.125,
            num_top_tokens=1,
            pool_kernel_size=1,
        )
        self.assertTrue(torch.allclose(scores, torch.tensor([[2.0, 0.0]])))

    def test_qk_norm_reshape_allows_different_query_and_key_lengths(self):
        class AttnWithNorms:
            q_norm = torch.nn.Identity()
            k_norm = torch.nn.Identity()

        query_states = torch.randn(1, 3, 4, 8)
        key_states = torch.randn(1, 5, 2, 8)
        query_out, key_out = reshape_and_apply_qk_norm(
            AttnWithNorms(),
            query_states.reshape(1, -1),
            key_states.reshape(1, -1),
            (1, 3, 4, 8),
            (1, 5, 2, 8),
        )
        self.assertEqual(query_out.shape, (1, 4, 3, 8))
        self.assertEqual(key_out.shape, (1, 2, 5, 8))

    def test_obs_layer_uses_global_compressed_history_length(self):
        cfg = _tiny_config(KVQwen2Config)
        cfg.num_hidden_layers = 4
        cfg.full_attn_layers = [0, 1]
        cfg.tail_token_size = 4
        cfg.num_sink_tokens = 1
        cache = create_deltakv_cache(cfg)
        cache._seen_tokens = 14
        self.assertEqual(cache.get_compressed_length(1), 0)
        self.assertEqual(cache.get_observable_compressed_length(current_q_len=1), 8)

    def test_legacy_seq_chunk_size_preserves_cluster_neighbor_count(self):
        cfg = KVQwen2Config(seq_chunk_size=4, use_cluster=True)
        self.assertEqual(cfg.compressor_token_group_size, 4)
        self.assertEqual(cfg.get_cluster_neighbor_count(), 4)

    def test_variant_class_names_match_current_cache_impls(self):
        cases = (
            (Qwen2DeltaCompressedLatentWFull, KVQwen2Config, DELTA_COMPRESSED_LATENT_W_FULL),
            (Qwen2DeltaOriginWoFull, KVQwen2Config, DELTA_ORIGIN_WO_FULL),
            (Qwen2DeltaOriginWFull, KVQwen2Config, DELTA_ORIGIN_W_FULL),
            (Qwen3DeltaCompressedLatentWFull, KVQwen3Config, DELTA_COMPRESSED_LATENT_W_FULL),
            (Qwen3DeltaOriginWoFull, KVQwen3Config, DELTA_ORIGIN_WO_FULL),
            (Qwen3DeltaOriginWFull, KVQwen3Config, DELTA_ORIGIN_W_FULL),
            (LlamaDeltaCompressedLatentWFull, KVLlamaConfig, DELTA_COMPRESSED_LATENT_W_FULL),
            (LlamaDeltaOriginWoFull, KVLlamaConfig, DELTA_ORIGIN_WO_FULL),
            (LlamaDeltaOriginWFull, KVLlamaConfig, DELTA_ORIGIN_W_FULL),
        )
        for variant_cls, config_cls, expected_impl in cases:
            with self.subTest(variant=variant_cls.__name__):
                cfg = _tiny_config(config_cls)
                variant_cls(cfg)
                self.assertEqual(cfg.deltakv_cache_impl, expected_impl)
                self.assertNotIn("AllOrigin", variant_cls.__name__)
                self.assertNotIn("ResidualQuant", variant_cls.__name__)


if __name__ == "__main__":
    unittest.main()
