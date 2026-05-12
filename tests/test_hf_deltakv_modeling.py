import unittest

import torch

from deltakv.configs.model_config_cls import KVLlamaConfig, KVQwen2Config, KVQwen3Config
from deltakv.modeling.cache_factory import (
    DELTA_COMPRESSED_LATENT_W_FULL,
    DELTA_ORIGIN_W_FULL,
    DELTA_ORIGIN_WO_FULL,
)
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
