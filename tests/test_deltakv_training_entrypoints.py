import unittest

import torch

from deltakv.configs.model_config_cls import KVLlamaConfig, KVQwen2Config, KVQwen3Config
from deltakv.modeling.llama_training import LlamaKVClusterCompress
from deltakv.modeling.qwen2_training import Qwen2KVClusterCompress
from deltakv.modeling.qwen3_training import Qwen3KVClusterCompress


def _training_config(config_cls):
    cfg = config_cls(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        use_cluster=True,
        use_compression=True,
        kv_compressed_size=8,
        deltakv_neighbor_count=1,
        cluster_ratio=0.25,
        cluster_metric="l2",
        cluster_on_kv=True,
        cluster_soft_assignment=True,
        collect_kv_before_rope=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    return cfg


class DeltaKVTrainingEntrypointsTest(unittest.TestCase):
    MODEL_CASES = (
        ("qwen2", Qwen2KVClusterCompress, KVQwen2Config),
        ("qwen3", Qwen3KVClusterCompress, KVQwen3Config),
        ("llama", LlamaKVClusterCompress, KVLlamaConfig),
    )

    def test_cluster_e2e_big_training_classes_construct_for_all_hf_models(self):
        for name, model_cls, config_cls in self.MODEL_CASES:
            with self.subTest(model=name):
                model = model_cls(_training_config(config_cls))
                trainable_names = [name for name, _ in model.named_parameters() if "compress" in name]
                self.assertTrue(trainable_names)

    def test_cluster_e2e_big_training_forward_smoke_for_all_hf_models(self):
        for name, model_cls, config_cls in self.MODEL_CASES:
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = model_cls(_training_config(config_cls)).train()
                input_ids = torch.arange(18, dtype=torch.long).unsqueeze(0) % model.config.vocab_size
                out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                self.assertIsNotNone(out.loss)
                self.assertEqual(out.logits.shape[:2], (1, 18))

    def test_train_compressor_only_exposes_cluster_e2e_big_modeling(self):
        source = open("src/deltakv/train_compressor.py", encoding="utf-8").read()
        self.assertIn("cluster_e2e_big", source)
        self.assertIn("qwen2_training", source)
        self.assertIn("qwen3_training", source)
        self.assertIn("llama_training", source)
        for removed_path in (
            "qwen2_e2e",
            "qwen2_e2e_cluster",
            "qwen3_e2e",
            "qwen3_e2e_cluster",
            "llama_e2e",
            "llama_e2e_cluster",
        ):
            self.assertNotIn(removed_path, source)


if __name__ == "__main__":
    unittest.main()
