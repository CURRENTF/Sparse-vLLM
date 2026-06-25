import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config


class SkipKVConfigTest(unittest.TestCase):
    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
        )

    def test_rejects_unsupported_model_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "SomeOtherModel"
            model_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "SkipKV is supported only"):
                Config(model=str(model_dir), vllm_sparse_method="skipkv")

    def test_supported_model_basename_passes_without_steering(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "DeepSeek-R1-Distill-Qwen-7B"
            model_dir.mkdir()
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                cfg = Config(model=str(model_dir), vllm_sparse_method="skipkv")
            self.assertEqual(cfg.vllm_sparse_method, "skipkv")

    def test_steering_requires_vector_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "DeepSeek-R1-Distill-Qwen-7B"
            model_dir.mkdir()
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                with self.assertRaisesRegex(ValueError, "requires skipkv_steering_vector_path"):
                    Config(
                        model=str(model_dir),
                        vllm_sparse_method="skipkv",
                        skipkv_enable_activation_steering=True,
                    )


if __name__ == "__main__":
    unittest.main()
