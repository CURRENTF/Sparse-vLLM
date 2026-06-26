import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config


class DecodeCudaGraphTPConfigTest(unittest.TestCase):
    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    def _config(self, method: str, *, model_name: str = "TinyModel", **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / model_name
            model_dir.mkdir()
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                return Config(
                    model=str(model_dir),
                    vllm_sparse_method=method,
                    decode_cuda_graph=True,
                    tensor_parallel_size=2,
                    max_decoding_seqs=4,
                    **kwargs,
                )

    def test_tp_decode_cuda_graph_accepts_v1_methods(self):
        for method in ["vanilla", "streamingllm", "snapkv", "pyramidkv", "omnikv", "rkv"]:
            with self.subTest(method=method):
                cfg = self._config(method)
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertEqual(cfg.tensor_parallel_size, 2)

        cfg = self._config("skipkv", model_name="DeepSeek-R1-Distill-Qwen-7B")
        self.assertEqual(cfg.vllm_sparse_method, "skipkv")

    def test_tp_decode_cuda_graph_rejects_deltakv_and_quest_v1(self):
        with self.assertRaisesRegex(ValueError, "v1 excludes DeltaKV and QuEST"):
            self._config("deltakv", allow_missing_deltakv_path=True)

        with self.assertRaisesRegex(ValueError, "v1 excludes DeltaKV and QuEST"):
            self._config("quest")

    def test_tp_decode_cuda_graph_rejects_capture_sampling(self):
        with self.assertRaisesRegex(ValueError, "capture_sampling is disabled"):
            self._config("snapkv", decode_cuda_graph_capture_sampling=True)


if __name__ == "__main__":
    unittest.main()
