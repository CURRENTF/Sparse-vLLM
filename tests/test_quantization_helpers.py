import unittest

import torch

from deltakv.quantization import build_model_load_kwargs, restore_modules_to_dtype


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        self.compress_down = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        )
        self.cluster = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        self.nested = torch.nn.Module()
        self.nested.v_compress_up = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)


class QuantizationHelperTests(unittest.TestCase):
    def test_build_model_load_kwargs_for_4bit(self):
        runtime_cfg, load_kwargs, target_dtype = build_model_load_kwargs(
            {
                "load_in_4bit": True,
                "torch_dtype": "fp16",
                "bnb_4bit_compute_dtype": "bf16",
                "quant_skip_modules": ["custom_head"],
                "chunk_prefill_size": 4096,
            },
            default_torch_dtype=torch.bfloat16,
        )

        self.assertEqual(runtime_cfg, {"chunk_prefill_size": 4096})
        self.assertEqual(target_dtype, torch.float16)
        self.assertIn("quantization_config", load_kwargs)
        quant_cfg = load_kwargs["quantization_config"]
        self.assertTrue(quant_cfg.load_in_4bit)
        self.assertEqual(quant_cfg.bnb_4bit_compute_dtype, torch.bfloat16)
        self.assertIn("compress_down", quant_cfg.llm_int8_skip_modules)
        self.assertIn("custom_head", quant_cfg.llm_int8_skip_modules)

    def test_preserves_chunk_prefill_size(self):
        runtime_cfg, load_kwargs, target_dtype = build_model_load_kwargs(
            {
                "load_in_4bit": True,
                "chunk_prefill_size": 204800000,
            },
            default_torch_dtype=torch.bfloat16,
        )

        self.assertEqual(runtime_cfg["chunk_prefill_size"], 204800000)
        self.assertIn("quantization_config", load_kwargs)
        self.assertEqual(target_dtype, torch.bfloat16)

    def test_restore_modules_to_dtype_skips_transformer(self):
        model = _DummyModel()

        restored = restore_modules_to_dtype(model, torch.bfloat16)

        self.assertIn("compress_down", restored)
        self.assertIn("cluster", restored)
        self.assertIn("nested.v_compress_up", restored)
        self.assertEqual(model.compress_down[0].weight.dtype, torch.bfloat16)
        self.assertEqual(model.cluster.weight.dtype, torch.bfloat16)
        self.assertEqual(model.nested.v_compress_up.weight.dtype, torch.bfloat16)
        self.assertEqual(model.transformer.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
