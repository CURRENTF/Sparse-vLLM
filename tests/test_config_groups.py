from types import SimpleNamespace

import torch

from sparsevllm.config import Config as PublicConfig
from sparsevllm.configs import Config
from sparsevllm.configs.model import _normalize_hf_config_dtype


def test_public_config_import_remains_compatible():
    assert PublicConfig is Config


def test_model_dtype_normalization_uses_current_transformers_field():
    class ModernConfig:
        dtype = torch.bfloat16

        @property
        def torch_dtype(self):
            raise AssertionError("deprecated dtype field was accessed")

    config = ModernConfig()

    assert _normalize_hf_config_dtype(config) is torch.bfloat16
    assert config.dtype is torch.bfloat16


def test_model_dtype_normalization_accepts_legacy_field():
    config = SimpleNamespace(torch_dtype=torch.float16)

    assert _normalize_hf_config_dtype(config) is torch.float16
    assert config.dtype is torch.float16
