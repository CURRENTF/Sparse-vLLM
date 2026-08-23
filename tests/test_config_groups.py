from sparsevllm.config import Config as PublicConfig
from sparsevllm.configs import Config


def test_public_config_import_remains_compatible():
    assert PublicConfig is Config
