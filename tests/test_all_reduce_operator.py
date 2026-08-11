from unittest.mock import Mock

import torch

from sparsevllm.operators.all_reduce import HopperTp2FlashInferAllReduceProvider


def test_flashinfer_all_reduce_dispatches_unsupported_shape_before_launch():
    provider = HopperTp2FlashInferAllReduceProvider.__new__(
        HopperTp2FlashInferAllReduceProvider
    )
    provider.fallback = Mock()
    tensor = torch.randn(1, 248320, dtype=torch.bfloat16)
    provider.fallback.run.return_value = tensor

    assert provider.run(tensor) is tensor
    provider.fallback.run.assert_called_once_with(tensor)
