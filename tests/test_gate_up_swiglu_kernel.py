import pytest
import torch

from sparsevllm.triton_kernel.gate_up_swiglu import (
    gate_up_swiglu,
    resolve_h20_gate_up_swiglu_config,
)


def _is_h20() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA H20"


@pytest.mark.skipif(not _is_h20(), reason="requires NVIDIA H20")
@pytest.mark.parametrize("intermediate_size", [256, 512])
def test_h20_gate_up_swiglu_matches_torch(intermediate_size):
    torch.manual_seed(0)
    inputs = torch.randn(1, 2048, dtype=torch.bfloat16, device="cuda")
    weight = 0.02 * torch.randn(
        2 * intermediate_size,
        2048,
        dtype=torch.bfloat16,
        device="cuda",
    )
    projected = torch.nn.functional.linear(inputs, weight)
    gate, up = projected.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate.float()) * up.float()

    actual = gate_up_swiglu(
        inputs,
        weight,
        resolve_h20_gate_up_swiglu_config(1, 2048, intermediate_size),
    )

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
