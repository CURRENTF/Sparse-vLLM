import pytest
import torch

from sparsevllm.operators.gate_up_swiglu import (
    GATE_UP_SWIGLU_REGISTRY,
    GateUpSwiGLUOpSpec,
    TorchGateUpSwiGLUProvider,
)
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _spec(**overrides) -> GateUpSwiGLUOpSpec:
    values = {
        "hidden_size": 2048,
        "intermediate_size": 512,
        "tp_size": 1,
        "activation_dtype": torch.bfloat16,
        "weight_dtype": torch.bfloat16,
        "cuda_graph": True,
    }
    values.update(overrides)
    return GateUpSwiGLUOpSpec(**values)


def _caps(device_name="NVIDIA H20", capability=(9, 0)) -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name=device_name,
        compute_capability=capability,
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        supports_native_fp8=True,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_h20_provider_requires_profiled_qwen36_shape(tp_size):
    resolved = OpResolver(GATE_UP_SWIGLU_REGISTRY).resolve(
        _spec(tp_size=tp_size), _caps()
    )

    assert resolved.provider.name == "h20_triton_decode"


@pytest.mark.parametrize(
    "spec,caps",
    [
        (_spec(weight_dtype=torch.float8_e4m3fn), _caps()),
        (_spec(intermediate_size=768), _caps()),
        (_spec(), _caps(device_name="NVIDIA H100 80GB HBM3")),
        (_spec(), _caps(capability=(8, 9))),
    ],
)
def test_unprofiled_shape_uses_torch_provider(spec, caps):
    resolved = OpResolver(GATE_UP_SWIGLU_REGISTRY).resolve(spec, caps)

    assert resolved.provider.name == "torch"


def test_torch_provider_matches_gate_up_swiglu_semantics():
    torch.manual_seed(0)
    inputs = torch.randn(3, 8)
    projection = torch.nn.Linear(8, 12, bias=False)
    with torch.inference_mode():
        projected = projection(inputs)
        gate, up = projected.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        actual = TorchGateUpSwiGLUProvider().run(
            _spec(
                hidden_size=8,
                intermediate_size=6,
                activation_dtype=torch.float32,
                weight_dtype=torch.float32,
                cuda_graph=False,
            ),
            inputs,
            projection,
        )

    torch.testing.assert_close(actual, expected)
