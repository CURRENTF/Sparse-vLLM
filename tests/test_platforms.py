import importlib
from types import SimpleNamespace

import pytest
import torch

from sparsevllm.platforms.interface import normalize_accelerator_identity


@pytest.mark.parametrize(
    ("device_name", "family", "variant"),
    [
        ("NVIDIA H100 80GB HBM3", "h100", "sxm"),
        ("NVIDIA H100-SXM5-80GB", "h100", "sxm"),
        ("NVIDIA H100 PCIe", "h100", "pcie"),
        ("nvidia h100 pcie", "h100", "pcie"),
        ("NVIDIA H200", "h200", None),
        ("NVIDIA H20", "h20", None),
        (
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "rtx_pro_6000",
            None,
        ),
        ("NVIDIA H1000", "unknown", None),
        ("unprofiled SM90 GPU", "unknown", None),
    ],
)
def test_accelerator_identity_is_normalized(
    device_name: str,
    family: str,
    variant: str | None,
) -> None:
    assert normalize_accelerator_identity(device_name) == (family, variant)


def test_explicit_cpu_platform_is_lazy_and_available(monkeypatch):
    monkeypatch.setenv("SPARSEVLLM_PLATFORM", "cpu")
    platforms = importlib.import_module("sparsevllm.platforms")
    platforms._set_current_platform_for_tests(None)

    platform = platforms.get_current_platform()

    assert platform.name == "cpu"
    assert platform.get_device(3) == torch.device("cpu")
    assert platform.get_distributed_backend() == "gloo"
    assert not platform.supports_graph_capture()
    caps = platform.get_device_caps()
    assert caps.platform.name == "CPU"
    assert caps.device_type == "cpu"
    assert caps.supports_bfloat16
    assert not caps.supports_native_fp8
    assert not platform.supports_inference()
    with pytest.raises(RuntimeError, match="inference is not supported"):
        platform.validate_inference()

    platforms._set_current_platform_for_tests(None)


def test_cuda_device_caps_are_the_capability_source(monkeypatch):
    from sparsevllm.platforms.cuda import CudaPlatform

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test H100")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=120),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    platform = CudaPlatform()

    caps = platform.get_device_caps(7)

    assert caps.device_index == 7
    assert caps.device_name == "Test H100"
    assert caps.accelerator_family == "h100"
    assert caps.accelerator_variant is None
    assert caps.compute_capability == (9, 0)
    assert caps.multi_processor_count == 120
    assert caps.supports_native_fp8
    assert platform.supports_fp8()


def test_unknown_explicit_platform_fails_fast(monkeypatch):
    monkeypatch.setenv("SPARSEVLLM_PLATFORM", "missing_test_platform")
    platforms = importlib.import_module("sparsevllm.platforms")
    platforms._set_current_platform_for_tests(None)

    with pytest.raises(RuntimeError, match="SPARSEVLLM_PLATFORM"):
        platforms.get_current_platform()

    platforms._set_current_platform_for_tests(None)


def test_rocm_detection_fails_fast_until_backend_exists(monkeypatch):
    monkeypatch.setenv("SPARSEVLLM_PLATFORM", "rocm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.0.0", raising=False)
    platforms = importlib.import_module("sparsevllm.platforms")
    platforms._set_current_platform_for_tests(None)

    with pytest.raises(RuntimeError, match="ROCm.*not supported"):
        platforms.get_current_platform()

    platforms._set_current_platform_for_tests(None)
