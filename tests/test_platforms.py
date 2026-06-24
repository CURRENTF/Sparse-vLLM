import importlib

import pytest
import torch


def test_explicit_cpu_platform_is_lazy_and_available(monkeypatch):
    monkeypatch.setenv("SPARSEVLLM_PLATFORM", "cpu")
    platforms = importlib.import_module("sparsevllm.platforms")
    platforms._set_current_platform_for_tests(None)

    platform = platforms.get_current_platform()

    assert platform.name == "cpu"
    assert platform.get_device(3) == torch.device("cpu")
    assert platform.get_distributed_backend() == "gloo"
    assert not platform.supports_graph_capture()
    assert not platform.supports_inference()
    with pytest.raises(RuntimeError, match="inference is not supported"):
        platform.validate_inference()

    platforms._set_current_platform_for_tests(None)


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
