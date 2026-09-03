from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sparsevllm.engine.llm_engine as llm_engine
import sparsevllm.engine.model_runner as model_runner
from sparsevllm.kernels.external.required import (
    validate_required_cuda_kernel_families,
    validate_required_cuda_kernel_metadata,
)
from sparsevllm.kernels.external.support import (
    KernelFamilyHealth,
    KernelFamilyState,
)


def _health(
    family: str,
    state: KernelFamilyState,
    reason: str,
) -> KernelFamilyHealth:
    return KernelFamilyHealth(
        family=family,
        state=state,
        version=None,
        reason=reason,
    )


def test_required_cuda_binary_validation_accepts_healthy_families() -> None:
    with (
        patch(
            "sparsevllm.kernels.external.required.flashinfer_kernel_health",
            return_value=_health(
                "flashinfer-python",
                KernelFamilyState.READY,
                "ready",
            ),
        ),
        patch(
            "sparsevllm.kernels.external.required.sgl_kernel_health",
            return_value=_health(
                "sglang-kernel",
                KernelFamilyState.READY,
                "ready",
            ),
        ),
    ):
        validate_required_cuda_kernel_families()


def test_required_cuda_binary_validation_reports_all_unhealthy_families() -> None:
    with (
        patch(
            "sparsevllm.kernels.external.required.flashinfer_kernel_health",
            return_value=_health(
                "flashinfer-python",
                KernelFamilyState.ABSENT,
                "not installed",
            ),
        ),
        patch(
            "sparsevllm.kernels.external.required.sgl_kernel_health",
            return_value=_health(
                "sglang-kernel",
                KernelFamilyState.BROKEN,
                "undefined symbol",
            ),
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        validate_required_cuda_kernel_families()

    message = str(exc_info.value)
    assert "flashinfer-python is absent: not installed" in message
    assert "sglang-kernel is broken: undefined symbol" in message
    assert 'pip install -e ".[cu129]"' in message
    assert 'pip install -e ".[cu130]"' in message


def test_required_cuda_metadata_validation_reports_unhealthy_families() -> None:
    with (
        patch(
            "sparsevllm.kernels.external.required.flashinfer_kernel_metadata_health",
            return_value=_health(
                "flashinfer-python",
                KernelFamilyState.ABSENT,
                "not installed",
            ),
        ),
        patch(
            "sparsevllm.kernels.external.required.sgl_kernel_metadata_health",
            return_value=_health(
                "sglang-kernel",
                KernelFamilyState.READY,
                "ready",
            ),
        ),
        pytest.raises(RuntimeError, match="startup metadata validation"),
    ):
        validate_required_cuda_kernel_metadata()


def test_gpu_engine_validates_metadata_before_starting_workers() -> None:
    config = SimpleNamespace()
    with (
        patch.object(llm_engine, "fields", return_value=()),
        patch.object(llm_engine, "Config", return_value=config),
        patch.object(
            llm_engine.platforms,
            "get_current_platform",
            return_value=SimpleNamespace(enum=llm_engine.PlatformEnum.CUDA),
        ),
        patch.object(
            llm_engine,
            "validate_required_cuda_kernel_metadata",
            side_effect=RuntimeError("required dependencies are unavailable"),
        ) as validate,
        patch.object(llm_engine.mp, "get_context") as get_context,
        pytest.raises(RuntimeError, match="required dependencies are unavailable"),
    ):
        llm_engine.LLMEngine("model")

    validate.assert_called_once_with()
    get_context.assert_not_called()


def test_model_runner_validates_binaries_after_selecting_rank_device() -> None:
    events: list[tuple[str, object]] = []

    class CudaPlatform:
        enum = model_runner.platforms.PlatformEnum.CUDA

        def validate_inference(self) -> None:
            events.append(("validate_inference", None))

        def init_backend(self) -> None:
            events.append(("init_backend", None))

        def get_device(self, rank: int) -> torch.device:
            return torch.device("cuda", rank)

        def set_device(self, device: torch.device) -> None:
            events.append(("set_device", device))

    def validate_binaries() -> None:
        events.append(("validate_binaries", None))
        raise RuntimeError("stop after binary validation")

    config = SimpleNamespace(
        enable_profiler=False,
        hf_config=SimpleNamespace(),
        world_size=1,
    )
    with (
        patch.object(model_runner.platforms, "_current_platform", CudaPlatform()),
        patch.object(
            model_runner,
            "validate_required_cuda_kernel_families",
            side_effect=validate_binaries,
        ),
        pytest.raises(RuntimeError, match="stop after binary validation"),
    ):
        model_runner.ModelRunner(config, rank=0, event=[])

    assert events == [
        ("validate_inference", None),
        ("init_backend", None),
        ("set_device", torch.device("cuda", 0)),
        ("validate_binaries", None),
    ]
