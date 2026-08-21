from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re

from sparsevllm.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
)

_MIN_VERSION = (0, 6, 15)
_MAX_VERSION = (0, 7, 0)
_DISTRIBUTION = "flashinfer-python"


def flashinfer_kernel_health() -> KernelFamilyHealth:
    """Inspect FlashInfer package metadata and its top-level import."""

    try:
        package_spec = importlib.util.find_spec("flashinfer")
    except (ImportError, ValueError) as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            None,
            f"package discovery failed: {type(error).__name__}: {error}",
        )
    if package_spec is None:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.ABSENT,
            None,
            f"{_DISTRIBUTION} is not installed",
        )
    try:
        version = importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            None,
            f"{_DISTRIBUTION} package metadata is unavailable",
        )
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    parsed = tuple(map(int, match.groups())) if match else None
    if parsed is None or not _MIN_VERSION <= parsed < _MAX_VERSION:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"requires {_DISTRIBUTION}>=0.6.15,<0.7, got {version}",
        )
    try:
        importlib.import_module("flashinfer")
    except Exception as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"package failed to load: {type(error).__name__}: {error}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} package family is ready",
    )


def flashinfer_kernel_support(feature: str) -> tuple[bool, str]:
    """Require a healthy FlashInfer family before probing one feature."""

    health = flashinfer_kernel_health()
    if not health.ready:
        raise ExternalKernelFamilyError(health, feature=feature)
    return True, f"{_DISTRIBUTION} {health.version} {feature} is available"


__all__ = ["flashinfer_kernel_health", "flashinfer_kernel_support"]
