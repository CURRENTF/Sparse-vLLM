from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

from sparsevllm.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
)

_DISTRIBUTION = "sglang-kernel"
_REQUIRED_VERSION = "0.4.5"


def sgl_kernel_health() -> KernelFamilyHealth:
    """Inspect the required SGL binary family independently of any feature."""
    try:
        package_spec = importlib.util.find_spec("sgl_kernel")
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
    if version != _REQUIRED_VERSION:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"requires {_DISTRIBUTION}=={_REQUIRED_VERSION}, got {version}",
        )
    try:
        importlib.import_module("sgl_kernel")
    except Exception as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"binary failed to load: {type(error).__name__}: {error}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} binary family is ready",
    )


def sgl_kernel_support(feature: str) -> tuple[bool, str]:
    """Require a healthy SGL binary family before checking a feature contract."""

    health = sgl_kernel_health()
    if not health.ready:
        raise ExternalKernelFamilyError(health, feature=feature)
    return True, f"{_DISTRIBUTION} {health.version} {feature} is available"


__all__ = ["sgl_kernel_health", "sgl_kernel_support"]
