from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re

_MIN_VERSION = (0, 3, 21)
_MAX_VERSION = (0, 4, 0)


def sgl_kernel_support(feature: str) -> tuple[bool, str]:
    """Check the declared version and load the architecture-specific extension."""

    try:
        package_spec = importlib.util.find_spec("sgl_kernel")
    except (ImportError, ValueError) as error:
        return False, f"sgl-kernel package discovery failed: {error}"
    if package_spec is None:
        return False, "sgl-kernel is not installed"
    try:
        version = importlib.metadata.version("sgl-kernel")
    except importlib.metadata.PackageNotFoundError:
        return False, "sgl-kernel package metadata is unavailable"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    parsed = tuple(map(int, match.groups())) if match else None
    if parsed is None or not _MIN_VERSION <= parsed < _MAX_VERSION:
        return False, f"requires sgl-kernel>=0.3.21,<0.4, got {version}"
    try:
        importlib.import_module("sgl_kernel")
    except Exception as error:
        return False, (
            f"sgl-kernel {version} {feature} failed to load: "
            f"{type(error).__name__}: {error}"
        )
    return True, f"sgl-kernel {version} {feature} is available"
