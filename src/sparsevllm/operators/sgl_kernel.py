from __future__ import annotations

import importlib.metadata
import importlib.util
import re


_MIN_VERSION = (0, 3, 21)
_MAX_VERSION = (0, 4, 0)


def sgl_kernel_support(feature: str) -> tuple[bool, str]:
    """Check the sgl-kernel API range declared by pyproject.toml."""

    if importlib.util.find_spec("sgl_kernel") is None:
        return False, "sgl-kernel is not installed"
    try:
        version = importlib.metadata.version("sgl-kernel")
    except importlib.metadata.PackageNotFoundError:
        return False, "sgl-kernel package metadata is unavailable"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    parsed = tuple(map(int, match.groups())) if match else None
    if parsed is None or not _MIN_VERSION <= parsed < _MAX_VERSION:
        return False, f"requires sgl-kernel>=0.3.21,<0.4, got {version}"
    return True, f"sgl-kernel {version} {feature} is available"
