from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re

_MIN_VERSION = (0, 4, 5)
_MAX_VERSION = (0, 4, 6)
_DISTRIBUTION = "sglang-kernel"


def sgl_kernel_support(feature: str) -> tuple[bool, str]:
    """Check the declared version and load the architecture-specific extension."""

    try:
        package_spec = importlib.util.find_spec("sgl_kernel")
    except (ImportError, ValueError) as error:
        return False, f"{_DISTRIBUTION} package discovery failed: {error}"
    if package_spec is None:
        return False, f"{_DISTRIBUTION} is not installed"
    try:
        version = importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return False, f"{_DISTRIBUTION} package metadata is unavailable"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    parsed = tuple(map(int, match.groups())) if match else None
    if parsed is None or not _MIN_VERSION <= parsed < _MAX_VERSION:
        return False, f"requires {_DISTRIBUTION}>=0.4.5,<0.4.6, got {version}"
    try:
        importlib.import_module("sgl_kernel")
    except Exception as error:
        return False, (
            f"{_DISTRIBUTION} {version} {feature} failed to load: "
            f"{type(error).__name__}: {error}"
        )
    return True, f"{_DISTRIBUTION} {version} {feature} is available"
