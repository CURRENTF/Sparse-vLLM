"""Lightweight compatibility checks for repository-owned TileLang kernels."""

from __future__ import annotations

from importlib import metadata


_SUPPORTED_VERSION_PAIR = ("0.1.9", "0.1.10")


def tilelang_dependency_support() -> tuple[bool, str]:
    """Check the validated dependency pair without importing the compiler."""

    try:
        tilelang_version = metadata.version("tilelang")
    except metadata.PackageNotFoundError:
        return False, "tilelang is not installed"
    try:
        tvm_ffi_version = metadata.version("apache-tvm-ffi")
    except metadata.PackageNotFoundError:
        return False, "apache-tvm-ffi is not installed"
    installed_pair = (tilelang_version, tvm_ffi_version)
    if installed_pair != _SUPPORTED_VERSION_PAIR:
        return False, (
            "requires the validated TileLang dependency pair "
            f"tilelang=={_SUPPORTED_VERSION_PAIR[0]}, "
            f"apache-tvm-ffi=={_SUPPORTED_VERSION_PAIR[1]}; "
            f"got tilelang=={tilelang_version}, apache-tvm-ffi=={tvm_ffi_version}"
        )

    return True, (
        f"tilelang {tilelang_version}, apache-tvm-ffi {tvm_ffi_version}"
    )


__all__ = ["tilelang_dependency_support"]
