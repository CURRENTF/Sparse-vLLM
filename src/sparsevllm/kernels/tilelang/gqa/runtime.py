# SPDX-License-Identifier: Apache-2.0
"""Lightweight support checks for the SM90 TileLang GQA kernels."""

from importlib import metadata

import torch


_VALIDATED_TILELANG_VERSION = "0.1.9"
_VALIDATED_TVM_FFI_VERSION = "0.1.10"


def tilelang_gqa_device_support(device_index: int | None = None) -> tuple[bool, str]:
    """Check dependencies and device support without importing TileLang."""

    try:
        tilelang_version = metadata.version("tilelang")
    except metadata.PackageNotFoundError:
        return False, "tilelang is not installed"
    if tilelang_version != _VALIDATED_TILELANG_VERSION:
        return False, (
            f"requires tilelang == {_VALIDATED_TILELANG_VERSION}, "
            f"got {tilelang_version}"
        )

    try:
        tvm_ffi_version = metadata.version("apache-tvm-ffi")
    except metadata.PackageNotFoundError:
        return False, "apache-tvm-ffi is not installed"
    if tvm_ffi_version != _VALIDATED_TVM_FFI_VERSION:
        return False, (
            f"requires apache-tvm-ffi == {_VALIDATED_TVM_FFI_VERSION}, "
            f"got {tvm_ffi_version}"
        )

    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if device_index is None:
        device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(int(device_index))
    if (props.major, props.minor) != (9, 0):
        return False, f"requires Hopper SM90 architecture, got SM{props.major}{props.minor}"

    return True, (
        f"tilelang {tilelang_version}, apache-tvm-ffi {tvm_ffi_version}, SM90"
    )


__all__ = ["tilelang_gqa_device_support"]
