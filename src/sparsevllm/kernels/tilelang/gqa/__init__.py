# SPDX-License-Identifier: Apache-2.0
"""TileLang GQA kernels with lightweight dependency probing."""

from importlib import import_module

from sparsevllm.kernels.tilelang.gqa.runtime import tilelang_gqa_device_support

__all__ = [
    "gqa_paged_prefill_attention_tilelang",
    "gqa_prefill_score_tilelang",
    "tilelang_gqa_device_support",
]


def __getattr__(name: str):
    if name in {
        "gqa_paged_prefill_attention_tilelang",
        "gqa_prefill_score_tilelang",
    }:
        prefill = import_module("sparsevllm.kernels.tilelang.gqa.prefill")
        return getattr(prefill, name)
    raise AttributeError(name)
