# Vendored Qwen3.5 GDN Kernels

This directory contains Qwen3Next/Qwen3.5 Gated DeltaNet Triton kernels copied
from LightLLM:

- Source: https://github.com/ModelTC/lightllm
- License: Apache-2.0

Some copied files also carry upstream SPDX notices for vLLM and
flash-linear-attention. Keep those headers intact when editing the vendored
kernel code.

Sparse-vLLM vendors these files so the Qwen3.5/Qwen3.6 mixed runtime can be
customized locally without depending on a matching LightLLM package install.

Install their CUDA-specific runtime dependencies with
`pip install -e ".[qwen35]"`.
