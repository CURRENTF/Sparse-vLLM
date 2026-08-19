# SPDX-License-Identifier: Apache-2.0
"""Theoretical FLOPs, Memory Bandwidth, MFU, and MBU Calculator for LLM Serving."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GPUHardwareProfile:
    name: str
    peak_tflops_bf16: float  # TFLOPS (FP16/BF16 Tensor Core, Dense)
    peak_bandwidth_tbs: float  # TB/s (HBM / GDDR)
    tdp_watts: float
    description: str = ""


# Standard GPU Hardware Database
GPU_HARDWARE_REGISTRY: dict[str, GPUHardwareProfile] = {
    "h100_sxm": GPUHardwareProfile(
        name="NVIDIA H100 80GB HBM3 (SXM5)",
        peak_tflops_bf16=989.0,
        peak_bandwidth_tbs=3.35,
        tdp_watts=700.0,
        description="Hopper SM90 SXM5 80GB HBM3",
    ),
    "h100_nvl": GPUHardwareProfile(
        name="NVIDIA H100 NVL 80GB (Dual-GPU)",
        peak_tflops_bf16=835.0,
        peak_bandwidth_tbs=3.90,
        tdp_watts=400.0,
        description="Hopper SM90 NVL 80GB HBM3",
    ),
    "h100_pcie": GPUHardwareProfile(
        name="NVIDIA H100 80GB PCIe",
        peak_tflops_bf16=756.0,
        peak_bandwidth_tbs=2.00,
        tdp_watts=350.0,
        description="Hopper SM90 PCIe 80GB HBM2e",
    ),
    "h800_sxm": GPUHardwareProfile(
        name="NVIDIA H800 SXM5",
        peak_tflops_bf16=989.0,
        peak_bandwidth_tbs=3.35,
        tdp_watts=700.0,
        description="Hopper SM90 H800 SXM5",
    ),
    "h20_sxm": GPUHardwareProfile(
        name="NVIDIA H20 96GB SXM",
        peak_tflops_bf16=148.0,
        peak_bandwidth_tbs=4.00,
        tdp_watts=400.0,
        description="Hopper SM90 H20 SXM 96GB",
    ),
    "a100_sxm": GPUHardwareProfile(
        name="NVIDIA A100 80GB SXM4",
        peak_tflops_bf16=312.0,
        peak_bandwidth_tbs=2.039,
        tdp_watts=400.0,
        description="Ampere SM80 SXM4 80GB HBM2e",
    ),
    "a100_pcie": GPUHardwareProfile(
        name="NVIDIA A100 80GB PCIe",
        peak_tflops_bf16=312.0,
        peak_bandwidth_tbs=1.555,
        tdp_watts=250.0,
        description="Ampere SM80 PCIe 80GB HBM2e",
    ),
}


def detect_gpu_hardware(device_name: str | None = None) -> GPUHardwareProfile:
    """Detect or match GPU hardware profile from device string or CUDA device."""
    if device_name is None:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    if not device_name:
        return GPU_HARDWARE_REGISTRY["h100_sxm"]

    d_lower = device_name.lower()
    if "h100" in d_lower:
        if "nvl" in d_lower:
            return GPU_HARDWARE_REGISTRY["h100_nvl"]
        elif "pcie" in d_lower:
            return GPU_HARDWARE_REGISTRY["h100_pcie"]
        else:
            return GPU_HARDWARE_REGISTRY["h100_sxm"]
    elif "h800" in d_lower:
        return GPU_HARDWARE_REGISTRY["h800_sxm"]
    elif "h20" in d_lower:
        return GPU_HARDWARE_REGISTRY["h20_sxm"]
    elif "a100" in d_lower:
        if "pcie" in d_lower:
            return GPU_HARDWARE_REGISTRY["a100_pcie"]
        else:
            return GPU_HARDWARE_REGISTRY["a100_sxm"]

    return GPU_HARDWARE_REGISTRY["h100_sxm"]


@dataclass
class ModelArchitectureSpecs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    moe_intermediate_size: int = 0
    dense_intermediate_size: int = 0
    bytes_per_param: int = 2  # default BF16

    @classmethod
    def from_config_dict(cls, cfg: dict[str, Any], bytes_per_param: int = 2) -> ModelArchitectureSpecs:
        hidden_size = int(cfg.get("hidden_size", 2048))
        num_hidden_layers = int(cfg.get("num_hidden_layers", 48))
        num_attention_heads = int(cfg.get("num_attention_heads", 16))
        num_key_value_heads = int(cfg.get("num_key_value_heads", num_attention_heads))
        head_dim = int(cfg.get("head_dim", hidden_size // num_attention_heads))
        vocab_size = int(cfg.get("vocab_size", 152064))

        # MoE parameters
        num_experts = int(cfg.get("num_experts", cfg.get("n_routed_experts", 1)))
        num_experts_per_tok = int(cfg.get("num_experts_per_tok", cfg.get("num_experts_per_token", 1)))
        is_moe = num_experts > 1

        moe_intermediate_size = int(
            cfg.get("moe_intermediate_size", cfg.get("expert_intermediate_size", 0))
        )
        dense_intermediate_size = int(cfg.get("intermediate_size", 0))

        if is_moe and moe_intermediate_size == 0:
            moe_intermediate_size = dense_intermediate_size

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            is_moe=is_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            dense_intermediate_size=dense_intermediate_size,
            bytes_per_param=bytes_per_param,
        )

    @classmethod
    def from_model_path_or_name(cls, model_path: str | Path) -> ModelArchitectureSpecs:
        path = Path(model_path)
        config_file = path / "config.json" if path.is_dir() else None

        if config_file and config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_config_dict(data)

        # Try transformers AutoConfig if model_path is a huggingface name or dir
        try:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            return cls.from_config_dict(hf_cfg.to_dict())
        except Exception:
            pass

        # Fallback to predefined Qwen3-30B-A3B if string matches
        name = str(model_path).lower()
        if "qwen3-30b" in name or "qwen3_30b" in name:
            # Qwen3-30B-A3B canonical specs
            return cls(
                hidden_size=2048,
                num_hidden_layers=48,
                num_attention_heads=32,
                num_key_value_heads=4,
                head_dim=128,
                vocab_size=152064,
                is_moe=True,
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=768,
                dense_intermediate_size=0,
                bytes_per_param=2,
            )
        elif "qwen3-8b" in name or "qwen3_8b" in name:
            return cls(
                hidden_size=4096,
                num_hidden_layers=36,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                vocab_size=152064,
                is_moe=False,
                dense_intermediate_size=11008,
                bytes_per_param=2,
            )
        elif "qwen2.5-7b" in name or "qwen25_7b" in name:
            return cls(
                hidden_size=3584,
                num_hidden_layers=28,
                num_attention_heads=28,
                num_key_value_heads=4,
                head_dim=128,
                vocab_size=152064,
                is_moe=False,
                dense_intermediate_size=18944,
                bytes_per_param=2,
            )

        # Default fallback
        return cls(
            hidden_size=2048,
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            vocab_size=152064,
            is_moe=True,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            dense_intermediate_size=0,
            bytes_per_param=2,
        )

    def calculate_prefill_flops(self, prompt_len: int, batch_size: int = 1) -> float:
        """Calculate theoretical FLOPs for prefilling a prompt of length N.

        FLOPs Breakdown per layer:
          - QKV Projection: 2 * N * H * (d_q + 2*d_kv)
          - Attention Matrix & Score: 2 * N^2 * H (causal attention triangle: ~2 * N^2 * H)
          - Out Projection: 2 * N * H * H
          - MLP:
            - MoE: 2 * N * H * num_experts (router) + 2 * 3 * N * H * intermediate * k_active (SwiGLU)
            - Dense: 2 * 3 * N * H * intermediate (SwiGLU)
          - LM Head: 2 * N * H * vocab_size
        """
        N = float(prompt_len)
        B = float(batch_size)
        H = float(self.hidden_size)
        L = float(self.num_hidden_layers)
        d_q = float(self.num_attention_heads * self.head_dim)
        d_kv = float(self.num_key_value_heads * self.head_dim)

        # 1. QKV Linear Proj
        qkv_flops = 2.0 * N * H * (d_q + 2.0 * d_kv)
        # 2. Causal Attention Matrix computation (QK^T + Softmax*V)
        # QK^T is N*N*d_q, Softmax*V is N*N*d_q -> each 2*N^2*d_q FLOPs / 2 (causal) = 2*N^2*d_q
        attn_flops = 2.0 * (N ** 2) * d_q
        # 3. Output Projection
        out_flops = 2.0 * N * d_q * H

        # 4. MLP computation (SwiGLU = 3 GEMMs: gate, up, down)
        if self.is_moe:
            router_flops = 2.0 * N * H * float(self.num_experts)
            inter_size = float(self.moe_intermediate_size)
            k_act = float(self.num_experts_per_tok)
            moe_mlp_flops = 2.0 * 3.0 * N * H * inter_size * k_act
            mlp_flops = router_flops + moe_mlp_flops
        else:
            inter_size = float(self.dense_intermediate_size)
            mlp_flops = 2.0 * 3.0 * N * H * inter_size

        layer_flops = qkv_flops + attn_flops + out_flops + mlp_flops
        total_layers_flops = L * layer_flops

        # 5. LM Head (calculated on last token or all tokens depending on loss vs inference)
        # For serving inference, LM Head is only projected on the final prompt token to sample the 1st token
        lm_head_flops = 2.0 * 1.0 * H * float(self.vocab_size)

        return (total_layers_flops + lm_head_flops) * B

    def calculate_decode_step_bytes(
        self,
        context_len: int,
        sparse_budget: int | None = None,
        batch_size: int = 1,
    ) -> float:
        """Calculate theoretical memory bytes accessed during a single decode step.

        Access Breakdown:
          1. Active Model Weights read:
             - Attention weights (QKV + Out): L * [H*(d_q + 2*d_kv) + d_q*H] * bytes_per_param
             - LayerNorms / RMSNorms: ~minor
             - MLP Active Weights:
               - If MoE: L * [Router + k_active * 3 * H * moe_inter] * bytes_per_param
               - If Dense: L * [3 * H * dense_inter] * bytes_per_param
             - LM Head / Embedding: H * vocab_size * bytes_per_param (or amortized)
          2. KV Cache Read:
             - 2 * effective_context_len * L * num_kv_heads * head_dim * bytes_per_param * batch_size
          3. KV Cache Write (new generated token):
             - 2 * 1 * L * num_kv_heads * head_dim * bytes_per_param * batch_size
        """
        B = float(batch_size)
        H = float(self.hidden_size)
        L = float(self.num_hidden_layers)
        d_q = float(self.num_attention_heads * self.head_dim)
        d_kv = float(self.num_key_value_heads * self.head_dim)
        bpp = float(self.bytes_per_param)

        # 1. Weights Memory (read per step in decode)
        attn_params = L * (H * (d_q + 2.0 * d_kv) + d_q * H)
        if self.is_moe:
            router_params = L * (H * float(self.num_experts))
            active_expert_params = L * float(self.num_experts_per_tok) * (3.0 * H * float(self.moe_intermediate_size))
            mlp_active_params = router_params + active_expert_params
        else:
            mlp_active_params = L * (3.0 * H * float(self.dense_intermediate_size))

        lm_head_params = H * float(self.vocab_size)
        total_active_params = attn_params + mlp_active_params + lm_head_params
        weights_bytes = total_active_params * bpp

        # 2. KV Cache Read & Write
        effective_ctx = float(sparse_budget) if (sparse_budget is not None and sparse_budget < context_len) else float(context_len)
        kv_bytes_per_token_total = 2.0 * L * d_kv * bpp

        kv_read_bytes = effective_ctx * kv_bytes_per_token_total * B
        kv_write_bytes = 1.0 * kv_bytes_per_token_total * B

        return weights_bytes + kv_read_bytes + kv_write_bytes


def calculate_mfu(
    flops: float,
    duration_seconds: float,
    tp_size: int,
    peak_tflops_bf16: float,
) -> float:
    """Calculate Model FLOPs Utilization (MFU) in percent."""
    if duration_seconds <= 0 or peak_tflops_bf16 <= 0 or tp_size <= 0:
        return 0.0
    achieved_tflops = (flops / duration_seconds) / 1e12
    total_peak_tflops = float(tp_size) * peak_tflops_bf16
    return (achieved_tflops / total_peak_tflops) * 100.0


def calculate_mbu(
    bytes_accessed: float,
    duration_seconds: float,
    tp_size: int,
    peak_bandwidth_tbs: float,
) -> float:
    """Calculate Model Bandwidth Utilization (MBU) in percent."""
    if duration_seconds <= 0 or peak_bandwidth_tbs <= 0 or tp_size <= 0:
        return 0.0
    achieved_tbs = (bytes_accessed / duration_seconds) / 1e12
    total_peak_tbs = float(tp_size) * peak_bandwidth_tbs
    return (achieved_tbs / total_peak_tbs) * 100.0
