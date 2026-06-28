from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sparsevllm.engine.expert_parallel.deepep_v2 import destroy_cached_buffers
from sparsevllm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from sparsevllm.utils.parallel_context import ParallelContext, parallel_context_scope


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qwen3-MoE DeepEP v2 dispatch/combine.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser.parse_args()


def tiny_config(args: argparse.Namespace) -> SimpleNamespace:
    if int(args.hidden_size) % 256 != 0:
        raise ValueError("DeepEP v2 smoke requires hidden_size to be a multiple of 256.")
    return SimpleNamespace(
        hidden_size=int(args.hidden_size),
        intermediate_size=int(args.intermediate_size) * 2,
        moe_intermediate_size=int(args.intermediate_size),
        hidden_act="silu",
        num_experts=int(args.num_experts),
        num_experts_per_tok=int(args.top_k),
        norm_topk_prob=True,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        head_dim=4,
        rope_theta=1000000,
        rope_scaling=None,
        mlp_chunk_size=16,
        mlp_only_layers=[],
        decoder_sparse_step=1,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=False,
        expert_parallel_backend="deepep_v2",
    )


def build_router_weight(num_experts: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    if num_experts != 4 or hidden_size < 4:
        raise ValueError("This smoke uses a fixed 4-expert routing pattern and hidden_size >= 4.")
    weight = torch.zeros((num_experts, hidden_size), dtype=torch.bfloat16, device=device)
    # Tokens on dimensions 0/1/2/3 route to expert pairs split across the two ranks.
    weight[0, 0] = 6.0
    weight[2, 0] = 5.0
    weight[1, 1] = 6.0
    weight[3, 1] = 5.0
    weight[2, 2] = 6.0
    weight[0, 2] = 5.0
    weight[3, 3] = 6.0
    weight[1, 3] = 5.0
    return weight


def build_hidden_states(rank: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    hidden = torch.zeros((3, hidden_size), dtype=torch.bfloat16, device=device)
    dims = [0, 1, 2] if rank == 0 else [2, 3, 0]
    for row, dim in enumerate(dims):
        hidden[row, dim] = 1.0
        hidden[row, 4 + row] = 0.25
    return hidden


def build_expert_weights(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up = torch.arange(
        num_experts * 2 * intermediate_size * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(num_experts, 2 * intermediate_size, hidden_size)
    gate_up = (gate_up / 200.0 - 0.5).to(torch.bfloat16)
    down = torch.arange(
        num_experts * hidden_size * intermediate_size,
        dtype=torch.float32,
        device=device,
    ).reshape(num_experts, hidden_size, intermediate_size)
    down = (down / 200.0 - 0.25).to(torch.bfloat16)
    return gate_up, down


def reference_moe(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    router_logits = F.linear(hidden_states, router_weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(router_probs, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)

    output = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        token = hidden_states[token_idx : token_idx + 1]
        for top_k_pos in range(top_k):
            expert_idx = int(selected_experts[token_idx, top_k_pos].item())
            gate, up = F.linear(token, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, down_proj[expert_idx])
            output[token_idx] += expert_out.squeeze(0) * routing_weights[token_idx, top_k_pos]
    return output


def main() -> None:
    args = parse_args()
    started_at = time.time()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError(f"DeepEP v2 smoke expects world_size=2, got {world_size}.")
    if int(args.num_experts) % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size.")

    import deep_ep  # type: ignore

    cfg = tiny_config(args)
    context = ParallelContext(
        global_rank=rank,
        global_world_size=world_size,
        tp_size=1,
        tp_rank=0,
        ep_size=world_size,
        ep_rank=rank,
        local_rank=local_rank,
        ep_group=dist.group.WORLD,
        dp_size=world_size,
        dp_rank=rank,
        dp_group=dist.group.WORLD,
        overlap_data_parallel=True,
    )

    router_weight = build_router_weight(cfg.num_experts, cfg.hidden_size, device)
    gate_up_proj, down_proj = build_expert_weights(
        cfg.num_experts,
        cfg.moe_intermediate_size,
        cfg.hidden_size,
        device,
    )
    local_experts = cfg.num_experts // world_size
    local_start = rank * local_experts
    hidden_states = build_hidden_states(rank, cfg.hidden_size, device)

    with parallel_context_scope(context):
        block = Qwen3MoeSparseMoeBlock(cfg).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            block.gate.weight.copy_(router_weight)
            block.experts.gate_up_proj.copy_(gate_up_proj[local_start : local_start + local_experts])
            block.experts.down_proj.copy_(down_proj[local_start : local_start + local_experts])
        actual = block(hidden_states)

    expected = reference_moe(
        hidden_states,
        router_weight,
        gate_up_proj,
        down_proj,
        cfg.num_experts_per_tok,
    )
    diff = (actual.float() - expected.float()).abs()
    rank_payload = {
        "rank": rank,
        "local_rank": local_rank,
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "actual": actual.detach().float().cpu().tolist(),
        "expected": expected.detach().float().cpu().tolist(),
    }
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(rank_payload, gathered, dst=0)
    failed = torch.tensor(
        [0 if torch.allclose(actual.float(), expected.float(), rtol=float(args.rtol), atol=float(args.atol)) else 1],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)

    if rank == 0:
        result = {
            "status": "success" if int(failed.item()) == 0 else "metric_failed",
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "world_size": world_size,
            "deep_ep_version": getattr(deep_ep, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl_version": str(torch.cuda.nccl.version()),
            "nvidia_nccl_cu13": package_version("nvidia-nccl-cu13"),
            "nvidia_nccl_cu12": package_version("nvidia-nccl-cu12"),
            "nvidia_nvshmem_cu13": package_version("nvidia-nvshmem-cu13"),
            "nvidia_nvshmem_cu12": package_version("nvidia-nvshmem-cu12"),
            "ep_disable_gin": os.environ.get("EP_DISABLE_GIN"),
            "ep_jit_cache_dir": os.environ.get("EP_JIT_CACHE_DIR"),
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "ranks": gathered,
            "elapsed_s": time.time() - started_at,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))

    destroy_cached_buffers()
    dist.barrier()
    dist.destroy_process_group()
    if int(failed.item()) != 0:
        raise AssertionError("Qwen3-MoE DeepEP v2 smoke output differs from local reference.")


if __name__ == "__main__":
    main()
