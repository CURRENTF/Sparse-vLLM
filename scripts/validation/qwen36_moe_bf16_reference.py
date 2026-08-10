from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sparsevllm.operators.moe import MoeOpSpec, resolve_moe_provider
from sparsevllm.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)


WEIGHT_PREFIX = "model.language_model.layers.0.mlp."
WEIGHT_NAMES = (
    "gate.weight",
    "experts.gate_up_proj",
    "experts.down_proj",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "shared_expert_gate.weight",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _load_layer_weights(model_path: Path) -> dict[str, torch.Tensor]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    loaded: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for local_name in WEIGHT_NAMES:
        checkpoint_name = WEIGHT_PREFIX + local_name
        shard = weight_map.get(checkpoint_name)
        if shard is None:
            raise KeyError(f"Missing checkpoint tensor {checkpoint_name!r}.")
        by_shard.setdefault(shard, []).append((checkpoint_name, local_name))
    for shard, names in by_shard.items():
        shard_path = model_path / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for checkpoint_name, local_name in names:
                tensor = handle.get_tensor(checkpoint_name)
                if tensor.dtype != torch.bfloat16:
                    raise TypeError(
                        f"{checkpoint_name} must be BF16, got {tensor.dtype}."
                    )
                loaded[local_name] = tensor
    return loaded


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value)
    float_value = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(finite.all().item()),
        "min": float(float_value.min().item()),
        "max": float(float_value.max().item()),
        "mean": float(float_value.mean().item()),
    }


def _error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1.0e-6)
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "max_rel": float((difference / denominator).max().item()),
    }


def _record_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    metrics = _error_metrics(actual, expected)
    passed = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
    return {
        "component": name,
        "status": "success" if passed else "metric_failed",
        "atol": float(atol),
        "rtol": float(rtol),
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3.6 MoE BF16 checkpoint math with Transformers."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError(f"--tokens must be positive, got {args.tokens}.")
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires one CUDA device.")

    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    index_path = model_path / "model.safetensors.index.json"
    config_path = model_path / "config.json"
    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "model": str(model_path),
        "model_config_sha256": _sha256(config_path),
        "model_index_sha256": _sha256(index_path),
        "seed": int(args.seed),
        "tokens": int(args.tokens),
        "dtype": "torch.bfloat16",
        "device": torch.cuda.get_device_name(0),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "tolerances": {
            "routing_weights": {"atol": 0.004, "rtol": 0.004},
            "routed_experts": {"atol": 0.05, "rtol": 0.05},
            "shared_expert": {"atol": 0.02, "rtol": 0.02},
            "moe_output": {"atol": 0.05, "rtol": 0.05},
        },
        "requested_moe_provider": os.getenv("SPARSEVLLM_MOE_PROVIDER", "auto"),
        "requested_moe_router_provider": os.getenv(
            "SPARSEVLLM_MOE_ROUTER_PROVIDER", "auto"
        ),
    }

    weights = _load_layer_weights(model_path)
    outer_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = outer_config.text_config
    if config.torch_dtype != torch.bfloat16:
        raise TypeError(f"Reference config must be BF16, got {config.torch_dtype}.")
    num_experts = int(config.num_experts)
    expert_intermediate_size = int(weights["experts.gate_up_proj"].shape[1] // 2)
    moe_spec = MoeOpSpec(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=int(config.hidden_size),
        intermediate_size=expert_intermediate_size,
        top_k=int(config.num_experts_per_tok),
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        ep_size=1,
        cuda_graph=False,
        tp_size=1,
        routing_method="softmax",
    )
    moe_provider = resolve_moe_provider(moe_spec, device_index=0)
    router_spec = MoeRouterOpSpec(
        num_experts=num_experts,
        top_k=int(config.num_experts_per_tok),
        activation_dtype=torch.bfloat16,
        norm_topk_prob=True,
        cuda_graph=False,
    )
    router_provider = resolve_moe_router_provider(router_spec, device_index=0)
    run_info["resolved_moe_provider"] = moe_provider.name
    run_info["resolved_moe_router_provider"] = router_provider.name
    _write_json(output_dir / "run_info.json", run_info)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reference = Qwen3_5MoeSparseMoeBlock(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    missing, unexpected = reference.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Transformers MoE weight mismatch: missing={missing}, unexpected={unexpected}."
        )
    reference = reference.to(device="cuda", dtype=torch.bfloat16).eval()
    weights = {
        name: tensor.to(device="cuda", non_blocking=False)
        for name, tensor in weights.items()
    }

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hidden = torch.randn(
        (args.tokens, int(config.hidden_size)),
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        reference_logits, reference_routing_weights, reference_ids = reference.gate(
            hidden
        )
        reference_routed = reference.experts(
            hidden, reference_ids, reference_routing_weights
        )
        reference_shared = reference.shared_expert(hidden)
        reference_shared = torch.sigmoid(reference.shared_expert_gate(hidden)) * reference_shared
        reference_output = reference_routed + reference_shared

        actual_logits = F.linear(hidden, weights["gate.weight"])
        actual_routing_weights, actual_ids = router_provider.run(
            router_spec,
            actual_logits,
        )
        actual_routed = moe_provider.run(
            moe_spec,
            hidden,
            actual_ids,
            actual_routing_weights,
            weights["experts.gate_up_proj"],
            weights["experts.down_proj"],
            None,
            None,
            local_expert_start=0,
            ep_rank=0,
        )
        shared_gate = F.linear(hidden, weights["shared_expert.gate_proj.weight"])
        shared_up = F.linear(hidden, weights["shared_expert.up_proj.weight"])
        actual_shared = F.linear(
            F.silu(shared_gate) * shared_up,
            weights["shared_expert.down_proj.weight"],
        )
        actual_shared *= torch.sigmoid(
            F.linear(hidden, weights["shared_expert_gate.weight"])
        )
        actual_output = actual_routed + actual_shared
        torch.cuda.synchronize()

    records = [
        _record_close(
            "router_logits",
            actual_logits,
            reference_logits,
            atol=0.0,
            rtol=0.0,
        ),
        {
            "component": "routing_ids",
            "status": "success" if torch.equal(actual_ids, reference_ids.to(torch.int32)) else "metric_failed",
            "mismatch_count": int((actual_ids != reference_ids.to(torch.int32)).sum().item()),
        },
        _record_close(
            "routing_weights",
            actual_routing_weights,
            reference_routing_weights,
            atol=0.004,
            rtol=0.004,
        ),
        _record_close(
            "routed_experts",
            actual_routed,
            reference_routed,
            atol=0.05,
            rtol=0.05,
        ),
        _record_close(
            "shared_expert",
            actual_shared,
            reference_shared,
            atol=0.02,
            rtol=0.02,
        ),
        _record_close(
            "moe_output",
            actual_output,
            reference_output,
            atol=0.05,
            rtol=0.05,
        ),
    ]
    raw_outputs = {
        "hidden": hidden.cpu(),
        "reference_logits": reference_logits.cpu(),
        "actual_logits": actual_logits.cpu(),
        "reference_ids": reference_ids.cpu(),
        "actual_ids": actual_ids.cpu(),
        "reference_routing_weights": reference_routing_weights.cpu(),
        "actual_routing_weights": actual_routing_weights.cpu(),
        "reference_routed": reference_routed.cpu(),
        "actual_routed": actual_routed.cpu(),
        "reference_shared": reference_shared.cpu(),
        "actual_shared": actual_shared.cpu(),
        "reference_output": reference_output.cpu(),
        "actual_output": actual_output.cpu(),
    }
    torch.save(raw_outputs, output_dir / "raw_outputs.pt")
    parsed = {
        name: _tensor_summary(tensor)
        for name, tensor in raw_outputs.items()
    }
    _write_json(output_dir / "parsed_outputs.json", parsed)
    with (output_dir / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    failed = [record for record in records if record["status"] != "success"]
    aggregate = {
        "status": "success" if not failed else "metric_failed",
        "num_checks": len(records),
        "success_checks": len(records) - len(failed),
        "failed_checks": len(failed),
        "records": records,
    }
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
