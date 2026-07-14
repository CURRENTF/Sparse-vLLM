from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import sparsevllm.triton_kernel.moe as triton_moe


DEFAULT_CONFIGS = (
    {
        "name": "m16n64k64w4s3",
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "name": "m16n128k64w8s3",
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    {
        "name": "m16n128k32w4s4",
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "name": "m32n64k64w8s3",
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_tokens(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"--tokens must contain positive integers, got {value!r}.")
    if len(set(values)) != len(values):
        raise ValueError(f"--tokens contains duplicates: {values}.")
    return values


def _load_configs(path: str | None) -> tuple[dict[str, int | str], ...]:
    if path is None:
        configs = [dict(config) for config in DEFAULT_CONFIGS]
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("--configs must point to a non-empty JSON list.")
        configs = payload
    required = {
        "name",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
    names = set()
    for config in configs:
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(
                f"Each kernel config must contain exactly {sorted(required)}, got {config}."
            )
        name = str(config["name"])
        if not name or name in names:
            raise ValueError(f"Kernel config names must be non-empty and unique, got {name!r}.")
        names.add(name)
        for key in required - {"name"}:
            config[key] = int(config[key])
            if int(config[key]) <= 0:
                raise ValueError(f"Kernel config {name!r} has non-positive {key}.")
        if int(config["BLOCK_SIZE_M"]) not in {16, 32, 64}:
            raise ValueError(f"Kernel config {name!r} has unsupported BLOCK_SIZE_M.")
    return tuple(configs)


def _oracle(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_start: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.zeros(
        hidden_states.shape,
        dtype=output_dtype,
        device=hidden_states.device,
    )
    for local_expert_id in range(int(w13_weight.shape[0])):
        global_expert_id = local_expert_start + local_expert_id
        token_ids, topk_slots = torch.where(topk_ids == global_expert_id)
        if token_ids.numel() == 0:
            continue
        gate_up = F.linear(hidden_states[token_ids], w13_weight[local_expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, w2_weight[local_expert_id])
        expert_output *= topk_weights[token_ids, topk_slots, None]
        output.index_add_(0, token_ids, expert_output.to(output.dtype))
    return output


def _time_ms(function, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / iterations


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item())
    denominator = expected.float().abs().clamp_min(1.0e-5)
    max_rel = float((difference / denominator).max().item())
    return max_abs, max_rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and validate generic Triton MoE kernels.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokens", default="1,16,128,512")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--oracle-iterations", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--configs", default=None)
    parser.add_argument(
        "--output-dtype",
        choices=("bfloat16", "float32", "float64"),
        default="bfloat16",
        help="Final TopK-sum dtype. Qwen3MoE EP uses float64 before its final cast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = _parse_tokens(args.tokens)
    configs = _load_configs(args.configs)
    for name in (
        "num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
        "ep_size",
        "warmup",
        "iterations",
        "oracle_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.ep_rank < args.ep_size:
        raise ValueError(f"ep_rank={args.ep_rank} is invalid for ep_size={args.ep_size}.")
    if args.num_experts % args.ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size.")
    if not 1 <= args.top_k <= args.num_experts:
        raise ValueError("top_k must be in [1, num_experts].")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton MoE benchmarking requires CUDA.")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    output_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.output_dtype]
    num_local_experts = args.num_experts // args.ep_size
    local_expert_start = args.ep_rank * num_local_experts
    w13_weight = (
        torch.randn(
            num_local_experts,
            2 * args.intermediate_size,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )
    w2_weight = (
        torch.randn(
            num_local_experts,
            args.hidden_size,
            args.intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )
    router_weight = (
        torch.randn(
            args.num_experts,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    raw_outputs: dict[str, object] = {}
    parsed_outputs = []
    per_case_results = []
    failed = False
    original_config_resolver = triton_moe._gemm_launch_config
    try:
        for num_tokens in tokens:
            hidden_states = torch.randn(
                num_tokens,
                args.hidden_size,
                device=device,
                dtype=dtype,
            )
            router_logits = F.linear(hidden_states, router_weight)
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(router_probs, args.top_k, dim=-1)
            topk_weights = (
                topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            ).to(dtype)
            expected = _oracle(
                hidden_states,
                w13_weight,
                w2_weight,
                topk_ids,
                topk_weights,
                local_expert_start,
                output_dtype,
            )
            oracle_ms = _time_ms(
                lambda: _oracle(
                    hidden_states,
                    w13_weight,
                    w2_weight,
                    topk_ids,
                    topk_weights,
                    local_expert_start,
                    output_dtype,
                ),
                warmup=1,
                iterations=args.oracle_iterations,
            )
            raw_outputs[str(num_tokens)] = {
                "hidden_states": hidden_states.detach().cpu(),
                "topk_ids": topk_ids.detach().cpu(),
                "topk_weights": topk_weights.detach().cpu(),
                "oracle_output": expected.detach().cpu(),
                "triton_outputs": {},
            }

            for candidate in configs:
                name = str(candidate["name"])
                launch_config = {
                    key: int(value)
                    for key, value in candidate.items()
                    if key != "name"
                }
                triton_moe._gemm_launch_config = (
                    lambda _num_assignments, _output_size, config=launch_config: dict(config)
                )
                case = {
                    "case_id": len(per_case_results),
                    "tokens": num_tokens,
                    "config": dict(candidate),
                    "oracle_ms": oracle_ms,
                    "status": "model_failed",
                }
                try:
                    first_call_started = time.perf_counter()
                    actual = triton_moe.fused_moe(
                        hidden_states,
                        w13_weight,
                        w2_weight,
                        topk_ids,
                        topk_weights,
                        num_experts=args.num_experts,
                        local_expert_start=local_expert_start,
                        output_dtype=output_dtype,
                    )
                    torch.cuda.synchronize()
                    first_call_ms = 1000.0 * (time.perf_counter() - first_call_started)
                    max_abs, max_rel = _max_errors(actual, expected)
                    matches = bool(
                        torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
                    )
                    latency_ms = _time_ms(
                        lambda: triton_moe.fused_moe(
                            hidden_states,
                            w13_weight,
                            w2_weight,
                            topk_ids,
                            topk_weights,
                            num_experts=args.num_experts,
                            local_expert_start=local_expert_start,
                            output_dtype=output_dtype,
                        ),
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    status = "success" if matches else "metric_failed"
                    case.update(
                        {
                            "status": status,
                            "first_call_ms": first_call_ms,
                            "latency_ms": latency_ms,
                            "speedup_vs_oracle": oracle_ms / latency_ms,
                            "max_abs_error": max_abs,
                            "max_rel_error": max_rel,
                        }
                    )
                    parsed_outputs.append(
                        {
                            "tokens": num_tokens,
                            "config": name,
                            "status": status,
                            "max_abs_error": max_abs,
                            "max_rel_error": max_rel,
                        }
                    )
                    raw_outputs[str(num_tokens)]["triton_outputs"][name] = (
                        actual.detach().cpu()
                    )
                    if not matches:
                        failed = True
                except Exception:
                    failed = True
                    error = traceback.format_exc()
                    case["error"] = error
                    parsed_outputs.append(
                        {
                            "tokens": num_tokens,
                            "config": name,
                            "status": "model_failed",
                            "error": error,
                        }
                    )
                per_case_results.append(case)
    finally:
        triton_moe._gemm_launch_config = original_config_resolver

    successful = [item for item in per_case_results if item["status"] == "success"]
    best_by_tokens = {}
    for num_tokens in tokens:
        candidates = [item for item in successful if item["tokens"] == num_tokens]
        if candidates:
            best = min(candidates, key=lambda item: item["latency_ms"])
            best_by_tokens[str(num_tokens)] = {
                "config": best["config"]["name"],
                "latency_ms": best["latency_ms"],
                "speedup_vs_oracle": best["speedup_vs_oracle"],
            }

    run_config = {
        "command": [sys.executable, *sys.argv],
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "output_dtype": str(output_dtype),
        "seed": args.seed,
        "tokens": list(tokens),
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "ep_size": args.ep_size,
        "ep_rank": args.ep_rank,
        "local_expert_start": local_expert_start,
        "local_expert_end": local_expert_start + num_local_experts,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "oracle_iterations": args.oracle_iterations,
        "atol": args.atol,
        "rtol": args.rtol,
        "configs": list(configs),
    }
    num_metric_failed = sum(
        item["status"] == "metric_failed" for item in per_case_results
    )
    num_model_failed = sum(
        item["status"] == "model_failed" for item in per_case_results
    )
    aggregate_status = (
        "model_failed"
        if num_model_failed
        else ("metric_failed" if num_metric_failed else "success")
    )
    aggregate = {
        "status": aggregate_status,
        "num_cases": len(per_case_results),
        "num_success": len(successful),
        "num_metric_failed": num_metric_failed,
        "num_model_failed": num_model_failed,
        "best_by_tokens": best_by_tokens,
    }
    torch.save(raw_outputs, output_dir / "raw_outputs.pt")
    _write_json(output_dir / "parsed_outputs.json", parsed_outputs)
    _write_json(output_dir / "per_case_results.json", per_case_results)
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    _write_json(output_dir / "run_config.json", run_config)
    if failed:
        raise RuntimeError(f"MoE kernel benchmark failed; inspect {output_dir}.")


if __name__ == "__main__":
    main()
