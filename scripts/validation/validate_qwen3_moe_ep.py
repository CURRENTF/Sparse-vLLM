from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from sparsevllm.distributed import init_parallel_context, reset_parallel_context
from sparsevllm.models.qwen3_moe import Qwen3MoeForCausalLM
from sparsevllm.utils.loader import load_model


def _parse_layers(value: str, num_layers: int) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("--layers must contain at least one layer index.")
    if len(set(layers)) != len(layers):
        raise ValueError(f"--layers contains duplicates: {layers}.")
    invalid = [layer_idx for layer_idx in layers if not 0 <= layer_idx < num_layers]
    if invalid:
        raise ValueError(
            f"--layers contains invalid indices {invalid}; num_hidden_layers={num_layers}."
        )
    return layers


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item())
    denominator = expected.float().abs().clamp_min(1.0e-5)
    max_rel = float((difference / denominator).max().item())
    return max_abs, max_rel


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Qwen3MoE packed expert loading and replicated-input EP outputs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", default=None, help="EP=1 raw_outputs.pt to compare against.")
    parser.add_argument("--layers", default="0,47")
    parser.add_argument("--num-tokens", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument(
        "--moe-backend",
        choices=("pytorch", "triton"),
        default="triton",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_tokens <= 0:
        raise ValueError(f"--num-tokens must be positive, got {args.num_tokens}.")
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(world_rank)))
    if world_size <= 0 or not 0 <= world_rank < world_size:
        raise ValueError(
            f"Invalid distributed environment WORLD_SIZE={world_size}, RANK={world_rank}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3MoE validation requires CUDA GPUs.")
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    parallel_context = init_parallel_context(
        tp_size=1,
        ep_size=world_size,
        dp_size=1,
    )

    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if str(getattr(hf_config, "model_type", "")) != "qwen3_moe":
        raise ValueError(
            f"Expected model_type='qwen3_moe', got {getattr(hf_config, 'model_type', None)!r}."
        )
    if int(hf_config.num_experts) % world_size != 0:
        raise ValueError(
            f"num_experts={hf_config.num_experts} is not divisible by EP={world_size}."
        )
    hf_config.moe_backend = args.moe_backend
    layers = _parse_layers(args.layers, int(hf_config.num_hidden_layers))

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device(torch.device("cuda", local_rank))
    load_started = time.perf_counter()
    model = Qwen3MoeForCausalLM(hf_config)
    load_model(model, str(model_path), tp_rank=0, tp_size=1)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    torch.set_default_device("cpu")
    torch.set_default_dtype(previous_dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    hidden_states = torch.randn(
        args.num_tokens,
        int(hf_config.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(device=torch.device("cuda", local_rank), dtype=hf_config.torch_dtype)
    reference = None
    if args.reference is not None:
        reference_path = Path(args.reference).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference artifact does not exist: {reference_path}")
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)

    os.environ["SPARSEVLLM_DEBUG_MOE"] = "1"
    raw_outputs: dict[str, object] = {
        "input": hidden_states.detach().cpu(),
        "layers": {},
    }
    per_layer_results = []
    validation_failed = False
    for layer_idx in layers:
        block = model.model.layers[layer_idx].mlp
        output = block(hidden_states)
        torch.cuda.synchronize()
        gathered_outputs = [torch.empty_like(output) for _ in range(world_size)]
        dist.all_gather(
            gathered_outputs,
            output,
            group=parallel_context.world.process_group,
        )
        rank_max_abs = max(
            float((candidate.float() - gathered_outputs[0].float()).abs().max().item())
            for candidate in gathered_outputs
        )

        output_cpu = output.detach().cpu()
        topk_ids_cpu = block.debug_last_topk_ids.detach().cpu()
        topk_weights_cpu = block.debug_last_topk_weights.detach().cpu()
        layer_raw = {
            "output": output_cpu,
            "topk_ids": topk_ids_cpu,
            "topk_weights": topk_weights_cpu,
        }
        raw_outputs["layers"][str(layer_idx)] = layer_raw
        result = {
            "layer_idx": layer_idx,
            "status": "success",
            "rank_max_abs_error": rank_max_abs,
            "reference_max_abs_error": None,
            "reference_max_rel_error": None,
            "topk_ids_match": None,
        }
        if rank_max_abs != 0.0:
            result["status"] = "metric_failed"
            validation_failed = True
        if reference is not None:
            reference_layer = reference["layers"][str(layer_idx)]
            max_abs, max_rel = _max_errors(output_cpu, reference_layer["output"])
            topk_ids_match = bool(torch.equal(topk_ids_cpu, reference_layer["topk_ids"]))
            result.update(
                {
                    "reference_max_abs_error": max_abs,
                    "reference_max_rel_error": max_rel,
                    "topk_ids_match": topk_ids_match,
                }
            )
            if not topk_ids_match or not torch.allclose(
                output_cpu.float(),
                reference_layer["output"].float(),
                atol=args.atol,
                rtol=args.rtol,
            ):
                result["status"] = "metric_failed"
                validation_failed = True
        per_layer_results.append(result)

    local_experts = model.model.layers[0].mlp.experts
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(local_rank))
    run_config = {
        "command": [sys.executable, *sys.argv],
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "model": str(model_path),
        "model_config": hf_config.to_dict(),
        "dtype": str(hf_config.torch_dtype),
        "moe_backend": args.moe_backend,
        "seed": args.seed,
        "num_tokens": args.num_tokens,
        "layers": list(layers),
        "world_size": world_size,
        "tensor_parallel_size": 1,
        "expert_parallel_size": world_size,
        "data_parallel_size": 1,
        "world_rank": world_rank,
        "local_rank": local_rank,
        "local_expert_start": local_experts.local_expert_start,
        "local_expert_end": local_experts.local_expert_end,
        "reference": str(Path(args.reference).resolve()) if args.reference else None,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    aggregate_metrics = {
        "status": "metric_failed" if validation_failed else "success",
        "num_layers": len(layers),
        "num_success": sum(item["status"] == "success" for item in per_layer_results),
        "num_metric_failed": sum(
            item["status"] == "metric_failed" for item in per_layer_results
        ),
        "load_seconds": load_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }
    rank_dir = output_dir / f"rank_{world_rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    torch.save(raw_outputs, rank_dir / "raw_outputs.pt")
    _write_json(rank_dir / "run_config.json", run_config)
    _write_json(rank_dir / "per_layer_results.json", per_layer_results)
    _write_json(rank_dir / "aggregate_metrics.json", aggregate_metrics)
    dist.barrier()

    del model
    torch.cuda.empty_cache()
    reset_parallel_context()
    dist.destroy_process_group()
    if validation_failed:
        raise RuntimeError(
            f"Qwen3MoE EP validation failed; inspect artifacts under {output_dir}."
        )


if __name__ == "__main__":
    main()
