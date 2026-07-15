#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item())
    denominator = expected.float().abs().clamp_min(1.0e-5)
    return max_abs, float((difference / denominator).max().item())


def _step_input_tokens(
    *,
    stage: str,
    prompt_token_ids: list[int],
    prompt_offset: int,
    chunk_size: int,
    next_decode_token: int | None,
) -> tuple[list[int], int]:
    if stage == "prefill":
        if prompt_offset >= len(prompt_token_ids):
            raise ValueError("Reference contains an extra prefill step after the prompt ended.")
        end = min(prompt_offset + int(chunk_size), len(prompt_token_ids))
        return prompt_token_ids[prompt_offset:end], end
    if stage == "decode":
        if next_decode_token is None:
            raise ValueError("Decode step has no preceding sampled token.")
        return [int(next_decode_token)], prompt_offset
    raise ValueError(f"Unknown engine reference stage {stage!r}.")


def _single_sampled_token(step: dict[str, Any]) -> int | None:
    outputs = step["sampled_token_outputs"]
    if not outputs:
        return None
    if len(outputs) != 1 or len(outputs[0][1]) != 1:
        raise ValueError(
            "HF reference replay currently requires one sequence and one sampled "
            f"token per generating step, got {outputs!r}."
        )
    return int(outputs[0][1][0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3MoE EP=1 engine logits with Hugging Face replay."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3MoE Hugging Face validation requires CUDA.")
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}.")
    reference_path = Path(args.engine_reference).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Engine reference does not exist: {reference_path}.")
    engine_run_config_path = reference_path.parent / "run_config.json"
    if not engine_run_config_path.is_file():
        raise FileNotFoundError(
            f"Engine run config does not exist: {engine_run_config_path}."
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_reference = torch.load(
        reference_path,
        map_location="cpu",
        weights_only=True,
    )
    engine_run_config = json.loads(engine_run_config_path.read_text(encoding="utf-8"))
    requests = engine_reference["requests"]
    if len(requests) != 1 or len(requests[0]["seq_ids"]) != 1:
        raise ValueError(
            "HF reference replay requires exactly one engine request and sequence, "
            f"got requests={len(requests)}."
        )
    prompt_token_ids = [int(token_id) for token_id in requests[0]["prompt_token_ids"][0]]
    chunk_size = int(engine_run_config["engine_kwargs"]["engine_prefill_chunk_size"])
    if chunk_size <= 0:
        raise ValueError(f"Engine prefill chunk size must be positive, got {chunk_size}.")

    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    if str(getattr(hf_config, "model_type", "")) != "qwen3_moe":
        raise ValueError(
            f"Expected model_type='qwen3_moe', got {hf_config.model_type!r}."
        )
    dtype = getattr(hf_config, "torch_dtype", None)
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Expected a BF16/FP16 checkpoint, got dtype={dtype}.")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    raw_steps: list[dict[str, Any]] = []
    per_step: list[dict[str, Any]] = []
    failure: BaseException | None = None
    model = None
    started = time.perf_counter()
    load_seconds = None
    prompt_offset = 0
    next_decode_token = None
    try:
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            config=hf_config,
            dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation=args.attention_implementation,
            trust_remote_code=False,
        ).eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started

        past_key_values = None
        for expected in engine_reference["steps"]:
            stage = str(expected["stage"])
            input_token_ids, prompt_offset = _step_input_tokens(
                stage=stage,
                prompt_token_ids=prompt_token_ids,
                prompt_offset=prompt_offset,
                chunk_size=chunk_size,
                next_decode_token=next_decode_token,
            )
            input_ids = torch.tensor(
                [input_token_ids],
                dtype=torch.int64,
                device="cuda:0",
            )
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :].detach().cpu().contiguous()
            expected_logits = expected["logits"].contiguous()
            if tuple(logits.shape) != tuple(expected_logits.shape):
                raise ValueError(
                    "HF and engine logit shapes differ: "
                    f"hf={tuple(logits.shape)} engine={tuple(expected_logits.shape)}."
                )
            max_abs, max_rel = _max_errors(logits, expected_logits)
            logits_close = bool(
                torch.allclose(
                    logits.float(),
                    expected_logits.float(),
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            sampled_token = _single_sampled_token(expected)
            greedy_token = int(torch.argmax(logits, dim=-1).item())
            token_matches = sampled_token is None or greedy_token == sampled_token
            status = "success" if logits_close and token_matches else "metric_failed"
            raw_steps.append(
                {
                    "case_name": expected["case_name"],
                    "step_idx": int(expected["step_idx"]),
                    "stage": stage,
                    "input_token_ids": input_token_ids,
                    "logits": logits,
                    "greedy_token_id": greedy_token,
                    "engine_sampled_token_id": sampled_token,
                }
            )
            per_step.append(
                {
                    "case_name": expected["case_name"],
                    "step_idx": int(expected["step_idx"]),
                    "stage": stage,
                    "status": status,
                    "input_token_count": len(input_token_ids),
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "logits_within_tolerance": logits_close,
                    "greedy_token_id": greedy_token,
                    "engine_sampled_token_id": sampled_token,
                    "token_matches": token_matches,
                }
            )
            next_decode_token = sampled_token
        if prompt_offset != len(prompt_token_ids):
            raise ValueError(
                f"HF replay consumed {prompt_offset}/{len(prompt_token_ids)} prompt tokens."
            )
    except BaseException as exc:
        failure = exc
    finally:
        del model
        torch.cuda.empty_cache()

    metric_errors = [
        (
            f"step={step['step_idx']} stage={step['stage']} "
            f"max_abs={step['max_abs_error']} max_rel={step['max_rel_error']} "
            f"token_matches={step['token_matches']}"
        )
        for step in per_step
        if step["status"] != "success"
    ]
    status = (
        "model_failed"
        if failure is not None
        else ("metric_failed" if metric_errors else "success")
    )
    torch.save({"steps": raw_steps}, output_dir / "raw_outputs.pt")
    _write_json(output_dir / "parsed_outputs.json", {"status": status, "steps": per_step})
    _write_json(output_dir / "per_step_results.json", per_step)
    _write_json(
        output_dir / "run_config.json",
        {
            "command": [sys.executable, *sys.argv],
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "model": str(model_path),
            "model_type": str(hf_config.model_type),
            "dtype": str(dtype),
            "attention_implementation": args.attention_implementation,
            "engine_reference": str(reference_path),
            "engine_commit": engine_run_config["git_commit"],
            "prompt_token_ids": prompt_token_ids,
            "prefill_chunk_size": chunk_size,
            "atol": args.atol,
            "rtol": args.rtol,
        },
    )
    _write_json(
        output_dir / "aggregate_metrics.json",
        {
            "status": status,
            "num_steps": len(per_step),
            "num_success": sum(step["status"] == "success" for step in per_step),
            "num_metric_failed": len(metric_errors),
            "metric_errors": metric_errors,
            "failure": repr(failure) if failure is not None else None,
            "traceback": (
                "".join(traceback.format_exception(failure))
                if failure is not None
                else None
            ),
            "load_seconds": load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "max_abs_error": max(
                (step["max_abs_error"] for step in per_step),
                default=None,
            ),
        },
    )
    if failure is not None:
        raise failure
    if metric_errors:
        raise RuntimeError(
            f"Qwen3MoE Hugging Face validation failed; inspect {output_dir}. "
            f"Errors: {metric_errors}"
        )


if __name__ == "__main__":
    main()
