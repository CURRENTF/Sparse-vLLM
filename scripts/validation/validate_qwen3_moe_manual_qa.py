#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sparsevllm import LLM, SamplingParams


QA_CASES = (
    {
        "case_id": "arithmetic",
        "prompt": "Compute 17 × 23. Give only the final integer.",
    },
    {
        "case_id": "factual",
        "prompt": "What is the capital of France? Answer in one short sentence.",
    },
    {
        "case_id": "format_following",
        "prompt": (
            'Return exactly this JSON object and nothing else: '
            '{"status":"ok","count":3}'
        ),
    },
    {
        "case_id": "chinese_explanation",
        "prompt": "请用一句中文解释混合专家模型（MoE）的基本工作方式。",
    },
)


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


def evaluate_answer(case_id: str, text: str) -> tuple[bool, str]:
    answer = text.strip()
    if case_id == "arithmetic":
        passed = bool(re.search(r"(?<!\d)391(?!\d)", answer))
        return passed, "answer contains the exact integer 391"
    if case_id == "factual":
        passed = "paris" in answer.casefold()
        return passed, "answer identifies Paris"
    if case_id == "format_following":
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            return False, "answer is not a standalone JSON value"
        passed = parsed == {"status": "ok", "count": 3}
        return passed, "answer is exactly the requested JSON object"
    if case_id == "chinese_explanation":
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", answer)
        passed = (
            len(chinese_chars) >= 10
            and "专家" in answer
            and any(term in answer for term in ("路由", "选择", "激活"))
        )
        return passed, "answer is a substantive Chinese MoE explanation"
    raise ValueError(f"Unknown QA case_id={case_id!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed, reviewable Qwen3MoE manual-QA smoke set."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-parallel-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3MoE manual QA requires CUDA.")
    if args.max_tokens <= 0:
        raise ValueError(f"--max-tokens must be positive, got {args.max_tokens}.")
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}.")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).resolve() if args.reference else None
    reference_by_case: dict[str, dict[str, Any]] = {}
    if reference_path is not None:
        if not reference_path.is_file():
            raise FileNotFoundError(f"QA reference does not exist: {reference_path}.")
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_by_case = {
            str(record["case_id"]): record
            for record in reference_payload["records"]
        }
        expected_case_ids = {str(case["case_id"]) for case in QA_CASES}
        if set(reference_by_case) != expected_case_ids:
            raise ValueError(
                "QA reference case IDs differ from the fixed suite: "
                f"reference={sorted(reference_by_case)} expected={sorted(expected_case_ids)}."
            )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    prompt_token_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": str(case["prompt"])}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for case in QA_CASES
    ]
    longest_request = max(len(tokens) for tokens in prompt_token_ids) + args.max_tokens
    if longest_request > args.max_model_len:
        raise ValueError(
            "QA prompt plus generation budget exceeds max_model_len: "
            f"required={longest_request} configured={args.max_model_len}."
        )

    engine_kwargs = {
        "sparse_method": "vanilla",
        "enforce_eager": True,
        "decode_cuda_graph": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "engine_prefill_chunk_size": 256,
        "max_model_len": args.max_model_len,
        "max_num_seqs_in_batch": len(QA_CASES),
        "max_decoding_seqs": len(QA_CASES),
        "tensor_parallel_size": 1,
        "expert_parallel_size": args.expert_parallel_size,
        "data_parallel_size": 1,
        "moe_backend": "triton",
    }
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
            max_tokens=args.max_tokens,
        )
        for _ in QA_CASES
    ]

    llm = None
    generated: list[dict[str, Any]] | None = None
    failure: BaseException | None = None
    started = time.perf_counter()
    try:
        llm = LLM(str(model_path), **engine_kwargs)
        generated = llm.generate(
            prompt_token_ids,
            sampling_params,
            use_tqdm=False,
        )
        if len(generated) != len(QA_CASES):
            raise RuntimeError(
                f"Engine returned {len(generated)} QA outputs for {len(QA_CASES)} prompts."
            )
    except BaseException as exc:
        failure = exc
    finally:
        if llm is not None:
            llm.exit()

    raw_records = []
    per_sample = []
    for index, case in enumerate(QA_CASES):
        output = generated[index] if generated is not None and index < len(generated) else None
        text = str(output["text"]) if output is not None else ""
        token_ids = [int(token_id) for token_id in output["token_ids"]] if output is not None else []
        automatic_passed = False
        criterion = "model execution failed before this sample produced output"
        reference_matches = None
        status = "model_failed" if failure is not None else "metric_failed"
        if output is not None:
            automatic_passed, criterion = evaluate_answer(str(case["case_id"]), text)
            if reference_path is not None:
                reference_matches = token_ids == [
                    int(token_id)
                    for token_id in reference_by_case[str(case["case_id"])]["generated_token_ids"]
                ]
            status = (
                "success"
                if automatic_passed and reference_matches is not False
                else "metric_failed"
            )
        raw_records.append(
            {
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids[index]],
                "generated_text": text,
                "generated_token_ids": token_ids,
            }
        )
        per_sample.append(
            {
                "case_id": case["case_id"],
                "status": status,
                "automatic_check_passed": automatic_passed,
                "criterion": criterion,
                "reference_token_ids_match": reference_matches,
                "generated_text": text,
            }
        )

    num_success = sum(record["status"] == "success" for record in per_sample)
    aggregate_status = (
        "model_failed"
        if failure is not None
        else ("success" if num_success == len(per_sample) else "metric_failed")
    )
    raw_payload = {"records": raw_records}
    _write_json(output_dir / "raw_outputs.json", raw_payload)
    _write_json(
        output_dir / "parsed_outputs.json",
        {"status": aggregate_status, "records": per_sample},
    )
    _write_json(output_dir / "per_sample_results.json", per_sample)
    _write_json(
        output_dir / "run_config.json",
        {
            "command": [sys.executable, *sys.argv],
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "model": str(model_path),
            "expert_parallel_size": args.expert_parallel_size,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "engine_kwargs": engine_kwargs,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "reference": str(reference_path) if reference_path else None,
        },
    )
    _write_json(
        output_dir / "aggregate_metrics.json",
        {
            "status": aggregate_status,
            "num_samples": len(per_sample),
            "num_success": num_success,
            "num_metric_failed": sum(
                record["status"] == "metric_failed" for record in per_sample
            ),
            "num_model_failed": sum(
                record["status"] == "model_failed" for record in per_sample
            ),
            "failure": repr(failure) if failure is not None else None,
            "traceback": (
                "".join(traceback.format_exception(failure))
                if failure is not None
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    if failure is not None:
        raise failure
    if aggregate_status != "success":
        raise RuntimeError(
            f"Qwen3MoE manual QA failed; inspect {output_dir}."
        )


if __name__ == "__main__":
    main()
