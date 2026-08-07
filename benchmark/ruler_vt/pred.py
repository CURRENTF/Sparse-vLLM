#!/usr/bin/env python3
"""Evaluate DeltaKV schedules on the RULER Variable Tracking task.

This is a small, self-contained VT runner that mirrors NVIDIA RULER's
synthetic variable-tracking generation and string-match-all scoring while
using this repo's native Sparse-vLLM inference path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from benchmark.model_adapters.sparsevllm import get_sparsevllm_generate_api


TASK_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
)
ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text above, "
    "{num_v} variables are assigned the value {query}, they are: "
)
HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def normalize_prediction(text: str) -> str:
    text = text.strip()
    return re.sub(r"[\x00-\x1f]", "\n", text).strip()


def string_match_all(prediction: str, references: list[str]) -> float:
    pred = normalize_prediction(prediction).lower()
    return sum(1.0 if ref.lower() in pred else 0.0 for ref in references) / len(references)


@dataclass
class VTSample:
    index: int
    context_length: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str
    query: str


class VariableTrackingGenerator:
    def __init__(
        self,
        tokenizer,
        *,
        num_chains: int = 1,
        num_hops: int = 4,
        tokens_to_generate: int = 30,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_chains = num_chains
        self.num_hops = num_hops
        self.tokens_to_generate = tokens_to_generate

    def generate_chains(self, *, is_icl: bool = False) -> tuple[list[list[str]], list[list[str]]]:
        var_len = 3 if is_icl else 5
        total_vars = (self.num_hops + 1) * self.num_chains
        vars_all = [
            "".join(random.choices(string.ascii_uppercase, k=var_len)).upper()
            for _ in range(total_vars)
        ]
        while len(set(vars_all)) < total_vars:
            vars_all.append("".join(random.choices(string.ascii_uppercase, k=var_len)).upper())

        vars_ret: list[list[str]] = []
        chains_ret: list[list[str]] = []
        for start in range(0, len(vars_all), self.num_hops + 1):
            this_vars = vars_all[start : start + self.num_hops + 1]
            vars_ret.append(this_vars)
            first_value = "12345" if is_icl else str(np.random.randint(10000, 99999))
            this_chain = [f"VAR {this_vars[0]} = {first_value}"]
            for hop in range(self.num_hops):
                this_chain.append(f"VAR {this_vars[hop + 1]} = VAR {this_vars[hop]} ")
            chains_ret.append(this_chain)
        return vars_ret, chains_ret

    @staticmethod
    def shuffle_sublists(chains: list[list[str]]) -> list[str]:
        heap: list[tuple[float, int, int]] = []
        import heapq

        for chain_idx in range(len(chains)):
            heapq.heappush(heap, (random.random(), chain_idx, 0))

        shuffled: list[str] = []
        while heap:
            _, chain_idx, elem_idx = heapq.heappop(heap)
            shuffled.append(chains[chain_idx][elem_idx])
            if elem_idx + 1 < len(chains[chain_idx]):
                heapq.heappush(heap, (random.random(), chain_idx, elem_idx + 1))
        return shuffled

    def generate_input_output(self, num_noises: int, *, is_icl: bool = False) -> tuple[str, list[str], str]:
        variables, chains = self.generate_chains(is_icl=is_icl)
        value = chains[0][0].split("=")[-1].strip()
        sentences = [HAYSTACK_SENTENCE] * num_noises
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences)), len(chain)))
            for insert_pos, hop_idx in zip(positions, range(len(chain))):
                sentences.insert(insert_pos + hop_idx, chain[hop_idx])
        context = "\n".join(sentences).replace(". \n", ".\n")
        text = (
            TASK_TEMPLATE.format(context=context, query=value)
            + ANSWER_PREFIX.format(num_v=self.num_hops + 1, query=value)
        )
        return text, variables[0], value

    def randomize_icl(self, icl_example: dict[str, Any]) -> str:
        icl = icl_example["input"] + " " + " ".join(icl_example["outputs"]) + "\n"
        for item in icl_example["outputs"]:
            icl = icl.replace(item, "".join(random.choices(string.ascii_uppercase, k=len(item))).upper())
        return icl.replace("12345", str(np.random.randint(10000, 99999)))

    def make_icl_example(self) -> dict[str, Any]:
        text, answer, _query = self.generate_input_output(5, is_icl=True)
        return {"input": text, "outputs": answer}

    def optimal_num_noises(self, max_seq_length: int, icl_example: dict[str, Any]) -> int:
        incremental = 10
        icl_tokens = count_tokens(
            self.tokenizer,
            icl_example["input"] + " " + " ".join(icl_example["outputs"]) + "\n",
        )
        sample_text, _answer, _query = self.generate_input_output(incremental, is_icl=False)
        tokens_per_haystack = count_tokens(self.tokenizer, sample_text) / incremental
        estimated_max_noises = int((max_seq_length / tokens_per_haystack) * 3)

        lower = incremental
        upper = max(estimated_max_noises, incremental * 2)
        optimal = incremental
        while lower <= upper:
            mid = (lower + upper) // 2
            text, _answer, _query = self.generate_input_output(mid, is_icl=False)
            total = count_tokens(self.tokenizer, text) + icl_tokens + self.tokens_to_generate
            if total <= max_seq_length:
                optimal = mid
                lower = mid + 1
            else:
                upper = mid - 1
        return optimal

    def generate_samples(
        self,
        *,
        context_lengths: list[int],
        samples_per_length: int,
    ) -> list[VTSample]:
        samples: list[VTSample] = []
        sample_idx = 0
        for context_length in context_lengths:
            icl_example = self.make_icl_example()
            num_noises = self.optimal_num_noises(context_length, icl_example)
            for _ in range(samples_per_length):
                used_noises = num_noises
                while True:
                    text, answer, query = self.generate_input_output(used_noises, is_icl=False)
                    cutoff = text.index(TASK_TEMPLATE[:20])
                    text = text[:cutoff] + self.randomize_icl(icl_example) + "\n" + text[cutoff:]
                    length = count_tokens(self.tokenizer, text) + self.tokens_to_generate
                    if length <= context_length or used_noises <= 10:
                        break
                    used_noises -= 10

                prefix_idx = text.rfind(ANSWER_PREFIX[:10])
                if prefix_idx < 0:
                    raise ValueError("Generated VT sample is missing the answer prefix.")
                answer_prefix = text[prefix_idx:]
                prompt = text[:prefix_idx]
                samples.append(
                    VTSample(
                        index=sample_idx,
                        context_length=context_length,
                        input=prompt,
                        outputs=answer,
                        length=length,
                        answer_prefix=answer_prefix,
                        query=query,
                    )
                )
                sample_idx += 1
        return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_from_row(row: dict[str, Any]) -> VTSample:
    return VTSample(
        index=int(row["index"]),
        context_length=int(row["context_length"]),
        input=str(row["input"]),
        outputs=list(row["outputs"]),
        length=int(row["length"]),
        answer_prefix=str(row["answer_prefix"]),
        query=str(row["others"]["query"]),
    )


def parse_context_lengths(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hyper-param", default=None)
    parser.add_argument("--sparse-method", default=None)
    parser.add_argument("--deltakv-checkpoint-path", default=None)
    parser.add_argument("--context-lengths", default="4096,8192,16384,32768,65536,98304")
    parser.add_argument("--samples-per-length", type=int, default=20)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--num-hops", type=int, default=4)
    parser.add_argument("--tokens-to-generate", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ws", type=int, default=1, help="Number of single-GPU worker processes.")
    parser.add_argument("--worker-rank", type=int, default=-1)
    parser.add_argument("--worker-world-size", type=int, default=1)
    parser.add_argument(
        "--no-answer-prefix",
        action="store_true",
        help="Do not append RULER's answer_prefix to the prompt before generation.",
    )
    return parser.parse_args()


def generate_dataset(args: argparse.Namespace, output_dir: Path, tokenizer) -> list[VTSample]:
    generator = VariableTrackingGenerator(
        tokenizer,
        num_chains=args.num_chains,
        num_hops=args.num_hops,
        tokens_to_generate=args.tokens_to_generate,
    )
    samples = generator.generate_samples(
        context_lengths=parse_context_lengths(args.context_lengths),
        samples_per_length=args.samples_per_length,
    )
    dataset_rows = [
        {
            "index": sample.index,
            "input": sample.input,
            "outputs": sample.outputs,
            "length": sample.length,
            "context_length": sample.context_length,
            "answer_prefix": sample.answer_prefix,
            "others": {"query": sample.query, "task": "ruler_vt"},
        }
        for sample in samples
    ]
    write_jsonl(output_dir / "dataset.jsonl", dataset_rows)
    return samples


def build_infer_config(args: argparse.Namespace, context_lengths: list[int]) -> dict[str, Any]:
    infer_config: dict[str, Any] = {}
    if args.hyper_param:
        with open(args.hyper_param, "r", encoding="utf-8") as f:
            infer_config.update(json.load(f))
    infer_config["max_model_len"] = args.max_model_len or (max(context_lengths) + args.tokens_to_generate + 1024)
    return infer_config


def write_run_info(args: argparse.Namespace, output_dir: Path, tokenizer_path: str, infer_config: dict[str, Any]) -> None:
    context_lengths = parse_context_lengths(args.context_lengths)

    run_info = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": "python " + " ".join(sys.argv),
        "cwd": os.getcwd(),
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "hyper_param": args.hyper_param,
        "infer_config": infer_config,
        "context_lengths": context_lengths,
        "samples_per_length": args.samples_per_length,
        "num_chains": args.num_chains,
        "num_hops": args.num_hops,
        "append_answer_prefix": not args.no_answer_prefix,
        "seed": args.seed,
        "cuda_device": args.cuda_device,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        },
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)


def evaluate_samples(args: argparse.Namespace, samples: list[VTSample], output_dir: Path, infer_config: dict[str, Any]) -> None:
    rank_suffix = "" if args.worker_rank < 0 else f"_rank{args.worker_rank}"
    raw_path = output_dir / f"raw_outputs{rank_suffix}.jsonl"
    parsed_path = output_dir / f"parsed_outputs{rank_suffix}.jsonl"
    result_path = output_dir / f"per_sample_results{rank_suffix}.jsonl"
    for path in [raw_path, parsed_path, result_path]:
        path.write_text("", encoding="utf-8")

    tokenizer_path = args.tokenizer_path or args.model_path
    generate = get_sparsevllm_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=args.deltakv_checkpoint_path,
        sparse_method=args.sparse_method,
    )

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="RULER-VT"):
        batch = samples[start : start + args.batch_size]
        prompts = [
            sample.input if args.no_answer_prefix else sample.input + sample.answer_prefix
            for sample in batch
        ]
        try:
            preds = generate(
                prompts if len(prompts) > 1 else prompts[0],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
            )
            if isinstance(preds, str):
                preds = [preds]
            status = "success"
        except Exception as exc:  # Fail per sample while preserving audit artifacts.
            preds = ["" for _ in batch]
            status = "model_failed"
            error = repr(exc)
        else:
            error = None

        for sample, pred in zip(batch, preds):
            parsed = normalize_prediction(pred)
            score = string_match_all(parsed, sample.outputs) if status == "success" else 0.0
            is_correct = score == 1.0
            raw_rows.append(
                {
                    "index": sample.index,
                    "context_length": sample.context_length,
                    "length": sample.length,
                    "prompt": sample.input if args.no_answer_prefix else sample.input + sample.answer_prefix,
                    "raw_output": pred,
                    "status": status,
                    "error": error,
                }
            )
            parsed_rows.append(
                {
                    "index": sample.index,
                    "context_length": sample.context_length,
                    "prediction": parsed,
                    "outputs": sample.outputs,
                    "status": "success" if status == "success" else "parse_failed",
                }
            )
            result_rows.append(
                {
                    "index": sample.index,
                    "context_length": sample.context_length,
                    "length": sample.length,
                    "prediction": parsed,
                    "outputs": sample.outputs,
                    "score": score,
                    "correct": is_correct,
                    "status": status,
                }
            )

            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(raw_rows[-1], ensure_ascii=False) + "\n")
            with parsed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(parsed_rows[-1], ensure_ascii=False) + "\n")
            with result_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result_rows[-1], ensure_ascii=False) + "\n")


def write_aggregate(output_dir: Path, context_lengths: list[int], elapsed_seconds: float) -> dict[str, Any]:
    result_rows = read_jsonl(output_dir / "per_sample_results.jsonl")
    by_length: dict[str, dict[str, Any]] = {}
    for context_length in context_lengths:
        rows = [row for row in result_rows if row["context_length"] == context_length]
        if not rows:
            continue
        by_length[str(context_length)] = {
            "score": round(100 * float(np.mean([row["score"] for row in rows])), 2),
            "exact_match": round(100 * float(np.mean([row["correct"] for row in rows])), 2),
            "num_samples": len(rows),
            "num_success": sum(row["status"] == "success" for row in rows),
            "mean_input_tokens": round(float(np.mean([row["length"] for row in rows])), 2),
        }

    aggregate = {
        "task": "ruler_vt",
        "metric": "string_match_all",
        "score_by_context_length": by_length,
        "overall_score": round(100 * float(np.mean([row["score"] for row in result_rows])), 2),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return aggregate


def merge_worker_outputs(output_dir: Path, world_size: int) -> None:
    for stem in ["raw_outputs", "parsed_outputs", "per_sample_results"]:
        rows: list[dict[str, Any]] = []
        for rank in range(world_size):
            path = output_dir / f"{stem}_rank{rank}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing worker output: {path}")
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: int(row["index"]))
        write_jsonl(output_dir / f"{stem}.jsonl", rows)


def launch_workers(args: argparse.Namespace, output_dir: Path) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_ids = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    else:
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    if len(gpu_ids) < args.ws:
        raise ValueError(f"Requested ws={args.ws}, but only {len(gpu_ids)} visible GPUs are available: {gpu_ids}")

    script_path = Path(__file__).resolve()
    child_base = sys.argv[1:]
    procs: list[subprocess.Popen] = []
    for rank in range(args.ws):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            *child_base,
            "--worker-rank",
            str(rank),
            "--worker-world-size",
            str(args.ws),
            "--cuda-device",
            "0",
        ]
        print(f"[Parent] launch rank={rank} gpu={gpu_ids[rank]} cmd={' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(script_path.parent.parent.parent)))

    failed: list[tuple[int, int]] = []
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed.append((rank, ret))
    if failed:
        raise RuntimeError("RULER-VT worker failed: " + ", ".join(f"rank={r}, exitcode={c}" for r, c in failed))


def main() -> None:
    args = parse_args()
    random.seed(args.seed + max(args.worker_rank, 0))
    np.random.seed(args.seed + max(args.worker_rank, 0))
    torch.manual_seed(args.seed + max(args.worker_rank, 0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_lengths = parse_context_lengths(args.context_lengths)
    tokenizer_path = args.tokenizer_path or args.model_path

    if args.worker_rank < 0:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        generate_dataset(args, output_dir, tokenizer)
        infer_config = build_infer_config(args, context_lengths)
        write_run_info(args, output_dir, tokenizer_path, infer_config)

        if args.ws > 1:
            start = time.time()
            launch_workers(args, output_dir)
            merge_worker_outputs(output_dir, args.ws)
            write_aggregate(output_dir, context_lengths, time.time() - start)
            return

        samples = [sample_from_row(row) for row in read_jsonl(output_dir / "dataset.jsonl")]
        start = time.time()
        evaluate_samples(args, samples, output_dir, infer_config)
        write_aggregate(output_dir, context_lengths, time.time() - start)
        return

    samples_all = [sample_from_row(row) for row in read_jsonl(output_dir / "dataset.jsonl")]
    samples = samples_all[args.worker_rank :: args.worker_world_size]
    infer_config = build_infer_config(args, context_lengths)
    evaluate_samples(args, samples, output_dir, infer_config)


if __name__ == "__main__":
    main()
