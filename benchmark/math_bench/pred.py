import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

from deltakv.get_chat_api import get_generate_api

# Keep defaults consistent with benchmark/long_bench/pred.py, but allow env overrides.
BASE_PATH = os.getenv("DELTAKV_OUTPUT_DIR", "/root/autodl-fs/deltakv_outputs")
DATA_PREFIX_PATH = os.getenv("DELTAKV_DATA_DIR", "/root/autodl-fs/datasets")
DEFAULT_GSM8K_DATASET = ("openai/gsm8k", "main", "test")
DEFAULT_AIME2024_DATASET = ("Maxwell-Jia/AIME_2024", None, "train")
DEFAULT_HMMT_NOV_DATASET = ("MathArena/hmmt_nov_2025", None, "train")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, prompt: str, no_chat_template: bool) -> str:
    if not no_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        msgs = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=os.getenv("ENABLE_THINKING", "1") not in ("0", "false", "False"),
        )
    if os.getenv("DEBUG"):
        print("input prompt:", prompt)
    return prompt


def _read_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}")
            return data
        return [json.loads(line) for line in f if line.strip()]


def _load_hf_dataset(dataset_name: str, config_name: str, split: str):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets is required for HF dataset loading. Install `datasets`.") from e

    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    return [ds[i] for i in range(len(ds))]


def _resolve_default_data_path(data_dir: str, dataset: str, split: str) -> str:
    candidates = []
    if dataset == "gsm8k":
        candidates = [
            f"gsm8k/{split}.jsonl",
            f"gsm8k/{split}.json",
            f"GSM8K/{split}.jsonl",
            f"GSM8K/{split}.json",
            f"gsm8k_{split}.jsonl",
            f"gsm8k_{split}.json",
        ]
    elif dataset == "aime2024":
        candidates = [
            f"aime2024/{split}.jsonl",
            f"aime2024/{split}.json",
            f"aime_2024/{split}.jsonl",
            f"aime_2024/{split}.json",
            f"AIME2024/{split}.jsonl",
            f"AIME2024/{split}.json",
            f"aime2024_{split}.jsonl",
            f"aime2024_{split}.json",
        ]
    elif dataset == "hmmt_nov":
        candidates = [
            f"hmmt_nov/{split}.jsonl",
            f"hmmt_nov/{split}.json",
            f"hmmt_nov_2025/{split}.jsonl",
            f"hmmt_nov_2025/{split}.json",
            f"hmmt_nov_{split}.jsonl",
            f"hmmt_nov_{split}.json",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for rel in candidates:
        path = os.path.join(data_dir, rel)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find dataset file for {dataset} (split={split}) under {data_dir}. "
        f"Tried: {', '.join(candidates)}. Use --data_path_{dataset} to override."
    )


def _get_problem_text(example: dict, dataset: str) -> str:
    # HF AIME_2024 uses "Problem"; GSM8K uses "question".
    keys = ["question", "Question", "problem", "Problem", "prompt", "input"]
    for k in keys:
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k].strip()
    raise KeyError(f"Cannot find problem text keys {keys} for dataset={dataset}. Keys: {list(example.keys())}")


def _get_example_id(example: dict, idx: int) -> str:
    for k in ("id", "idx", "index", "qid", "uid"):
        if k in example:
            return str(example[k])
    return str(idx)


def load_model_and_tokenizer(rank: int, args):
    infer_config = {
        "max_model_len": args.max_model_len,
    }

    if args.hyper_param:
        if os.path.exists(args.hyper_param):
            with open(args.hyper_param, "r") as f:
                extra_config = json.load(f)
            infer_config.update(extra_config)
            print(f"Loaded hyper-parameters from {args.hyper_param}: {extra_config}")
        else:
            try:
                extra_config = json.loads(args.hyper_param)
                infer_config.update(extra_config)
                print(f"Parsed hyper-parameters from string: \n{extra_config}")
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse --hyper_param '{args.hyper_param}'. "
                    f"It is neither a valid file path nor a valid JSON string. Error: {e}"
                )

    generate_fn = get_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        compressor_path=args.compressor_path,
        tokenizer_path=args.tokenizer_path,
        model_cls=args.model_cls,
        cuda_device=rank,
        backend=args.backend,
    )

    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return generate_fn, tokenizer


def get_pred(rank: int, data, dataset: str, args, model, tokenizer, out_path: str) -> None:
    prompt_format = {
        "gsm8k": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "Begin your response with \"<think>\\n\" and do not output an empty think block.\n\n"
            "Problem:\n{problem}\n"
        ),
        "aime2024": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "The final answer is an integer.\n"
            "Begin your response with \"<think>\\n\" and do not output an empty think block.\n\n"
            "Problem:\n{problem}\n"
        ),
        "hmmt_nov": (
            "Please reason step by step, and put your final answer within \\\\boxed{{}}.\n"
            "Begin your response with \"<think>\\n\" and do not output an empty think block.\n\n"
            "Problem:\n{problem}\n"
        ),
    }[dataset]

    max_gen = args.max_new_tokens
    batch_size = args.batch_size
    max_prompt_len = max(1, int(args.max_model_len) - int(max_gen) - 32)

    for i in tqdm(range(0, len(data), batch_size), desc=f"[Rank {rank}] {dataset}"):
        batch_data = data[i : i + batch_size]
        prompts = []
        meta = []
        for j, example in enumerate(batch_data):
            problem = _get_problem_text(example, dataset)
            prompt = prompt_format.format(problem=problem)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_prompt_len:
                half = int(max_prompt_len / 2)
                prompt = (
                    tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                    + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                )
            prompts.append(build_chat(tokenizer, prompt, args.no_chat_template))
            meta.append({"id": _get_example_id(example, i + j)})

        eos_token_id = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "eot_token_id"):
            eos_token_id.append(tokenizer.eot_token_id)

        preds = model(
            prompts,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
        )
        if isinstance(preds, str):
            preds = [preds]

        for example, pred, info in zip(batch_data, preds, meta):
            if args.force_think_prefix:
                if pred.startswith("<think>") and not pred.startswith("<think>\n"):
                    pred = "<think>\n" + pred[len("<think>") :].lstrip("\n")
                elif not pred.startswith("<think>\n"):
                    pred = args.think_prefix + pred
            record = {
                "id": info["id"],
                "pred": pred,
                "gold": example,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")


def worker(rank: int, world_size: int, datasets: List[str], args, out_root: str) -> None:
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(rank, args)

    for dataset in datasets:
        if dataset == "gsm8k":
            if args.data_path_gsm8k:
                data = _read_json_or_jsonl(args.data_path_gsm8k)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_gsm8k, args.hf_config_gsm8k, args.hf_split_gsm8k)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[gsm8k] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "gsm8k", args.split)
                    data = _read_json_or_jsonl(data_path)
        elif dataset == "aime2024":
            if args.data_path_aime2024:
                data = _read_json_or_jsonl(args.data_path_aime2024)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_aime2024, args.hf_config_aime2024, args.hf_split_aime2024)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[aime2024] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "aime2024", args.split)
                    data = _read_json_or_jsonl(data_path)
        elif dataset == "hmmt_nov":
            if args.data_path_hmmt_nov:
                data = _read_json_or_jsonl(args.data_path_hmmt_nov)
            else:
                try:
                    data = _load_hf_dataset(args.hf_dataset_hmmt_nov, args.hf_config_hmmt_nov, args.hf_split_hmmt_nov)
                except Exception as e:
                    if os.getenv("DEBUG"):
                        print(f"[hmmt_nov] HF load failed ({e}); falling back to local files under {args.data_dir}")
                    data_path = _resolve_default_data_path(args.data_dir, "hmmt_nov", args.split)
                    data = _read_json_or_jsonl(data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if args.num_samples:
            data = data[: args.num_samples]

        data_subset = data[rank::world_size]
        if not data_subset:
            continue

        out_path = os.path.join(out_root, f"{dataset}.jsonl")
        get_pred(rank, data_subset, dataset, args, model, tokenizer, out_path)
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="my_model")
    parser.add_argument("--ws", default=1, type=int, help="world size")
    parser.add_argument("--task", default="gsm8k,aime2024", type=str, help="Comma-separated: gsm8k,aime2024,hmmt_nov")
    parser.add_argument("--split", default="test", type=str, help="Dataset split name (used for default path resolution)")
    parser.add_argument("--data_dir", default=DATA_PREFIX_PATH, type=str, help="Root folder for datasets")
    parser.add_argument("--data_path_gsm8k", default=None, type=str)
    parser.add_argument("--data_path_aime2024", default=None, type=str)
    parser.add_argument("--data_path_hmmt_nov", default=None, type=str)
    parser.add_argument("--hf_dataset_gsm8k", default=DEFAULT_GSM8K_DATASET[0], type=str)
    parser.add_argument("--hf_config_gsm8k", default=DEFAULT_GSM8K_DATASET[1], type=str)
    parser.add_argument("--hf_split_gsm8k", default=DEFAULT_GSM8K_DATASET[2], type=str)
    parser.add_argument("--hf_dataset_aime2024", default=DEFAULT_AIME2024_DATASET[0], type=str)
    parser.add_argument("--hf_config_aime2024", default=DEFAULT_AIME2024_DATASET[1], type=str)
    parser.add_argument("--hf_split_aime2024", default=DEFAULT_AIME2024_DATASET[2], type=str)
    parser.add_argument("--hf_dataset_hmmt_nov", default=DEFAULT_HMMT_NOV_DATASET[0], type=str)
    parser.add_argument("--hf_config_hmmt_nov", default=DEFAULT_HMMT_NOV_DATASET[1], type=str)
    parser.add_argument("--hf_split_hmmt_nov", default=DEFAULT_HMMT_NOV_DATASET[2], type=str)

    # DeltaKV related arguments (aligned with benchmark/long_bench/pred.py)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--compressor_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--model_cls", type=str, default="deltakv")
    parser.add_argument("--backend", type=str, default="deltakv", choices=["hf", "sparsevllm"])
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples per task")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--no_chat_template", action="store_true", help="Do not use chat template")
    parser.add_argument("--hyper_param", type=str, default=None, help="Path to JSON file or inline JSON string")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=131000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--think_prefix", type=str, default="<think>\n")
    parser.add_argument("--no_force_think_prefix", action="store_false", dest="force_think_prefix")
    parser.set_defaults(force_think_prefix=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    if not (0.5 <= float(args.temperature) <= 0.7):
        raise ValueError(f"--temperature must be within [0.5, 0.7], got {args.temperature}")

    model_name = args.model
    compressor_name = os.path.basename(args.compressor_path.rstrip("/")) if args.compressor_path else "None"

    datasets = [d.strip() for d in args.task.split(",") if d.strip()]
    time_tag = datetime.now().strftime("%m%d_%H%M")
    out_root = os.path.join(BASE_PATH, f"benchmark/math_bench/pred/{model_name}/{compressor_name}_{time_tag}")
    os.makedirs(out_root, exist_ok=True)
    print(f"Results will be saved in: {out_root}")

    for dataset in datasets:
        with open(os.path.join(out_root, f"{dataset}.jsonl"), "w", encoding="utf-8") as f:
            f.write("")

    if args.ws > 1:
        processes = []
        for rank in range(args.ws):
            p = mp.Process(target=worker, args=(rank, args.ws, datasets, args, out_root))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        worker(0, 1, datasets, args, out_root)

    log_path = os.path.join(BASE_PATH, "mathbench_eval.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: python {' '.join(sys.argv)}\n")
        f.write(f"Output Root: {out_root}\n")
        f.write(f"Args: {json.dumps(vars(args), indent=2)}\n")
        f.write("-" * 80 + "\n")

    print(f"Evaluating {out_root} ...")
    eval_cmd = [sys.executable, "benchmark/math_bench/eval.py", "--path", out_root]
    try:
        subprocess.run(eval_cmd, check=True)
        result_path = os.path.join(out_root, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("Evaluation Results (pass@1):\n")
                f.write(json.dumps(scores, indent=4, ensure_ascii=False))
                f.write("\n" + "=" * 80 + "\n\n")
            print(f"Wrote eval results to: {log_path}")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
