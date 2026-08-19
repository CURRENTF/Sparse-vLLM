# SPDX-License-Identifier: Apache-2.0
"""LongBench prediction runner using upstream vLLM."""
import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_PREFIX_PATH = os.getenv("SPARSEVLLM_LONGBENCH_DATA_DIR") or os.getenv("SPARSEVLLM_DATA_DIR") or "data/LongBench"
NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

def get_longbench_data_path(dataset, use_longbench_e=False):
    suffix = "_e" if use_longbench_e else ""
    return os.path.join(DATA_PREFIX_PATH, "data", f"{dataset}{suffix}.jsonl")

def build_chat_prompt(tokenizer, prompt_template, input_text, context_text):
    formatted = prompt_template.format(input=input_text, context=context_text)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": formatted}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return formatted
    return formatted

def parse_args():
    parser = argparse.ArgumentParser(description="Run LongBench evaluation on upstream vLLM.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--task", type=str, default="qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--samples_per_task", type=int, default=-1)
    parser.add_argument("--min_required_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--e", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.makedirs(args.output_root, exist_ok=True)
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Initializing upstream vLLM (TP={args.tensor_parallel_size}, max_model_len={args.max_model_len})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    dataset2prompt = json.load(open("benchmark/long_bench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/long_bench/config/dataset2maxlen.json", "r"))
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]

    for dataset in tasks:
        print(f"\n[{dataset}] Starting LongBench evaluation...")
        prompt_template = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_path = get_longbench_data_path(dataset, args.e)
        
        with open(data_path, "r", encoding="utf-8") as f:
            all_lines = [json.loads(l) for l in f if l.strip()]
            if args.samples_per_task > 0:
                lines = all_lines[:args.samples_per_task]
            else:
                lines = all_lines
        
        prompts = []
        for item in lines:
            context = item["context"]
            input_text = item["input"]
            if dataset in NO_CHAT_TEMPLATE_DATASETS:
                p = prompt_template.format(input=input_text, context=context)
            else:
                p = build_chat_prompt(tokenizer, prompt_template, input_text, context)
            prompts.append(p)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_gen,
        )

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        out_file = os.path.join(args.output_root, f"{dataset}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for item, out in zip(lines, outputs):
                pred_text = out.outputs[0].text
                rec = {
                    "pred": pred_text,
                    "answers": item.get("answers", []),
                    "all_classes": item.get("all_classes", None),
                    "length": item.get("length", 0),
                    "_id": item.get("_id", ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{dataset}] Finished {len(lines)} samples -> {out_file}")

    print(f"\nRunning LongBench automatic evaluation on {args.output_root}...")
    eval_python = "/data2/haojitai/conda_envs/sparse-vllm-glm47-torch211/bin/python"
    eval_cmd = [eval_python, "benchmark/long_bench/eval.py", "--path", args.output_root]
    if args.e:
        eval_cmd.append("--e")
    subprocess.run(eval_cmd, check=True)

if __name__ == "__main__":
    main()
