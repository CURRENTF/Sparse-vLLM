import argparse
import gc
import json
import re
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sparsevllm import LLM, SamplingParams


if not hasattr(DynamicCache, "get_max_length"):
    def _dynamic_cache_get_max_length(self):
        return None


    DynamicCache.get_max_length = _dynamic_cache_get_max_length


def build_chat(tokenizer, prompt: str, no_chat_template: bool) -> str:
    if no_chat_template or not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def first_diff_index(left: str, right: str) -> int:
    for idx, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return idx
    return min(len(left), len(right))


def load_hotpotqa_records(
    model_path: str,
    tokenizer_path: str,
    dataset_path: str,
    prompt_config_path: str,
    num_samples: int,
    no_chat_template: bool,
    target_prompt_tokens: int | None,
) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_length = int(getattr(config, "max_position_embeddings", 32000))

    with open(prompt_config_path, "r", encoding="utf-8") as file:
        prompt_format = json.load(file)["hotpotqa"]

    records: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= num_samples:
                break

            sample = json.loads(line)
            prompt = prompt_format.format(**sample)
            tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            if tokenized.shape[0] > max_length:
                half = max_length // 2
                prompt = (
                    tokenizer.decode(tokenized[:half], skip_special_tokens=True)
                    + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
                )
                tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            if target_prompt_tokens is not None and tokenized.shape[0] > target_prompt_tokens:
                half = target_prompt_tokens // 2
                prompt = (
                    tokenizer.decode(tokenized[:half], skip_special_tokens=True)
                    + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
                )
                tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            prompt = build_chat(tokenizer, prompt, no_chat_template=no_chat_template)
            records.append(
                {
                    "sample_id": index,
                    "prompt": prompt,
                    "answers": sample["answers"],
                    "context_length": sample.get("length"),
                    "prompt_tokens": int(tokenized.shape[0]),
                }
            )

    return records


def run_hf_generation(
    model_path: str,
    tokenizer_path: str,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    outputs: list[str] = []
    device = torch.device("cuda:0")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = generated[0, inputs["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_sparsevllm_generation(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    max_model_len: int,
    chunk_prefill_size: int,
    gpu_memory_utilization: float,
) -> list[str]:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    llm = LLM(
        model_path,
        enforce_eager=True,
        max_model_len=max_model_len,
        chunk_prefill_size=chunk_prefill_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs: list[str] = []
    try:
        for prompt_batch in batched(prompts, batch_size):
            batch_outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
            outputs.extend(item["text"] for item in batch_outputs)
    finally:
        llm.exit()

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/root/autodl-fs/datasets/LongBench/data/hotpotqa.jsonl",
    )
    parser.add_argument(
        "--prompt_config_path",
        type=str,
        default="benchmark/long_bench/config/dataset2prompt.json",
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=121000)
    parser.add_argument("--chunk_prefill_size", type=int, default=16384)
    parser.add_argument("--target_prompt_tokens", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path
    records = load_hotpotqa_records(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        dataset_path=args.dataset_path,
        prompt_config_path=args.prompt_config_path,
        num_samples=args.num_samples,
        no_chat_template=args.no_chat_template,
        target_prompt_tokens=args.target_prompt_tokens,
    )
    prompts = [record["prompt"] for record in records]

    hf_outputs = run_hf_generation(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )
    sparse_outputs = run_sparsevllm_generation(
        model_path=args.model_path,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        chunk_prefill_size=args.chunk_prefill_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    comparisons = []
    raw_exact_matches = 0
    normalized_exact_matches = 0
    for record, hf_output, sparse_output in zip(records, hf_outputs, sparse_outputs):
        raw_equal = hf_output == sparse_output
        normalized_equal = normalize_text(hf_output) == normalize_text(sparse_output)
        raw_exact_matches += int(raw_equal)
        normalized_exact_matches += int(normalized_equal)

        comparisons.append(
            {
                "sample_id": record["sample_id"],
                "answers": record["answers"],
                "context_length": record["context_length"],
                "prompt_tokens": record["prompt_tokens"],
                "hf_output": hf_output,
                "sparsevllm_output": sparse_output,
                "raw_equal": raw_equal,
                "normalized_equal": normalized_equal,
                "first_diff_index": first_diff_index(hf_output, sparse_output),
            }
        )

    summary = {
        "model_path": args.model_path,
        "num_samples": len(comparisons),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "chunk_prefill_size": args.chunk_prefill_size,
        "target_prompt_tokens": args.target_prompt_tokens,
        "raw_exact_matches": raw_exact_matches,
        "normalized_exact_matches": normalized_exact_matches,
        "comparisons": comparisons,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
