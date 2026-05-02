#!/usr/bin/env python3
import argparse
import gc
import io
import json
import re
import string
import time
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor


PAPER_AI2D_TARGETS = {
    "llava-onevision-qwen2-0.5b-ov-hf": 57.1,
    "llava-onevision-qwen2-7b-ov-hf": 81.4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark vanilla HF LLaVA-OneVision on AI2D with the LMMs-Eval "
            "default prompt format used by the LLaVA-OneVision paper."
        )
    )
    parser.add_argument("--model_path", default="/data2/haojitai/models/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--dataset_path", default="lmms-lab/ai2d")
    parser.add_argument("--dataset_dir", default="/data2/haojitai/datasets/lmms-lab_ai2d")
    parser.add_argument("--dataset_cache_dir", default="/data2/haojitai/datasets/hf_cache")
    parser.add_argument("--output_dir", default="/data2/haojitai/datasets/llava_onevision_ai2d_vanilla")
    parser.add_argument("--num_samples", type=int, default=32, help="Use -1 for the full AI2D test split.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--print_records", action="store_true", help="Print per-sample records in the terminal summary.")
    return parser.parse_args()


def resolve_local_parquet(dataset_dir: Path):
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return []
    return sorted(str(path) for path in data_dir.glob("test-*.parquet"))


def load_ai2d_rows(args):
    dataset_dir = Path(args.dataset_dir)
    parquet_files = resolve_local_parquet(dataset_dir)
    if parquet_files:
        dataset = load_dataset("parquet", data_files={"test": parquet_files}, split="test")
    else:
        dataset = load_dataset(args.dataset_path, split="test", cache_dir=args.dataset_cache_dir)

    if args.num_samples >= 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    return [dataset[idx] for idx in range(len(dataset))]


def normalize_for_match(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(str(text).lower().translate(table).split())


def extract_ai2d_choice(prediction: str, choices: list[str]) -> str:
    text = prediction.strip()
    direct = re.match(r"^\s*([A-Z])(?:[\.\):\s]|$)", text, flags=re.IGNORECASE)
    if direct:
        return direct.group(1).upper()

    normalized = normalize_for_match(text)
    for idx, choice in enumerate(choices):
        if normalized == normalize_for_match(choice):
            return chr(ord("A") + idx)

    for idx, choice in enumerate(choices):
        choice_norm = normalize_for_match(choice)
        if choice_norm and choice_norm in normalized:
            return chr(ord("A") + idx)
    return text[:32]


def to_rgb_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    raise TypeError(f"Unsupported AI2D image value: {type(image)!r}")


def build_ai2d_prompt(processor, row):
    choices = list(row["options"])
    choices_str = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
    text = (
        f"{row['question']}\n"
        f"{choices_str}\n"
        "Answer with the option's letter from the given choices directly."
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def batch_to_device(inputs, device, dtype):
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
    return inputs


def ensure_left_padding(processor):
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def iter_batches(rows, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


def infer_paper_target(model_path: str):
    model_name = Path(model_path.rstrip("/")).name.lower()
    return PAPER_AI2D_TARGETS.get(model_name)


@torch.inference_mode()
def run_benchmark(args):
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    rows = load_ai2d_rows(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] rows={len(rows)} dataset_dir={args.dataset_dir} output_dir={output_dir}", flush=True)

    processor = LlavaOnevisionProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    ensure_left_padding(processor)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    torch.cuda.reset_peak_memory_stats(device)
    records = []
    total_new_tokens = 0
    total_time = 0.0
    total_batches = 0
    log_every = max(1, int(args.log_every))
    batch_size = max(1, int(args.batch_size))

    for batch_start, batch_rows in iter_batches(rows, batch_size):
        images = [to_rgb_image(row["image"]) for row in batch_rows]
        prompts = [build_ai2d_prompt(processor, row) for row in batch_rows]
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
        visual_token_counts = (inputs["input_ids"] == model.config.image_token_id).sum(dim=1).tolist()
        inputs = batch_to_device(inputs, device, dtype)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[:, input_len:]
        decoded_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
        new_tokens = int(generated_ids.shape[1])
        batch_new_tokens = new_tokens * len(batch_rows)
        total_new_tokens += batch_new_tokens
        total_time += elapsed
        total_batches += 1
        batch_tok_s = batch_new_tokens / elapsed if elapsed > 0 else 0.0

        for offset, (row, decoded) in enumerate(zip(batch_rows, decoded_batch)):
            sample_idx = batch_start + offset + 1
            choices = list(row["options"])
            prediction = decoded.strip()
            pred_choice = extract_ai2d_choice(prediction, choices)
            target_choice = chr(ord("A") + int(row["answer"]))
            correct = pred_choice == target_choice
            record = {
                "sample_idx": sample_idx - 1,
                "question": row["question"],
                "options": choices,
                "target_choice": target_choice,
                "prediction": prediction,
                "pred_choice": pred_choice,
                "correct": correct,
                "input_tokens": int(input_token_counts[offset]),
                "padded_input_tokens": input_len,
                "visual_tokens": int(visual_token_counts[offset]),
                "new_tokens": new_tokens,
                "batch_seconds": elapsed,
                "batch_new_tokens_per_s": batch_tok_s,
            }
            records.append(record)
            if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
                print(
                    f"[ai2d_vanilla] {sample_idx}/{len(rows)} batch={len(batch_rows)} "
                    f"input={input_token_counts[offset]} padded={input_len} visual={visual_token_counts[offset]} "
                    f"pred={pred_choice!r} target={target_choice!r} correct={correct} "
                    f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} raw={prediction[:80]!r}",
                    flush=True,
                )

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    accuracy = sum(record["correct"] for record in records) / max(len(records), 1)
    paper_target = infer_paper_target(args.model_path)
    result = {
        "benchmark": "ai2d",
        "method": "vanilla",
        "model_path": args.model_path,
        "num_samples": len(records),
        "batch_size": batch_size,
        "total_batches": total_batches,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "paper_ai2d_target_percent": paper_target,
        "delta_vs_paper_points": (accuracy * 100.0 - paper_target) if paper_target is not None else None,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "peak_memory_gb": peak_gb,
        "records": records,
    }

    out_path = output_dir / "last_ai2d_vanilla_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("[summary]")
    if args.print_records:
        printable_result = result
    else:
        printable_result = dict(result)
        printable_result["records"] = f"{len(records)} records saved to {out_path}"
    print(json.dumps(printable_result, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
