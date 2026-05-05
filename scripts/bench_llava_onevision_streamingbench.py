#!/usr/bin/env python3
import argparse
import ast
import csv
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaOnevisionProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_llava_onevision_visual_prune import (  # noqa: E402
    batch_to_device,
    ensure_left_padding,
    load_llava_delta_quant_model,
    load_vanilla_model,
)


TASK_CSV_FILES = {
    "real": "Real_Time_Visual_Understanding.csv",
    "omni": "Omni_Source_Understanding.csv",
    "contextual": "Contextual_Understanding.csv",
    "sqa": "Sequential_Question_Answering.csv",
}

TASK_VIDEO_HINTS = {
    "real": ("real", "visual", "real-time"),
    "omni": ("omni", "emotion", "alignment", "source"),
    "contextual": ("context", "anomaly", "misleading", "scene"),
    "sqa": ("sqa", "sequential"),
}

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
SAMPLE_RE = re.compile(r"sample[_ -]?(\d+)", re.IGNORECASE)


PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided.
Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question}
Options:
{options}

The best option is:"""

SQA_PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with a video and a multiple-choice question related to the video. Your task is to carefully analyze the video and the provided context to answer the question, choosing from the four options provided.
Respond with only the letter (A, B, C, or D) of the correct option.
{context}
Here is the question. Answer it and don't confuse it with the previous conversation.
Question: {question}
Options:
{options}

The best option is:"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LLaVA-OneVision on StreamingBench multiple-choice video QA "
            "with vanilla HF generation and no-checkpoint DeltaKV delta quantization."
        )
    )
    parser.add_argument("--model_path", default="/data2/haojitai/models/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--deltakv_checkpoint_path", default="none")
    parser.add_argument("--dataset_dir", default="/data2/haojitai/datasets/StreamingBench_hf")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--output_dir", default="/data2/haojitai/datasets/llava_onevision_streamingbench")
    parser.add_argument("--tasks", default="real", help="Comma-separated tasks: real, omni, contextual, sqa.")
    parser.add_argument("--methods", default="vanilla,deltakv_delta_quant")
    parser.add_argument("--num_samples", type=int, default=16, help="Rows to evaluate after filtering missing videos. Use -1 for all available rows.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument(
        "--context_seconds",
        type=float,
        default=60.0,
        help="Seconds before each query timestamp. Use -1 for all context from the start of the video.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--recent_keep_tokens", type=int, default=128)
    parser.add_argument("--sink_keep_tokens", type=int, default=8)
    parser.add_argument("--decode_keep_tokens", type=int, default=1024)
    parser.add_argument("--prefill_keep_tokens", type=int, default=4096)
    parser.add_argument("--hf_prefill_chunk_size", type=int, default=100000000)
    parser.add_argument("--chunk_prefill_accel_omnikv", action="store_true")
    parser.add_argument("--full_attention_layers", default="0,1,2,3,8,16,22")
    parser.add_argument("--visual_keep_ratio", type=float, default=1.0)
    parser.add_argument("--delta_quant_bits", type=int, default=4, choices=[4])
    parser.add_argument("--deltakv_center_ratio", type=float, default=0.1)
    parser.add_argument("--deltakv_neighbor_count", type=int, default=1)
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--reuse_frame_cache", action="store_true")
    parser.add_argument("--strict_videos", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def parse_timestamp(value: str) -> float:
    parts = [part.strip() for part in str(value).split(":") if part.strip()]
    if not parts:
        return 0.0
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def parse_options(value: str) -> list[str]:
    options = ast.literal_eval(value) if isinstance(value, str) else value
    options = [str(option).strip() for option in options]
    labels = ["A", "B", "C", "D"]
    normalized = []
    for label, option in zip(labels, options):
        if re.match(r"^[ABCD]\s*\.", option, re.IGNORECASE):
            normalized.append(option)
        else:
            normalized.append(f"{label}. {option}")
    return normalized


def extract_sample_id(question_id: str) -> int:
    match = SAMPLE_RE.search(question_id)
    if not match:
        raise ValueError(f"Cannot parse sample id from question_id={question_id!r}")
    return int(match.group(1))


def extract_question_index(question_id: str) -> int:
    match = re.search(r"_(\d+)$", question_id)
    return int(match.group(1)) if match else 0


def list_tasks(tasks: str) -> list[str]:
    selected = [task.strip().lower() for task in tasks.split(",") if task.strip()]
    if selected == ["all_mc"]:
        selected = ["real", "omni", "contextual", "sqa"]
    unknown = [task for task in selected if task not in TASK_CSV_FILES]
    if unknown:
        raise ValueError(f"Unknown StreamingBench task(s): {unknown}. Supported: {sorted(TASK_CSV_FILES)}")
    return selected


def score_video_candidate(path: Path, task: str, sample_id: int) -> tuple[int, str]:
    lower = str(path).lower()
    score = 0
    for hint in TASK_VIDEO_HINTS.get(task, ()):
        if hint in lower:
            score += 2
    if re.search(fr"sample[_ -]?{sample_id}\b", path.stem, re.IGNORECASE):
        score += 4
    return -score, str(path)


def build_video_index(video_dir: Path) -> dict[int, list[Path]]:
    index: dict[int, list[Path]] = {}
    if not video_dir.exists():
        return index
    for path in video_dir.rglob("*"):
        if path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        match = SAMPLE_RE.search(str(path))
        if match:
            index.setdefault(int(match.group(1)), []).append(path)
    return index


def resolve_video_path(video_index: dict[int, list[Path]], task: str, sample_id: int) -> Path | None:
    candidates = video_index.get(sample_id, [])
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: score_video_candidate(path, task, sample_id))[0]


def build_sqa_context(previous: list[dict]) -> str:
    if not previous:
        return ""
    parts = [
        "Here are the contextual information related to the video. "
        "Please answer the questions based on the contextual information:"
    ]
    for item in previous:
        parts.append(
            f"At timestamp {item['time_stamp']}, the following question and answer occurred: "
            f"Question: {item['question']}; Options: {', '.join(item['options'])}; Answer: {item['answer']};"
        )
    return " ".join(parts)


def load_streamingbench_rows(args):
    dataset_dir = Path(args.dataset_dir)
    csv_dir = Path(args.csv_dir) if args.csv_dir else dataset_dir / "StreamingBench"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "videos"
    video_index = build_video_index(video_dir)
    selected_rows = []
    missing_videos = []

    for task in list_tasks(args.tasks):
        csv_path = csv_dir / TASK_CSV_FILES[task]
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing StreamingBench CSV for task={task}: {csv_path}")

        task_rows = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                question_id = raw["question_id"]
                sample_id = extract_sample_id(question_id)
                video_path = resolve_video_path(video_index, task, sample_id)
                if video_path is None:
                    missing_videos.append({"task": task, "question_id": question_id, "sample_id": sample_id})
                    continue
                task_rows.append(
                    {
                        "task": task,
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "question_index": extract_question_index(question_id),
                        "task_type": raw["task_type"],
                        "question": raw["question"],
                        "time_stamp": raw["time_stamp"],
                        "timestamp_seconds": parse_timestamp(raw["time_stamp"]),
                        "answer": str(raw["answer"]).strip().upper()[:1],
                        "options": parse_options(raw["options"]),
                        "frames_required": raw.get("frames_required", ""),
                        "temporal_clue_type": raw.get("temporal_clue_type", ""),
                        "video_path": str(video_path),
                    }
                )

        if task == "sqa":
            history: dict[int, list[dict]] = {}
            task_rows.sort(key=lambda item: (item["sample_id"], item["question_index"]))
            for row in task_rows:
                previous = history.setdefault(row["sample_id"], [])
                row["context"] = build_sqa_context(previous)
                previous.append(row)
        else:
            for row in task_rows:
                row["context"] = ""

        selected_rows.extend(task_rows)

    if args.strict_videos and missing_videos:
        raise FileNotFoundError(f"Missing videos for {len(missing_videos)} rows; first missing: {missing_videos[:5]}")

    start = max(0, int(args.sample_start))
    rows = selected_rows[start:]
    if args.num_samples >= 0:
        rows = rows[: args.num_samples]
    return rows, {
        "csv_dir": str(csv_dir),
        "video_dir": str(video_dir),
        "indexed_video_count": sum(len(items) for items in video_index.values()),
        "missing_video_rows": len(missing_videos),
        "missing_video_examples": missing_videos[:10],
    }


def ffmpeg_extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(timestamp, 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command)
    return completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


def ensure_frame(video_path: Path, timestamp: float, output_path: Path, fallback_timestamps: list[float]) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    if output_path.exists():
        output_path.unlink()
    for candidate in [timestamp, *fallback_timestamps]:
        if ffmpeg_extract_frame(video_path, candidate, output_path):
            return True
        if output_path.exists():
            output_path.unlink()
    return False


def frame_cache_key(video_path: Path, start_seconds: float, end_seconds: float, num_frames: int) -> str:
    raw = f"{video_path.resolve()}:{start_seconds:.3f}:{end_seconds:.3f}:{num_frames}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def load_video_frames(row: dict, args) -> list[Image.Image]:
    video_path = Path(row["video_path"])
    end_seconds = max(0.0, float(row["timestamp_seconds"]))
    if args.context_seconds < 0:
        start_seconds = 0.0
    else:
        start_seconds = max(0.0, end_seconds - float(args.context_seconds))
    if end_seconds <= start_seconds:
        end_seconds = start_seconds + 0.2

    cache_root = Path(args.frame_cache_dir) if args.frame_cache_dir else Path(args.output_dir) / "frame_cache"
    cache_key = frame_cache_key(video_path, start_seconds, end_seconds, args.num_video_frames)
    cache_dir = cache_root / cache_key
    if cache_dir.exists() and not args.reuse_frame_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.num_video_frames <= 1:
        timestamps = [(start_seconds + end_seconds) / 2.0]
    else:
        step = (end_seconds - start_seconds) / (args.num_video_frames - 1)
        timestamps = [start_seconds + step * idx for idx in range(args.num_video_frames)]

    frame_paths = [cache_dir / f"frame_{idx:03d}.jpg" for idx in range(len(timestamps))]
    first_valid_frame = None
    for timestamp, frame_path in zip(timestamps, frame_paths):
        if frame_path.exists() and args.reuse_frame_cache:
            first_valid_frame = first_valid_frame or frame_path
            continue
        fallback_timestamps = [
            max(start_seconds, timestamp - 0.5),
            max(start_seconds, timestamp - 1.0),
            start_seconds,
            max(0.0, end_seconds - 0.5),
            0.0,
        ]
        if ensure_frame(video_path, timestamp, frame_path, fallback_timestamps):
            first_valid_frame = first_valid_frame or frame_path
            continue
        if first_valid_frame is not None:
            shutil.copyfile(first_valid_frame, frame_path)
        else:
            Image.new("RGB", (384, 384), color=(0, 0, 0)).save(frame_path)
            first_valid_frame = frame_path

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def build_prompt(processor, row: dict):
    options = "\n".join(row["options"])
    if row["task"] == "sqa":
        text = SQA_PROMPT_TEMPLATE.format(context=row.get("context", ""), question=row["question"], options=options)
    else:
        text = PROMPT_TEMPLATE.format(question=row["question"], options=options)
    conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def extract_choice(text: str) -> str:
    stripped = text.strip()
    if stripped[:1].upper() in {"A", "B", "C", "D"}:
        return stripped[:1].upper()
    match = CHOICE_RE.search(stripped)
    return match.group(1).upper() if match else ""


def iter_batches(rows, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


@torch.inference_mode()
def run_method(method: str, model, processor, rows: list[dict], args, dtype, device, policy=None):
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    total_new_tokens = 0
    total_time = 0.0
    total_batches = 0
    effective_batch_size = max(1, int(args.batch_size))
    if effective_batch_size > 1:
        ensure_left_padding(processor)
    elif getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        ensure_left_padding(processor)

    log_every = max(1, int(args.log_every))
    for batch_start, batch_rows in iter_batches(rows, effective_batch_size):
        videos = [load_video_frames(row, args) for row in batch_rows]
        prompts = [build_prompt(processor, row) for row in batch_rows]
        processor_kwargs = {"text": prompts, "videos": videos, "return_tensors": "pt"}
        if len(batch_rows) > 1:
            processor_kwargs["padding"] = True
        inputs = processor(**processor_kwargs)
        input_len = int(inputs["input_ids"].shape[1])
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_token_counts = attention_mask.sum(dim=1).tolist()
        else:
            input_token_counts = [input_len for _ in batch_rows]
        video_token_counts = (inputs["input_ids"] == model.config.video_token_id).sum(dim=1).tolist()
        inputs = batch_to_device(inputs, device, dtype)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
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
            decoded = decoded.strip()
            prediction = extract_choice(decoded)
            correct = prediction == row["answer"]
            records.append(
                {
                    "task": row["task"],
                    "task_type": row["task_type"],
                    "question_id": row["question_id"],
                    "sample_id": row["sample_id"],
                    "time_stamp": row["time_stamp"],
                    "video_path": row["video_path"],
                    "input_tokens": int(input_token_counts[offset]),
                    "padded_input_tokens": input_len,
                    "video_tokens": int(video_token_counts[offset]),
                    "new_tokens": new_tokens,
                    "seconds": elapsed / len(batch_rows),
                    "batch_seconds": elapsed,
                    "new_tokens_per_s": new_tokens / (elapsed / len(batch_rows)) if elapsed > 0 else 0.0,
                    "batch_new_tokens_per_s": batch_tok_s,
                    "answer": row["answer"],
                    "prediction": prediction,
                    "raw_prediction": decoded,
                    "correct": correct,
                }
            )
            if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
                print(
                    f"[{method}] {sample_idx}/{len(rows)} task={row['task']} qid={row['question_id']} "
                    f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                    f"video_tokens={video_token_counts[offset]} new={new_tokens} "
                    f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} "
                    f"ans={row['answer']} pred={prediction or '?'} ok={correct} raw={decoded[:80]!r}",
                    flush=True,
                )

    total = max(len(records), 1)
    task_stats = {}
    for record in records:
        stats = task_stats.setdefault(record["task"], {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(record["correct"])
    for stats in task_stats.values():
        stats["accuracy"] = stats["correct"] / max(stats["total"], 1)

    result = {
        "method": method,
        "num_samples": len(records),
        "batch_size": effective_batch_size,
        "num_video_frames": args.num_video_frames,
        "context_seconds": args.context_seconds,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "mean_seconds": total_time / total,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "accuracy": sum(record["correct"] for record in records) / total,
        "task_stats": task_stats,
        "records": records,
    }
    if policy is not None:
        result["visual_cache_policy"] = policy
    return result


def iter_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield "vanilla", "vanilla"
        elif method in {"deltakv_delta_quant", "delta_quant", "llava_deltakv_delta_quant"}:
            yield raw_method, "deltakv_delta_quant"
        else:
            raise ValueError("StreamingBench script currently supports methods: vanilla, deltakv_delta_quant.")


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    rows, dataset_info = load_streamingbench_rows(args)
    if not rows:
        raise RuntimeError(
            "No StreamingBench rows with available videos were found. "
            f"Dataset info: {json.dumps(dataset_info, ensure_ascii=False)}"
        )
    print(
        "[dataset] "
        f"rows={len(rows)} tasks={args.tasks} csv_dir={dataset_info['csv_dir']} "
        f"video_dir={dataset_info['video_dir']} indexed_videos={dataset_info['indexed_video_count']} "
        f"missing_video_rows={dataset_info['missing_video_rows']}",
        flush=True,
    )

    processor = LlavaOnevisionProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    results = []
    for requested_method, method_kind in iter_methods(args.methods):
        if method_kind == "vanilla":
            model = load_vanilla_model(args, dtype, device)
            method_label = "vanilla"
            policy = None
        else:
            model, policy = load_llava_delta_quant_model(args, dtype, device)
            method_label = policy["method"]

        result = run_method(method_label, model, processor, rows, args, dtype, device, policy=policy)
        result["requested_method"] = requested_method
        result["dataset_info"] = dataset_info
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(results) == 2:
        base = next((item for item in results if item["method"] == "vanilla"), None)
        candidate = next((item for item in results if item["method"] != "vanilla"), None)
        if base and candidate:
            candidate["accuracy_delta_vs_vanilla"] = candidate["accuracy"] - base["accuracy"]
            candidate["speedup_vs_vanilla"] = candidate["new_tokens_per_s"] / base["new_tokens_per_s"]
            candidate["memory_delta_gb_vs_vanilla"] = candidate["peak_memory_gb"] - base["peak_memory_gb"]

    out_path = Path(args.output_dir) / "last_streamingbench_result.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print("[summary]")
    if args.print_records:
        printable_results = results
    else:
        printable_results = []
        for result in results:
            item = dict(result)
            item["records"] = f"{len(result.get('records', []))} records saved to {out_path}"
            printable_results.append(item)
    print(json.dumps(printable_results, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
