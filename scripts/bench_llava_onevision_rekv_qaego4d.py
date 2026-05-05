#!/usr/bin/env python3
import argparse
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


CHOICE_LETTERS = "ABCDEFGH"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLaVA-OneVision on QAEGO4D-test-mc using the ReKV paper "
            "multiple-choice protocol: 0.5 FPS video stream, 64 QA context frames, "
            "ReKV-style prompt, and accuracy."
        )
    )
    parser.add_argument("--model_path", default="/data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--deltakv_checkpoint_path", default="none")
    parser.add_argument("--dataset_dir", default="/data2/haojitai/datasets/rekv_qaego4d")
    parser.add_argument("--anno_path", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--output_dir", default="/data2/haojitai/datasets/llava_onevision_rekv_qaego4d")
    parser.add_argument("--methods", default="vanilla,deltakv_delta_quant")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of QA pairs to evaluate. Use -1 for all 500.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--max_context_frames", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="float16", choices=["bfloat16", "float16"])
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
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(completed.stdout)


def ffprobe_video_info(video_path: Path) -> tuple[float, float, int]:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,duration,nb_frames",
            "-of",
            "json",
            str(video_path),
        ]
    )
    stream = data["streams"][0]
    duration = float(stream.get("duration") or 0.0)
    rate = stream.get("avg_frame_rate") or "0/1"
    num, den = rate.split("/")
    fps = float(num) / max(float(den), 1.0)
    nb_frames = stream.get("nb_frames")
    if nb_frames and str(nb_frames).isdigit():
        frame_count = int(nb_frames)
    elif duration > 0 and fps > 0:
        frame_count = max(1, int(round(duration * fps)))
    else:
        frame_count = 1
    return duration, fps, frame_count


def ffprobe_video(video_path: Path) -> tuple[float, float]:
    duration, fps, _ = ffprobe_video_info(video_path)
    return duration, fps


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


def ensure_frame(video_path: Path, timestamp: float, output_path: Path, duration: float) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    if output_path.exists():
        output_path.unlink()
    fallback_timestamps = [
        timestamp,
        max(0.0, timestamp - 0.5),
        min(max(duration - 0.1, 0.0), timestamp + 0.5),
        0.0,
        max(duration - 0.5, 0.0),
    ]
    for candidate in fallback_timestamps:
        if ffmpeg_extract_frame(video_path, candidate, output_path):
            return True
        if output_path.exists():
            output_path.unlink()
    return False


def ffmpeg_extract_frames_by_index(video_path: Path, frame_indices: list[int], output_paths: list[Path]) -> bool:
    if not frame_indices:
        return False
    if len(frame_indices) != len(output_paths):
        raise ValueError("frame_indices and output_paths must have the same length.")

    temp_dir = output_paths[0].parent / ".extract_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # ReKV samples by frame index through decord. A single select pass is much
        # faster than spawning ffmpeg once per frame and keeps the same indexing.
        select_expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)
        temp_pattern = temp_dir / "frame_%03d.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select={select_expr}",
            "-vsync",
            "0",
            "-q:v",
            "2",
            str(temp_pattern),
        ]
        completed = subprocess.run(command)
        if completed.returncode != 0:
            return False
        extracted = sorted(temp_dir.glob("frame_*.jpg"))
        if not extracted:
            return False
        first_valid = extracted[0]
        for idx, output_path in enumerate(output_paths):
            source = extracted[idx] if idx < len(extracted) else first_valid
            shutil.copyfile(source, output_path)
        return all(path.exists() and path.stat().st_size > 0 for path in output_paths)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def decord_context_frame_indices(video_path: Path, sample_fps: float, max_context_frames: int):
    try:
        from decord import VideoReader, cpu
    except Exception:
        return None
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        avg_fps = float(vr.get_avg_fps())
        rounded_fps = max(round(avg_fps), 1)
        step_frames = max(int(rounded_fps / sample_fps), 1)
        frame_indices = list(range(0, len(vr), step_frames)) or [0]
        if len(frame_indices) > max_context_frames:
            keep_indices = torch.linspace(0, len(frame_indices) - 1, steps=max_context_frames).round().long().tolist()
            frame_indices = [frame_indices[idx] for idx in keep_indices]
        frame_indices = [min(max(0, idx), max(len(vr) - 1, 0)) for idx in frame_indices]
        duration = len(vr) / max(avg_fps, 1e-6)
        return duration, avg_fps, frame_indices
    except Exception:
        return None


def decord_extract_frames_by_index(video_path: Path, frame_indices: list[int], output_paths: list[Path]) -> bool:
    try:
        from decord import VideoReader, cpu
    except Exception:
        return False
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        safe_indices = [min(max(0, idx), max(len(vr) - 1, 0)) for idx in frame_indices]
        batch = vr.get_batch(safe_indices).asnumpy()
        if len(batch) == 0:
            return False
        for frame, output_path in zip(batch, output_paths):
            Image.fromarray(frame).save(output_path, quality=95)
        return all(path.exists() and path.stat().st_size > 0 for path in output_paths)
    except Exception:
        return False


def frame_cache_key(video_path: Path, sample_fps: float, max_context_frames: int) -> str:
    raw = f"{video_path.resolve()}:{sample_fps:.6f}:{max_context_frames}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def rekv_context_frame_indices(video_path: Path, sample_fps: float, max_context_frames: int) -> tuple[float, float, list[int]]:
    duration, fps, frame_count = ffprobe_video_info(video_path)
    if duration <= 0:
        duration = 1.0
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    # ReKV uses frame_idx = range(0, len(video), int(fps / sample_fps)).
    rounded_fps = max(round(fps), 1)
    step_frames = max(int(rounded_fps / sample_fps), 1)
    frame_indices = list(range(0, max(frame_count, 1), step_frames)) or [0]
    if len(frame_indices) > max_context_frames:
        keep_indices = torch.linspace(0, len(frame_indices) - 1, steps=max_context_frames).round().long().tolist()
        frame_indices = [frame_indices[idx] for idx in keep_indices]
    frame_indices = [min(max(0, idx), max(frame_count - 1, 0)) for idx in frame_indices]
    return duration, fps, frame_indices


def rekv_context_timestamps(video_path: Path, sample_fps: float, max_context_frames: int) -> list[float]:
    _, fps, frame_indices = rekv_context_frame_indices(video_path, sample_fps, max_context_frames)
    return [idx / max(fps, 1e-6) for idx in frame_indices]


def load_video_frames(video_path: Path, args) -> list[Image.Image]:
    cache_root = Path(args.frame_cache_dir) if args.frame_cache_dir else Path(args.output_dir) / "frame_cache"
    cache_dir = cache_root / frame_cache_key(video_path, args.sample_fps, args.max_context_frames)
    if cache_dir.exists() and not args.reuse_frame_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    decord_indices = decord_context_frame_indices(video_path, args.sample_fps, args.max_context_frames)
    if decord_indices is not None:
        duration, fps, frame_indices = decord_indices
    else:
        duration, fps, frame_indices = rekv_context_frame_indices(video_path, args.sample_fps, args.max_context_frames)
    timestamps = [idx / max(fps, 1e-6) for idx in frame_indices]
    frame_paths = [cache_dir / f"frame_{idx:03d}.jpg" for idx in range(len(frame_indices))]
    if not (args.reuse_frame_cache and all(path.exists() and path.stat().st_size > 0 for path in frame_paths)):
        for frame_path in frame_paths:
            if frame_path.exists():
                frame_path.unlink()
        extracted = decord_extract_frames_by_index(video_path, frame_indices, frame_paths)
        if not extracted and not ffmpeg_extract_frames_by_index(video_path, frame_indices, frame_paths):
            first_valid_frame = None
            for timestamp, frame_path in zip(timestamps, frame_paths):
                if ensure_frame(video_path, timestamp, frame_path, duration):
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


def resolve_video_path(video_path: str, dataset_dir: Path, video_dir: Path) -> Path:
    raw = Path(video_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(dataset_dir / raw)
    candidates.append(video_dir / raw.name)
    candidates.append(video_dir / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve video path {video_path!r}; tried {[str(c) for c in candidates]}")


def load_qaego4d_rows(args):
    dataset_dir = Path(args.dataset_dir)
    anno_path = Path(args.anno_path) if args.anno_path else dataset_dir / "test_mc.json"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "videos"
    data = json.loads(anno_path.read_text())
    rows = []
    for video_sample in data:
        video_path = resolve_video_path(video_sample["video_path"], dataset_dir, video_dir)
        for conv_idx, sample in enumerate(video_sample["conversations"]):
            choices = list(sample["choices"])
            answer = sample["answer"] if sample["answer"] is not None else choices[0]
            correct_choice = CHOICE_LETTERS[choices.index(answer)]
            rows.append(
                {
                    "video_id": video_sample["video_id"],
                    "conv_idx": conv_idx,
                    "video_path": str(video_path),
                    "duration": float(video_sample.get("duration", 0.0) or 0.0),
                    "question": sample["question"],
                    "choices": choices,
                    "answer": answer,
                    "correct_choice": correct_choice,
                    "temporal_windows": sample.get("temporal_windows", []),
                }
            )

    start = max(0, int(args.sample_start))
    rows = rows[start:]
    if args.num_samples >= 0:
        rows = rows[: args.num_samples]
    return rows, {"anno_path": str(anno_path), "video_dir": str(video_dir), "num_videos": len(data)}


def build_rekv_prompt(processor, question: str, choices: list[str]) -> str:
    formatted_choices = "\n".join(
        f"({CHOICE_LETTERS[idx]}) {candidate}" for idx, candidate in enumerate(choices)
    )
    formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": formatted_question}]},
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True) + "Best option: ("


def extract_rekv_choice(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if ")" in text:
        index = text.index(")")
        if index > 0:
            return text[index - 1 : index].upper()
    return text[0].upper()


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
        videos = [load_video_frames(Path(row["video_path"]), args) for row in batch_rows]
        prompts = [build_rekv_prompt(processor, row["question"], row["choices"]) for row in batch_rows]
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
            prediction = extract_rekv_choice(decoded)
            correct = prediction == row["correct_choice"]
            record = {
                "video_id": row["video_id"],
                "conv_idx": row["conv_idx"],
                "video_path": row["video_path"],
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "correct_choice": row["correct_choice"],
                "pred_answer": decoded,
                "pred_choice": prediction,
                "qa_acc": float(correct) * 100.0,
                "input_tokens": int(input_token_counts[offset]),
                "padded_input_tokens": input_len,
                "video_tokens": int(video_token_counts[offset]),
                "new_tokens": new_tokens,
                "seconds": elapsed / len(batch_rows),
                "batch_seconds": elapsed,
                "new_tokens_per_s": new_tokens / (elapsed / len(batch_rows)) if elapsed > 0 else 0.0,
                "batch_new_tokens_per_s": batch_tok_s,
            }
            records.append(record)
            if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
                print(
                    f"[{method}] {sample_idx}/{len(rows)} video={row['video_id']} "
                    f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                    f"video_tokens={video_token_counts[offset]} new={new_tokens} "
                    f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} "
                    f"ans={row['correct_choice']} pred={prediction or '?'} ok={correct} raw={decoded[:80]!r}",
                    flush=True,
                )

    total = max(len(records), 1)
    result = {
        "method": method,
        "num_samples": len(records),
        "sample_fps": args.sample_fps,
        "max_context_frames": args.max_context_frames,
        "batch_size": effective_batch_size,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "mean_seconds": total_time / total,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "qa_acc": sum(record["qa_acc"] for record in records) / total,
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
            raise ValueError("QAEGO4D ReKV protocol script supports methods: vanilla, deltakv_delta_quant.")


def write_official_style_csv(result: dict, output_dir: Path):
    csv_path = output_dir / f"{result['method']}_results.csv"
    fieldnames = [
        "video_id",
        "question",
        "choices",
        "answer",
        "correct_choice",
        "pred_answer",
        "pred_choice",
        "qa_acc",
        "retrieve_size",
        "chunk_size",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in result["records"]:
            writer.writerow(
                {
                    "video_id": record["video_id"],
                    "question": record["question"],
                    "choices": record["choices"],
                    "answer": record["answer"],
                    "correct_choice": record["correct_choice"],
                    "pred_answer": record["pred_answer"],
                    "pred_choice": record["pred_choice"],
                    "qa_acc": record["qa_acc"],
                    "retrieve_size": result["max_context_frames"],
                    "chunk_size": 1,
                }
            )
    return csv_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    rows, dataset_info = load_qaego4d_rows(args)
    if not rows:
        raise RuntimeError("No QAEGO4D rows selected.")
    print(
        "[dataset] "
        f"rows={len(rows)} anno_path={dataset_info['anno_path']} video_dir={dataset_info['video_dir']} "
        f"sample_fps={args.sample_fps} max_context_frames={args.max_context_frames}",
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
        result["official_rekv_protocol"] = {
            "benchmark": "QAEGO4Dtest-mc",
            "metric": "accuracy",
            "sample_fps": args.sample_fps,
            "n_local": 15000,
            "retrieve_size_or_context_frames": args.max_context_frames,
            "prompt_style": "ReKV official MC prompt ending with 'Best option: ('",
        }
        result["csv_path"] = str(write_official_style_csv(result, output_dir))
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(results) == 2:
        base = next((item for item in results if item["method"] == "vanilla"), None)
        candidate = next((item for item in results if item["method"] != "vanilla"), None)
        if base and candidate:
            candidate["qa_acc_delta_vs_vanilla"] = candidate["qa_acc"] - base["qa_acc"]
            candidate["speedup_vs_vanilla"] = candidate["new_tokens_per_s"] / base["new_tokens_per_s"]
            candidate["memory_delta_gb_vs_vanilla"] = candidate["peak_memory_gb"] - base["peak_memory_gb"]

    out_path = output_dir / "last_rekv_qaego4d_result.json"
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
