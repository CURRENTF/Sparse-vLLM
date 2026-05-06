#!/usr/bin/env python3
import argparse
import ast
import csv
import gc
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
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

LIVEVLM_TABLE4_TASKS = ("real", "omni", "contextual")
LIVEVLM_TABLE4_DISPLAY_SUBTASKS = {
    "Object Perception": ("OP", 80.38),
    "Causal Reasoning": ("CR", 74.22),
    "Clips Summarize": ("CS", 76.03),
    "Attribute Perception": ("ATP", 80.72),
    "Event Understanding": ("EU", 72.67),
    "Text-Rich Understanding": ("TR", 71.65),
    "Prospective Reasoning": ("PR", 67.59),
    "Spatial Understanding": ("SU", 65.45),
    "Action Perception": ("ACP", 65.72),
    "Counting": ("CT", 45.08),
    "Emotion Recognition": ("ER", 40.80),
    "Scene Understanding": ("SCU", 37.20),
    "Source Discrimination": ("SD", 33.60),
    "Multimodal Alignment": ("MA", 44.80),
}
LIVEVLM_TABLE4_OVERALL_EXTRA_SUBTASKS = {
    "Anomaly Context Understanding": ("ACU", None),
    "Misleading Context Recognition": ("MCU", None),
}
LIVEVLM_TABLE4_OVERALL_SUBTASKS = {
    **LIVEVLM_TABLE4_DISPLAY_SUBTASKS,
    **LIVEVLM_TABLE4_OVERALL_EXTRA_SUBTASKS,
}
LIVEVLM_TABLE4_EXPECTED_OVERALL = 58.85
LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS = 4000
LIVEVLM_TABLE4_REFERENCE = "LiveVLM arXiv:2505.15269 Table 4, LLaVA-OneVision-7B row"

TASK_VIDEO_HINTS = {
    "real": ("real", "visual", "real-time"),
    "omni": ("omni", "emotion", "alignment", "source", "scene"),
    "contextual": ("context", "anomaly", "misleading"),
    "sqa": ("sqa", "sequential"),
}
TASK_TYPE_VIDEO_HINTS = {
    "Emotion Recognition": ("emotion",),
    "Multimodal Alignment": ("multimodal", "alignment"),
    "Scene Understanding": ("scene",),
    "Source Discrimination": ("source",),
    "Anomaly Context Understanding": ("anomaly",),
    "Misleading Context Recognition": ("misleading",),
}

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
SAMPLE_RE = re.compile(r"sample[_ -]?(\d+)", re.IGNORECASE)


PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

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
    parser.add_argument(
        "--tasks",
        default="real",
        help="Comma-separated tasks: real, omni, contextual, sqa. Use all_mc or livevlm_table4 for aliases.",
    )
    parser.add_argument("--methods", default="vanilla,deltakv_delta_quant")
    parser.add_argument("--num_samples", type=int, default=16, help="Rows to evaluate after filtering missing videos. Use -1 for all available rows.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=32,
        help="Uniform video frames per query. StreamingBench's LLaVA-OneVision baseline uses 32.",
    )
    parser.add_argument(
        "--context_seconds",
        type=float,
        default=60.0,
        help="Seconds before each query timestamp. Use -1 for all context from the start of the video.",
    )
    parser.add_argument(
        "--streamingbench_profile",
        default="custom",
        choices=["custom", "official_60s", "official_all_context", "livevlm_table4"],
        help=(
            "Convenience profile. official_60s sets 32 frames and 60s query context, matching "
            "the StreamingBench main leaderboard. official_all_context sets 32 frames and all "
            "preceding context. livevlm_table4 matches LiveVLM Table 4's LLaVA-OneVision-7B "
            "baseline scope: real+omni MCQA, 32 frames, all preceding context."
        ),
    )
    parser.add_argument(
        "--frame_sampling_backend",
        default="decord",
        choices=["decord", "ffmpeg"],
        help="Use decord uniform frame indices to match the official model adapter, or ffmpeg timestamp extraction.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--choice_parse_mode",
        default="official_first_char",
        choices=["official_first_char", "robust"],
        help=(
            "official_first_char matches StreamingBench's count.py behavior after stripping whitespace; "
            "robust searches for A/B/C/D anywhere in the response for diagnostic runs."
        ),
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow_missing_videos",
        action="store_true",
        help="Explicitly allow shard-level evaluation by skipping annotation rows whose videos are absent.",
    )
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def apply_streamingbench_profile(args):
    if args.streamingbench_profile == "official_60s":
        args.num_video_frames = 32
        args.context_seconds = 60.0
    elif args.streamingbench_profile == "official_all_context":
        args.num_video_frames = 32
        args.context_seconds = -1.0
    elif args.streamingbench_profile == "livevlm_table4":
        args.num_video_frames = 32
        args.context_seconds = -1.0
        args.frame_sampling_backend = "decord"
        if args.tasks == "real":
            args.tasks = "livevlm_table4"
    return args


def parse_timestamp(value: str) -> float:
    parts = [part.strip() for part in str(value).split(":") if part.strip()]
    if not parts:
        raise ValueError(f"StreamingBench row has an empty timestamp: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def parse_options(value: str) -> list[str]:
    options = ast.literal_eval(value) if isinstance(value, str) else value
    options = [str(option).strip() for option in options]
    if len(options) != 4:
        raise ValueError(f"StreamingBench options must contain exactly 4 choices, got {len(options)}: {value!r}")
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
    elif selected == ["livevlm_table4"]:
        selected = list(LIVEVLM_TABLE4_TASKS)
    unknown = [task for task in selected if task not in TASK_CSV_FILES]
    if unknown:
        aliases = ["all_mc", "livevlm_table4"]
        raise ValueError(
            f"Unknown StreamingBench task(s): {unknown}. "
            f"Supported: {sorted(TASK_CSV_FILES)}; aliases: {aliases}"
        )
    return selected


def video_hints_for(task: str, task_type: str) -> tuple[str, ...]:
    return TASK_TYPE_VIDEO_HINTS.get(task_type, TASK_VIDEO_HINTS.get(task, ()))


def score_video_candidate(path: Path, hints: tuple[str, ...], sample_id: int) -> tuple[int, str]:
    lower = str(path).lower()
    score = 0
    for hint in hints:
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


def resolve_video_path(video_index: dict[int, list[Path]], task: str, task_type: str, sample_id: int) -> Path | None:
    candidates = video_index.get(sample_id, [])
    if not candidates:
        return None
    hints = video_hints_for(task, task_type)
    return sorted(candidates, key=lambda path: score_video_candidate(path, hints, sample_id))[0]


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


def count_by_key(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def load_streamingbench_rows(args):
    dataset_dir = Path(args.dataset_dir)
    csv_dir = Path(args.csv_dir) if args.csv_dir else dataset_dir / "StreamingBench"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"StreamingBench video directory does not exist: {video_dir}")
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
                video_path = resolve_video_path(video_index, task, raw["task_type"], sample_id)
                if video_path is None:
                    missing_videos.append({"task": task, "question_id": question_id, "sample_id": sample_id})
                    continue
                answer = str(raw["answer"]).strip().upper()[:1]
                if answer not in {"A", "B", "C", "D"}:
                    raise ValueError(
                        f"StreamingBench row has invalid answer={raw['answer']!r} "
                        f"for question_id={question_id!r}."
                    )
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
                        "answer": answer,
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

    if missing_videos and not args.allow_missing_videos:
        raise FileNotFoundError(
            f"Missing videos for {len(missing_videos)} StreamingBench rows; "
            f"first missing: {missing_videos[:5]}. "
            "Download the required video shards or pass --allow_missing_videos "
            "when intentionally running a partial shard."
        )

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
        "selected_rows_before_slice": len(selected_rows),
        "evaluated_task_counts": count_by_key(rows, "task"),
        "evaluated_task_type_counts": count_by_key(rows, "task_type"),
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


def extract_required_frame(video_path: Path, timestamp: float, output_path: Path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    if output_path.exists():
        output_path.unlink()
    if ffmpeg_extract_frame(video_path, timestamp, output_path):
        return
    if output_path.exists():
        output_path.unlink()
    raise RuntimeError(
        f"Failed to extract required StreamingBench frame: "
        f"video={video_path} timestamp={timestamp:.3f}s output={output_path}"
    )


def frame_cache_key(video_path: Path, start_seconds: float, end_seconds: float, num_frames: int, backend: str) -> str:
    raw = f"{video_path.resolve()}:{start_seconds:.3f}:{end_seconds:.3f}:{num_frames}:{backend}:v3"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def context_bounds(row: dict, args) -> tuple[float, float]:
    end_seconds = max(0.0, float(row["timestamp_seconds"]))
    if args.context_seconds < 0:
        start_seconds = 0.0
    else:
        start_seconds = max(0.0, end_seconds - float(args.context_seconds))
    if end_seconds <= start_seconds:
        end_seconds = start_seconds + 0.2
    return start_seconds, end_seconds


def decord_extract_context_frames(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    num_frames: int,
    frame_paths: list[Path],
) -> None:
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise RuntimeError(
            "StreamingBench decord frame sampling requested, but decord is not installed. "
            "Install decord or pass --frame_sampling_backend ffmpeg."
        ) from e

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        fps = max(float(vr.get_avg_fps()), 1e-6)
        start_idx = min(max(int(round(start_seconds * fps)), 0), total_frames - 1)
        end_idx = min(max(int(round(end_seconds * fps)) - 1, start_idx), total_frames - 1)
        if num_frames <= 1:
            frame_indices = [(start_idx + end_idx) // 2]
        else:
            # Match the official StreamingBench LLaVA-OneVision adapter:
            # np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).
            denom = max(num_frames - 1, 1)
            frame_indices = [
                int(start_idx + (end_idx - start_idx) * idx / denom)
                for idx in range(num_frames)
            ]
        batch = vr.get_batch(frame_indices).asnumpy()
        if len(batch) != len(frame_paths):
            raise RuntimeError(
                f"Decord returned {len(batch)} frames, expected {len(frame_paths)} "
                f"for video={video_path}."
            )
        for frame, frame_path in zip(batch, frame_paths):
            Image.fromarray(frame).save(frame_path, quality=95)
        missing = [str(path) for path in frame_paths if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Decord extraction produced missing/empty frame files: {missing[:5]}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract StreamingBench frames with decord: "
            f"video={video_path} start={start_seconds:.3f}s end={end_seconds:.3f}s frames={num_frames}"
        ) from e


def load_video_frames(row: dict, args) -> list[Image.Image]:
    video_path = Path(row["video_path"])
    start_seconds, end_seconds = context_bounds(row, args)

    cache_root = Path(args.frame_cache_dir) if args.frame_cache_dir else Path(args.output_dir) / "frame_cache"
    cache_key = frame_cache_key(
        video_path,
        start_seconds,
        end_seconds,
        args.num_video_frames,
        args.frame_sampling_backend,
    )
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
    if args.reuse_frame_cache and all(path.exists() and path.stat().st_size > 0 for path in frame_paths):
        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        return frames

    for frame_path in frame_paths:
        if frame_path.exists():
            frame_path.unlink()

    if args.frame_sampling_backend == "decord":
        decord_extract_context_frames(video_path, start_seconds, end_seconds, len(frame_paths), frame_paths)
        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        return frames

    for timestamp, frame_path in zip(timestamps, frame_paths):
        extract_required_frame(video_path, timestamp, frame_path)

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


def extract_choice(text: str, mode: str) -> str:
    stripped = text.strip()
    first = stripped[:1].upper()
    if first in {"A", "B", "C", "D"}:
        return first
    if mode == "official_first_char":
        return first
    match = CHOICE_RE.search(stripped)
    return match.group(1).upper() if match else ""


def iter_batches(rows, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


def init_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to read git commit: {completed.stderr.strip()}")
    return completed.stdout.strip()


def build_run_info(args, dataset_info: dict, row_count: int) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "model_path": args.model_path,
        "methods": args.methods,
        "dataset_dir": args.dataset_dir,
        "csv_dir": dataset_info["csv_dir"],
        "video_dir": dataset_info["video_dir"],
        "tasks": args.tasks,
        "streamingbench_profile": args.streamingbench_profile,
        "num_video_frames": args.num_video_frames,
        "context_seconds": args.context_seconds,
        "frame_sampling_backend": args.frame_sampling_backend,
        "choice_parse_mode": args.choice_parse_mode,
        "prompt_template": PROMPT_TEMPLATE,
        "sqa_prompt_template": SQA_PROMPT_TEMPLATE,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
        "seed": args.seed,
        "sample_start": args.sample_start,
        "num_samples_arg": args.num_samples,
        "evaluated_sample_count": row_count,
        "dataset_info": dataset_info,
    }


def status_for_prediction(prediction: str) -> str:
    return "success" if prediction in {"A", "B", "C", "D"} else "parse_failed"


def add_accuracy(stats: dict, correct: bool):
    stats["total"] += 1
    stats["correct"] += int(correct)


def finalize_accuracy_stats(stats: dict):
    for item in stats.values():
        item["accuracy"] = item["correct"] / max(item["total"], 1)
        item["accuracy_pct"] = 100.0 * item["accuracy"]


def compute_livevlm_table4_stats(records: list[dict]) -> dict:
    subtasks = []
    overall_extra_subtasks = []
    overall = {"total": 0, "correct": 0, "status_counts": {}}
    by_task_type: dict[str, dict] = {}
    for record in records:
        if record["task_type"] not in LIVEVLM_TABLE4_OVERALL_SUBTASKS:
            continue
        stats = by_task_type.setdefault(record["task_type"], {"total": 0, "correct": 0, "status_counts": {}})
        add_accuracy(stats, bool(record["correct"]))
        add_accuracy(overall, bool(record["correct"]))
        stats["status_counts"][record["status"]] = stats["status_counts"].get(record["status"], 0) + 1
        overall["status_counts"][record["status"]] = overall["status_counts"].get(record["status"], 0) + 1

    for task_type, (abbr, expected) in LIVEVLM_TABLE4_DISPLAY_SUBTASKS.items():
        stats = by_task_type.get(task_type, {"total": 0, "correct": 0, "status_counts": {}})
        accuracy = stats["correct"] / max(stats["total"], 1)
        accuracy_pct = 100.0 * accuracy
        subtasks.append(
            {
                "abbr": abbr,
                "task_type": task_type,
                "total": stats["total"],
                "correct": stats["correct"],
                "status_counts": stats["status_counts"],
                "accuracy": accuracy,
                "accuracy_pct": accuracy_pct,
                "expected_llava_onevision_7b_pct": expected,
                "delta_vs_expected_pct": accuracy_pct - expected if stats["total"] > 0 else None,
            }
        )

    for task_type, (abbr, expected) in LIVEVLM_TABLE4_OVERALL_EXTRA_SUBTASKS.items():
        stats = by_task_type.get(task_type, {"total": 0, "correct": 0, "status_counts": {}})
        accuracy = stats["correct"] / max(stats["total"], 1)
        accuracy_pct = 100.0 * accuracy
        overall_extra_subtasks.append(
            {
                "abbr": abbr,
                "task_type": task_type,
                "total": stats["total"],
                "correct": stats["correct"],
                "status_counts": stats["status_counts"],
                "accuracy": accuracy,
                "accuracy_pct": accuracy_pct,
                "expected_llava_onevision_7b_pct": expected,
                "used_for_paper_overall": True,
            }
        )

    overall_accuracy = overall["correct"] / max(overall["total"], 1)
    overall_accuracy_pct = 100.0 * overall_accuracy
    return {
        "reference": LIVEVLM_TABLE4_REFERENCE,
        "metric": (
            "MCQA accuracy over all evaluated rows; parse_failed predictions are counted as incorrect. "
            "The displayed subtask list mirrors LiveVLM Table 4. The overall score also includes ACU/MCU "
            "rows from the current StreamingBench contextual split to match the paper's 4000-row scope."
        ),
        "expected_llava_onevision_7b_overall_pct": LIVEVLM_TABLE4_EXPECTED_OVERALL,
        "expected_overall_row_count": LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS,
        "overall": {
            "total": overall["total"],
            "correct": overall["correct"],
            "status_counts": overall["status_counts"],
            "accuracy": overall_accuracy,
            "accuracy_pct": overall_accuracy_pct,
            "matches_expected_row_count": overall["total"] == LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS,
            "delta_vs_expected_pct": overall_accuracy_pct - LIVEVLM_TABLE4_EXPECTED_OVERALL
            if overall["total"] > 0
            else None,
        },
        "subtasks": subtasks,
        "overall_extra_subtasks": overall_extra_subtasks,
    }


def validate_livevlm_table4_rows(args, rows: list[dict]) -> None:
    if args.streamingbench_profile != "livevlm_table4":
        return
    is_full_run = args.num_samples < 0 and int(args.sample_start) == 0 and not args.allow_missing_videos
    if not is_full_run:
        return

    counts = count_by_key(rows, "task_type")
    missing = [task_type for task_type in LIVEVLM_TABLE4_OVERALL_SUBTASKS if counts.get(task_type, 0) == 0]
    overall_total = sum(counts.get(task_type, 0) for task_type in LIVEVLM_TABLE4_OVERALL_SUBTASKS)
    if overall_total != LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS or missing:
        raise RuntimeError(
            "LiveVLM Table 4 full baseline requires the 4000-row StreamingBench scope. "
            f"Got overall_rows={overall_total}, expected={LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS}, "
            f"missing_task_types={missing}, counts={counts}."
        )


def save_method_artifacts(output_dir: Path, result: dict, run_info: dict):
    method = result["method"]
    raw_path = output_dir / f"{method}_raw_outputs.jsonl"
    parsed_path = output_dir / f"{method}_parsed_outputs.jsonl"
    records_path = output_dir / f"{method}_per_sample_results.jsonl"
    metrics_path = output_dir / f"{method}_aggregate_metrics.json"
    run_info_path = output_dir / "run_info.json"

    with raw_path.open("w", encoding="utf-8") as handle:
        for record in result["records"]:
            handle.write(
                json.dumps(
                    {
                        "question_id": record["question_id"],
                        "sample_id": record["sample_id"],
                        "task": record["task"],
                        "task_type": record["task_type"],
                        "raw_prediction": record["raw_prediction"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with parsed_path.open("w", encoding="utf-8") as handle:
        for record in result["records"]:
            handle.write(
                json.dumps(
                    {
                        "question_id": record["question_id"],
                        "prediction": record["prediction"],
                        "answer": record["answer"],
                        "status": record["status"],
                        "correct": record["correct"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with records_path.open("w", encoding="utf-8") as handle:
        for record in result["records"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {key: value for key, value in result.items() if key != "records"}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "raw_outputs": str(raw_path),
        "parsed_outputs": str(parsed_path),
        "per_sample_results": str(records_path),
        "aggregate_metrics": str(metrics_path),
        "run_info": str(run_info_path),
    }


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
            raw_decoded = decoded
            decoded = raw_decoded.strip()
            prediction = extract_choice(decoded, args.choice_parse_mode)
            status = status_for_prediction(prediction)
            correct = prediction == row["answer"]
            context_start, context_end = context_bounds(row, args)
            records.append(
                {
                    "status": status,
                    "task": row["task"],
                    "task_type": row["task_type"],
                    "question_id": row["question_id"],
                    "sample_id": row["sample_id"],
                    "time_stamp": row["time_stamp"],
                    "video_path": row["video_path"],
                    "context_start_seconds": context_start,
                    "context_end_seconds": context_end,
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
                    "raw_prediction": raw_decoded,
                    "parsed_text": decoded,
                    "correct": correct,
                }
            )
            if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
                print(
                    f"[{method}] {sample_idx}/{len(rows)} task={row['task']} qid={row['question_id']} "
                    f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                    f"video_tokens={video_token_counts[offset]} new={new_tokens} "
                    f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} "
                    f"status={status} ans={row['answer']} pred={prediction or '?'} ok={correct} raw={decoded[:80]!r}",
                    flush=True,
                )

    total = max(len(records), 1)
    task_stats = {}
    task_type_stats = {}
    status_counts = {}
    for record in records:
        stats = task_stats.setdefault(record["task"], {"total": 0, "correct": 0})
        add_accuracy(stats, bool(record["correct"]))
        type_stats = task_type_stats.setdefault(record["task_type"], {"total": 0, "correct": 0})
        add_accuracy(type_stats, bool(record["correct"]))
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
    finalize_accuracy_stats(task_stats)
    finalize_accuracy_stats(task_type_stats)

    result = {
        "method": method,
        "num_samples": len(records),
        "status_counts": status_counts,
        "batch_size": effective_batch_size,
        "streamingbench_profile": args.streamingbench_profile,
        "num_video_frames": args.num_video_frames,
        "context_seconds": args.context_seconds,
        "frame_sampling_backend": args.frame_sampling_backend,
        "choice_parse_mode": args.choice_parse_mode,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "mean_seconds": total_time / total,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "accuracy": sum(record["correct"] for record in records) / total,
        "accuracy_pct": 100.0 * sum(record["correct"] for record in records) / total,
        "task_stats": task_stats,
        "task_type_stats": task_type_stats,
        "livevlm_table4_stats": compute_livevlm_table4_stats(records),
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
    args = apply_streamingbench_profile(args)
    init_seed(args.seed)
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
    validate_livevlm_table4_rows(args, rows)
    print(
        "[dataset] "
        f"rows={len(rows)} tasks={args.tasks} csv_dir={dataset_info['csv_dir']} "
        f"video_dir={dataset_info['video_dir']} indexed_videos={dataset_info['indexed_video_count']} "
        f"missing_video_rows={dataset_info['missing_video_rows']}",
        flush=True,
    )
    run_info = build_run_info(args, dataset_info, len(rows))

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
        result["artifact_paths"] = save_method_artifacts(Path(args.output_dir), result, run_info)
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
