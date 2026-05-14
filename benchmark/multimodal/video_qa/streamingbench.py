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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaOnevisionProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS = {
    "Object Perception": 369,
    "Causal Reasoning": 128,
    "Clips Summarize": 317,
    "Attribute Perception": 312,
    "Event Understanding": 159,
    "Text-Rich Understanding": 321,
    "Prospective Reasoning": 108,
    "Spatial Understanding": 246,
    "Action Perception": 352,
    "Counting": 188,
    "Emotion Recognition": 250,
    "Scene Understanding": 250,
    "Source Discrimination": 250,
    "Multimodal Alignment": 250,
    "Anomaly Context Understanding": 250,
    "Misleading Context Recognition": 250,
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
    parser.add_argument(
        "--image_processor_use_fast",
        action="store_true",
        help="Use the fast Transformers image/video processor implementation when available.",
    )
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
    parser.add_argument("--svllm_chunk_prefill_size", type=int, default=8192)
    parser.add_argument("--svllm_max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--svllm_max_num_seqs_in_batch", type=int, default=32)
    parser.add_argument("--svllm_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--svllm_mlp_seq_chunk_size", type=int, default=0)
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--reuse_frame_cache", action="store_true")
    parser.add_argument(
        "--frame_load_workers",
        type=int,
        default=1,
        help="Thread workers for loading cached frame images within each evaluation batch.",
    )
    parser.add_argument(
        "--preprocess_prefetch_batches",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Prefetch the next processed batch on a background thread while the current batch "
            "is generating. 0 disables prefetch; 1 overlaps frame loading and processor work "
            "for the next batch. The prefetch worker is intentionally single-threaded to keep "
            "processor calls ordered and auditable."
        ),
    )
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


def validate_args(args) -> None:
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 for all rows or a non-negative count.")
    if args.num_samples == 0:
        raise ValueError("--num_samples=0 does not evaluate any rows.")
    if args.sample_start < 0:
        raise ValueError("--sample_start must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.num_video_frames < 1:
        raise ValueError("--num_video_frames must be >= 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1.")
    if args.frame_load_workers < 1:
        raise ValueError("--frame_load_workers must be >= 1.")
    if args.preprocess_prefetch_batches not in {0, 1}:
        raise ValueError("--preprocess_prefetch_batches must be 0 or 1.")
    if args.context_seconds < 0 and args.context_seconds != -1:
        raise ValueError("--context_seconds must be -1 for all context or a non-negative window in seconds.")
    if args.visual_keep_ratio <= 0.0 or args.visual_keep_ratio > 1.0:
        raise ValueError("--visual_keep_ratio must be in (0, 1].")
    if args.deltakv_center_ratio <= 0.0 or args.deltakv_center_ratio > 1.0:
        raise ValueError("--deltakv_center_ratio must be in (0, 1].")
    if args.svllm_chunk_prefill_size < 1:
        raise ValueError("--svllm_chunk_prefill_size must be >= 1.")
    if args.svllm_max_num_batched_tokens < 1:
        raise ValueError("--svllm_max_num_batched_tokens must be >= 1.")
    if args.svllm_max_num_seqs_in_batch < 1:
        raise ValueError("--svllm_max_num_seqs_in_batch must be >= 1.")
    if args.svllm_gpu_memory_utilization <= 0.0 or args.svllm_gpu_memory_utilization > 1.0:
        raise ValueError("--svllm_gpu_memory_utilization must be in (0, 1].")
    if args.svllm_mlp_seq_chunk_size < 0:
        raise ValueError("--svllm_mlp_seq_chunk_size must be >= 0.")
    for name in (
        "recent_keep_tokens",
        "sink_keep_tokens",
        "decode_keep_tokens",
        "prefill_keep_tokens",
        "deltakv_neighbor_count",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.hf_prefill_chunk_size < 1:
        raise ValueError("--hf_prefill_chunk_size must be >= 1.")


def parse_timestamp(value: str) -> float:
    parts = [part.strip() for part in str(value).split(":") if part.strip()]
    if not parts:
        raise ValueError(f"StreamingBench row has an empty timestamp: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


OPTION_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([ABCD])\s*\.\s*", re.IGNORECASE)
OPTION_LABELS = ["A", "B", "C", "D"]


def _normalize_labeled_options(options: list[str]) -> list[str]:
    normalized = []
    for label, option in zip(OPTION_LABELS, options):
        if re.match(r"^[ABCD]\s*\.", option, re.IGNORECASE):
            normalized.append(option)
        else:
            normalized.append(f"{label}. {option}")
    return normalized


def _split_labeled_option_fragments(options: list[str]) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    current_label = ""
    current_parts: list[str] = []

    def flush_current() -> None:
        nonlocal current_label, current_parts
        if current_label:
            text = " ".join(part for part in current_parts if part).strip()
            fragments.append((current_label, text))
        current_label = ""
        current_parts = []

    for option in options:
        text = str(option).strip()
        if not text:
            continue
        matches = list(OPTION_LABEL_RE.finditer(text))
        if not matches:
            if not current_label:
                raise ValueError(f"StreamingBench options contain an unlabeled leading fragment: {option!r}")
            current_parts.append(text)
            continue
        prefix = text[: matches[0].start()].strip()
        if prefix:
            if not current_label:
                raise ValueError(f"StreamingBench options contain an unlabeled leading fragment: {option!r}")
            current_parts.append(prefix)
        for index, match in enumerate(matches):
            flush_current()
            current_label = match.group(1).upper()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            current_parts = [text[match.start() : end].strip()]
    flush_current()
    return fragments


def _repair_labeled_option_fragments(options: list[str]) -> tuple[list[str], str]:
    fragments = _split_labeled_option_fragments(options)
    labels = [label for label, _ in fragments]
    complete_windows = [
        index
        for index in range(0, len(labels) - len(OPTION_LABELS) + 1)
        if labels[index : index + len(OPTION_LABELS)] == OPTION_LABELS
    ]
    if not complete_windows:
        raise ValueError(
            "StreamingBench options must contain a recoverable A/B/C/D choice sequence, "
            f"got labels={labels}."
        )
    start = complete_windows[-1]
    repaired = [text for _, text in fragments[start : start + len(OPTION_LABELS)]]
    reason = "reconstructed_repeated_or_fragmented_labeled_options"
    if start == 0 and len(fragments) > len(OPTION_LABELS):
        reason = "dropped_trailing_extra_labeled_options"
    return _normalize_labeled_options(repaired), reason


def parse_options_with_repair(value: str) -> tuple[list[str], str | None]:
    options = ast.literal_eval(value) if isinstance(value, str) else value
    options = [str(option).strip() for option in options]
    if len(options) != 4:
        try:
            return _repair_labeled_option_fragments(options)
        except ValueError as exc:
            raise ValueError(
                f"StreamingBench options must contain exactly 4 choices, got {len(options)}: {value!r}"
            ) from exc
    return _normalize_labeled_options(options), None


def parse_options(value: str) -> list[str]:
    options, _ = parse_options_with_repair(value)
    return options


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
    option_repairs: list[dict] = []

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
                options, option_repair = parse_options_with_repair(raw["options"])
                if option_repair:
                    option_repairs.append(
                        {
                            "task": task,
                            "question_id": question_id,
                            "reason": option_repair,
                        }
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
                        "options": options,
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

    start = int(args.sample_start)
    rows = selected_rows[start:]
    if args.num_samples >= 0:
        rows = rows[: args.num_samples]
    return rows, {
        "csv_dir": str(csv_dir),
        "video_dir": str(video_dir),
        "indexed_video_count": sum(len(items) for items in video_index.values()),
        "missing_video_rows": len(missing_videos),
        "missing_video_examples": missing_videos[:10],
        "option_repair_rows": len(option_repairs),
        "option_repair_examples": option_repairs[:10],
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
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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


def ffmpeg_reencode_context_clip(video_path: Path, start_seconds: float, end_seconds: float, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    duration_seconds = max(int(end_seconds) - int(start_seconds), 1)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(int(start_seconds)),
        "-i",
        str(video_path),
        "-t",
        str(duration_seconds),
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        str(output_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(
            f"Failed to re-encode StreamingBench context clip: video={video_path} "
            f"start={start_seconds:.3f}s end={end_seconds:.3f}s rc={completed.returncode} "
            f"stderr={stderr[-2000:]}"
        )
    return completed.stderr or ""


def official_clip_extract_context_frames(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    num_frames: int,
    frame_paths: list[Path],
) -> str:
    clip_path = frame_paths[0].parent / "official_context_clip.mp4"
    stderr = ffmpeg_reencode_context_clip(video_path, start_seconds, end_seconds, clip_path)
    decord_extract_context_frames(clip_path, 0.0, max(end_seconds - start_seconds, 1.0), num_frames, frame_paths)
    return stderr


def frame_cache_paths_for_row(row: dict, args) -> tuple[Path, list[Path], float, float]:
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
    return cache_dir, frame_paths, start_seconds, end_seconds


def ensure_frame_cache(row: dict, args) -> list[Path]:
    video_path = Path(row["video_path"])
    cache_dir, frame_paths, start_seconds, end_seconds = frame_cache_paths_for_row(row, args)
    if cache_dir.exists() and not args.reuse_frame_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_frame_cache and all(path.exists() and path.stat().st_size > 0 for path in frame_paths):
        return frame_paths

    for frame_path in frame_paths:
        if frame_path.exists():
            frame_path.unlink()

    if args.frame_sampling_backend == "decord":
        decord_extract_context_frames(video_path, start_seconds, end_seconds, len(frame_paths), frame_paths)
        return frame_paths

    if args.num_video_frames <= 1:
        timestamps = [(start_seconds + end_seconds) / 2.0]
    else:
        step = (end_seconds - start_seconds) / (args.num_video_frames - 1)
        timestamps = [start_seconds + step * idx for idx in range(args.num_video_frames)]
    for timestamp, frame_path in zip(timestamps, frame_paths):
        extract_required_frame(video_path, timestamp, frame_path)

    return frame_paths


def load_video_frames(row: dict, args) -> list[Image.Image]:
    frame_paths = ensure_frame_cache(row, args)
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def load_batch_video_frames(batch_rows: list[dict], args, executor: ThreadPoolExecutor | None) -> list[list[Image.Image]]:
    if executor is None or len(batch_rows) <= 1:
        return [load_video_frames(row, args) for row in batch_rows]
    return list(executor.map(lambda row: load_video_frames(row, args), batch_rows))


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


def prepare_generation_batch(
    batch_start: int,
    batch_rows: list[dict],
    processor,
    args,
    frame_load_executor: ThreadPoolExecutor | None,
    video_token_id: int,
) -> dict:
    frame_load_start = time.perf_counter()
    videos = load_batch_video_frames(batch_rows, args, frame_load_executor)
    frame_load_elapsed = time.perf_counter() - frame_load_start

    prompt_start = time.perf_counter()
    prompts = [build_prompt(processor, row) for row in batch_rows]
    prompt_elapsed = time.perf_counter() - prompt_start

    processor_kwargs = {"text": prompts, "videos": videos, "return_tensors": "pt"}
    if len(batch_rows) > 1:
        processor_kwargs["padding"] = True
    processor_start = time.perf_counter()
    inputs = processor(**processor_kwargs)
    processor_elapsed = time.perf_counter() - processor_start
    del videos

    input_len = int(inputs["input_ids"].shape[1])
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        input_token_counts = attention_mask.sum(dim=1).tolist()
    else:
        input_token_counts = [input_len for _ in batch_rows]
    video_token_counts = (inputs["input_ids"] == video_token_id).sum(dim=1).tolist()

    return {
        "batch_start": batch_start,
        "batch_rows": batch_rows,
        "inputs": inputs,
        "input_len": input_len,
        "input_token_counts": input_token_counts,
        "video_token_counts": video_token_counts,
        "frame_load_elapsed": frame_load_elapsed,
        "prompt_elapsed": prompt_elapsed,
        "processor_elapsed": processor_elapsed,
    }


class ReadyBatch:
    def __init__(self, value: dict):
        self.value = value

    def result(self) -> dict:
        return self.value


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
        "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
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
        "runtime_params": {
            "recent_keep_tokens": args.recent_keep_tokens,
            "sink_keep_tokens": args.sink_keep_tokens,
            "decode_keep_tokens": args.decode_keep_tokens,
            "prefill_keep_tokens": args.prefill_keep_tokens,
            "hf_prefill_chunk_size": args.hf_prefill_chunk_size,
            "full_attention_layers": args.full_attention_layers,
            "visual_keep_ratio": args.visual_keep_ratio,
            "delta_quant_bits": args.delta_quant_bits,
            "deltakv_center_ratio": args.deltakv_center_ratio,
            "deltakv_neighbor_count": args.deltakv_neighbor_count,
            "svllm_chunk_prefill_size": args.svllm_chunk_prefill_size,
            "svllm_max_num_batched_tokens": args.svllm_max_num_batched_tokens,
            "svllm_max_num_seqs_in_batch": args.svllm_max_num_seqs_in_batch,
            "svllm_gpu_memory_utilization": args.svllm_gpu_memory_utilization,
            "svllm_mlp_seq_chunk_size": args.svllm_mlp_seq_chunk_size,
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
            "frame_load_workers": args.frame_load_workers,
            "preprocess_prefetch_batches": args.preprocess_prefetch_batches,
        },
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
                "expected_rows": LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS[task_type],
                "matches_expected_rows": stats["total"] == LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS[task_type],
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
                "expected_rows": LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS[task_type],
                "matches_expected_rows": stats["total"] == LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS[task_type],
                "correct": stats["correct"],
                "status_counts": stats["status_counts"],
                "accuracy": accuracy,
                "accuracy_pct": accuracy_pct,
                "expected_llava_onevision_7b_pct": expected,
                "used_for_paper_overall": True,
            }
        )

    expected_display_total = sum(item["total"] for item in subtasks)
    expected_extra_total = sum(item["total"] for item in overall_extra_subtasks)
    expected_display_weighted_pct = None
    implied_extra_expected_pct = None
    if expected_display_total > 0:
        expected_display_weighted_pct = sum(
            item["expected_llava_onevision_7b_pct"] * item["total"] for item in subtasks
        ) / expected_display_total
    if expected_extra_total > 0 and expected_display_total > 0:
        implied_extra_expected_pct = (
            LIVEVLM_TABLE4_EXPECTED_OVERALL * (expected_display_total + expected_extra_total)
            - sum(item["expected_llava_onevision_7b_pct"] * item["total"] for item in subtasks)
        ) / expected_extra_total

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
        "expected_display_weighted_accuracy_pct": expected_display_weighted_pct,
        "implied_expected_extra_subtasks_accuracy_pct": implied_extra_expected_pct,
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
    mismatched = {
        task_type: {"got": counts.get(task_type, 0), "expected": expected}
        for task_type, expected in LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS.items()
        if counts.get(task_type, 0) != expected
    }
    overall_total = sum(counts.get(task_type, 0) for task_type in LIVEVLM_TABLE4_OVERALL_SUBTASKS)
    if overall_total != LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS or mismatched:
        raise RuntimeError(
            "LiveVLM Table 4 full baseline requires the 4000-row StreamingBench scope. "
            f"Got overall_rows={overall_total}, expected={LIVEVLM_TABLE4_EXPECTED_OVERALL_ROWS}, "
            f"mismatched_task_type_counts={mismatched}, counts={counts}."
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
    from benchmark.multimodal.model_adapters.llava_onevision import batch_to_device, ensure_left_padding

    is_sparsevllm = policy is not None and policy.get("backend") == "sparsevllm"
    generation_device = getattr(model, "svllm_device", device)
    torch.cuda.reset_peak_memory_stats(generation_device)
    records = []
    total_new_tokens = 0
    total_time = 0.0
    total_frame_load_time = 0.0
    total_prompt_time = 0.0
    total_processor_time = 0.0
    total_preprocess_wait_time = 0.0
    total_batches = 0
    wall_start = time.perf_counter()
    effective_batch_size = max(1, int(args.batch_size))
    if effective_batch_size > 1:
        ensure_left_padding(processor)
    elif getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        ensure_left_padding(processor)

    log_every = max(1, int(args.log_every))
    frame_load_executor = (
        ThreadPoolExecutor(max_workers=args.frame_load_workers)
        if int(args.frame_load_workers) > 1
        else None
    )
    prefetch_executor = (
        ThreadPoolExecutor(max_workers=1)
        if int(args.preprocess_prefetch_batches) == 1
        else None
    )

    def submit_preprocess(batch_item):
        if batch_item is None:
            return None
        batch_start, batch_rows = batch_item
        if prefetch_executor is None:
            return prepare_generation_batch(
                batch_start,
                batch_rows,
                processor,
                args,
                frame_load_executor,
                int(model.config.video_token_id),
            )
        return prefetch_executor.submit(
            prepare_generation_batch,
            batch_start,
            batch_rows,
            processor,
            args,
            frame_load_executor,
            int(model.config.video_token_id),
        )

    try:
        batch_iterator = iter(iter_batches(rows, effective_batch_size))
        pending = submit_preprocess(next(batch_iterator, None))
        while pending is not None:
            if prefetch_executor is None:
                prepared = pending
            else:
                wait_start = time.perf_counter()
                prepared = pending.result()
                total_preprocess_wait_time += time.perf_counter() - wait_start

            next_item = next(batch_iterator, None)
            next_pending = submit_preprocess(next_item)

            batch_start = prepared["batch_start"]
            batch_rows = prepared["batch_rows"]
            inputs = prepared["inputs"]
            input_len = prepared["input_len"]
            input_token_counts = prepared["input_token_counts"]
            video_token_counts = prepared["video_token_counts"]
            frame_load_elapsed = prepared["frame_load_elapsed"]
            prompt_elapsed = prepared["prompt_elapsed"]
            processor_elapsed = prepared["processor_elapsed"]
            total_frame_load_time += frame_load_elapsed
            total_prompt_time += prompt_elapsed
            total_processor_time += processor_elapsed
            inputs = batch_to_device(inputs, generation_device, dtype)

            torch.cuda.synchronize(generation_device)
            start = time.perf_counter()
            if is_sparsevllm:
                from sparsevllm import SamplingParams

                svllm_outputs = model.generate_multimodal(
                    dict(inputs),
                    SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0),
                    use_tqdm=False,
                )
                decoded_batch = [item["text"] for item in svllm_outputs]
                new_tokens_by_sample = [len(item["token_ids"]) for item in svllm_outputs]
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
                )
                generated_ids = output_ids[:, input_len:]
                decoded_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
                new_tokens_by_sample = [int(generated_ids.shape[1])] * len(batch_rows)
            torch.cuda.synchronize(generation_device)
            elapsed = time.perf_counter() - start

            if prefetch_executor is not None and next_pending is not None:
                wait_start = time.perf_counter()
                next_prepared = next_pending.result()
                total_preprocess_wait_time += time.perf_counter() - wait_start
                next_pending = ReadyBatch(next_prepared)

            batch_new_tokens = sum(new_tokens_by_sample)
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
                new_tokens = int(new_tokens_by_sample[offset])
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
                        f"frame_load={frame_load_elapsed:.3f}s processor={processor_elapsed:.3f}s "
                        f"prefetch_wait={total_preprocess_wait_time:.3f}s "
                        f"status={status} ans={row['answer']} pred={prediction or '?'} ok={correct} raw={decoded[:80]!r}",
                        flush=True,
                    )
            pending = next_pending
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if frame_load_executor is not None:
            frame_load_executor.shutdown(wait=True)

    total = max(len(records), 1)
    wall_time = time.perf_counter() - wall_start
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
        "total_frame_load_seconds": total_frame_load_time,
        "total_prompt_seconds": total_prompt_time,
        "total_processor_seconds": total_processor_time,
        "total_preprocess_wait_seconds": total_preprocess_wait_time,
        "end_to_end_seconds": wall_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "end_to_end_examples_per_s": len(records) / wall_time if wall_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "mean_seconds": total_time / total,
        "peak_memory_gb": torch.cuda.max_memory_allocated(generation_device) / (1024**3),
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


def add_vanilla_comparison(results: list[dict]) -> None:
    if len(results) != 2:
        return
    base = next((item for item in results if item["method"] == "vanilla"), None)
    candidate = next((item for item in results if item["method"] != "vanilla"), None)
    if base is None or candidate is None:
        return
    if base["new_tokens_per_s"] <= 0:
        raise RuntimeError("Cannot compute speedup_vs_vanilla because vanilla new_tokens_per_s is not positive.")
    candidate["accuracy_delta_vs_vanilla"] = candidate["accuracy"] - base["accuracy"]
    candidate["speedup_vs_vanilla"] = candidate["new_tokens_per_s"] / base["new_tokens_per_s"]
    candidate["memory_delta_gb_vs_vanilla"] = candidate["peak_memory_gb"] - base["peak_memory_gb"]


def iter_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield "vanilla", "vanilla"
        elif method in {"deltakv_delta_quant", "delta_quant", "llava_deltakv_delta_quant"}:
            yield raw_method, "deltakv_delta_quant"
        elif method in {"svllm_deltakv_delta_quant", "sparsevllm_deltakv_delta_quant"}:
            yield raw_method, "svllm_deltakv_delta_quant"
        else:
            raise ValueError(
                "StreamingBench script currently supports methods: "
                "vanilla, deltakv_delta_quant, svllm_deltakv_delta_quant."
            )


def main():
    args = parse_args()
    args = apply_streamingbench_profile(args)
    validate_args(args)
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

    processor = LlavaOnevisionProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=args.image_processor_use_fast,
    )
    results = []
    for requested_method, method_kind in iter_methods(args.methods):
        if method_kind == "vanilla":
            from benchmark.multimodal.model_adapters.llava_onevision import load_vanilla_model

            model = load_vanilla_model(args, dtype, device)
            method_label = "vanilla"
            policy = None
        elif method_kind == "deltakv_delta_quant":
            from benchmark.multimodal.model_adapters.llava_onevision import load_llava_delta_quant_model

            model, policy = load_llava_delta_quant_model(args, dtype, device)
            method_label = policy["method"]
        else:
            from benchmark.multimodal.model_adapters.llava_onevision import load_svllm_delta_quant_model

            model, policy = load_svllm_delta_quant_model(args, dtype, device)
            method_label = policy["method"]

        result = run_method(method_label, model, processor, rows, args, dtype, device, policy=policy)
        result["requested_method"] = requested_method
        result["dataset_info"] = dataset_info
        results.append(result)
        if hasattr(model, "exit"):
            model.exit()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    add_vanilla_comparison(results)
    output_dir = Path(args.output_dir)
    for result in results:
        result["artifact_paths"] = save_method_artifacts(output_dir, result, run_info)

    out_path = output_dir / "last_streamingbench_result.json"
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
