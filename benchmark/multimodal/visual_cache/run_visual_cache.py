#!/usr/bin/env python3
import argparse
import gc
import json
import os
import random
import re
import shlex
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from deltakv.modeling.llava_ov.llava_onevision_deltakv import (
    LlavaOnevisionDeltaKVForConditionalGeneration,
    load_deltakv_compressor_into_llava,
)


CUSTOM_CONFIG_KEYS = {
    "kv_compressed_size",
    "seq_chunk_size",
    "k_neighbors",
    "compressor_token_group_size",
    "deltakv_neighbor_count",
    "layer_chunk_size",
    "recon_mode",
    "ref_mode",
    "use_nonlinear_compressor",
    "compressor_intermediate_size",
    "compressor_down_type",
    "compressor_up_type",
    "compressor_down_intermediate_size",
    "compressor_up_intermediate_size",
    "collect_kv_before_rope",
    "compressor_linear_bias",
    "split_kv",
    "cluster_metric",
    "cluster_on_kv",
    "cluster_ratio",
    "stride_alpha",
    "cluster_temp",
    "cluster_soft_assignment",
    "tail_token_size",
    "num_recent_tokens",
    "full_attn_layers",
    "num_top_tokens",
    "num_top_tokens_in_prefill",
    "num_sink_tokens",
    "omnikv_score_method",
    "deltakv_use_omnikv_selection",
    "use_compression",
    "use_cluster",
    "chunk_prefill_size",
    "snapkv_window_size",
    "pool_kernel_size",
    "chunk_prefill_accel_omnikv",
    "kv_quant_bits",
}


VISUAL_PRUNE_METHOD_ALIASES = {
    "visual_uniform_keep",
    "visual_keep",
    "visual_prune",
    "uniform_keep",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HF LLaVA-OneVision vanilla, standard DeltaKV, direct "
            "Delta residual quantization, and the no-checkpoint visual-token "
            "uniform-pruning baseline."
        )
    )
    parser.add_argument("--model_path", default="/data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument(
        "--deltakv_checkpoint_path",
        default="none",
        help=(
            "Use 'none' for vanilla, deltakv_delta_quant, and visual_uniform_keep. "
            "Set a trained DeltaKV compressor checkpoint path only when benchmarking "
            "method deltakv."
        ),
    )
    parser.add_argument("--dataset_dir", default="/data2/haojitai/datasets/llava_onevision_visual_prune_bench")
    parser.add_argument("--source_vqa_dir", default="/data2/haojitai/datasets/VQAv2")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory for benchmark artifacts. Defaults to --dataset_dir for backward-compatible local runs.",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Generation batch size. Supported by vanilla, deltakv, and "
            "deltakv_delta_quant. visual_uniform_keep is still single-example."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument(
        "--methods",
        default="vanilla,visual_uniform_keep",
        help=(
            "Comma-separated methods. Supported: vanilla, deltakv, "
            "deltakv_delta_quant, visual_uniform_keep."
        ),
    )
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
    parser.add_argument("--quantize_visual_kv", action="store_true")
    parser.add_argument("--limit_text_tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--print_records", action="store_true", help="Print per-sample records in the terminal summary.")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 for all rows or a non-negative count.")
    if args.num_samples == 0:
        raise ValueError("--num_samples=0 does not evaluate any samples.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1.")
    if args.limit_text_tokens < 0:
        raise ValueError("--limit_text_tokens must be non-negative.")
    if args.visual_keep_ratio <= 0.0 or args.visual_keep_ratio > 1.0:
        raise ValueError("--visual_keep_ratio must be in (0, 1].")
    if args.deltakv_center_ratio <= 0.0 or args.deltakv_center_ratio > 1.0:
        raise ValueError("--deltakv_center_ratio must be in (0, 1].")
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


def init_seed(seed: int) -> None:
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


def build_run_info(args, output_dir: Path, row_count: int) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "model_path": args.model_path,
        "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
        "methods": args.methods,
        "dataset_dir": args.dataset_dir,
        "source_vqa_dir": args.source_vqa_dir,
        "output_dir": str(output_dir),
        "num_samples_arg": args.num_samples,
        "evaluated_sample_count": row_count,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {
            "do_sample": False,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
        "seed": args.seed,
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
            "quantize_visual_kv": bool(args.quantize_visual_kv),
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
            "limit_text_tokens": args.limit_text_tokens,
        },
    }


def validate_vqa_row(row: dict, *, source: str) -> dict:
    required = ("question_id", "image_id", "question", "answer", "answers", "image_path")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"VQA row from {source} is missing fields: {missing}")
    image_path = Path(row["image_path"])
    if not image_path.exists() or image_path.stat().st_size == 0:
        raise FileNotFoundError(f"VQA row from {source} references a missing/empty image: {image_path}")
    if not str(row["question"]).strip():
        raise ValueError(f"VQA row from {source} has an empty question: question_id={row['question_id']!r}")
    if not str(row["answer"]).strip():
        raise ValueError(f"VQA row from {source} has an empty answer: question_id={row['question_id']!r}")
    if not isinstance(row["answers"], list) or not [answer for answer in row["answers"] if str(answer).strip()]:
        raise ValueError(f"VQA row from {source} has no annotator answers: question_id={row['question_id']!r}")
    return row


def prepare_vqa_subset(source_vqa_dir: Path, dataset_dir: Path, num_samples: int):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    use_all = num_samples < 0
    manifest_path = dataset_dir / ("vqa_validation_all.jsonl" if use_all else f"vqa_subset_{num_samples}.jsonl")
    if manifest_path.exists():
        rows = [
            validate_vqa_row(json.loads(line), source=str(manifest_path))
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]
        if use_all:
            return rows
        if len(rows) >= num_samples:
            return rows[:num_samples]

    parquet_files = sorted((source_vqa_dir / "data").glob("validation-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No VQAv2 validation parquet files found under {source_vqa_dir / 'data'}")

    rows = []
    for parquet_file in parquet_files:
        table = pq.read_table(
            parquet_file,
            columns=["question_id", "image_id", "question", "multiple_choice_answer", "answers", "image"],
        )
        batch = table.to_pydict()
        for question_id, image_id, question, answer, answers, image in zip(
            batch["question_id"],
            batch["image_id"],
            batch["question"],
            batch["multiple_choice_answer"],
            batch["answers"],
            batch["image"],
        ):
            if not image or image.get("bytes") is None:
                continue
            if answer is None or not str(answer).strip():
                raise ValueError(f"VQAv2 row has an empty multiple_choice_answer: question_id={question_id!r}")
            answer_list = [item["answer"] for item in answers or [] if item and item.get("answer") is not None]
            if not [item for item in answer_list if str(item).strip()]:
                raise ValueError(f"VQAv2 row has no annotator answers: question_id={question_id!r}")
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                image_path.write_bytes(image["bytes"])
            if image_path.stat().st_size == 0:
                raise FileNotFoundError(f"VQAv2 image is empty after materialization: {image_path}")
            rows.append(
                {
                    "question_id": int(question_id),
                    "image_id": int(image_id),
                    "question": question,
                    "answer": answer,
                    "answers": answer_list,
                    "image_path": str(image_path),
                }
            )
            if not use_all and len(rows) >= num_samples:
                manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
                return rows

    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return rows


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


VQA_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "lets": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "thats": "that's",
    "thered": "there'd",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whats": "what's",
    "whenll": "when'll",
    "whens": "when's",
    "whered": "where'd",
    "wherell": "where'll",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whyd": "why'd",
    "whyll": "why'll",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

VQA_DIGIT_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

VQA_ARTICLES = {"a", "an", "the"}
VQA_PUNCT = set(string.punctuation)
VQA_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")
VQA_COMMA_STRIP = re.compile(r"(?<=\d)(,)(?=\d)")


def normalize_vqa_answer(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\t", " ").strip().lower()
    text = VQA_COMMA_STRIP.sub("", text)
    text = VQA_PERIOD_STRIP.sub("", text)
    chars = []
    for char in text:
        if char in VQA_PUNCT and char not in {"'", ":"}:
            chars.append(" ")
        else:
            chars.append(char)
    words = []
    for word in " ".join("".join(chars).split()).split():
        mapped = VQA_DIGIT_MAP.get(word, word)
        if mapped not in VQA_ARTICLES:
            words.append(VQA_CONTRACTIONS.get(mapped, mapped))
    return " ".join(words)


def vqa_score(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    pred = normalize_vqa_answer(prediction)
    normalized_answers = [normalize_vqa_answer(answer) for answer in answers]
    if len(normalized_answers) == 1:
        return float(pred == normalized_answers[0])
    scores = []
    for idx, answer in enumerate(normalized_answers):
        other_answers = normalized_answers[:idx] + normalized_answers[idx + 1 :]
        matching = sum(pred == other_answer for other_answer in other_answers)
        scores.append(min(1.0, matching / 3.0))
    return sum(scores) / len(scores)


def batch_to_device(inputs, device, dtype):
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def build_prompt(processor, question: str, limit_text_tokens: int):
    if limit_text_tokens > 0:
        question = " ".join(question.split()[:limit_text_tokens])
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question + " Answer with a short phrase."},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def iter_batches(rows, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


def ensure_left_padding(processor):
    if getattr(processor, "tokenizer", None) is None:
        return
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def resolve_compressor_path(args):
    checkpoint_path = str(args.deltakv_checkpoint_path)
    return Path(checkpoint_path) if checkpoint_path.lower() not in {"", "none", "null"} else None


def build_llava_deltakv_policy(infer_config):
    uses_cluster = bool(infer_config.get("use_cluster", False))
    uses_learned_compressor = bool(infer_config.get("use_compression", False))
    kv_quant_bits = int(
        infer_config.get("deltakv_latent_quant_bits", infer_config.get("kv_quant_bits", 0)) or 0
    )

    return {
        "method": "llava_deltakv",
        "selection_policy": "standard_deltakv_cache",
        "uses_deltakv_wrapper": True,
        "uses_learned_compressor": uses_learned_compressor,
        "uses_cluster": uses_cluster,
        "uses_ref_tokens": uses_cluster,
        "uses_visual_uniform_pruning": False,
        "supports_batch_generation": True,
        "kv_quant_bits": kv_quant_bits,
        "note": (
            "Uses the LLaVA-OneVision DeltaKV wrapper with the same standard "
            "DeltaKV cache/compressor path as text inference. It is not visual "
            "uniform pruning; whether it uses cluster/ref behavior depends on "
            "the checkpoint config."
        ),
    }


def build_llava_delta_quant_policy(infer_config):
    kv_quant_bits = int(
        infer_config.get("deltakv_latent_quant_bits", infer_config.get("kv_quant_bits", 0)) or 0
    )
    return {
        "method": "llava_deltakv_delta_quant",
        "selection_policy": "cluster_ref_delta_quant",
        "uses_deltakv_wrapper": True,
        "uses_learned_compressor": False,
        "uses_cluster": True,
        "uses_ref_tokens": True,
        "uses_visual_uniform_pruning": False,
        "supports_batch_generation": True,
        "kv_quant_bits": kv_quant_bits,
        "note": (
            "Uses DeltaKV-style cluster/ref reconstruction and stores token-space "
            "residuals with direct quantization. It does not load or use a learned "
            "compressor checkpoint."
        ),
    }


def build_visual_uniform_policy(args, infer_config):
    kv_quant_bits = int(
        infer_config.get("deltakv_latent_quant_bits", infer_config.get("kv_quant_bits", 0)) or 0
    )
    if kv_quant_bits == 4:
        method = "visual_uniform_keep_int4"
        note = (
            "No DeltaKV compressor, no cluster, no ref tokens. Uniformly keeps "
            "visual tokens then stores kept visual KV with direct int4 min/max quantization."
        )
    else:
        method = "visual_uniform_keep"
        note = (
            "No DeltaKV compressor, no cluster, no ref tokens, no SnapKV-style "
            "attention scoring. Uniformly keeps a fixed ratio of visual KV tokens."
        )

    return {
        "method": method,
        "selection_policy": "uniform_visual_subsampling",
        "uses_deltakv_wrapper": True,
        "uses_learned_compressor": False,
        "uses_cluster": False,
        "uses_ref_tokens": False,
        "uses_visual_uniform_pruning": True,
        "supports_batch_generation": False,
        "kv_quant_bits": kv_quant_bits,
        "note": note,
    }


def load_vanilla_model(args, dtype, device):
    return LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()


def migrate_checkpoint_infer_config(infer_config: dict) -> dict:
    migrated = dict(infer_config)
    if "seq_chunk_size" in migrated:
        value = migrated.pop("seq_chunk_size")
        if "compressor_token_group_size" in migrated and migrated["compressor_token_group_size"] != value:
            raise ValueError("Checkpoint config has conflicting seq_chunk_size/compressor_token_group_size.")
        migrated["compressor_token_group_size"] = value
    if "k_neighbors" in migrated:
        value = migrated.pop("k_neighbors")
        if "deltakv_neighbor_count" in migrated and migrated["deltakv_neighbor_count"] != value:
            raise ValueError("Checkpoint config has conflicting k_neighbors/deltakv_neighbor_count.")
        migrated["deltakv_neighbor_count"] = value
    return migrated


def _load_llava_deltakv_wrapper(args, dtype, device, config, infer_config, infer_config_is_native, policy):
    print(
        "[llava_cache_policy] "
        f"method={policy['method']} selection={policy['selection_policy']} "
        f"cluster={policy['uses_cluster']} compressor={policy['uses_learned_compressor']} "
        f"ref_tokens={policy['uses_ref_tokens']} uniform_visual={policy['uses_visual_uniform_pruning']} "
        f"kv_quant_bits={policy['kv_quant_bits']} visual_keep_ratio={args.visual_keep_ratio}",
        flush=True,
    )
    config.deltakv_infer_config = infer_config
    config.deltakv_infer_config_is_native = infer_config_is_native
    model = LlavaOnevisionDeltaKVForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    return model, policy


def load_llava_deltakv_model(args, dtype, device):
    config = LlavaOnevisionConfig.from_pretrained(args.model_path, trust_remote_code=True)
    compressor_path = resolve_compressor_path(args)
    if compressor_path is None:
        raise ValueError("deltakv requires a real --deltakv_checkpoint_path, not 'none'.")
    compressor_config = json.loads((compressor_path / "config.json").read_text())
    infer_config = migrate_checkpoint_infer_config(
        {key: compressor_config[key] for key in CUSTOM_CONFIG_KEYS if key in compressor_config}
    )
    infer_config.update(
        {
            "visual_token_prune_only": False,
            "num_recent_tokens": args.recent_keep_tokens,
            "num_sink_tokens": args.sink_keep_tokens,
            "num_top_tokens": args.decode_keep_tokens,
            "num_top_tokens_in_prefill": args.prefill_keep_tokens,
            "chunk_prefill_size": args.hf_prefill_chunk_size,
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
        }
    )
    policy = build_llava_deltakv_policy(infer_config)
    model, policy = _load_llava_deltakv_wrapper(args, dtype, device, config, infer_config, True, policy)
    incompatible = load_deltakv_compressor_into_llava(model, str(compressor_path), device="cpu")
    compressor_missing = [key for key in incompatible.missing_keys if "compress_" in key]
    if compressor_missing:
        raise RuntimeError(f"DeltaKV compressor weights were not fully loaded; missing examples: {compressor_missing[:8]}")
    return model, policy


def load_llava_delta_quant_model(args, dtype, device):
    if resolve_compressor_path(args) is not None:
        raise ValueError(
            "deltakv_delta_quant does not use a learned compressor checkpoint. "
            "Set --deltakv_checkpoint_path none."
        )
    config = LlavaOnevisionConfig.from_pretrained(args.model_path, trust_remote_code=True)
    infer_config = {
        "use_compression": False,
        "use_cluster": True,
        "deltakv_latent_quant_bits": args.delta_quant_bits,
        "deltakv_center_ratio": args.deltakv_center_ratio,
        "deltakv_neighbor_count": args.deltakv_neighbor_count,
        "full_attention_layers": args.full_attention_layers,
        "deltakv_use_omnikv_selection": True,
        "omnikv_score_method": "last",
        "visual_token_prune_only": False,
        "recent_keep_tokens": args.recent_keep_tokens,
        "sink_keep_tokens": args.sink_keep_tokens,
        "decode_keep_tokens": args.decode_keep_tokens,
        "prefill_keep_tokens": args.prefill_keep_tokens,
        "hf_prefill_chunk_size": args.hf_prefill_chunk_size,
        "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
    }
    policy = build_llava_delta_quant_policy(infer_config)
    model, policy = _load_llava_deltakv_wrapper(args, dtype, device, config, infer_config, False, policy)
    return model, policy


def load_visual_uniform_model(args, dtype, device):
    if resolve_compressor_path(args) is not None:
        raise ValueError(
            "visual_uniform_keep is the no-checkpoint uniform baseline. "
            "Use --deltakv_checkpoint_path none, or request method deltakv for the standard DeltaKV path."
        )
    config = LlavaOnevisionConfig.from_pretrained(args.model_path, trust_remote_code=True)
    infer_config = {
        "use_compression": False,
        "use_cluster": False,
        "deltakv_latent_quant_bits": 4 if args.quantize_visual_kv else 0,
        "full_attention_layers": args.full_attention_layers,
        "deltakv_use_omnikv_selection": True,
        "omnikv_score_method": "last",
        "visual_token_prune_only": True,
        "visual_token_keep_ratio": args.visual_keep_ratio,
        "recent_keep_tokens": args.recent_keep_tokens,
        "sink_keep_tokens": args.sink_keep_tokens,
        "decode_keep_tokens": args.decode_keep_tokens,
        "prefill_keep_tokens": args.prefill_keep_tokens,
        "hf_prefill_chunk_size": args.hf_prefill_chunk_size,
        "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
    }
    policy = build_visual_uniform_policy(args, infer_config)
    model, policy = _load_llava_deltakv_wrapper(args, dtype, device, config, infer_config, False, policy)
    return model, policy


@torch.inference_mode()
def run_method(method, model, processor, rows, args, dtype, device, policy=None, requested_method=None):
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    total_new_tokens = 0
    total_time = 0.0
    total_batches = 0

    if policy is None:
        supports_batch = method == "vanilla"
    else:
        supports_batch = bool(policy.get("supports_batch_generation", False))
    effective_batch_size = max(1, int(args.batch_size)) if supports_batch else 1
    if effective_batch_size > 1:
        ensure_left_padding(processor)
    elif getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        ensure_left_padding(processor)

    log_every = max(1, int(args.log_every))
    for batch_start, batch_rows in iter_batches(rows, effective_batch_size):
        images = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB").copy())
        prompts = [build_prompt(processor, row["question"], args.limit_text_tokens) for row in batch_rows]
        processor_kwargs = {"text": prompts, "images": images, "return_tensors": "pt"}
        if len(batch_rows) > 1:
            processor_kwargs["padding"] = True
        inputs = processor(**processor_kwargs)
        input_len = int(inputs["input_ids"].shape[1])
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_token_counts = attention_mask.sum(dim=1).tolist()
        else:
            input_token_counts = [input_len for _ in batch_rows]
        visual_token_counts = (inputs["input_ids"] == model.config.image_token_id).sum(dim=1).tolist()
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
            answer = row["answer"]
            answers = row["answers"]
            hit = normalize_text(answer) in normalize_text(decoded)
            score = vqa_score(decoded, answers)
            records.append(
                {
                    "status": "success",
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "input_tokens": int(input_token_counts[offset]),
                    "padded_input_tokens": input_len,
                    "visual_tokens": int(visual_token_counts[offset]),
                    "new_tokens": new_tokens,
                    "seconds": elapsed / len(batch_rows),
                    "batch_seconds": elapsed,
                    "new_tokens_per_s": new_tokens / (elapsed / len(batch_rows)) if elapsed > 0 else 0.0,
                    "batch_new_tokens_per_s": batch_tok_s,
                    "answer": answer,
                    "answers": answers,
                    "prediction": decoded,
                    "raw_prediction": raw_decoded,
                    "parsed_text": decoded,
                    "contains_answer": hit,
                    "vqa_score": score,
                }
            )
            if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
                print(
                    f"[{method}] {sample_idx}/{len(rows)} qid={row['question_id']} "
                    f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                    f"visual={visual_token_counts[offset]} new={new_tokens} "
                    f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} "
                    f"vqa={score:.3f} hit={hit} pred={decoded[:80]!r}",
                    flush=True,
                )

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    contains_acc = sum(record["contains_answer"] for record in records) / max(len(records), 1)
    mean_vqa_score = sum(record["vqa_score"] for record in records) / max(len(records), 1)
    status_counts = {}
    for record in records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
    uses_visual_uniform = bool(policy and policy.get("uses_visual_uniform_pruning", False))
    visual_keep_ratio = args.visual_keep_ratio if uses_visual_uniform else 1.0
    visual_storage_ratio = 1.0
    if uses_visual_uniform:
        visual_storage_ratio = args.visual_keep_ratio
        if int(policy.get("kv_quant_bits", 0)) == 4:
            visual_storage_ratio *= 4.0 / 16.0

    result = {
        "method": method,
        "requested_method": requested_method or method,
        "visual_keep_ratio": visual_keep_ratio,
        "visual_storage_ratio": visual_storage_ratio,
        "num_samples": len(records),
        "status_counts": status_counts,
        "batch_size": effective_batch_size,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "mean_seconds": total_time / max(len(records), 1),
        "peak_memory_gb": peak_gb,
        "contains_answer_acc": contains_acc,
        "mean_vqa_score": mean_vqa_score,
        "records": records,
    }
    if policy is not None:
        result["visual_cache_policy"] = policy
    return result


def save_method_artifacts(output_dir: Path, result: dict, run_info: dict) -> dict:
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
                        "image_id": record["image_id"],
                        "image_path": record["image_path"],
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
                        "answers": record["answers"],
                        "status": record["status"],
                        "contains_answer": record["contains_answer"],
                        "vqa_score": record["vqa_score"],
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


def add_vanilla_comparison(results: list[dict]) -> None:
    if len(results) != 2:
        return
    base = next((item for item in results if item["method"] == "vanilla"), None)
    candidate = next((item for item in results if item["method"] != "vanilla"), None)
    if base is None or candidate is None:
        return
    if base["new_tokens_per_s"] <= 0:
        raise RuntimeError("Cannot compute speedup_vs_vanilla because vanilla new_tokens_per_s is not positive.")
    candidate["speedup_vs_vanilla"] = candidate["new_tokens_per_s"] / base["new_tokens_per_s"]
    candidate["memory_delta_gb_vs_vanilla"] = candidate["peak_memory_gb"] - base["peak_memory_gb"]
    candidate["vqa_score_delta_vs_vanilla"] = candidate["mean_vqa_score"] - base["mean_vqa_score"]
    candidate["contains_answer_delta_vs_vanilla"] = candidate["contains_answer_acc"] - base["contains_answer_acc"]


def iter_requested_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield raw_method, "vanilla"
        elif method in VISUAL_PRUNE_METHOD_ALIASES:
            yield raw_method, "visual_cache"
        elif method in {"deltakv_delta_quant", "delta_quant", "llava_deltakv_delta_quant"}:
            yield raw_method, "deltakv_delta_quant"
        elif method in {"deltakv", "llava_deltakv"}:
            yield raw_method, "deltakv"
        elif method == "visual_deltakv_compressor":
            raise ValueError("visual_deltakv_compressor is deprecated. Use method deltakv for the standard DeltaKV path.")
        else:
            raise ValueError(
                f"Unknown method: {raw_method}. Supported: vanilla, deltakv, "
                "deltakv_delta_quant, or visual_uniform_keep."
            )


def main():
    args = parse_args()
    validate_args(args)
    init_seed(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    rows = prepare_vqa_subset(Path(args.source_vqa_dir), Path(args.dataset_dir), args.num_samples)
    if not rows:
        raise RuntimeError(f"No VQA rows were prepared from source_vqa_dir={args.source_vqa_dir}")
    print(f"[dataset] rows={len(rows)} dataset_dir={args.dataset_dir} output_dir={output_dir}", flush=True)
    run_info = build_run_info(args, output_dir, len(rows))
    processor = LlavaOnevisionProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    results = []
    for requested_method, method_kind in iter_requested_methods(args.methods):
        if method_kind == "vanilla":
            model = load_vanilla_model(args, dtype, device)
            method_label = "vanilla"
            policy = None
        elif method_kind == "deltakv" and resolve_compressor_path(args) is None:
            raise ValueError("deltakv requires a real --deltakv_checkpoint_path, not 'none'.")
        elif method_kind == "deltakv_delta_quant":
            model, policy = load_llava_delta_quant_model(args, dtype, device)
            method_label = policy["method"]
        elif method_kind == "visual_cache" and resolve_compressor_path(args) is not None:
            raise ValueError(
                "visual_uniform_keep is the no-checkpoint uniform baseline. "
                "Use --deltakv_checkpoint_path none, or request method deltakv for the standard DeltaKV path."
            )
        elif method_kind == "deltakv":
            model, policy = load_llava_deltakv_model(args, dtype, device)
            method_label = policy["method"]
        else:
            model, policy = load_visual_uniform_model(args, dtype, device)
            method_label = policy["method"]

        result = run_method(
            method_label,
            model,
            processor,
            rows,
            args,
            dtype,
            device,
            policy=policy,
            requested_method=requested_method,
        )
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    add_vanilla_comparison(results)
    for result in results:
        result["artifact_paths"] = save_method_artifacts(output_dir, result, run_info)

    out_path = output_dir / "last_benchmark_result.json"
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
