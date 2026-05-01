#!/usr/bin/env python3
import argparse
import gc
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from deltakv.modeling.llava_onevision_deltakv import (
    LlavaOnevisionDeltaKVForConditionalGeneration,
    load_deltakv_compressor_into_llava,
)


CUSTOM_CONFIG_KEYS = {
    "kv_compressed_size",
    "seq_chunk_size",
    "k_neighbors",
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
    "deltakv",  # legacy alias; only true DeltaKV if a compressor checkpoint is supplied.
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HF LLaVA-OneVision with a visual-token uniform-pruning "
            "baseline. Supplying --compressor_path enables the experimental "
            "DeltaKV-wrapper path; --compressor_path none is not DeltaKV cluster "
            "or learned compression."
        )
    )
    parser.add_argument("--model_path", default="/data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument(
        "--compressor_path",
        default="none",
        help=(
            "Use 'none' for visual-token uniform pruning. Set a trained DeltaKV "
            "compressor checkpoint path only when benchmarking the experimental "
            "visual DeltaKV-compressor path."
        ),
    )
    parser.add_argument("--dataset_dir", default="/data2/haojitai/datasets/llava_onevision_visual_prune_bench")
    parser.add_argument("--source_vqa_dir", default="/data2/haojitai/datasets/VQAv2")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument(
        "--methods",
        default="vanilla,visual_uniform_keep",
        help=(
            "Comma-separated methods. Supported: vanilla, visual_uniform_keep. "
            "'deltakv' is accepted as a legacy alias for visual_uniform_keep when "
            "--compressor_path none."
        ),
    )
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--num_recent_tokens", type=int, default=128)
    parser.add_argument("--num_sink_tokens", type=int, default=8)
    parser.add_argument("--num_top_tokens", type=int, default=1024)
    parser.add_argument("--num_top_tokens_in_prefill", type=int, default=4096)
    parser.add_argument("--chunk_prefill_size", type=int, default=100000000)
    parser.add_argument("--chunk_prefill_accel_omnikv", action="store_true")
    parser.add_argument("--full_attn_layers", default="0,1,2,3,8,16,22")
    parser.add_argument("--visual_keep_ratio", type=float, default=1.0)
    parser.add_argument("--quantize_visual_kv", action="store_true")
    parser.add_argument("--limit_text_tokens", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def prepare_vqa_subset(source_vqa_dir: Path, dataset_dir: Path, num_samples: int):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    use_all = num_samples < 0
    manifest_path = dataset_dir / ("vqa_validation_all.jsonl" if use_all else f"vqa_subset_{num_samples}.jsonl")
    if manifest_path.exists():
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if use_all:
            return rows
        if len(rows) >= num_samples:
            return rows[:num_samples]

    parquet_files = sorted((source_vqa_dir / "data").glob("validation-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No VQAv2 validation parquet files found under {source_vqa_dir / 'data'}")

    rows = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["question_id", "image_id", "question", "multiple_choice_answer", "image"])
        batch = table.to_pydict()
        for question_id, image_id, question, answer, image in zip(
            batch["question_id"],
            batch["image_id"],
            batch["question"],
            batch["multiple_choice_answer"],
            batch["image"],
        ):
            if not image or image.get("bytes") is None:
                continue
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                image_path.write_bytes(image["bytes"])
            rows.append(
                {
                    "question_id": int(question_id),
                    "image_id": int(image_id),
                    "question": question,
                    "answer": answer,
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


def resolve_compressor_path(args):
    return Path(args.compressor_path) if args.compressor_path.lower() not in {"", "none", "null"} else None


def build_visual_cache_policy(args, infer_config, compressor_path):
    uses_compressor = compressor_path is not None
    uses_cluster = bool(infer_config.get("use_cluster", False))
    uses_learned_compressor = uses_compressor and bool(infer_config.get("use_compression", False))
    kv_quant_bits = int(infer_config.get("kv_quant_bits", 0) or 0)

    if uses_compressor:
        method = "visual_deltakv_compressor"
        selection_policy = "checkpoint_config"
        note = (
            "Uses the DeltaKV wrapper with a supplied compressor checkpoint. "
            "Whether this is cluster/ref based depends on the checkpoint config."
        )
    elif kv_quant_bits == 4:
        method = "visual_uniform_keep_int4"
        selection_policy = "uniform_visual_subsampling"
        note = (
            "No DeltaKV compressor, no cluster, no ref tokens. Uniformly keeps "
            "visual tokens then stores kept visual KV with direct int4 min/max quantization."
        )
    else:
        method = "visual_uniform_keep"
        selection_policy = "uniform_visual_subsampling"
        note = (
            "No DeltaKV compressor, no cluster, no ref tokens, no SnapKV-style "
            "attention scoring. Uniformly keeps a fixed ratio of visual KV tokens."
        )

    return {
        "method": method,
        "selection_policy": selection_policy,
        "uses_deltakv_wrapper": True,
        "uses_learned_compressor": uses_learned_compressor,
        "uses_cluster": uses_cluster,
        "uses_ref_tokens": uses_cluster,
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


def load_visual_cache_model(args, dtype, device):
    config = LlavaOnevisionConfig.from_pretrained(args.model_path, trust_remote_code=True)
    compressor_path = resolve_compressor_path(args)
    if compressor_path is not None:
        compressor_config = json.loads((compressor_path / "config.json").read_text())
        infer_config = {key: compressor_config[key] for key in CUSTOM_CONFIG_KEYS if key in compressor_config}
    else:
        # This fallback is a visual-token uniform-pruning baseline. It is not
        # DeltaKV clustering, learned compressor inference, ref-token residuals,
        # or SnapKV attention-score selection.
        infer_config = {
            "use_compression": False,
            "use_cluster": False,
            "kv_quant_bits": 4 if args.quantize_visual_kv else 0,
            "full_attn_layers": args.full_attn_layers,
            "deltakv_use_omnikv_selection": True,
            "omnikv_score_method": "last",
        }
    infer_config.update(
        {
            "visual_token_prune_only": True,
            "visual_token_keep_ratio": args.visual_keep_ratio,
            "num_recent_tokens": args.num_recent_tokens,
            "num_sink_tokens": args.num_sink_tokens,
            "num_top_tokens": args.num_top_tokens,
            "num_top_tokens_in_prefill": args.num_top_tokens_in_prefill,
            "chunk_prefill_size": args.chunk_prefill_size,
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
        }
    )
    policy = build_visual_cache_policy(args, infer_config, compressor_path)
    print(
        "[visual_cache_policy] "
        f"method={policy['method']} selection={policy['selection_policy']} "
        f"cluster={policy['uses_cluster']} compressor={policy['uses_learned_compressor']} "
        f"ref_tokens={policy['uses_ref_tokens']} kv_quant_bits={policy['kv_quant_bits']} "
        f"visual_keep_ratio={args.visual_keep_ratio}",
        flush=True,
    )
    config.deltakv_infer_config = infer_config
    model = LlavaOnevisionDeltaKVForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    if compressor_path is not None:
        incompatible = load_deltakv_compressor_into_llava(model, str(compressor_path), device="cpu")
        compressor_missing = [key for key in incompatible.missing_keys if "compress_" in key]
        if compressor_missing:
            raise RuntimeError(f"DeltaKV compressor weights were not fully loaded; missing examples: {compressor_missing[:8]}")
    return model, policy


@torch.inference_mode()
def run_method(method, model, processor, rows, args, dtype, device, policy=None, requested_method=None):
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    total_new_tokens = 0
    total_time = 0.0

    log_every = max(1, int(args.log_every))
    for sample_idx, row in enumerate(rows, 1):
        image = Image.open(row["image_path"]).convert("RGB")
        prompt = build_prompt(processor, row["question"], args.limit_text_tokens)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        visual_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
        inputs = batch_to_device(inputs, device, dtype)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        new_tokens = int(output_ids.shape[1] - input_len)
        total_new_tokens += new_tokens
        total_time += elapsed
        decoded = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        answer = row["answer"]
        hit = normalize_text(answer) in normalize_text(decoded)
        records.append(
            {
                "question_id": row["question_id"],
                "input_tokens": input_len,
                "visual_tokens": visual_tokens,
                "new_tokens": new_tokens,
                "seconds": elapsed,
                "new_tokens_per_s": new_tokens / elapsed if elapsed > 0 else 0.0,
                "answer": answer,
                "prediction": decoded,
                "contains_answer": hit,
            }
        )
        if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
            print(
                f"[{method}] {sample_idx}/{len(rows)} qid={row['question_id']} "
                f"input={input_len} visual={visual_tokens} new={new_tokens} "
                f"time={elapsed:.3f}s tok/s={records[-1]['new_tokens_per_s']:.2f} "
                f"hit={hit} pred={decoded[:80]!r}",
                flush=True,
            )

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    contains_acc = sum(record["contains_answer"] for record in records) / max(len(records), 1)
    visual_keep_ratio = args.visual_keep_ratio if policy is not None else 1.0
    visual_storage_ratio = 1.0
    if policy is not None:
        visual_storage_ratio = args.visual_keep_ratio
        if int(policy.get("kv_quant_bits", 0)) == 4:
            visual_storage_ratio *= 4.0 / 16.0

    result = {
        "method": method,
        "requested_method": requested_method or method,
        "visual_keep_ratio": visual_keep_ratio,
        "visual_storage_ratio": visual_storage_ratio,
        "num_samples": len(records),
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "mean_seconds": total_time / max(len(records), 1),
        "peak_memory_gb": peak_gb,
        "contains_answer_acc": contains_acc,
        "records": records,
    }
    if policy is not None:
        result["visual_cache_policy"] = policy
    return result


def iter_requested_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield raw_method, "vanilla"
        elif method in VISUAL_PRUNE_METHOD_ALIASES:
            yield raw_method, "visual_cache"
        elif method == "visual_deltakv_compressor":
            yield raw_method, "visual_deltakv_compressor"
        else:
            raise ValueError(
                f"Unknown method: {raw_method}. Supported: vanilla, visual_uniform_keep "
                "(legacy alias: deltakv)."
            )


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    rows = prepare_vqa_subset(Path(args.source_vqa_dir), Path(args.dataset_dir), args.num_samples)
    print(f"[dataset] rows={len(rows)} dataset_dir={args.dataset_dir}", flush=True)
    processor = LlavaOnevisionProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    results = []
    for requested_method, method_kind in iter_requested_methods(args.methods):
        if method_kind == "vanilla":
            model = load_vanilla_model(args, dtype, device)
            method_label = "vanilla"
            policy = None
        elif method_kind == "visual_deltakv_compressor" and resolve_compressor_path(args) is None:
            raise ValueError("visual_deltakv_compressor requires a real --compressor_path, not 'none'.")
        else:
            model, policy = load_visual_cache_model(args, dtype, device)
            method_label = policy["method"]
            if requested_method.lower() == "deltakv" and resolve_compressor_path(args) is None:
                print(
                    "[method_alias] 'deltakv' was requested, but --compressor_path none "
                    "runs visual_uniform_keep, not DeltaKV cluster/compressor inference.",
                    flush=True,
                )

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

    if len(results) == 2:
        base = next((item for item in results if item["method"] == "vanilla"), None)
        candidate = next((item for item in results if item["method"] != "vanilla"), None)
        if base and candidate:
            candidate["speedup_vs_vanilla"] = candidate["new_tokens_per_s"] / base["new_tokens_per_s"]
            candidate["memory_delta_gb_vs_vanilla"] = candidate["peak_memory_gb"] - base["peak_memory_gb"]

    out_path = Path(args.dataset_dir) / "last_benchmark_result.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print("[summary]")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
