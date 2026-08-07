from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deltakv.configs.model_config_cls import KVQwen3Config
from deltakv.configs.runtime_params import normalize_runtime_params
from deltakv.get_chat_api import _load_deltakv_checkpoint_config, load_compressor
from deltakv.modeling.cache_factory import DELTA_COMPRESSED_QUANT_KIVI_FULL_FP8_REF
from deltakv.modeling.qwen3_inference import Qwen3KVCompress


NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _build_chat(tokenizer, prompt: str, dataset: str, *, thinking_mode: str) -> str:
    if dataset in NO_CHAT_TEMPLATE_DATASETS:
        return prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode != "off",
        )
        if thinking_mode == "off" and prompt.endswith("<think>\n"):
            prompt += "</think>\n"
    return prompt


def build_longbench_input_ids(
    *,
    tokenizer,
    data_root: str,
    dataset: str,
    sample_index: int,
    max_prompt_tokens: int,
    thinking_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    data_root_path = Path(data_root)
    prompt_path = Path("benchmark/long_bench/config/dataset2prompt.json")
    data_path = data_root_path / "data" / f"{dataset}.jsonl"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing LongBench prompt config: {prompt_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing LongBench data file: {data_path}")

    dataset2prompt = _load_json(prompt_path)
    rows = _load_jsonl(data_path)
    if sample_index < 0 or sample_index >= len(rows):
        raise IndexError(f"sample_index={sample_index} is outside dataset size {len(rows)}")
    row = rows[sample_index]
    prompt = dataset2prompt[dataset].format(**row)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if tokenized_prompt.numel() > max_prompt_tokens:
        half = max_prompt_tokens // 2
        if half <= 0:
            raise ValueError(f"max_prompt_tokens must be positive, got {max_prompt_tokens}")
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized_prompt[-half:], skip_special_tokens=True
        )
    prompt = _build_chat(tokenizer, prompt, dataset, thinking_mode=thinking_mode)
    input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids
    return input_ids, {
        "dataset": dataset,
        "sample_index": sample_index,
        "length_field": row.get("length"),
        "answers": row.get("answers"),
        "prompt_tokens_before_chat_truncation": int(tokenized_prompt.numel()),
        "input_tokens": int(input_ids.shape[1]),
    }


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


def load_vanilla_model(model_path: str, *, device: str, dtype: torch.dtype, attn_implementation: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model


def load_deltakv_model(
    model_path: str,
    *,
    hyper_params: dict[str, Any],
    checkpoint_path: str | None,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
):
    params = dict(hyper_params)
    params.setdefault("sparse_method", DELTA_COMPRESSED_QUANT_KIVI_FULL_FP8_REF)
    normalized = normalize_runtime_params(params, backend="hf")
    if normalized.hf_model_cls != DELTA_COMPRESSED_QUANT_KIVI_FULL_FP8_REF:
        raise ValueError(
            "This diagnostic script currently expects HF "
            f"{DELTA_COMPRESSED_QUANT_KIVI_FULL_FP8_REF}, got {normalized.hf_model_cls!r}."
        )

    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if base_config.model_type != "qwen3":
        raise ValueError(f"Expected a Qwen3 model, got model_type={base_config.model_type!r}.")

    if checkpoint_path:
        config = _load_deltakv_checkpoint_config(KVQwen3Config, checkpoint_path)
    else:
        config = KVQwen3Config.from_pretrained(model_path)
    config.set_native_args(**normalized.infer_config)
    config.deltakv_cache_impl = DELTA_COMPRESSED_QUANT_KIVI_FULL_FP8_REF
    config.finalize_cluster_args()

    model = Qwen3KVCompress.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if checkpoint_path:
        comp_state_dict = load_compressor(checkpoint_path, device=device)
        _, unexpected = model.load_state_dict(comp_state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected DeltaKV checkpoint keys while loading compressor: {unexpected}")
        del comp_state_dict
    model.eval()
    return model, tuple(normalized.warnings), config


def _hidden_label(index: int, total: int) -> str:
    if index == 0:
        return "embed"
    if index == total - 1:
        return "final_norm"
    return f"layer_{index - 1:02d}"


@torch.inference_mode()
def trace_model(model, input_ids: torch.Tensor, *, prefill_tokens: int, decode_steps: int, device: str):
    if input_ids.shape[1] < prefill_tokens + decode_steps:
        raise ValueError(
            f"Need at least prefill_tokens + decode_steps tokens "
            f"({prefill_tokens + decode_steps}), got {input_ids.shape[1]}."
        )

    input_ids = input_ids.to(device)
    steps: list[dict[str, Any]] = []
    past = None

    chunks = [("prefill", input_ids[:, :prefill_tokens])]
    for i in range(decode_steps):
        pos = prefill_tokens + i
        chunks.append((f"decode_{i:03d}", input_ids[:, pos : pos + 1]))

    for step_index, (step_name, chunk_ids) in enumerate(chunks):
        outputs = model(
            input_ids=chunk_ids,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        hidden_last = []
        total = len(outputs.hidden_states)
        for layer_index, hidden in enumerate(outputs.hidden_states):
            hidden_last.append(
                {
                    "label": _hidden_label(layer_index, total),
                    "tensor": hidden[:, -1, :].detach().to(dtype=torch.float32, device="cpu").contiguous(),
                }
            )
        logits = outputs.logits[:, -1, :].detach().to(dtype=torch.float32, device="cpu").contiguous()
        steps.append(
            {
                "step_index": step_index,
                "step_name": step_name,
                "chunk_tokens": int(chunk_ids.shape[1]),
                "absolute_token_index": int(prefill_tokens - 1 if step_index == 0 else prefill_tokens + step_index - 1),
                "hidden_last": hidden_last,
                "logits": logits,
            }
        )
    return steps


def _mse_stats(delta: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    err = delta - ref
    mse = torch.mean(err * err).item()
    ref_mse = torch.mean(ref * ref).item()
    rel_mse = mse / ref_mse if ref_mse > 0 else math.inf
    return {
        "mse": float(mse),
        "rel_mse": float(rel_mse),
        "err_rms": float(math.sqrt(max(mse, 0.0))),
        "ref_rms": float(math.sqrt(max(ref_mse, 0.0))),
        "max_abs": float(torch.max(torch.abs(err)).item()),
    }


def compare_traces(vanilla_steps, delta_steps) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(vanilla_steps) != len(delta_steps):
        raise ValueError(f"Trace step count differs: vanilla={len(vanilla_steps)}, delta={len(delta_steps)}")

    records: list[dict[str, Any]] = []
    max_hidden = None
    max_logit = None
    first_rel_above_1e_2 = None

    for vanilla_step, delta_step in zip(vanilla_steps, delta_steps):
        if vanilla_step["step_name"] != delta_step["step_name"]:
            raise ValueError(f"Trace step mismatch: {vanilla_step['step_name']} vs {delta_step['step_name']}")
        logit_stats = _mse_stats(delta_step["logits"], vanilla_step["logits"])
        logit_record = {
            "step_index": vanilla_step["step_index"],
            "step_name": vanilla_step["step_name"],
            "absolute_token_index": vanilla_step["absolute_token_index"],
            "scope": "logits",
            **logit_stats,
        }
        if max_logit is None or logit_record["rel_mse"] > max_logit["rel_mse"]:
            max_logit = logit_record

        for vanilla_hidden, delta_hidden in zip(vanilla_step["hidden_last"], delta_step["hidden_last"]):
            if vanilla_hidden["label"] != delta_hidden["label"]:
                raise ValueError(f"Layer mismatch: {vanilla_hidden['label']} vs {delta_hidden['label']}")
            stats = _mse_stats(delta_hidden["tensor"], vanilla_hidden["tensor"])
            record = {
                "step_index": vanilla_step["step_index"],
                "step_name": vanilla_step["step_name"],
                "absolute_token_index": vanilla_step["absolute_token_index"],
                "scope": "hidden_last_token",
                "layer": vanilla_hidden["label"],
                **stats,
            }
            records.append(record)
            if max_hidden is None or record["rel_mse"] > max_hidden["rel_mse"]:
                max_hidden = record
            if first_rel_above_1e_2 is None and record["rel_mse"] >= 1e-2:
                first_rel_above_1e_2 = record

    summary = {
        "max_hidden_rel_mse": max_hidden,
        "first_hidden_rel_mse_ge_1e_2": first_rel_above_1e_2,
        "max_logit_rel_mse": max_logit,
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--hyper_param", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_prompt_tokens", type=int, default=8192)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--decode_steps", type=int, default=8)
    parser.add_argument("--thinking_mode", default="off", choices=["off", "on"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _dtype_from_name(args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hyper_params = _load_json(args.hyper_param)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    input_ids, sample_meta = build_longbench_input_ids(
        tokenizer=tokenizer,
        data_root=args.data_root,
        dataset=args.dataset,
        sample_index=args.sample_index,
        max_prompt_tokens=args.max_prompt_tokens,
        thinking_mode=args.thinking_mode,
    )
    required_tokens = args.prefill_tokens + args.decode_steps
    if input_ids.shape[1] < required_tokens:
        raise ValueError(
            f"Prompt has {input_ids.shape[1]} tokens after chat formatting; "
            f"need at least {required_tokens}. Reduce prefill_tokens/decode_steps."
        )

    meta = {
        "model_path": args.model_path,
        "hyper_param": str(args.hyper_param),
        "checkpoint_path": args.checkpoint_path,
        "data_root": args.data_root,
        "prefill_tokens": args.prefill_tokens,
        "decode_steps": args.decode_steps,
        "max_prompt_tokens": args.max_prompt_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "sample": sample_meta,
    }

    vanilla_model = load_vanilla_model(
        args.model_path,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    vanilla_steps = trace_model(
        vanilla_model,
        input_ids,
        prefill_tokens=args.prefill_tokens,
        decode_steps=args.decode_steps,
        device=args.device,
    )
    del vanilla_model
    gc.collect()
    torch.cuda.empty_cache()

    deltakv_model, normalization_warnings, resolved_config = load_deltakv_model(
        args.model_path,
        hyper_params=hyper_params,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    delta_steps = trace_model(
        deltakv_model,
        input_ids,
        prefill_tokens=args.prefill_tokens,
        decode_steps=args.decode_steps,
        device=args.device,
    )
    records, summary = compare_traces(vanilla_steps, delta_steps)

    meta["normalization_warnings"] = list(normalization_warnings)
    meta["resolved_config"] = {
        "full_attn_layers": list(getattr(resolved_config, "full_attn_layers", [])),
        "use_compression": bool(getattr(resolved_config, "use_compression", False)),
        "use_cluster": bool(getattr(resolved_config, "use_cluster", False)),
        "kv_quant_bits": int(getattr(resolved_config, "kv_quant_bits", 0) or 0),
        "kv_quant_group_size": int(getattr(resolved_config, "kv_quant_group_size", 0) or 0),
        "full_layer_kv_quant_bits": int(getattr(resolved_config, "full_layer_kv_quant_bits", 0) or 0),
        "enable_full_layer_kivi_quant": bool(getattr(resolved_config, "enable_full_layer_kivi_quant", False)),
        "enable_sparse_ref_fp8": bool(getattr(resolved_config, "enable_sparse_ref_fp8", False)),
        "cluster_ratio": float(getattr(resolved_config, "cluster_ratio", 0.0) or 0.0),
        "deltakv_neighbor_count": int(getattr(resolved_config, "deltakv_neighbor_count", 0) or 0),
        "num_top_tokens": int(getattr(resolved_config, "num_top_tokens", 0) or 0),
        "num_top_tokens_in_prefill": int(getattr(resolved_config, "num_top_tokens_in_prefill", 0) or 0),
        "num_recent_tokens": int(getattr(resolved_config, "num_recent_tokens", 0) or 0),
        "num_sink_tokens": int(getattr(resolved_config, "num_sink_tokens", 0) or 0),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": records}, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(output_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
