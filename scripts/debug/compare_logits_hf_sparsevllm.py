#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from deltakv.configs.runtime_params import normalize_runtime_params
from deltakv.get_chat_api import get_generate_api
from sparsevllm.config import Config
from sparsevllm.engine.model_runner import ModelRunner
from sparsevllm.engine.sequence import Sequence
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.utils.context import get_context, reset_context


DEFAULT_MODEL = "/data2/haojitai/models/Qwen2.5-7B-Instruct-1M"
DEFAULT_COMPRESSOR = "/data2/haojitai/checkpoints/compressor/Qwen2.5-7B-Instruct-1M-Compressor"
DEFAULT_OUTPUT_ROOT = "/data2/haojitai/outputs/sparsevllm_logits_align"


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def _git_status_short() -> str:
    import subprocess

    return subprocess.check_output(["git", "status", "--short"], text=True).strip()


def _require_path(path: str, kind: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _parse_cases(value: str) -> list[str]:
    cases = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"short", "long"}
    bad = sorted(set(cases) - allowed)
    if bad:
        raise ValueError(f"Unsupported cases: {bad}. Allowed: {sorted(allowed)}")
    return cases


def _build_prompt(tokenizer, case_name: str, target_tokens: int) -> tuple[str, list[int]]:
    if case_name == "short":
        prompt = "The capital of France is"
    elif case_name == "long":
        unit = (
            "Sparse long-context inference compares cache layouts, attention masks, "
            "position ids, and compressed DeltaKV reconstruction. "
        )
        prompt = unit
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        while len(token_ids) < target_tokens:
            prompt += unit
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt = tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=False)
    else:
        raise ValueError(f"Unsupported case_name={case_name!r}")

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"{case_name} prompt tokenized to an empty sequence.")
    return prompt, token_ids


def _compare_logits(
    hf_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
    *,
    tokenizer,
    topk_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, Any]:
    hf = hf_logits.detach().float().cpu().view(-1)
    sv = sparse_logits.detach().float().cpu().view(-1)
    if hf.shape != sv.shape:
        raise ValueError(f"Logit shape mismatch: hf={tuple(hf.shape)} sparse={tuple(sv.shape)}")

    diff = (hf - sv).abs()
    hf_argmax = int(torch.argmax(hf).item())
    sv_argmax = int(torch.argmax(sv).item())
    metrics: dict[str, Any] = {
        "shape": list(hf.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "median_abs_diff": float(torch.quantile(diff, 0.5).item()),
        "p99_abs_diff": float(torch.quantile(diff, 0.99).item()),
        "argmax_match": hf_argmax == sv_argmax,
        "hf_argmax": hf_argmax,
        "sparse_argmax": sv_argmax,
        "hf_argmax_text": tokenizer.decode([hf_argmax], skip_special_tokens=False),
        "sparse_argmax_text": tokenizer.decode([sv_argmax], skip_special_tokens=False),
        "hf_argmax_logit": float(hf[hf_argmax].item()),
        "sparse_argmax_logit": float(sv[sv_argmax].item()),
        "topk_overlap": {},
    }

    vocab = int(hf.numel())
    for k in topk_values:
        kk = min(int(k), vocab)
        hf_top = set(int(x) for x in torch.topk(hf, kk).indices.tolist())
        sv_top = set(int(x) for x in torch.topk(sv, kk).indices.tolist())
        metrics["topk_overlap"][str(k)] = {
            "intersection": len(hf_top & sv_top),
            "ratio": len(hf_top & sv_top) / max(kk, 1),
        }

    union_top = []
    for token_id in torch.topk(hf, min(10, vocab)).indices.tolist():
        if int(token_id) not in union_top:
            union_top.append(int(token_id))
    for token_id in torch.topk(sv, min(10, vocab)).indices.tolist():
        if int(token_id) not in union_top:
            union_top.append(int(token_id))

    metrics["top_token_diffs"] = [
        {
            "token_id": token_id,
            "text": tokenizer.decode([token_id], skip_special_tokens=False),
            "hf_logit": float(hf[token_id].item()),
            "sparse_logit": float(sv[token_id].item()),
            "abs_diff": float(diff[token_id].item()),
        }
        for token_id in union_top[:20]
    ]
    return metrics


def _hf_infer_config(args: argparse.Namespace, method: str, prompt_len: int) -> dict[str, Any]:
    if method == "vanilla":
        return {
            "sparse_method": "vanilla",
        }

    if not isinstance(args.decode_keep_tokens, int) or not isinstance(args.prefill_keep_tokens, int):
        raise TypeError("HF/Sparse-VLLM top token budgets must be explicit integers.")

    # HF DeltaKV should not chunk this logits-alignment prefill unless explicitly requested.
    hf_prefill_chunk_size = int(args.hf_prefill_chunk_size)
    if hf_prefill_chunk_size <= prompt_len:
        raise ValueError(
            "hf_prefill_chunk_size must exceed the prompt length for this alignment run. "
            f"got hf_prefill_chunk_size={hf_prefill_chunk_size}, prompt_len={prompt_len}"
        )

    if method == "omnikv":
        return {
            "sparse_method": "omnikv",
            "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
            "decode_keep_tokens": int(args.decode_keep_tokens),
            "prefill_keep_tokens": int(args.prefill_keep_tokens),
            "sink_keep_tokens": int(args.sink_keep_tokens),
            "recent_keep_tokens": int(args.recent_keep_tokens),
            "full_attention_layers": args.full_attention_layers,
            "hf_prefill_chunk_size": hf_prefill_chunk_size,
        }

    return {
        "sparse_method": "deltakv",
        "deltakv_checkpoint_path": args.compressor_path,
        "use_cluster": True,
        "use_compression": True,
        "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
        "decode_keep_tokens": int(args.decode_keep_tokens),
        "prefill_keep_tokens": int(args.prefill_keep_tokens),
        "sink_keep_tokens": int(args.sink_keep_tokens),
        "recent_keep_tokens": int(args.recent_keep_tokens),
        "full_attention_layers": args.full_attention_layers,
        "deltakv_center_ratio": float(args.deltakv_center_ratio),
        "deltakv_latent_dim": int(args.deltakv_latent_dim),
        "deltakv_latent_quant_bits": int(args.deltakv_latent_quant_bits),
        "deltakv_neighbor_count": int(args.deltakv_neighbor_count),
        "hf_prefill_chunk_size": hf_prefill_chunk_size,
    }


def _sparse_infer_config(args: argparse.Namespace, method: str) -> dict[str, Any]:
    config = {
        "max_model_len": args.max_model_len,
        "max_num_seqs_in_batch": 1,
        "max_decoding_seqs": 1,
        "max_num_batched_tokens": max(args.long_tokens + 8, args.engine_prefill_chunk_size * 2 + 8),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "throughput_log_interval_s": 0.0,
        "sparse_method": method,
        "engine_prefill_chunk_size": int(args.engine_prefill_chunk_size),
        "mlp_chunk_size": int(args.mlp_chunk_size),
    }
    if method == "omnikv":
        config.update(
            {
                "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "prefill_keep_tokens": int(args.prefill_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
                "full_attention_layers": args.full_attention_layers,
            }
        )
    elif method != "vanilla":
        config.update(
            {
                "deltakv_checkpoint_path": args.compressor_path,
                "chunk_prefill_accel_omnikv": bool(args.chunk_prefill_accel_omnikv),
                "decode_keep_tokens": int(args.decode_keep_tokens),
                "prefill_keep_tokens": int(args.prefill_keep_tokens),
                "sink_keep_tokens": int(args.sink_keep_tokens),
                "recent_keep_tokens": int(args.recent_keep_tokens),
                "full_attention_layers": args.full_attention_layers,
                "deltakv_center_ratio": float(args.deltakv_center_ratio),
                "deltakv_latent_dim": int(args.deltakv_latent_dim),
                "deltakv_latent_quant_bits": int(args.deltakv_latent_quant_bits),
                "deltakv_neighbor_count": int(args.deltakv_neighbor_count),
                "deltakv_full_pool_reserve_ratio": float(args.deltakv_full_pool_reserve_ratio),
                "deltakv_cluster_gather_chunk_size": int(args.deltakv_cluster_gather_chunk_size),
            }
        )
    return config


def _load_hf_model(args: argparse.Namespace, method: str, prompt_len: int):
    infer_config = _hf_infer_config(args, method, prompt_len)
    _, model = get_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=None,
        sparse_method=None,
        cuda_device=0,
        backend="hf",
        return_model=True,
    )
    model.eval()
    return model, infer_config


def _hf_logits_for_prompt(model, input_ids: list[int], forced_token_id: int) -> dict[str, torch.Tensor]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    forced = torch.tensor([[forced_token_id]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        prefill = model(input_ids=ids, use_cache=True, return_dict=True)
        prefill_logits = prefill.logits[:, -1, :].detach().cpu()
        decode = model(
            input_ids=forced,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        decode_logits = decode.logits[:, -1, :].detach().cpu()
    return {"prefill": prefill_logits, "decode": decode_logits}


def _make_sparse_runner(args: argparse.Namespace, method: str) -> tuple[ModelRunner, dict[str, Any]]:
    public_config = _sparse_infer_config(args, method)
    normalized = normalize_runtime_params(public_config, backend="sparsevllm")
    config_fields = {field.name for field in fields(Config)}
    unknown = sorted(set(normalized.infer_config) - config_fields)
    if unknown:
        raise ValueError(f"Unknown Sparse-VLLM config keys after normalization: {unknown}")
    config = Config(args.model_path, **normalized.infer_config)
    runner = ModelRunner(config, 0, [])
    return runner, public_config


def _sparse_prefill(
    runner: ModelRunner,
    input_ids: list[int],
) -> tuple[torch.Tensor, Sequence]:
    seq = Sequence(input_ids, SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    seq.current_chunk_size = len(input_ids)
    try:
        input_tensor, positions = runner.prepare_step([seq], is_prefill=True)
        ctx = get_context()
        ctx.sparse_controller = runner.sparse_controller
        runner.sparse_controller.prepare_forward([seq], is_prefill=True)
        logits = runner.run_model(input_tensor, positions, is_prefill=True).detach().cpu()
        runner.sparse_controller.post_forward([seq], is_prefill=True)
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        return logits[-1:].contiguous(), seq
    finally:
        reset_context()


def _sparse_decode(
    runner: ModelRunner,
    seq: Sequence,
    forced_token_id: int,
) -> torch.Tensor:
    seq.append_token(forced_token_id)
    try:
        input_tensor, positions = runner.prepare_step([seq], is_prefill=False)
        ctx = get_context()
        ctx.sparse_controller = runner.sparse_controller
        runner.sparse_controller.prepare_forward([seq], is_prefill=False)
        logits = runner.run_model(input_tensor, positions, is_prefill=False).detach().cpu()
        runner.sparse_controller.post_forward([seq], is_prefill=False)
        return logits[-1:].contiguous()
    finally:
        reset_context()


def _sparse_logits_for_prompt(args: argparse.Namespace, method: str, input_ids: list[int], forced_token_id: int):
    runner = None
    try:
        runner, public_config = _make_sparse_runner(args, method)
        cache_manager_class = type(runner.cache_manager).__name__
        prefill_logits, seq = _sparse_prefill(runner, input_ids)
        decode_logits = _sparse_decode(runner, seq, forced_token_id)
        return {"prefill": prefill_logits, "decode": decode_logits}, public_config, runner.config, cache_manager_class
    finally:
        if runner is not None:
            runner.call("exit")
        _cleanup_cuda()


def _run_one(args: argparse.Namespace, tokenizer, case_name: str, method: str, output_dir: Path) -> dict[str, Any]:
    prompt, input_ids = _build_prompt(tokenizer, case_name, args.long_tokens)
    print(f"[Case] {case_name}/{method}: prompt_tokens={len(input_ids)}")

    hf_method = method if method in {"vanilla", "omnikv"} else "deltakv"
    hf_model, hf_config = _load_hf_model(args, hf_method, len(input_ids))
    try:
        with torch.inference_mode():
            tmp = hf_model(input_ids=torch.tensor([input_ids], dtype=torch.long, device=hf_model.device), use_cache=True)
            hf_prefill_logits = tmp.logits[:, -1, :].detach().cpu()
        forced_token_id = int(torch.argmax(hf_prefill_logits[0]).item())
        del tmp, hf_prefill_logits
        _cleanup_cuda()
        hf_logits = _hf_logits_for_prompt(hf_model, input_ids, forced_token_id)
    finally:
        del hf_model
        _cleanup_cuda()

    sparse_method = method if method in {"vanilla", "omnikv"} else args.sparse_method
    sparse_logits, sparse_public_config, sparse_resolved_config, sparse_cache_manager_class = _sparse_logits_for_prompt(
        args,
        sparse_method,
        input_ids,
        forced_token_id,
    )

    comparisons = {
        stage: _compare_logits(hf_logits[stage], sparse_logits[stage], tokenizer=tokenizer)
        for stage in ("prefill", "decode")
    }
    result = {
        "case": case_name,
        "method": method,
        "status": "success",
        "prompt_tokens": len(input_ids),
        "prompt_preview": prompt[:240],
        "forced_decode_token_id": forced_token_id,
        "forced_decode_token_text": tokenizer.decode([forced_token_id], skip_special_tokens=False),
        "hf_config": hf_config,
        "sparse_public_config": sparse_public_config,
        "sparse_resolved_config": {
            "vllm_sparse_method": sparse_resolved_config.vllm_sparse_method,
            "cache_manager_class": sparse_cache_manager_class,
            "prefill_schedule_policy": sparse_resolved_config.prefill_schedule_policy,
            "num_top_tokens": sparse_resolved_config.num_top_tokens,
            "num_top_tokens_in_prefill": sparse_resolved_config.num_top_tokens_in_prefill,
            "num_sink_tokens": sparse_resolved_config.num_sink_tokens,
            "num_recent_tokens": sparse_resolved_config.num_recent_tokens,
            "chunk_prefill_size": sparse_resolved_config.chunk_prefill_size,
            "cluster_ratio": sparse_resolved_config.cluster_ratio,
            "kv_compressed_size": sparse_resolved_config.kv_compressed_size,
            "kv_quant_bits": sparse_resolved_config.kv_quant_bits,
            "deltakv_k_neighbors": sparse_resolved_config.deltakv_k_neighbors,
            "full_attn_layers": sparse_resolved_config.full_attn_layers,
        },
        "comparisons": comparisons,
    }
    _json_dump(output_dir / f"{case_name}_{method}.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HF and Sparse-VLLM logits on GPU 6.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--compressor_path", default=DEFAULT_COMPRESSOR)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cases", default="short,long")
    parser.add_argument("--methods", default="vanilla,deltakv")
    parser.add_argument("--cuda_visible_devices", default="6")
    parser.add_argument("--master_port", type=int, default=29561)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--long_tokens", type=int, default=9000)
    parser.add_argument("--sparse_method", default="deltakv-triton-v4")
    parser.add_argument("--decode_keep_tokens", type=int, default=4096)
    parser.add_argument("--prefill_keep_tokens", type=int, default=4096)
    parser.add_argument("--sink_keep_tokens", type=int, default=8)
    parser.add_argument("--recent_keep_tokens", type=int, default=128)
    parser.add_argument("--full_attention_layers", default="0,1,2,4,7,14")
    parser.add_argument("--deltakv_center_ratio", type=float, default=0.1)
    parser.add_argument("--deltakv_latent_dim", type=int, default=256)
    parser.add_argument("--deltakv_latent_quant_bits", type=int, default=0)
    parser.add_argument("--deltakv_neighbor_count", type=int, default=4)
    parser.add_argument("--engine_prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--hf_prefill_chunk_size", type=int, default=100_000_000)
    parser.add_argument("--chunk_prefill_accel_omnikv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--deltakv_full_pool_reserve_ratio", type=float, default=0.2)
    parser.add_argument("--deltakv_cluster_gather_chunk_size", type=int, default=16384)
    parser.add_argument("--mlp_chunk_size", type=int, default=16384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.cuda_visible_devices:
        raise RuntimeError(
            "This script must be launched with the intended visible GPU. "
            f"Expected CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r}, got {visible!r}."
        )

    os.environ.setdefault("SPARSEVLLM_MASTER_PORT", str(args.master_port))
    _require_path(args.model_path, "model_path")
    _require_path(args.compressor_path, "compressor_path")
    if not isinstance(args.decode_keep_tokens, int) or not isinstance(args.prefill_keep_tokens, int):
        raise TypeError("decode_keep_tokens and prefill_keep_tokens must be integers.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_info = {
        "status": "running",
        "created_at": timestamp,
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "cuda_visible_devices": visible,
        "torch_version": torch.__version__,
        "model_path": args.model_path,
        "compressor_path": args.compressor_path,
        "args": vars(args),
    }
    _json_dump(output_dir / "run_info.json", run_info)

    cases = _parse_cases(args.cases)
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    allowed_methods = {"vanilla", "deltakv", "omnikv"}
    bad_methods = sorted(set(methods) - allowed_methods)
    if bad_methods:
        raise ValueError(f"Unsupported methods: {bad_methods}. Allowed: {sorted(allowed_methods)}")

    results = []
    try:
        for method in methods:
            for case_name in cases:
                results.append(_run_one(args, tokenizer, case_name, method, output_dir))
        run_info["status"] = "completed"
        run_info["completed_at"] = time.strftime("%Y%m%d_%H%M%S")
        run_info["results"] = results
        _json_dump(output_dir / "summary.json", run_info)
        _json_dump(output_dir / "run_info.json", run_info)
    except Exception as exc:
        run_info["status"] = "failed"
        run_info["error"] = repr(exc)
        _json_dump(output_dir / "run_info.json", run_info)
        raise

    print(f"[Done] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
