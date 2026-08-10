from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNormGated,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runtime_validation import collect_worker_runtime_status

DEFAULT_PROMPTS = (
    "Sparse attention keeps the most useful context because",
    "请用一句话解释专家并行的作用：",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _tokenize_prompts(tokenizer, prompts: tuple[str, ...]) -> list[list[int]]:
    token_ids = []
    for prompt in prompts:
        add_special_tokens = True
        if tokenizer.bos_token is None or prompt.startswith(tokenizer.bos_token):
            add_special_tokens = False
        encoded = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if not encoded:
            raise ValueError(f"Prompt tokenized to an empty sequence: {prompt!r}.")
        token_ids.append([int(token_id) for token_id in encoded])
    return token_ids


def _run_transformers(
    args,
    tokenizer,
    prompt_token_ids: list[list[int]],
) -> tuple[
    list[dict[str, Any]],
    list[torch.Tensor],
    dict[str, Any],
    list[dict[int, torch.Tensor]],
    list[dict[int, torch.Tensor]],
    list[dict[int, dict[str, torch.Tensor]]],
]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=("eager" if args.torch_reference_kernels else "sdpa"),
    ).to("cuda").eval()
    language_model = model.model
    if hasattr(language_model, "language_model"):
        language_model = language_model.language_model
    if args.torch_reference_kernels:
        for layer in language_model.layers:
            linear_attn = getattr(layer, "linear_attn", None)
            if linear_attn is None:
                continue
            linear_attn.causal_conv1d_fn = None
            linear_attn.causal_conv1d_update = torch_causal_conv1d_update
            linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            linear_attn.recurrent_gated_delta_rule = (
                torch_recurrent_gated_delta_rule
            )
            torch_norm = Qwen3_5MoeRMSNormGated(
                linear_attn.head_v_dim,
                eps=linear_attn.layer_norm_epsilon,
            ).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                torch_norm.weight.copy_(linear_attn.norm.weight)
            linear_attn.norm = torch_norm
    rows = []
    logits = []
    hidden_snapshots: list[dict[int, torch.Tensor]] = []
    cached_hidden_snapshots: list[dict[int, torch.Tensor]] = []
    selected_layers = tuple(args.debug_hidden_layers)
    live_hidden: dict[int, torch.Tensor] = {}
    handles = []

    def capture(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            live_hidden[layer_idx] = tensor[:, -1].detach().cpu()

        return hook

    if selected_layers:
        for layer_idx in selected_layers:
            handles.append(
                language_model.layers[layer_idx].register_forward_hook(
                    capture(layer_idx)
                )
            )
        handles.append(
            language_model.norm.register_forward_hook(
                capture(len(language_model.layers))
            )
        )
    try:
        for sample_id, (prompt, input_ids) in enumerate(
            zip(DEFAULT_PROMPTS, prompt_token_ids)
        ):
            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device="cuda"
            )
            with torch.inference_mode():
                generated = model.generate(
                    input_tensor,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                output_ids = generated[0, input_tensor.shape[1] :].tolist()
                if len(output_ids) != args.max_new_tokens:
                    raise RuntimeError(
                        f"Transformers sample {sample_id} generated {len(output_ids)} "
                        f"tokens, expected {args.max_new_tokens}."
                    )
                if selected_layers:
                    live_hidden[-1] = (
                        language_model.embed_tokens(generated[:, -2])
                        .detach()
                        .cpu()
                    )
                    cached_hidden_snapshots.append(
                        dict(sorted(live_hidden.items()))
                    )
                final_input = generated[:, :-1]
                live_hidden.clear()
                final_logits = model(
                    input_ids=final_input,
                    use_cache=False,
                    return_dict=True,
                ).logits[0, -1].detach().cpu()
                if selected_layers:
                    live_hidden[-1] = (
                        language_model.embed_tokens(final_input[:, -1])
                        .detach()
                        .cpu()
                    )
                    hidden_snapshots.append(dict(sorted(live_hidden.items())))
            logits.append(final_logits)
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "prompt": prompt,
                    "prompt_token_ids": input_ids,
                    "output_token_ids": [int(token_id) for token_id in output_ids],
                    "output_text": tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ),
                }
            )
    finally:
        for handle in handles:
            handle.remove()
    return (
        rows,
        logits,
        {"backend": "transformers", "worker_runtime_status": []},
        hidden_snapshots,
        cached_hidden_snapshots,
        [],
    )


def _run_sparsevllm(
    args,
    tokenizer,
    prompt_token_ids: list[list[int]],
) -> tuple[
    list[dict[str, Any]],
    list[torch.Tensor],
    dict[str, Any],
    list[dict[int, torch.Tensor]],
    list[dict[int, torch.Tensor]],
    list[dict[int, dict[str, torch.Tensor]]],
]:
    os.environ["SPARSEVLLM_DEBUG_RUNTIME"] = "1"
    os.environ["SPARSEVLLM_DEBUG_MOE"] = "1"
    if args.debug_hidden_layers:
        os.environ["SPARSEVLLM_DEBUG_HIDDEN_LAYERS"] = ",".join(
            str(layer_idx) for layer_idx in args.debug_hidden_layers
        )
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from sparsevllm import LLM, SamplingParams

    max_prompt_len = max(len(item) for item in prompt_token_ids)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        data_parallel_size=1,
        enforce_eager=not args.decode_cuda_graph,
        decode_cuda_graph=args.decode_cuda_graph,
        gpu_memory_utilization=args.gpu_memory_utilization,
        weight_loading_workers=args.weight_loading_workers,
        max_model_len=max_prompt_len + args.max_new_tokens + 32,
        max_num_seqs_in_batch=1,
        max_decoding_seqs=1,
        engine_prefill_chunk_size=max(64, max_prompt_len),
        enable_profiler=False,
    )
    rows = []
    logits = []
    debug_summaries = []
    hidden_snapshots = []
    moe_snapshots = []
    try:
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        )
        for sample_id, (prompt, input_ids) in enumerate(
            zip(DEFAULT_PROMPTS, prompt_token_ids)
        ):
            result = llm.generate(
                [input_ids], sampling_params, use_tqdm=False
            )[0]
            output_ids = [int(token_id) for token_id in result["token_ids"]]
            if len(output_ids) != args.max_new_tokens:
                raise RuntimeError(
                    f"Sparse-vLLM sample {sample_id} generated {len(output_ids)} "
                    f"tokens, expected {args.max_new_tokens}."
                )
            logits.append(llm.debug_last_logits().detach().cpu()[0])
            debug_summaries.append(llm.debug_sparse_state_summaries())
            if args.debug_hidden_layers:
                hidden_snapshots.append(llm.debug_hidden_states())
            if args.debug_moe_states:
                moe_snapshots.append(llm.debug_moe_states())
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "prompt": prompt,
                    "prompt_token_ids": input_ids,
                    "output_token_ids": output_ids,
                    "output_text": result["text"],
                }
            )
        worker_status = collect_worker_runtime_status(llm)
        if args.decode_cuda_graph and not all(
            bool(status["decode_cuda_graph_active"]) for status in worker_status
        ):
            raise RuntimeError(
                "Decode CUDA Graph is not active on every rank: "
                f"{worker_status!r}."
            )
    finally:
        llm.exit()
    return (
        rows,
        logits,
        {
            "backend": "sparsevllm",
            "worker_runtime_status": worker_status,
            "debug_sparse_state_summaries": debug_summaries,
        },
        hidden_snapshots,
        hidden_snapshots,
        moe_snapshots,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Produce deterministic Qwen3.6 MoE end-to-end reference artifacts."
    )
    parser.add_argument("--backend", choices=("transformers", "sparsevllm"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--decode-cuda-graph", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--weight-loading-workers", type=int, default=16)
    parser.add_argument(
        "--torch-reference-kernels",
        action="store_true",
        help=(
            "For the Transformers backend, force eager full attention and the "
            "explicit Torch Gated DeltaNet conv/chunk/recurrent/norm functions."
        ),
    )
    parser.add_argument(
        "--debug-hidden-layers",
        type=int,
        nargs="*",
        default=(),
        help="Capture last-token hidden states after the selected decoder layers.",
    )
    parser.add_argument(
        "--debug-moe-states",
        action="store_true",
        help="Capture rank-0 per-layer MoE inputs, routing, and outputs.",
    )
    parser.add_argument(
        "--forced-prefix-artifact",
        type=Path,
        default=None,
        help=(
            "Append all but the last generated token from a prior "
            "per_sample_results.jsonl artifact to each fixed prompt."
        ),
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.torch_reference_kernels and args.backend != "transformers":
        raise ValueError("--torch-reference-kernels requires --backend transformers.")
    invalid_hidden_layers = [
        layer_idx
        for layer_idx in args.debug_hidden_layers
        if layer_idx < 0 or layer_idx >= 40
    ]
    if invalid_hidden_layers:
        raise ValueError(
            "--debug-hidden-layers must be in [0, 39], got "
            f"{invalid_hidden_layers}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires CUDA.")

    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.forced_prefix_artifact is not None:
        args.forced_prefix_artifact = args.forced_prefix_artifact.resolve()
        if not args.forced_prefix_artifact.is_file():
            raise FileNotFoundError(
                "Forced-prefix artifact does not exist: "
                f"{args.forced_prefix_artifact}."
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "backend": args.backend,
        "model": str(args.model),
        "model_config_sha256": _sha256(args.model / "config.json"),
        "model_index_sha256": _sha256(
            args.model / "model.safetensors.index.json"
        ),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "tensor_parallel_size": args.tensor_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "decode_cuda_graph": args.decode_cuda_graph,
        "torch_reference_kernels": args.torch_reference_kernels,
        "forced_prefix_artifact": (
            None
            if args.forced_prefix_artifact is None
            else str(args.forced_prefix_artifact)
        ),
        "forced_prefix_artifact_sha256": (
            None
            if args.forced_prefix_artifact is None
            else _sha256(args.forced_prefix_artifact)
        ),
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "requested_moe_provider": os.getenv("SPARSEVLLM_MOE_PROVIDER", "auto"),
        "requested_moe_router_provider": os.getenv(
            "SPARSEVLLM_MOE_ROUTER_PROVIDER", "auto"
        ),
    }
    _write_json(args.output_dir / "run_info.json", run_info)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_token_ids = _tokenize_prompts(tokenizer, DEFAULT_PROMPTS)
    if args.forced_prefix_artifact is not None:
        forced_rows = [
            json.loads(line)
            for line in args.forced_prefix_artifact.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(forced_rows) != len(prompt_token_ids):
            raise ValueError(
                "Forced-prefix artifact must contain one row per fixed prompt: "
                f"expected={len(prompt_token_ids)} got={len(forced_rows)}."
            )
        for sample_id, (token_ids, row) in enumerate(
            zip(prompt_token_ids, forced_rows)
        ):
            output_token_ids = row.get("output_token_ids")
            if not isinstance(output_token_ids, list) or len(output_token_ids) < 2:
                raise ValueError(
                    f"Forced-prefix sample {sample_id} needs at least two "
                    "output_token_ids."
                )
            token_ids.extend(int(token_id) for token_id in output_token_ids[:-1])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        if args.backend == "transformers":
            (
                rows,
                logits,
                parsed,
                hidden_snapshots,
                cached_hidden_snapshots,
                moe_snapshots,
            ) = _run_transformers(
                args, tokenizer, prompt_token_ids
            )
        else:
            (
                rows,
                logits,
                parsed,
                hidden_snapshots,
                cached_hidden_snapshots,
                moe_snapshots,
            ) = _run_sparsevllm(
                args, tokenizer, prompt_token_ids
            )
    except Exception as exc:
        failure = {
            "sample_id": None,
            "status": "model_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _write_jsonl(args.output_dir / "raw_outputs.jsonl", [failure])
        _write_jsonl(args.output_dir / "parsed_outputs.jsonl", [failure])
        _write_jsonl(args.output_dir / "per_sample_results.jsonl", [failure])
        _write_json(args.output_dir / "aggregate_metrics.json", failure)
        raise

    _write_jsonl(args.output_dir / "raw_outputs.jsonl", rows)
    _write_jsonl(args.output_dir / "parsed_outputs.jsonl", rows)
    _write_jsonl(args.output_dir / "per_sample_results.jsonl", rows)
    torch.save(logits, args.output_dir / "raw_logits.pt")
    if hidden_snapshots:
        torch.save(
            hidden_snapshots,
            args.output_dir / "raw_hidden_states.pt",
        )
    if cached_hidden_snapshots:
        torch.save(
            cached_hidden_snapshots,
            args.output_dir / "raw_cached_hidden_states.pt",
        )
    if moe_snapshots:
        torch.save(
            moe_snapshots,
            args.output_dir / "raw_moe_states.pt",
        )
    _write_json(args.output_dir / "runtime_status.json", parsed)
    aggregate = {
        "status": "success",
        "num_samples": len(rows),
        "success_samples": sum(row["status"] == "success" for row in rows),
        "failed_samples": sum(row["status"] != "success" for row in rows),
    }
    _write_json(args.output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
