from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path
from typing import Any

import torch


def _load_json_arg(value: str) -> dict[str, Any]:
    if value is None:
        return {}
    value = str(value).strip()
    if value.startswith("@"):
        value = Path(value[1:]).expanduser().read_text(encoding="utf-8")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--hyper_params must be a JSON object.")
    return parsed


def _sparse_kwargs(method: str) -> dict[str, Any]:
    return {"sparse_method": "vanilla" if method == "vanilla" else method}


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> dict[str, float | int]:
    k = min(int(k), int(a.numel()), int(b.numel()))
    a_top = set(a.topk(k).indices.tolist())
    b_top = set(b.topk(k).indices.tolist())
    intersection = len(a_top & b_top)
    return {"intersection": intersection, "ratio": float(intersection / k if k else 1.0)}


def _compare_logits(eager: torch.Tensor, graph: torch.Tensor) -> dict[str, Any]:
    if eager.shape != graph.shape:
        raise ValueError(f"Logit shape mismatch: eager={tuple(eager.shape)} graph={tuple(graph.shape)}")
    diff = (eager - graph).abs()
    result: dict[str, Any] = {
        "shape": list(eager.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "argmax_match": eager.argmax(dim=-1).tolist() == graph.argmax(dim=-1).tolist(),
        "eager_argmax": eager.argmax(dim=-1).tolist(),
        "graph_argmax": graph.argmax(dim=-1).tolist(),
        "rows": [],
        "topk_overlap": {},
    }
    for row in range(eager.shape[0]):
        row_diff = diff[row]
        result["rows"].append(
            {
                "row": row,
                "max_abs_diff": float(row_diff.max().item()),
                "mean_abs_diff": float(row_diff.mean().item()),
                "argmax_match": int(eager[row].argmax().item()) == int(graph[row].argmax().item()),
                "eager_argmax": int(eager[row].argmax().item()),
                "graph_argmax": int(graph[row].argmax().item()),
                "top5": _topk_overlap(eager[row], graph[row], 5),
                "top10": _topk_overlap(eager[row], graph[row], 10),
            }
        )
    for k in (1, 5, 10, 50):
        row_scores = [
            _topk_overlap(eager[row], graph[row], k)
            for row in range(eager.shape[0])
        ]
        result["topk_overlap"][str(k)] = {
            "min_ratio": min(item["ratio"] for item in row_scores),
            "avg_ratio": sum(item["ratio"] for item in row_scores) / len(row_scores),
        }
    return result


def _tensor_summary(tensor: torch.Tensor | None, *, limit: int = 16) -> dict[str, Any] | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    flat = detached.flatten()
    preview = flat[:limit].cpu()
    out: dict[str, Any] = {
        "shape": [int(x) for x in detached.shape],
        "dtype": str(detached.dtype),
        "numel": int(flat.numel()),
    }
    if flat.numel() == 0:
        out.update({"sum": 0, "min": None, "max": None, "preview": []})
        return out
    if detached.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.bool):
        values = flat.to(torch.int64)
        out.update(
            {
                "sum": int(values.sum().item()),
                "min": int(values.min().item()),
                "max": int(values.max().item()),
                "preview": [int(x) for x in preview.to(torch.int64).tolist()],
            }
        )
    else:
        values = flat.float()
        out.update(
            {
                "sum": float(values.sum().item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "preview": [float(x) for x in preview.float().tolist()],
            }
        )
    return out


def _capture_selection_trace(llm, *, step: int, use_graph: bool) -> dict[str, Any]:
    sparse_controller = llm.model_runner.sparse_controller
    cache_manager = llm.model_runner.cache_manager
    layers: dict[str, Any] = {}
    for layer_idx, state in sparse_controller.layer_batch_sparse_states.items():
        active = state.active_compressed_indices
        attn_score = state.attn_score
        should_record = (
            active is not None
            or attn_score is not None
            or int(layer_idx) in set(getattr(sparse_controller, "obs_layer_ids", []))
        )
        if not should_record:
            continue
        layers[str(int(layer_idx))] = {
            "active_compressed_indices": _tensor_summary(active),
            "attn_score": _tensor_summary(attn_score, limit=8),
            "context_lens": _tensor_summary(state.context_lens),
            "req_indices": _tensor_summary(state.req_indices),
            "max_context_len": None if state.max_context_len is None else int(state.max_context_len),
        }

    compressed_lens = getattr(cache_manager, "_deltakv_decode_static_compressed_lens", None)
    return {
        "step": int(step),
        "use_graph": bool(use_graph),
        "compressed_lens": _tensor_summary(compressed_lens),
        "layers": layers,
    }


def _run_decode_logits(
    *,
    model_path: str,
    method: str,
    prompt_lens: list[int],
    batch_size: int,
    max_tokens: int,
    hyper_params: dict[str, Any],
    use_graph: bool,
    trace_selection: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    from sparsevllm import LLM, SamplingParams

    if os.getenv("SPARSEVLLM_DEBUG_SKIP_ENGINE_WARMUP", "0") == "1":
        LLM._warmup = lambda self: None

    engine_kwargs = {
        **hyper_params,
        **_sparse_kwargs(method),
        "max_model_len": max(prompt_lens) + max_tokens + 100,
        "max_num_seqs_in_batch": batch_size,
        "max_decoding_seqs": batch_size,
        "decode_cuda_graph": bool(use_graph),
        "decode_cuda_graph_capture_sampling": False,
        "throughput_log_interval_s": 0.0,
    }
    llm = LLM(model_path, **engine_kwargs)
    captured: list[torch.Tensor] = []
    trace: list[dict[str, Any]] = []
    decode_step = 0

    if not use_graph:
        runner = llm.model_runner
        original_run_model = runner.run_model

        def wrapped_run_model(input_ids, positions, is_prefill):
            logits = original_run_model(input_ids, positions, is_prefill)
            if not is_prefill:
                captured.append(logits.detach().float().cpu())
            return logits

        runner.run_model = wrapped_run_model
        if runner.decode_cuda_graph_runner is not None:
            runner.decode_cuda_graph_runner.run_model = wrapped_run_model

    try:
        for round_idx, prompt_len in enumerate(prompt_lens):
            prompt_token_ids = []
            for batch_idx in range(batch_size):
                # Use deterministic non-uniform prompts so sparse selection and
                # graph-state reuse bugs are not hidden by identical rows.
                base = 100 + 997 * round_idx + 131 * batch_idx
                prompt_token_ids.append([base + (pos % 127) for pos in range(prompt_len)])
            sampling_params = [
                SamplingParams(temperature=0.0, top_p=1.0, ignore_eos=True, max_tokens=max_tokens)
                for _ in range(batch_size)
            ]
            for prompt, params in zip(prompt_token_ids, sampling_params):
                llm.add_request(prompt, params)

            while not llm.is_finished():
                _, num_tokens = llm.step()
                if num_tokens < 0:
                    decode_step += 1
                    if use_graph:
                        runner = llm.model_runner.decode_cuda_graph_runner
                        if runner is None:
                            raise RuntimeError("decode_cuda_graph runner was not initialized.")
                        if runner.last_state_key is None or runner.last_real_batch_size is None:
                            raise RuntimeError("No graph logits were captured.")
                        state = runner._graphs[runner.last_state_key]
                        if state.logits is None:
                            raise RuntimeError("Last graph state has no logits.")
                        captured.append(state.logits[:runner.last_real_batch_size].detach().float().cpu())
                    if trace_selection:
                        trace.append(_capture_selection_trace(llm, step=decode_step, use_graph=use_graph))
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    if not captured:
        raise RuntimeError("No decode logits captured. Use max_tokens >= 3.")
    return torch.cat(captured, dim=0), trace


def _run_decode_logits_worker(result_queue, kwargs: dict[str, Any]):
    try:
        logits, trace = _run_decode_logits(**kwargs)
        result_queue.put(("ok", {"logits": logits.numpy(), "trace": trace}))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _run_decode_logits_isolated(**kwargs) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_run_decode_logits_worker, args=(result_queue, kwargs))
    process.start()
    try:
        status, payload = result_queue.get(timeout=900)
    except queue.Empty as exc:
        process.terminate()
        process.join(timeout=30)
        raise TimeoutError("Timed out waiting for decode logits worker.") from exc
    process.join()
    if process.exitcode != 0 or status != "ok":
        raise RuntimeError(f"Decode logits worker failed with exitcode={process.exitcode}:\n{payload}")
    return torch.from_numpy(payload["logits"]), payload["trace"]


def main():
    parser = argparse.ArgumentParser(description="Compare Sparse-VLLM eager decode logits with decode CUDA Graph logits.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--method",
        default="vanilla",
        choices=(
            "vanilla",
            "streamingllm",
            "attention-sink",
            "attention_sink",
            "snapkv",
            "pyramidkv",
            "rkv",
            "skipkv",
            "quest",
            "omnikv",
            "deltakv",
            "deltakv-less-memory",
            "deltakv-less-memory-cudagraph",
        ),
    )
    parser.add_argument("--prompt_len", type=int, default=2048)
    parser.add_argument(
        "--second_prompt_len",
        type=int,
        default=None,
        help="Run a second generate() on the same LLM instance to test graph reuse.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--hyper_params", default="{}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace_selection", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 3:
        raise ValueError("--max_tokens must be >= 3 to force at least one decode step.")

    hyper_params = _load_json_arg(args.hyper_params)
    prompt_lens = [args.prompt_len]
    if args.second_prompt_len is not None:
        prompt_lens.append(args.second_prompt_len)
    eager_logits, eager_trace = _run_decode_logits_isolated(
        model_path=args.model_path,
        method=args.method,
        prompt_lens=prompt_lens,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        hyper_params=hyper_params,
        use_graph=False,
        trace_selection=args.trace_selection,
    )
    graph_logits, graph_trace = _run_decode_logits_isolated(
        model_path=args.model_path,
        method=args.method,
        prompt_lens=prompt_lens,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        hyper_params=hyper_params,
        use_graph=True,
        trace_selection=args.trace_selection,
    )

    output = {
        "status": "success",
        "method": args.method,
        "prompt_lens": prompt_lens,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "hyper_params": hyper_params,
        "comparison": _compare_logits(eager_logits, graph_logits),
    }
    if args.trace_selection:
        output["selection_trace"] = {
            "eager": eager_trace,
            "graph": graph_trace,
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["comparison"], indent=2))


if __name__ == "__main__":
    main()
