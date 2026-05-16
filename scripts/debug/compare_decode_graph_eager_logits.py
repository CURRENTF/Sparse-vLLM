from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
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
        "topk_overlap": {},
    }
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


def _run_decode_logits(
    *,
    model_path: str,
    method: str,
    prompt_len: int,
    batch_size: int,
    max_tokens: int,
    hyper_params: dict[str, Any],
    use_graph: bool,
) -> torch.Tensor:
    from sparsevllm import LLM, SamplingParams

    engine_kwargs = {
        **hyper_params,
        **_sparse_kwargs(method),
        "max_model_len": prompt_len + max_tokens + 100,
        "max_num_seqs_in_batch": batch_size,
        "max_decoding_seqs": batch_size,
        "decode_cuda_graph": bool(use_graph),
        "decode_cuda_graph_capture_sampling": False,
        "throughput_log_interval_s": 0.0,
    }
    llm = LLM(model_path, **engine_kwargs)
    captured: list[torch.Tensor] = []

    if not use_graph:
        runner = llm.model_runner
        original_run_model = runner.run_model

        def wrapped_run_model(input_ids, positions, is_prefill):
            logits = original_run_model(input_ids, positions, is_prefill)
            if not is_prefill and not captured:
                captured.append(logits.detach().float().cpu())
            return logits

        runner.run_model = wrapped_run_model

    try:
        prompt_token_ids = [[100] * prompt_len for _ in range(batch_size)]
        sampling_params = [
            SamplingParams(temperature=0.0, top_p=1.0, ignore_eos=True, max_tokens=max_tokens)
            for _ in range(batch_size)
        ]
        for prompt, params in zip(prompt_token_ids, sampling_params):
            llm.add_request(prompt, params)

        while not llm.is_finished():
            _, num_tokens = llm.step()
            if num_tokens < 0:
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
                break
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    if not captured:
        raise RuntimeError("No decode logits captured. Use max_tokens >= 3.")
    return captured[0]


def _run_decode_logits_worker(result_queue, kwargs: dict[str, Any]):
    try:
        logits = _run_decode_logits(**kwargs)
        result_queue.put(("ok", logits.numpy()))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _run_decode_logits_isolated(**kwargs) -> torch.Tensor:
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
    return torch.from_numpy(payload)


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
            "quest",
            "omnikv",
        ),
    )
    parser.add_argument("--prompt_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--hyper_params", default="{}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.max_tokens < 3:
        raise ValueError("--max_tokens must be >= 3 to force at least one decode step.")

    hyper_params = _load_json_arg(args.hyper_params)
    eager_logits = _run_decode_logits_isolated(
        model_path=args.model_path,
        method=args.method,
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        hyper_params=hyper_params,
        use_graph=False,
    )
    graph_logits = _run_decode_logits_isolated(
        model_path=args.model_path,
        method=args.method,
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        hyper_params=hyper_params,
        use_graph=True,
    )

    output = {
        "status": "success",
        "method": args.method,
        "prompt_len": args.prompt_len,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "hyper_params": hyper_params,
        "comparison": _compare_logits(eager_logits, graph_logits),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["comparison"], indent=2))


if __name__ == "__main__":
    main()
