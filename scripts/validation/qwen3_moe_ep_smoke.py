from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sparsevllm import LLM, SamplingParams


def jsonable_capture_sizes(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Qwen3-MoE Sparse-vLLM smoke.")
    parser.add_argument("--model", required=True, help="Local Qwen3-MoE checkpoint directory.")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-backend", choices=["torch", "deepep_v2"], default="torch")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to run. Can be passed multiple times.",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--engine-prefill-chunk-size", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs-in-batch", type=int, default=None)
    parser.add_argument("--max-decoding-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--use-compression", action="store_true")
    parser.add_argument("--decode-cuda-graph", action="store_true")
    parser.add_argument("--decode-cuda-graph-capture-sizes", default="auto")
    parser.add_argument("--distributed-master-port", type=int, default=None)
    parser.add_argument("--throughput-log-interval-s", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    prompts = args.prompt or ["The capital of France is"]
    max_num_seqs = int(args.max_num_seqs_in_batch or len(prompts))
    max_decoding_seqs = int(args.max_decoding_seqs or len(prompts))
    result = {
        "status": "model_failed",
        "model": args.model,
        "data_parallel_size": int(args.data_parallel_size),
        "expert_parallel_size": int(args.expert_parallel_size),
        "expert_parallel_backend": str(args.expert_parallel_backend),
        "prompts": prompts,
        "max_tokens": int(args.max_tokens),
        "decode_cuda_graph": bool(args.decode_cuda_graph),
        "use_compression": bool(args.use_compression),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "started_at": started_at,
    }
    llm = None
    try:
        llm = LLM(
            args.model,
            tensor_parallel_size=1,
            data_parallel_size=int(args.data_parallel_size),
            expert_parallel_size=int(args.expert_parallel_size),
            expert_parallel_backend=str(args.expert_parallel_backend),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            engine_prefill_chunk_size=int(args.engine_prefill_chunk_size),
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            max_num_seqs_in_batch=max_num_seqs,
            max_decoding_seqs=max_decoding_seqs,
            decode_cuda_graph=bool(args.decode_cuda_graph),
            decode_cuda_graph_capture_sizes=args.decode_cuda_graph_capture_sizes,
            enforce_eager=not bool(args.decode_cuda_graph),
            use_compression=bool(args.use_compression),
            distributed_master_port=args.distributed_master_port,
            throughput_log_interval_s=float(args.throughput_log_interval_s),
        )
        result["resolved_config"] = {
            "data_parallel_size": int(llm.config.data_parallel_size),
            "expert_parallel_size": int(llm.config.expert_parallel_size),
            "expert_parallel_backend": str(llm.config.expert_parallel_backend),
            "parallel_world_size": int(llm.config.parallel_world_size),
            "decode_cuda_graph": bool(llm.config.decode_cuda_graph),
            "decode_cuda_graph_capture_sizes": jsonable_capture_sizes(
                llm.config.decode_cuda_graph_capture_sizes
            ),
            "enforce_eager": bool(llm.config.enforce_eager),
            "use_compression": bool(llm.config.use_compression),
            "distributed_master_port": llm.config.distributed_master_port,
        }
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=int(args.max_tokens)),
            use_tqdm=False,
        )
        result.update(
            {
                "status": "success",
                "outputs": outputs,
            }
        )
        print("SMOKE_OK")
        for index, output in enumerate(outputs):
            print(f"output[{index}].token_ids", output["token_ids"])
            print(f"output[{index}].text", repr(output["text"]))
    except BaseException as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        if llm is not None:
            llm.exit()
        result["finished_at"] = time.time()
        result["elapsed_s"] = result["finished_at"] - started_at
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
