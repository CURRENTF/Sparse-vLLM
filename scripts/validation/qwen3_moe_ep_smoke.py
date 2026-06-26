from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sparsevllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Qwen3-MoE Sparse-vLLM smoke.")
    parser.add_argument("--model", required=True, help="Local Qwen3-MoE checkpoint directory.")
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--engine-prefill-chunk-size", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    result = {
        "status": "model_failed",
        "model": args.model,
        "expert_parallel_size": int(args.expert_parallel_size),
        "prompt": args.prompt,
        "max_tokens": int(args.max_tokens),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "started_at": started_at,
    }
    llm = None
    try:
        llm = LLM(
            args.model,
            tensor_parallel_size=1,
            expert_parallel_size=int(args.expert_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            engine_prefill_chunk_size=int(args.engine_prefill_chunk_size),
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            max_num_seqs_in_batch=1,
            max_decoding_seqs=1,
            decode_graph=False,
            enforce_eager=True,
        )
        outputs = llm.generate(
            prompts=[args.prompt],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=int(args.max_tokens)),
            use_tqdm=False,
        )
        result.update(
            {
                "status": "success",
                "token_ids": outputs[0]["token_ids"],
                "text": outputs[0]["text"],
            }
        )
        print("SMOKE_OK")
        print("token_ids", outputs[0]["token_ids"])
        print("text", repr(outputs[0]["text"]))
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
