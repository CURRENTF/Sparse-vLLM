import os
import time
import torch
import argparse
import multiprocessing as mp
import traceback
import json
from pathlib import Path
from typing import Any
from time import perf_counter


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) # GB


def _load_json_arg(value: str) -> dict[str, Any]:
    """Load a JSON object from a CLI arg.

    Supports:
      - Inline JSON: '{"gpu_memory_utilization": 0.9}'
    """
    if value is None:
        return {}
    value = str(value).strip()

    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"Invalid JSON for --hyper_params: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("--hyper_params must be a JSON object (dict).")
    return parsed


def _build_engine_hyper_params(args) -> dict[str, Any]:
    # Keep benchmark defaults stable (do not rely on sparsevllm.Config defaults).
    hyper_params: dict[str, Any] = {
        "enforce_eager": True,
        "gpu_memory_utilization": 0.8,
        "chunk_prefill_size": 4096,
        "tensor_parallel_size": 1,
    }

    hyper_params.update(_load_json_arg(args.hyper_params))

    return hyper_params


def benchmark_task(method, length, bs, args, results_dict):
    # 为每个子进程重置显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    print(f"\n>>> Starting: {method.upper()} | Context: {length} | Batch: {bs}...")
    
    base_hyper_params = args.hyper_params_dict
    sparse_kwargs: dict[str, Any] = {"vllm_sparse_method": ""}
    if method == "vanilla":
        sparse_kwargs["vllm_sparse_method"] = ""
    elif method in ("snapkv", "omnikv", "deltakv"):
        sparse_kwargs["vllm_sparse_method"] = method
    elif "deltakv" in method:
        sparse_kwargs["vllm_sparse_method"] = method
    
    llm = None
    try:
        m_len = length + args.output_len + 100
        # Note: max_model_len are derived from (length, bs, output_len, chunk_prefill_size).
        # They can be passed in --hyper_params, but will be overwritten here to keep the benchmark consistent.
        hyper_params = dict(base_hyper_params)
        hyper_params.pop("max_model_len", None)
        chunk_prefill_size = int(hyper_params.get("chunk_prefill_size", 4096))
        
        from sparsevllm import LLM, SamplingParams
        engine_kwargs = {
            **hyper_params,
            "max_model_len": m_len,
            **sparse_kwargs,
        }
        llm = LLM(args.model_path, **engine_kwargs)

        prompt_token_ids = [[100] * length for _ in range(bs)]
        sampling_params = [SamplingParams(temperature=0.1, ignore_eos=True, max_tokens=args.output_len) for _ in range(bs)]

        # --- 关键修改：重置并开始正式测量 ---
        from sparsevllm.utils.profiler import profiler
        profiler.reset()
        
        torch.cuda.synchronize()

        prefill_tokens = 0
        decode_tokens = 0
        prefill_times = []
        decode_times = []
        ttft = None
        
        t_start = perf_counter()
        decode_started = False

        # Manually run the generation loop to get detailed stats
        for p, sp in zip(prompt_token_ids, sampling_params):
            llm.add_request(p, sp)

        has_queued = False
        zero_steps = 0
        while not llm.is_finished():
            step_start = perf_counter()
            finished_outputs, num_tokens = llm.step()
            step_dt = perf_counter() - step_start
            
            if num_tokens > 0:
                prefill_tokens += num_tokens
                prefill_times.append(step_dt)
                if decode_started:
                    has_queued = True
                zero_steps = 0
                # In this engine, the first completion token is sampled during the *last* prefill
                # step of each sequence (Scheduler.postprocess appends it in the prefill branch).
                # So TTFT should be captured on a prefill step, not on the first decode step.
                if ttft is None and (llm.scheduler.decoding or any(tids for _, tids in finished_outputs)):
                    ttft = perf_counter() - t_start
            elif num_tokens < 0:
                # print(f'one decode step ... {perf_counter() - last_time}')
                decode_started = True
                decode_times.append(step_dt)
                decode_tokens += (-num_tokens)
                zero_steps = 0
                # Fallback: if output_len==0/1 or internal behavior changes, ensure TTFT is set.
                if ttft is None:
                    ttft = perf_counter() - t_start
            else:
                zero_steps += 1
                if zero_steps >= 50:
                    raise RuntimeError("llm.step() returned 0 tokens repeatedly; scheduler may be stuck.")

        print(f'@@@ {decode_tokens=}')
                
        torch.cuda.synchronize()
        t_end = perf_counter()
        
        duration = t_end - t_start
        peak_mem = get_peak_memory()

        ttft = float(ttft or 0.0)
        prefill_s = sum(prefill_times)
        decode_s = sum(decode_times)

        print(f'[debug] {prefill_tokens=} {prefill_s=} {ttft=} {decode_tokens=} {decode_s=} {has_queued=}')
        prefill_tp = prefill_tokens / prefill_s if prefill_s > 0 else 0
        decode_tp = decode_tokens / decode_s if decode_s > 0 else 0
        # ITL (Inter-token Latency) 是用户感知的生成速度：总解码时间 / 单序列平均生成的 token 数
        avg_itl = (decode_s / (decode_tokens / bs) * 1000) if decode_tokens > 0 else 0
        avg_active_bs = decode_tokens / len(decode_times) if decode_times else 0
        
        print(f"[{method.upper()}] TTFT: {ttft:.2f}s | Prefill: {prefill_tp:.2f} tok/s | Decode: {decode_tp:.2f} tok/s | ITL: {avg_itl:.2f}ms | AvgBS: {avg_active_bs:.1f} | Mem: {peak_mem:.2f} GB")
        
        results_dict[(method, length, bs)] = {
            "prefill_tp": prefill_tp,
            "decode_tp": decode_tp,
            "ttft": ttft,
            "itl": avg_itl,
            "avg_bs": avg_active_bs,
            "mem": peak_mem,
            "has_queued": has_queued,
            "status": "SUCCESS"
        }

    except Exception as e:
        print(f"Error at {method}/{length}/{bs}: {e}")
        traceback.print_exc()
        results_dict[(method, length, bs)] = {"status": "FAILED"}
    finally:
        if llm is not None and hasattr(llm, "exit"):
            llm.exit()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Professional benchmark for sparsevllm.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lengths", type=str, default="16000,32000,64000", help="Context lengths to test")
    parser.add_argument("--batch_sizes", type=str, default="4", help="Batch sizes to test")
    parser.add_argument(
        "--methods",
        type=str,
        default="vanilla,snapkv,omnikv",
        help="Methods to test (vanilla, snapkv, omnikv, deltakv, deltakv-triton, deltakv-triton-v2, deltakv-triton-v3, deltakv-triton-v3-offload, deltakv-triton-v3-cuda-offload)",
    )
    parser.add_argument("--output_len", type=int, default=512, help="Output tokens per request")
    parser.add_argument(
        "--hyper_params",
        type=str,
        default="{}",
        help=(
            "LLMEngine/Config hyper-params as JSON (string or @file.json). "
            'Example: \'{"gpu_memory_utilization":0.9,"chunk_prefill_size":4096,"tensor_parallel_size":1,"num_top_tokens":2048}\''
        ),
    )
    # Deprecated (kept for backward compatibility; prefer --hyper_params)
    parser.add_argument("--gpu_util", type=float, default=None, help="[DEPRECATED] use --hyper_params.gpu_memory_utilization")
    parser.add_argument("--chunk_size", type=int, default=None, help="[DEPRECATED] use --hyper_params.chunk_prefill_size")
    parser.add_argument("--tp", type=int, default=None, help="[DEPRECATED] use --hyper_params.tensor_parallel_size")
    parser.add_argument("--enforce_eager", type=str2bool, default=None, help="[DEPRECATED] use --hyper_params.enforce_eager")
    
    args = parser.parse_args()
    try:
        args.hyper_params_dict = _build_engine_hyper_params(args)
    except ValueError as e:
        parser.error(str(e))
    
    test_lengths = [int(x) for x in args.lengths.split(",")]
    test_methods = args.methods.split(",")
    test_batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    manager = mp.Manager()
    results_dict = manager.dict()

    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                p = mp.Process(target=benchmark_task, args=(method, length, bs, args, results_dict))
                p.start()
                p.join()

    # 打印最终报表
    print(f"\n\n{'='*140}")
    print(f"{ 'Method':<12} {'Len':<8} {'BS':<4} {'TTFT(s)':<10} {'PreTP':<12} {'DecTP':<12} {'ITL(ms)':<10} {'AvgBS':<8} {'Mem(GB)':<10} {'Speedup'}")
    print("-" * 140)
    
    # 获取 Vanilla 作为基准计算加速比 (按 length 和 BS 匹配)
    vanilla_stats = {}
    for length in test_lengths:
        for bs in test_batch_sizes:
            v_res = results_dict.get(("vanilla", length, bs))
            if v_res and v_res["status"] == "SUCCESS":
                vanilla_stats[(length, bs)] = v_res["decode_tp"]

    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                res = results_dict.get((method, length, bs))
                if not res or res["status"] in ["FAILED", "OOM"]:
                    status_str = res["status"] if res else "UNKNOWN"
                    print(f"{method:<12} {length:<8} {bs:<4} {status_str:<10} {'-':<12} {'-':<12} {'-':<10} {'-':<8} {'-':<10} {'-'}")
                    continue
                
                ttft = res["ttft"]
                pre_tp = res["prefill_tp"]
                dec_tp = res["decode_tp"]
                itl = res["itl"]
                avg_bs = res["avg_bs"]
                mem = res["mem"]
                has_queued = res.get("has_queued", False)
                
                bs_str = f"{bs}*" if has_queued else f"{bs}"
                
                speedup = 1.0
                if (length, bs) in vanilla_stats:
                    speedup = dec_tp / vanilla_stats[(length, bs)]
                
                speedup_str = f"{speedup:.2f}x"
                print(f"{method:<12} {length:<8} {bs_str:<4} {ttft:<10.2f} {pre_tp:<12.1f} {dec_tp:<12.1f} {itl:<10.2f} {avg_bs:<8.1f} {mem:<10.2f} {speedup_str}")
    print(f"{ '='*140}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
