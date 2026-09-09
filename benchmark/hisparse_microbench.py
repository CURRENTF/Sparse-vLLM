"""Synchronous SGLang HiSparse PR stage adapter for the canonical microbench.

No HTTP/event-window rates: measure scheduling through result processing with
CUDA completion at each step. Overlap is explicitly disabled for this diagnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from time import perf_counter
import traceback


def measured_scheduler_process(*args, **kwargs):
    """Install timing in the spawned worker; do not alter attention algorithms."""
    import torch
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    output = Path(os.environ["HISPARSE_STAGE_STEPS"])
    batch_target = int(os.environ["HISPARSE_STAGE_BATCH"])
    warmup = int(os.environ["HISPARSE_STAGE_WARMUP"])
    original_next = Scheduler.get_next_batch_to_run
    original_run = Scheduler.run_batch
    original_process = Scheduler.process_batch_result
    state = {"full_steps": 0}
    handle = output.open("x", buffering=1)

    def next_batch(self, *a, **kw):
        if self.enable_overlap:
            raise RuntimeError("HiSparse stage timing requires disabled overlap scheduling")
        torch.cuda.synchronize()
        state["start"] = perf_counter()
        return original_next(self, *a, **kw)

    def run_batch(self, batch, *a, **kw):
        pure = batch.forward_mode.is_decode()
        state["row"] = {
            "pure_decode": pure,
            "pure_prefill": batch.forward_mode.is_extend() and not batch.forward_mode.is_mixed(),
            # This upstream version materializes input_ids inside run_batch.
            # Its non-speculative scheduler owns these pre-materialization counts.
            "tokens": len(batch.reqs) if pure else int(batch.extend_num_tokens),
            "active": len(batch.reqs),
            "preempted": sum(req.retraction_count for req in batch.reqs),
        }
        return original_run(self, batch, *a, **kw)

    def process_result(self, batch, result, *a, **kw):
        row = state["row"]
        graph = bool(result.can_run_cuda_graph)
        ret = original_process(self, batch, result, *a, **kw)
        torch.cuda.synchronize()
        elapsed = perf_counter() - state["start"]
        full = row["pure_decode"] and row["active"] == batch_target
        if full:
            state["full_steps"] += 1
        measured = full and state["full_steps"] > warmup
        row.update(elapsed_s=elapsed, measured=measured, cuda_graph=graph,
                   peak_memory_gb=torch.cuda.max_memory_allocated() / 1024**3)
        handle.write(json.dumps(row) + "\n")
        if row["preempted"]:
            raise RuntimeError("Full decode batch capacity exceeded: scheduler preemption")
        if measured and (not graph or row["tokens"] != batch_target):
            raise RuntimeError("HiSparse stage requires graph-only one-token full-batch decode")
        return ret

    Scheduler.get_next_batch_to_run = next_batch
    Scheduler.run_batch = run_batch
    Scheduler.process_batch_result = process_result
    try:
        return run_scheduler_process(*args, **kwargs)
    finally:
        handle.close()


def benchmark_decode_stage(method, length, bs, args, results_dict):
    from benchmark.efficiency.metrics import stage_throughput
    from benchmark.vllm_microbench import _write_jsonl

    row = {"engine": "hisparse", "backend_label": "hisparse-quest-pr-series",
           "method": method, "length": length, "batch_size": bs,
           "output_len": args.output_len, "status": "FAILED"}
    engine = None
    outputs = []
    case = Path(args.output_dir) / f"{method}-{length}-{bs}"
    case.mkdir(parents=True, exist_ok=True)
    try:
        hp = args.hyper_params_dict
        if method != "quest" or args.backend_label != "hisparse-quest-pr-series":
            raise ValueError("HiSparse stage adapter requires explicitly labelled QuEST PR series")
        if (not args.synchronize_step_timing or not args.require_full_decode_batch
                or args.admission_wave_size or args.wave_decode_gap_steps
                or args.max_decode_steps_after_full or args.require_prefix_cache_hit):
            raise ValueError("HiSparse stage requires untruncated synchronized full batch, no waves/prefix cache")
        allowed_hp = {"tensor_parallel_size", "expert_parallel_size", "data_parallel_size",
                      "gpu_memory_utilization", "max_num_batched_tokens", "decode_graph",
                      "engine_prefill_chunk_size", "enable_prefix_caching"}
        if set(hp) - allowed_hp:
            raise ValueError(f"Unmapped HiSparse stage parameters: {sorted(set(hp) - allowed_hp)}")
        if any(hp.get(key, 1) != 1 for key in ("tensor_parallel_size", "expert_parallel_size", "data_parallel_size")):
            raise ValueError("HiSparse stage adapter currently requires TP1/EP1/DP1")
        if hp.get("enable_prefix_caching", False) or not hp.get("decode_graph", True):
            raise ValueError("HiSparse stage requires Graph and disabled prefix cache")
        if length <= 0 or bs <= 0 or args.output_len <= 1 or args.decode_warmup_steps_after_full < 0:
            raise ValueError("Invalid HiSparse workload dimensions/warmup")
        import sglang
        from sglang.srt.entrypoints.engine import Engine

        class StageEngine(Engine):
            run_scheduler_process_func = staticmethod(measured_scheduler_process)

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        config = dict(model_path=args.model_path, tp_size=1, random_seed=42,
            # SGLang reserves two positions internally when admitting a request.
            context_length=args.max_model_len_override or length + args.output_len + 2,
            max_running_requests=bs, mem_fraction_static=hp.get("gpu_memory_utilization", .9),
            chunked_prefill_size=hp.get("engine_prefill_chunk_size", 8192),
            max_prefill_tokens=hp.get("max_num_batched_tokens", 8192),
            disable_radix_cache=True, disable_overlap_schedule=True,
            skip_tokenizer_init=True, enable_hisparse=True, attention_backend="fa3", page_size=16,
            cuda_graph_config={"prefill": {"backend": "disabled"}, "decode": {"backend": "breakable", "bs": [bs]}},
            port=port)
        extra = args.engine_kwargs_dict
        if set(extra) - {"hisparse_config", "dtype"}:
            raise ValueError("HiSparse extra options may only specify hisparse_config and dtype")
        if extra.get("hisparse_config", {}).get("algorithm") != "quest":
            raise ValueError("HiSparse configuration must enable QuEST")
        config.update(extra)
        config["hisparse_config"] = json.dumps(config["hisparse_config"])
        row.update(engine_hyper_params=config, sglang_version=sglang.__version__)
        row["package_source"] = str(Path(sglang.__file__).resolve())
        os.environ.update(HISPARSE_STAGE_STEPS=str(case / "steps.jsonl"),
                          HISPARSE_STAGE_BATCH=str(bs), HISPARSE_STAGE_WARMUP=str(args.decode_warmup_steps_after_full))
        from sglang.srt.server_args import parse_cuda_graph_config_arg
        runtime_config = {**config, "cuda_graph_config": parse_cuda_graph_config_arg(json.dumps(config["cuda_graph_config"]))}
        engine = StageEngine(**runtime_config)
        generated = engine.generate(input_ids=[[100] * length for _ in range(bs)],
            sampling_params={"temperature": args.temperature, "top_p": args.top_p,
                             "max_new_tokens": args.output_len, "ignore_eos": True})
        if isinstance(generated, dict):
            generated = [generated]
        for index, item in enumerate(generated):
            tokens = item["output_ids"]
            outputs.append({"request_id": str(index), "token_ids": tokens,
                            "status": "success" if len(tokens) == args.output_len else "model_failed",
                            "meta_info": item["meta_info"]})
        engine.shutdown(); engine = None
        steps = [json.loads(line) for line in (case / "steps.jsonl").read_text().splitlines()]
        peak = max((s["active"] for s in steps if s["pure_decode"]), default=0)
        if len(outputs) != bs or any(x["status"] != "success" for x in outputs):
            raise RuntimeError("Incomplete output: every request must finish the requested output length")
        if peak != bs:
            raise RuntimeError("Full decode batch capacity exceeded: incomplete full residency")
        if any(s["preempted"] for s in steps):
            raise RuntimeError("Full decode batch capacity exceeded: scheduler preemption")
        measured = [s for s in steps if s["measured"]]
        if not measured or any(not s["pure_decode"] or not s["cuda_graph"] or s["tokens"] != bs for s in measured):
            raise RuntimeError("Invalid synchronized full-batch Graph-only stage window")
        elapsed = sum(s["elapsed_s"] for s in measured)
        tokens = sum(s["tokens"] for s in measured)
        prefill = [s for s in steps if s["pure_prefill"]]
        rate = stage_throughput(tokens, elapsed)
        prefill_rate = stage_throughput(sum(s["tokens"] for s in prefill), sum(s["elapsed_s"] for s in prefill))
        if rate is None or prefill_rate is None:
            raise RuntimeError("Missing independently timed stage work")
        row.update(status="SUCCESS", stage_metrics_status="success",
            stage_timing_scope="sum_synchronized_scheduler_steps",
            synchronization_boundary="Scheduler.get_next_batch_to_run through process_batch_result plus CUDA synchronization; excludes IPC receive and timing-log writes",
            measurement_scope="full_batch_pure_decode_steps", require_full_decode_batch=True,
            synchronize_step_timing=True, decode_stage_tokens=tokens, decode_stage_elapsed_s=elapsed,
            decode_stage_throughput_tps=rate, actual_decode_peak=peak, completed_requests=len(outputs),
            full_admission_reached=True, scheduler_preemptions=0, measured_decode_steps_after_full=len(measured),
            decode_warmup_steps_after_full=args.decode_warmup_steps_after_full,
            decode_tp=rate, prefill_tp=prefill_rate, avg_bs=bs, mem=max(s["peak_memory_gb"] for s in steps),
            ttft=sum(s["elapsed_s"] for s in prefill), itl=elapsed/len(measured)*1000,
            request_metrics_status="not_measured")
    except Exception as error:
        row.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        _write_jsonl(case / "raw_outputs.jsonl", outputs)
        results_dict[(method, length, bs)] = row
        if engine is not None:
            engine.shutdown()
