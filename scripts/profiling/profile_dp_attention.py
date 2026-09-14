"""Diagnostic GLM decode trace; component event runs are not throughput results.

Wrappers live only in this diagnostic process (including spawned workers).
Use --component-events for CUDA-event attribution, and a separate uninstrumented
run under Nsight Systems with --capture-range=nvtx --nvtx-capture=svllm_decode_profile.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

from sparsevllm import LLM, SamplingParams
from sparsevllm.distributed.collective_runtime import ParallelAllReduceHandle
from sparsevllm.distributed.moe_communication import (
    AllGatherReduceScatterMoeCommunication,
    AllReduceMoeCommunication,
)
from sparsevllm.engine import dp_step
from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparsevllm.engine.dp_engine import DPAttentionEngine
from sparsevllm.engine.llm_engine import LLMEngine
from sparsevllm.engine.model_runner import ModelRunner
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.kernels.external.deepep import DeepEPV1Normal
from sparsevllm.models.glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLitePackedExperts,
    Glm4MoeLiteRouter,
    Glm4MoeLiteSparseMoeBlock,
)
from sparsevllm.utils.context import get_context

_EVENT_PAIRS = {}
_SCOPES = []
_OWNERS = []
_LAYER_IDS = []
_HOST_LAST = {}
_HOST_TIMELINE = []


def _wrap(cls, method, label):
    original = getattr(cls, method)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        # The MoE all-reduce wrapper already attributes the nested handle call.
        if label == "attention_all_reduce" and "moe_all_reduce" in _SCOPES:
            return original(self, *args, **kwargs)
        events = os.environ.get("SVLLM_DIAG_COMPONENT_EVENTS") == "1"
        stage = (
            "other_all_reduce"
            if label == "attention_all_reduce" and "attention_inclusive" not in _SCOPES
            else label
        )
        selected_layer = int(os.environ.get("SVLLM_DIAG_EVENT_LAYER", "23"))
        if (
            events
            and not get_context().is_prefill
            and (label == "model_inclusive" or "model_inclusive" in _SCOPES)
            and (
                label == "model_inclusive"
                or selected_layer < 0
                or (_LAYER_IDS and _LAYER_IDS[-1] == selected_layer)
            )
        ):
            # A shared collective handle is called by multiple layers. Include
            # the owner call path so its events are not overwritten per layer.
            key = (*_OWNERS, id(self), stage)
            if key not in _EVENT_PAIRS:
                _EVENT_PAIRS[key] = (
                    torch.cuda.Event(enable_timing=True, external=True),
                    torch.cuda.Event(enable_timing=True, external=True),
                )
            start, end = _EVENT_PAIRS[key]
            start.record()
        else:
            start = end = None
        torch.cuda.nvtx.range_push(label)
        _SCOPES.append(label)
        _OWNERS.append(id(self))
        try:
            return original(self, *args, **kwargs)
        finally:
            _SCOPES.pop()
            _OWNERS.pop()
            torch.cuda.nvtx.range_pop()
            if end is not None:
                end.record()

    setattr(cls, method, wrapped)


def _wrap_host(owner, method, label):
    original = getattr(owner, method)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        start = perf_counter()
        torch.cuda.nvtx.range_push(label)
        try:
            return original(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
            end = perf_counter()
            _HOST_LAST[label] = (end - start) * 1000
            if os.environ.get("SVLLM_DIAG_HOST_TIMELINE") == "1":
                _HOST_TIMELINE.append((label, start, end))

    setattr(owner, method, wrapped)


def _snapshot(self):
    self.platform.synchronize()
    totals = defaultdict(float)
    counts = defaultdict(int)
    for key, (start, end) in _EVENT_PAIRS.items():
        label = key[-1]
        totals[label] += start.elapsed_time(end)
        counts[label] += 1
    result = {
        "rank": self.rank,
        "gpu_ms": dict(totals),
        "calls": dict(counts),
        "host_ms": dict(_HOST_LAST),
        "host_timeline": list(_HOST_TIMELINE),
    }
    _HOST_TIMELINE.clear()
    if self.independent_scheduler or self.world_size == 1:
        return [result]
    records = [None] * self.world_size
    dist.all_gather_object(
        records, result, group=self.parallel_context.world.process_group
    )
    return records


def _engine_snapshot(self):
    return self.model_runner.call("profile_snapshot")


def _dp_snapshot(self):
    records = self._call(range(len(self.ps)), "profile_snapshot")
    return [row for local in records.values() for row in local]


_layer_init = Glm4MoeLiteDecoderLayer.__init__
_layer_forward = Glm4MoeLiteDecoderLayer.forward


def _tag_layer(self, config, layer_idx, *args, **kwargs):
    _layer_init(self, config, layer_idx, *args, **kwargs)
    self._diagnostic_layer_idx = layer_idx


def _layer_scope(self, *args, **kwargs):
    _LAYER_IDS.append(self._diagnostic_layer_idx)
    _OWNERS.append(id(self))
    torch.cuda.nvtx.range_push(f"layer_{self._diagnostic_layer_idx}")
    try:
        return _layer_forward(self, *args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()
        _OWNERS.pop()
        _LAYER_IDS.pop()


Glm4MoeLiteDecoderLayer.__init__ = _tag_layer
Glm4MoeLiteDecoderLayer.forward = _layer_scope


# Spawn re-imports this module, so every worker gets identical diagnostic hooks.
for _cls, _method, _label in [
    (Glm4MoeLiteAttention, "forward", "attention_inclusive"),
    (Glm4MoeLiteRouter, "forward", "router"),
    (Glm4MoeLitePackedExperts, "forward", "routed_mlp"),
    (Glm4MoeLiteSparseMoeBlock, "_shared_chunk", "shared_mlp"),
    (AllGatherReduceScatterMoeCommunication, "dispatch", "all_gather"),
    (AllGatherReduceScatterMoeCommunication, "combine", "reduce_scatter"),
    (AllReduceMoeCommunication, "combine", "moe_all_reduce"),
    (DeepEPV1Normal, "dispatch", "all2all_dispatch"),
    (DeepEPV1Normal, "combine", "all2all_combine"),
    (ParallelAllReduceHandle, "run", "attention_all_reduce"),
    (ModelRunner, "run_model", "model_inclusive"),
]:
    _wrap(_cls, _method, _label)
_wrap_host(dp_step, "coordinate_dp_step", "host_dp_coordination")
_wrap_host(Scheduler, "schedule", "host_schedule")
_wrap_host(DecodeCudaGraphRunner, "run", "host_graph_prepare_submit")
_wrap_host(ModelRunner, "_sample_model_outputs", "host_sampling_including_gpu_wait")
_wrap_host(ModelRunner, "run", "host_runner")
_wrap_host(LLMEngine, "step", "host_worker_step")
_wrap_host(LLMEngine, "worker_routing_load", "host_routing_load")
_wrap_host(LLMEngine, "prefix_cache_routing_snapshot", "host_prefix_snapshot")
_wrap_host(LLMEngine, "chain_cache_routing_snapshot", "host_chain_snapshot")
_wrap_host(DPAttentionEngine, "_receive", "host_frontend_receive")
ModelRunner.profile_snapshot = _snapshot
LLMEngine.profile_snapshot = _engine_snapshot
DPAttentionEngine.profile_snapshot = _dp_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--layout", choices=["dp_ep", "tp_ep", "replicated_ep"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Configured request capacity; defaults to the measured concurrency.",
    )
    parser.add_argument("--parallel-size", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=16)
    instrumentation = parser.add_mutually_exclusive_group()
    instrumentation.add_argument("--component-events", action="store_true")
    instrumentation.add_argument(
        "--host-timeline", action="store_true",
        help="Buffer worker host spans; collect once after the decode window.",
    )
    parser.add_argument(
        "--event-layer",
        type=int,
        default=23,
        help="Sample one layer to limit instrumentation overhead; -1 instruments every layer.",
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--moe-communication", choices=["auto", "agrs", "all2all"], default="auto")
    args = parser.parse_args()
    configured_concurrency = (
        args.concurrency if args.max_concurrency is None else args.max_concurrency
    )
    if configured_concurrency < args.concurrency:
        parser.error("--max-concurrency must be at least --concurrency")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_text(Path(__file__).read_text())
    if args.host_timeline:
        os.environ["SVLLM_DIAG_HOST_TIMELINE"] = "1"
    if args.component_events:
        os.environ["SVLLM_DIAG_COMPONENT_EVENTS"] = "1"
        os.environ["SVLLM_DIAG_EVENT_LAYER"] = str(args.event_layer)
    dp = args.parallel_size if args.layout == "dp_ep" else 1
    local_rows = (args.concurrency + dp - 1) // dp
    local_capacity = (configured_concurrency + dp - 1) // dp
    prefill_budget = 2048 // dp
    # Early requests can decode while later requests still prefill. Reserve
    # enough outputs to keep every request alive through the measured window.
    prefill_allowance = (args.context * args.concurrency + prefill_budget - 1) // prefill_budget
    max_tokens = args.warmup + args.steps + 3 + prefill_allowance
    config = {
        "tensor_parallel_size": args.parallel_size if args.layout == "tp_ep" else 1,
        "expert_parallel_size": args.parallel_size,
        "data_parallel_size": dp,
        "moe_communication_backend": args.moe_communication,
        "decode_graph": not args.eager,
        "decode_graph_capture_sizes": sorted({local_rows, local_capacity}),
        "max_num_seqs_in_batch": local_capacity,
        "max_decoding_seqs": local_capacity,
        "max_model_len": args.context + max_tokens + 1,
        "max_num_batched_tokens": prefill_budget,
        "gpu_memory_utilization": 0.75,
        "throughput_log_interval_s": 0,
    }
    manifest = {
        "status": "running",
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "config": config,
        "command": sys.argv,
        "torch": torch.__version__,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "package_path": str(Path(sys.modules["sparsevllm"].__file__).resolve()),
        "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing": "CUDA events include external graph event nodes; diagnostic only"
        if args.component_events
        else "host step wall time; no extra per-step device synchronization",
    }
    llm = None
    raw = []
    try:
        llm = LLM(args.model, **config)
        manifest["operators"] = llm.operator_runtime_stats()
        generator = torch.Generator().manual_seed(42)
        prompts = [
            torch.randint(100, 10000, (args.context,), generator=generator).tolist()
            for _ in range(args.concurrency)
        ]
        manifest["prompts"] = prompts
        params = SamplingParams(
            temperature=0, ignore_eos=True, max_tokens=max_tokens
        )
        for prompt in prompts:
            llm.add_request(prompt, params)
        # No decode window begins until all requests have published a first token.
        seen = set()
        while len(seen) < args.concurrency:
            llm.step()
            seen.update(seq_id for seq_id, _ in llm.last_step_token_outputs)
        for _ in range(args.warmup):
            llm.step()
        manifest["graph_before"] = llm.debug_sparse_state_summaries(synchronize=True)
        if args.host_timeline:
            llm.profile_snapshot()  # Drain startup records outside the timed window.
            _HOST_TIMELINE.clear()
        torch.cuda.nvtx.range_push("svllm_decode_profile")
        for step in range(args.steps):
            start = perf_counter()
            finished, signed_tokens = llm.step()
            elapsed_ms = (perf_counter() - start) * 1000
            assert not finished and signed_tokens == -args.concurrency, (
                finished,
                signed_tokens,
            )
            row = {
                "step": step,
                "host_ms": elapsed_ms,
                "start_s": start,
                "tokens": llm.last_step_token_outputs,
            }
            if args.component_events:
                row["components"] = llm.profile_snapshot()
            raw.append(row)
        torch.cuda.nvtx.range_pop()
        if args.host_timeline:
            timeline = llm.profile_snapshot()
            (args.output / "host_timeline.json").write_text(json.dumps({
                "workers": timeline,
                "frontend": _HOST_TIMELINE,
                "clock": "process-shared perf_counter seconds",
            }))
        manifest["graph_after"] = llm.debug_sparse_state_summaries(synchronize=True)
        while not llm.is_finished():
            llm.step()
        manifest["status"] = "success"
        summary = {"host_median_ms": statistics.median(row["host_ms"] for row in raw)}
        if args.component_events:
            values = defaultdict(list)
            for row in raw:
                for rank in row["components"]:
                    for label, ms in rank["gpu_ms"].items():
                        values[f"rank{rank['rank']}/{label}"].append(ms)
            summary["component_median_ms"] = {
                key: statistics.median(v) for key, v in values.items()
            }
            host_values = defaultdict(list)
            for row in raw:
                for rank in row["components"]:
                    for label, ms in rank["host_ms"].items():
                        host_values[f"rank{rank['rank']}/{label}"].append(ms)
            summary["host_component_median_ms"] = {
                key: statistics.median(v) for key, v in host_values.items()
            }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        (args.output / "raw_samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in raw)
        )
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
