#!/usr/bin/env python3
"""Freeze and execute MiniSWE campaigns through the canonical adapter.

No engine imports in planning/collection; server and Docker driver may run on
different hosts. All subprocesses use argv arrays and explicit environment bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shlex
import socket
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def capture(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True, timeout=30).strip()


def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=15) as response:
        return json.load(response)


def prepare(args):
    setting = read(args.setting)
    # Each concurrency has an immutable root; methods are never silently retuned.
    concurrency = args.concurrency
    if not 1 <= concurrency <= 256:
        raise ValueError("Concurrency must be between 1 and 256")
    engine_concurrency = (
        concurrency
        if args.engine_concurrency is None
        else int(args.engine_concurrency)
    )
    if not 1 <= engine_concurrency <= 256:
        raise ValueError("Engine concurrency must be between 1 and 256")
    setting["agent"]["mini_workers"] = concurrency
    for key in ("max_num_seqs_in_batch", "max_decoding_seqs"):
        setting["engine"][key] = engine_concurrency
    resident_concurrency = (
        engine_concurrency
        if args.engine_resident_concurrency is None
        else int(args.engine_resident_concurrency)
    )
    if not engine_concurrency <= resident_concurrency <= 256:
        raise ValueError(
            "Engine resident concurrency must be between engine concurrency and 256"
        )
    setting["engine"]["max_num_seqs_in_gpu"] = resident_concurrency
    buckets = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    setting["engine"]["decode_graph_capture_sizes"] = sorted(
        {b for b in buckets if b <= engine_concurrency} | {engine_concurrency}
    )
    setting["slow_baseline_policy"]["pilot_instances"] = max(64, concurrency)
    root = args.root.resolve()
    if root == HERE or HERE in root.parents:
        raise ValueError("Place artifacts outside the recipe source directory")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "setting.json", setting)
    write(root / "source.json", {
        "commit": capture(["git", "rev-parse", "HEAD"], REPO),
        "dirty": capture(["git", "status", "--porcelain=v1"], REPO),
        "protocol": setting["protocol"], "prepared_at": time.time(),
    })
    for method, overrides in setting["methods"].items():
        write(root / method / "engine.json", {**setting["engine"], **overrides})
    print(f"Prepared {root}; server and benchmark commands are in README.md")


def environment(python):
    python = str(Path(python).resolve(strict=True))
    env = os.environ.copy()
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")))
    env["OPENAI_API_KEY"] = "local-sparsevllm"
    return python, env


def stop(process):
    # The process was started in a new session by this script; never pkill by name.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_start_ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw[raw.rfind(")") + 2:].split()[19])


def run_logged(argv, env, directory, name, timeout, process_record=None):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / f"{name}.invocation.json", {"argv": argv, "started": time.time()})
    if process_record is not None and process_record.exists():
        raise FileExistsError(f"Refusing to replace process identity record: {process_record}")
    started = time.monotonic()
    result = {"status": "running"}
    with (directory / f"{name}.log").open("x") as log:
        process = subprocess.Popen(argv, env=env, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if process_record is not None:
                write(process_record, {
                    "pid": process.pid,
                    "process_group": os.getpgid(process.pid),
                    "start_ticks": process_start_ticks(process.pid),
                    "hostname": socket.gethostname(),
                    "boot_id": boot_id(),
                    "command": shlex.join(argv),
                })
        except BaseException:
            stop(process)
            raise
        try:
            code = process.wait(timeout=timeout)
            stop_record = directory / f"{name}_stop.json"
            requested = stop_record.exists() and read(stop_record).get("status") == "stop_requested"
            result = {
                "status": "success" if code == 0 or requested else "failed",
                "exit_code": code,
            }
            if requested:
                result["stop_reason"] = "requested_after_benchmark"
        except subprocess.TimeoutExpired:
            result = {"status": "timeout", "timeout_seconds": timeout}
        except BaseException:
            result = {"status": "aborted"}
            raise
        finally:
            stop(process)
            result["elapsed_seconds"] = time.monotonic() - started
            write(directory / f"{name}.result.json", result)
    if result["status"] != "success":
        raise RuntimeError(f"{name}: {result}; inspect {directory}")


def idle_pair(gpus):
    ids = gpus.split(",")
    if len(ids) != 2 or len(set(ids)) != 2 or not all(x.isdigit() for x in ids):
        raise ValueError("--gpus must name exactly two distinct numeric GPU indices")
    raw = capture(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    rows = {x[0]: x for line in raw.splitlines() if (x := [v.strip() for v in line.split(",")])}
    apps = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    for gpu in ids:
        row = rows[gpu]
        if row[1] in apps or int(row[2]) > 128 or int(row[3]) != 0:
            raise RuntimeError(f"GPU {gpu} is occupied; wait, do not share or kill its processes")
    return {"gpus": raw, "processes": apps}


def serve(args):
    directory = args.root / args.method / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    python, env = environment(args.python)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["LOG_LEVEL"] = "INFO"
    model = args.model.resolve(strict=True)
    config = read(args.root / args.method / "engine.json")
    advertised = f"glm47-{args.method}"
    # Bind test avoids inadvertently attaching the driver to an older service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    hardware = idle_pair(args.gpus)
    command = [python, "-m", "sparsevllm.entrypoints.openai.api_server", "--model", str(model),
               "--served-model-name", advertised, "--host", "127.0.0.1", "--port", str(args.port),
               "--engine-kwargs", str((args.root / args.method / "engine.json").resolve()),
               "--request-log-dir", str((directory / "server_requests").resolve())]
    write(directory / "server_manifest.json", {
        "command": shlex.join(command), "model_path": str(model), "served_model_name": advertised,
        "cuda_visible_devices": args.gpus, "server_port": args.port, "engine_kwargs": config,
        "git_commit": capture(["git", "rev-parse", "HEAD"], REPO),
        "git_dirty": capture(["git", "status", "--porcelain=v1"], REPO),
        "hardware": hardware, "model_config": read(model / "config.json"),
    })
    # Server lifetime is explicit, one fresh process per method AND phase.
    run_logged(
        command,
        env,
        directory,
        "server",
        args.timeout,
        process_record=directory / "server_process.json",
    )


def stop_server(args):
    directory = args.root / args.method / args.phase
    result_path = directory / "server_stop.json"
    if result_path.exists():
        print(json.dumps(read(result_path), indent=2))
        return
    record = read(directory / "server_process.json")
    if record["hostname"] != socket.gethostname() or record["boot_id"] != boot_id():
        raise RuntimeError("Server process record belongs to another host or boot")
    pid = int(record["pid"])
    try:
        current_start = process_start_ticks(pid)
    except FileNotFoundError:
        write(result_path, {"status": "already_stopped", "pid": pid, "stopped_at": time.time()})
        print(json.dumps(read(result_path), indent=2))
        return
    if current_start != int(record["start_ticks"]) or os.getpgid(pid) != int(record["process_group"]):
        raise RuntimeError("Server PID identity changed; refusing to signal it")
    if int(record["process_group"]) != pid:
        raise RuntimeError("Recorded server is not the leader of its private process group")
    write(result_path, {"status": "stop_requested", "pid": pid, "stopped_at": time.time()})
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            process_start_ticks(pid)
        except FileNotFoundError:
            break
        time.sleep(.1)
    else:
        os.killpg(pid, signal.SIGKILL)
    print(json.dumps(read(result_path), indent=2))


def gate(root):
    policy = read(root / "setting.json")["slow_baseline_policy"]
    baseline = read(root / "snapkv-no-chain/pilot/generate.result.json")
    reference = read(root / "snapkv-chain/pilot/generate.result.json")
    if reference["status"] != "success":
        raise ValueError("The matched SnapKV Chain Cache pilot must complete first")
    # This is a cost gate, never an estimate of task quality or a matched-trace speedup.
    estimate = baseline["elapsed_seconds"] * 300 / policy["pilot_instances"]
    ratio = baseline["elapsed_seconds"] / reference["elapsed_seconds"]
    if baseline["status"] not in {"success", "timeout"}:
        raise ValueError("A failed pilot is an implementation failure, not a slow baseline")
    skip = baseline["status"] == "timeout" or estimate > policy["full_generation_limit_seconds"] or ratio > policy["slowdown_ratio"]
    return {"status": "skipped_by_policy" if skip else "eligible", "pilot_status": baseline["status"],
            "projected_full_generation_seconds": estimate, "pilot_wall_time_ratio": ratio,
            "policy": policy, "scope": "closed_loop_cost_gate_only"}


def benchmark(args):
    setting = read(args.root / "setting.json")
    directory = args.root / args.method / args.phase
    if args.method == "snapkv-no-chain" and args.phase == "full" and args.stage in {"all", "generate"}:
        decision = gate(args.root)
        write(directory / "slow_baseline_decision.json", decision)
        if decision["status"] == "skipped_by_policy":
            print(json.dumps(decision, indent=2))
            return
    python, env = environment(args.python)
    server_manifest = args.server_manifest or directory / "server_manifest.json"
    frozen = read(args.root / args.method / "engine.json")
    if read(server_manifest)["engine_kwargs"] != frozen:
        raise ValueError("Server manifest and frozen method settings differ")
    if args.stage in {"prepare", "generate", "all"}:
        get(args.api_base.removesuffix("/v1") + "/readyz")
    agent = setting["agent"].copy()
    if args.phase == "smoke":
        agent.update(mini_workers=1, eval_workers=1, batch_size=1)
    command = [python, "-m", "benchmark.swe_bench_lite.run", "--stage", args.stage,
               "--mini-command", shlex.join([python, str(HERE / "timed_mini.py")]),
               "--swe-bench-dir", str(args.swe_bench_dir.resolve(strict=True)),
               "--run-dir", str((directory / "benchmark").resolve()),
               "--model", f"openai/glm47-{args.method}", "--served-model-name", f"glm47-{args.method}",
               "--api-base", args.api_base, "--server-manifest", str(server_manifest.resolve()),
               "--chain-cache" if frozen.get("prefix_cache_mode") == "chain" else "--no-chain-cache"]
    for key, value in agent.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(option if value else "--no-" + key.replace("_", "-"))
        else:
            command.extend([option, str(value)])
    if args.phase != "full":
        pilot_count = setting["slow_baseline_policy"]["pilot_instances"]
        command.extend(["--slice", "0:1" if args.phase == "smoke" else f"0:{pilot_count}"])
    # No hidden downloading, concurrency reduction, cache fallback, or retries.
    run_logged(command, env, directory, args.stage, args.timeout)


def collect(args):
    directory = args.root / args.method / args.phase
    summary = read(directory / "benchmark/final_summary.json")
    setting = read(args.root / "setting.json")
    expected = {"smoke": 1, "pilot": setting["slow_baseline_policy"]["pilot_instances"], "full": 300}[args.phase]
    if summary["total_instances"] != expected:
        raise ValueError("Official summary has the wrong instance count")
    selected = set((directory / "benchmark/instances.txt").read_text().split())
    for filename in ("generation_results.jsonl", "per_sample_results.jsonl"):
        rows = [json.loads(line) for line in (directory / "benchmark" / filename).read_text().splitlines()]
        ids = [row["instance_id"] for row in rows]
        if len(ids) != expected or len(set(ids)) != expected or set(ids) != selected:
            raise ValueError(f"Missing, duplicate, or unexpected samples in {filename}")
    per_sample = {row["instance_id"]: row for row in rows}
    timings = sorted((directory / "benchmark/batches").glob("batch_*/*/*.wall_time.json"))
    task_rows = []
    for path in timings:
        row = read(path)
        if not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0:
            raise ValueError(f"Invalid task duration in {path}")
        outcome = per_sample[row["instance_id"]]
        task_rows.append({**row, "status": outcome["status"], "resolved": outcome["resolved"],
                          "generation_exit_status": outcome.get("generation_exit_status"),
                          "model_stats": outcome.get("model_stats")})
    if len(task_rows) != expected or {r["instance_id"] for r in task_rows} != selected:
        raise ValueError("Missing or duplicate task timings; instrumentation was not active or run was interrupted")
    with (directory / "task_samples.jsonl").open("x") as stream:
        for row in task_rows:
            stream.write(json.dumps(row) + "\n")
    # Shared percentile implementation; do not create another quantile formula.
    import sys
    sys.path.insert(0, str(REPO))
    from benchmark.efficiency.metrics import percentile
    task_summary = {}
    for label, subset in (("all_attempts", task_rows), ("resolved", [r for r in task_rows if r["resolved"]])):
        durations = [r["elapsed_s"] for r in subset]
        task_summary[label] = {"count": len(subset),
                               "p50_s": percentile(durations, .5) if durations else None,
                               "p95_s": percentile(durations, .95) if durations else None}
    generation = read(directory / "generate.result.json")
    evaluation = read(directory / "evaluate.result.json")
    logs = sorted((directory / "server_requests").glob("*.json"))
    if not logs:
        raise ValueError("No server request logs; copy them from the GPU host before collecting")
    totals = dict(requests=0, prompt_tokens=0, completion_tokens=0, reused_tokens=0,
                  uncached_prompt_tokens=0, summed_request_seconds=0.0)
    with (directory / "request_samples.jsonl").open("x") as stream:
        for path in logs:
            record = read(path)
            if record["status"] != "success":
                raise ValueError(f"Non-success request log: {path}; inspect the failure before aggregation")
            usage = record["response"]["usage"]
            prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
            reused = usage["prompt_tokens_details"]["cached_tokens"]
            if not 0 <= reused <= prompt or output < 0 or record["elapsed_s"] < 0:
                raise ValueError(f"Invalid usage or timing: {path}")
            row = {"status": "success", "request_id": record["request_id"],
                   "elapsed_s": record["elapsed_s"], "prompt_tokens": prompt,
                   "completion_tokens": output, "reused_tokens": reused,
                   "uncached_prompt_tokens": prompt - reused, "source": str(path),
                   "ttft_ms": None, "tpot_ms": None}
            stream.write(json.dumps(row) + "\n")
            totals["requests"] += 1
            for key in ("prompt_tokens", "completion_tokens", "reused_tokens", "uncached_prompt_tokens"):
                totals[key] += row[key]
            totals["summed_request_seconds"] += row["elapsed_s"]
    write(directory / "report.json", {
        "official": summary, "requests": totals,
        "task_completion": task_summary,
        "generation_stage": generation, "official_evaluation_stage": evaluation,
        "instance_ids_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(),
        "ttft_status": "not_measured_nonstreaming_api", "tpot_status": "not_measured",
        "prefill_scope": "logical_uncached_prompt_tokens_excludes_unobserved_recompute",
        "time_scope": "sum_of_overlapping_server_request_durations_not_wall_time_or_gpu_time",
        "comparison_scope": "closed_loop_same_tasks_different_generated_trajectories",
    })
    print(json.dumps({"official": summary, "requests": totals}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--setting", type=Path, default=HERE / "setting.json")
    p.add_argument("--concurrency", type=int, default=64,
                   help="Target concurrent MiniSWE agents; use a fresh root for each value")
    p.add_argument("--engine-concurrency", type=int,
                   help="Server prefill/decode sequence limit; defaults to --concurrency")
    p.add_argument("--engine-resident-concurrency", type=int,
                   help="Resident sequence rows; defaults to --engine-concurrency")
    for action in ("serve", "bench", "collect", "stop"):
        p = sub.add_parser(action)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--method", choices=list(read(HERE / "setting.json")["methods"]), required=True)
        p.add_argument("--phase", choices=("smoke", "pilot", "full"), required=True)
        if action in {"serve", "bench"}:
            p.add_argument("--python", required=True, help="Environment Python; its bin is added to PATH")
            p.add_argument("--timeout", type=int, default=86400)
        if action == "serve":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--gpus", required=True)
            p.add_argument("--port", type=int, default=18147)
        elif action == "bench":
            p.add_argument("--swe-bench-dir", type=Path, required=True)
            p.add_argument("--stage", choices=("prepare", "generate", "evaluate", "summarize"), required=True)
            p.add_argument("--api-base", default="http://127.0.0.1:18147/v1")
            p.add_argument("--server-manifest", type=Path)
    args = parser.parse_args()
    if hasattr(args, "timeout") and args.timeout <= 0:
        parser.error("--timeout must be positive")
    {"prepare": prepare, "serve": serve, "bench": benchmark,
     "collect": collect, "stop": stop_server}[args.action](args)


if __name__ == "__main__":
    main()
