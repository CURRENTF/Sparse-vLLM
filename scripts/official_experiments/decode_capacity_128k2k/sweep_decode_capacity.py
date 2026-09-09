"""Sweep the canonical microbench to a verified integer concurrency boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

PACKAGE = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def capacity_failure(error, log_path):
    if any(text in error.lower() for text in ("out of memory", "full decode batch capacity exceeded", "no runnable sequences", "cannot fit", "cannot admit")):
        return True
    if "Full decode batch capacity exceeded: scheduler preemption" in log_path.read_text():
        # SGLang terminates its parent when a scheduler worker fails; retain the
        # worker's explicit capacity diagnosis instead of inferring from SIGKILL.
        return True
    # A missing graph is not generally a capacity error. Require the allocator's
    # explicit evidence that this run skipped capture for insufficient KV.
    return ("no startup-captured graph" in error
            and bool(re.search(r"long=0 skipped_for_kv_capacity=[1-9][0-9]*", log_path.read_text())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True,
                        help="Compatible benchmark checkout (see README); not inferred from this package")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate paths and required adapter options without touching GPUs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--lanes", default="svllm-vanilla,svllm-snapkv,svllm-quest,svllm-omnikv,vllm-vanilla")
    parser.add_argument("--wait-for-release", action="store_true")
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--reuse-smoke-from", type=Path)
    parser.add_argument("--reuse-cases-from", type=Path)
    parser.add_argument("--reuse-additional-cases-from", type=Path, action="append", default=[],
                        help="Additional raw-validated probe roots, in deterministic priority order")
    parser.add_argument("--probe-concurrency", type=int)
    parser.add_argument("--probe-only", action="store_true",
                        help="Run one guarded probe; do not claim a complete capacity boundary")
    args = parser.parse_args()
    if args.probe_only and (not args.probe_concurrency or args.probe_concurrency < 1 or "," in args.lanes):
        raise ValueError("probe-only requires one lane and a positive probe concurrency")
    REPO = args.repo.resolve()
    config_text = os.path.expandvars(args.config.read_text())
    unresolved = re.findall(r"\$\{[^}]+\}", config_text)
    if unresolved:
        raise ValueError(f"Set these environment variables first: {sorted(set(unresolved))}")
    config = json.loads(config_text)
    for key in ("output_root", "scratch_root", "conda", "native_env", "vllm_env"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    for key in ("conda", "native_env", "vllm_env"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])
    model_config = Path(config["models"][args.model]["path"]) / "config.json"
    if not model_config.is_absolute() or not model_config.is_file():
        raise FileNotFoundError(f"Model config must exist at an absolute path: {model_config}")
    adapter = REPO / "benchmark" / "microbench.py"
    required = ("--engine", "--require_full_decode_batch", "--synchronize_step_timing",
                "--decode_warmup_steps_after_full", "--admission_wave_size")
    source = adapter.read_text()
    missing = [option for option in required if option not in source]
    if missing:
        raise RuntimeError(f"{adapter} lacks measured stage adapters: {missing}. "
                           "Use the compatible experiment checkout described in README.")
    if args.check_only:
        print("Paths and adapter options validated; no GPU execution or numerical validation performed.")
        return
    model = config["models"][args.model]
    root = Path(config["output_root"]) / args.model / (args.lanes.replace(",", "_") + "-" + args.attempt)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "campaign.json", config)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus,
           "PYTHONPATH": str(REPO) + ":" + str(REPO / "src"),
           "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for key, name in (("VLLM_CACHE_ROOT", "vllm"), ("TRITON_CACHE_DIR", "triton"),
                      ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("CUDA_CACHE_PATH", "cuda"), ("TMPDIR", "scratch")):
        directory = Path(config["scratch_root"]) if key == "TMPDIR" else Path(config["output_root"]) / "cache" / name
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    native_env = config["native_env"]
    active = None
    guard = None
    case_results = {}

    def stop(signum, frame):
        raise InterruptedError(f"Queue interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status(stage, state, **extra):
        value = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage,
                 "status": state, "gpus": args.gpus, **extra}
        print(json.dumps(value), flush=True)
        with (root / "status.tsv").open("a") as out:
            out.write("\t".join(str(value.get(key, "")) for key in ("time", "stage", "status", "gpus")) + "\t" + json.dumps(extra) + "\n")

    def ensure_guard():
        if guard.poll() is not None:
            raise RuntimeError("GPU reservation exited; inspect guard.log")
        if (root / "guard.contention.json").exists():
            raise RuntimeError("External GPU contention invalidates the run")

    def run_case(lane, batch, smoke=False):
        nonlocal active
        ensure_guard()
        if (lane, batch, smoke) in case_results:
            return case_results[(lane, batch, smoke)]
        case = root / lane / ("smoke" if smoke else f"bs{batch}")
        if case.exists():
            raise FileExistsError(f"Refusing to overwrite existing attempt: {case}")
        engine, method = ("vllm", "vanilla") if lane == "vllm-vanilla" else ("sparsevllm", lane.removeprefix("svllm-"))
        external = config.get("external_lanes", {}).get(lane)
        if external:
            engine, method = external["engine"], external["method"]
        hp = dict(tensor_parallel_size=model["tp"], expert_parallel_size=model["ep"],
                  data_parallel_size=1, decode_graph=True, gpu_memory_utilization=config["gpu_memory_utilization"],
                  max_num_batched_tokens=8192, engine_prefill_chunk_size=8192,
                  enable_prefix_caching=False)
        if external:
            chunk = external.get("max_num_batched_tokens", 8192)
            hp.update(max_num_batched_tokens=chunk, engine_prefill_chunk_size=chunk)
        if engine == "sparsevllm":
            hp["decode_graph_capture_sizes"] = [batch]
            if method in config["methods"]:
                hp.update(config["methods"][method])
        length, output = (4096, 64) if smoke else (config["input_len"], config["output_len"])
        reuse_roots = ([args.reuse_cases_from] if args.reuse_cases_from else []) + args.reuse_additional_cases_from
        available = [root / lane / f"bs{batch}" for root in reuse_roots
                     if (root / lane / f"bs{batch}" / "performance.jsonl").is_file()]
        if available and not smoke:
            previous = available[0]
            artifact = previous / "performance.jsonl"
            if artifact.exists():
                old_hp = json.loads((previous / "hyper_params.json").read_text())
                identity = json.loads((previous / "identity.json").read_text())
                source_matches = all((REPO / name).is_file() and hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
                                     for name, digest in identity["source_sha256"].items())
                old_command = identity["command"]
                expected_args = {"--model_path": model["path"], "--lengths": str(length),
                                 "--output_len": str(output), "--batch_sizes": str(batch),
                                 "--decode_warmup_steps_after_full": "32", "--engine": engine, "--methods": method}
                if external:
                    expected_args["--backend_label"] = external["backend_label"]
                    if json.loads((previous / "engine_kwargs.json").read_text()) != external["engine_kwargs"]:
                        raise RuntimeError(f"Refusing changed external algorithm parameters: {previous}")
                if (old_hp != hp or not source_matches
                        or any(old_command[old_command.index(key) + 1] != value for key, value in expected_args.items())
                        or "--require_full_decode_batch" not in old_command
                        or "--synchronize_step_timing" not in old_command):
                    raise RuntimeError(f"Refusing incompatible formal reuse: {previous}")
                rows = [json.loads(line) for line in artifact.read_text().splitlines()]
                if len(rows) != 1:
                    raise RuntimeError(f"Invalid reusable artifact: {artifact}")
                row = rows[0]
                if row.get("status") == "success":
                    from plot_decode_capacity import validate_measurement
                    validate_measurement(artifact, batch, config)
                    result = {"concurrency": batch, "artifact": str(artifact), "status": "success"}
                elif capacity_failure(str(row.get("error", "")), previous / "run.log"):
                    result = {"concurrency": batch, "artifact": str(artifact), "status": "capacity_exceeded", "error": row["error"]}
                else:
                    raise RuntimeError(f"Unclassified previous failure: {artifact}")
                status(lane + f"/bs{batch}", "reused_validated", artifact=str(artifact), result=result["status"])
                case_results[(lane, batch, smoke)] = result
                return result
        case.mkdir(parents=True)
        write(case / "hyper_params.json", hp)
        command = [config["conda"], "run", "--no-capture-output", "-p",
                   external["env"] if external else native_env if engine == "sparsevllm" else config["vllm_env"],
                   "python", "-u", "benchmark/microbench.py", "--engine", engine,
                   "--model_path", model["path"], "--lengths", str(length),
                   "--output_len", str(output), "--batch_sizes", str(batch),
                   "--methods", method, "--hyper_params", "@" + str(case / "hyper_params.json"),
                   "--synchronize_step_timing", "--require_full_decode_batch",
                   "--decode_warmup_steps_after_full", "8" if smoke else "32",
                   "--output_dir", str(case)]
        if external:
            write(case / "engine_kwargs.json", external["engine_kwargs"])
            if external.get("environment_kind") == "venv":
                prefix = Path(external["env"])
                if not (prefix / "pyvenv.cfg").is_file():
                    raise ValueError(f"Configured venv lacks pyvenv.cfg: {prefix}")
                command = [str(prefix / "bin/python")] + command[6:]
                env["PATH"] = str(prefix / "bin") + os.pathsep + os.environ["PATH"]
                env["VIRTUAL_ENV"] = str(prefix)
            command += ["--engine_kwargs", "@" + str(case / "engine_kwargs.json"),
                        "--backend_label", external["backend_label"]]
        if method == "snapkv" and engine == "sparsevllm":
            command += ["--admission_wave_size", "1", "--wave_decode_gap_steps", "1"]
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            env["SPARSEVLLM_MASTER_PORT"] = str(listener.getsockname()[1])
        snapshot_manifest = REPO.parent / "manifest.json"
        if (REPO / ".git").exists():
            tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0")
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        else:
            snapshot = json.loads(snapshot_manifest.read_text())
            tracked, commit = list(snapshot["sha256"]), snapshot["git_head"]
        hashes = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                  for name in tracked if name and (REPO / name).is_file()}
        write(case / "identity.json", {"command": command, "env": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "VLLM_ENABLE_V1_MULTIPROCESSING", "SPARSEVLLM_MASTER_PORT")},
              "git_commit": commit,
              "orchestrator_sha256": {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                                      for name in ("sweep_decode_capacity.py", "decode_capacity_guard.py", "plot_decode_capacity.py")},
              "source_sha256": hashes, "model_config_sha256": hashlib.sha256((Path(model["path"]) / "config.json").read_bytes()).hexdigest()})
        status(str(case.relative_to(root)), "running")
        with (case / "run.log").open("w") as log:
            active = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = active.wait(timeout=config["case_timeout_s"])
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGTERM)
                active.wait(timeout=30)
                raise RuntimeError(f"Case timeout: {case}")
            finally:
                if active.poll() is not None:
                    active = None
        ensure_guard()
        artifact = case / "performance.jsonl"
        rows = [json.loads(line) for line in artifact.read_text().splitlines()] if artifact.exists() else []
        row = rows[0] if len(rows) == 1 else {}
        if code == 0 and row.get("status") == "success":
            from plot_decode_capacity import validate_measurement
            validate_measurement(artifact, batch, {"input_len": length, "output_len": output})
            if row.get("actual_decode_peak") != batch or row.get("completed_requests") != batch:
                raise RuntimeError(f"Missing actual concurrency/completion evidence: {case}")
            if row.get("measurement_scope") != "full_batch_pure_decode_steps" or not row.get("decode_stage_throughput_tps", 0) > 0:
                raise RuntimeError(f"Invalid stage metric: {case}")
            status(str(case.relative_to(root)), "success", throughput=row["decode_stage_throughput_tps"])
            result = {"concurrency": batch, "artifact": str(artifact), "status": "success"}
            case_results[(lane, batch, smoke)] = result
            return result
        error = str(row.get("error", ""))
        capacity = capacity_failure(error, case / "run.log")
        status(str(case.relative_to(root)), "capacity_exceeded" if capacity else "failed", error=error, exit_code=code)
        if smoke or not capacity:
            raise RuntimeError(f"Benchmark failed, inspect {case / 'run.log'}")
        result = {"concurrency": batch, "artifact": str(artifact), "status": "capacity_exceeded", "error": error}
        case_results[(lane, batch, smoke)] = result
        return result

    try:
        status("resource", "waiting_for_idle")
        deadline = time.monotonic() + config["idle_timeout_s"]
        while time.monotonic() < deadline:
            if args.gpus.startswith("auto:"):
                count = int(args.gpus.split(":")[1])
                devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15)
                occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=15).splitlines()
                idle = []
                for device in devices.splitlines():
                    index, uuid, memory, utilization = [part.strip() for part in device.split(",")]
                    if uuid not in occupied and int(memory) < 100 and int(utilization) == 0:
                        idle.append(index)
                if len(idle) >= count:
                    args.gpus = ",".join(idle[:count])
                    env["CUDA_VISIBLE_DEVICES"] = args.gpus
                    break
                time.sleep(15)
                continue
            pids = subprocess.check_output(["nvidia-smi", "-i", args.gpus, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
            if not pids.strip():
                break
            time.sleep(15)
        else:
            raise RuntimeError("No idle GPU before resource deadline")
        with (root / "guard.log").open("w") as log:
            guard = subprocess.Popen([config["conda"], "run", "--no-capture-output", "-p", native_env,
                "python", "-u", str(PACKAGE / "decode_capacity_guard.py"), "--parent", str(os.getpid()),
                "--ready", str(root / "guard.json")], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            ensure_guard()
            if (root / "guard.json").exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError("GPU reservation readiness timeout")
        status("resource", "reserved")
        lanes = args.lanes.split(",")
        for lane in lanes:
            if args.reuse_smoke_from:
                previous = args.reuse_smoke_from / lane / "smoke" / "performance.jsonl"
                if previous.exists() and json.loads(previous.read_text().splitlines()[0]).get("status") == "success":
                    from plot_decode_capacity import validate_measurement
                    validate_measurement(previous, 2, {"input_len": 4096, "output_len": 64})
                    status(lane + "/smoke", "reused_validated", artifact=str(previous))
                    continue
            run_case(lane, 2, smoke=True)
        if args.wait_for_release:
            status("full_sweep", "waiting_for_validated_smoke_release")
            deadline = time.monotonic() + 7200
            while not (Path(config["output_root"]) / "full.ready").exists():
                ensure_guard()
                if time.monotonic() >= deadline:
                    raise RuntimeError("Full sweep release deadline exceeded")
                time.sleep(5)
        for lane in lanes:
            attempts = []
            if args.probe_concurrency and lane == lanes[0]:
                run_case(lane, args.probe_concurrency)
                if args.probe_only:
                    status("queue", "probe_completed", concurrency=args.probe_concurrency)
                    return
            lower, upper, batch = 0, None, 1
            while batch <= config["safety_concurrency_limit"]:
                result = run_case(lane, batch)
                attempts.append(result)
                write(root / lane / "capacity.json", {"status": "running", "attempts": attempts})
                if result["status"] != "success":
                    upper = batch
                    break
                lower, batch = batch, batch * 2
            if upper is None or lower == 0:
                raise RuntimeError(f"No validated nonzero capacity boundary for {lane}")
            while upper - lower > 1:
                batch = (upper + lower) // 2
                result = run_case(lane, batch)
                attempts.append(result)
                if result["status"] == "success":
                    lower = batch
                else:
                    upper = batch
                write(root / lane / "capacity.json", {"status": "running", "attempts": attempts})
            write(root / lane / "capacity.json", {"status": "completed", "model": args.model,
                  "lane": lane, "max_concurrency": lower, "first_failed_concurrency": upper, "attempts": attempts})
            status(lane, "completed", max_concurrency=lower)
        status("queue", "completed")
    except BaseException as error:
        status("queue", "failed", error=repr(error))
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            active.wait(timeout=30)
        if guard is not None and guard.poll() is None:
            os.killpg(guard.pid, signal.SIGTERM)
            guard.wait(timeout=30)


if __name__ == "__main__":
    main()
