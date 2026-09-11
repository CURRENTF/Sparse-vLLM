"""Prepare missing Vortex curves; use the canonical guarded capacity runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_export_namespaces(jobs):
    """Two length jobs must not compete for the same per-BS export files."""
    destinations = {}
    for job in jobs:
        command = job["command"]
        def option(flag):
            if command.count(flag) != 1:
                raise ValueError(f"Expected one {flag}: {job['name']}")
            return command[command.index(flag) + 1]
        destination = (Path(option("--export-measurements-dir")).resolve()
                       / option("--model") / option("--attempt"))
        if destination in destinations:
            raise ValueError(f"Export namespace collision: {destinations[destination]} and {job['name']}: {destination}")
        destinations[destination] = job["name"]


def freeze(repo, destination, metadata):
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo).decode().split("\0")
    for name in sorted(set(names)):
        path = repo / name
        if (not name or not path.is_file() or "tmp" in Path(name).parts
                or "data" in Path(name).parts or "__pycache__" in Path(name).parts
                or path.stat().st_size > 5_000_000):
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    manifest = dict(repo=str(repo), git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()))
    write(metadata, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, help="Execute the already frozen jobs sequentially")
    parser.add_argument("--prepare-plot", type=Path, help="Build raw grid sources from a previous portable 2x2 export")
    parser.add_argument("--job-names", help="Comma-separated unstarted jobs; does not restart existing queues")
    parser.add_argument("--after-queues", type=Path, nargs="*", default=[], help="Wait for these guarded sweep directories")
    parser.add_argument("--status-name", default="status")
    parser.add_argument("--recover-case", type=Path, help="Validate a completed case whose export failed; no GPU execution")
    parser.add_argument("--recovery-export-dir", type=Path)
    if "--run-root" in sys.argv:
        args = parser.parse_args()
        if args.recover_case:
            if args.recovery_export_dir is None:
                parser.error("--recover-case requires --recovery-export-dir")
            recover_case(args.recover_case.resolve(), args.recovery_export_dir.resolve())
        elif args.prepare_plot:
            prepare_plot(args.run_root.resolve(), args.prepare_plot)
        else:
            run(args.run_root.resolve(), args.job_names, args.after_queues, args.status_name)
        return
    parser.add_argument("--base-config", type=Path, required=True, help="Resolved existing boundary-sync config")
    parser.add_argument("--vortex-repo", type=Path, required=True)
    parser.add_argument("--vortex-env", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_config.read_text())
    if base["measurement_protocol"] != "boundary_sync_v2" or base["output_len"] != 2048:
        raise ValueError("Expected the existing boundary-sync 2K-output protocol")
    source, vortex = root / "source", root / "vortex"
    freeze(REPO, source, root / "manifest.json")
    freeze(args.vortex_repo.resolve(), vortex, root / "vortex_manifest.json")
    write(root / "prepare.json", {key: str(value) for key, value in vars(args).items()})
    scripts = source / PACKAGE.relative_to(REPO)
    jobs = []
    for model in ("qwen3-30b-fp8", "glm4.7-flash"):
        for length in (32768, 131072):
            name = f"{model}-{length}"
            cache = root / "cache" / name
            cache.mkdir(parents=True)
            mla = model == "glm4.7-flash"
            config = {**base, "models": {model: base["models"][model]}, "input_len": length,
                      "output_root": str(root / name), "unsupported": {}, "external_lanes": {}}
            config["external_lanes"]["vortex-quest"] = dict(
                engine="vortex", method="quest", backend_label="vortex-quest", env=str(args.vortex_env.resolve()),
                pythonpath=[str(args.overlay.resolve()), str(args.overlay.resolve() / "nvidia_cutlass_dsl/dsl_packages"),
                            str(vortex), str(vortex / "third_party/sglang/v0.5.9/sglang/python")],
                environment=dict(CUDA_HOME="/usr/local/cuda-13.0", SGLANG_ENABLE_TORCH_COMPILE="0",
                    VORTEX_CUDA_MLA_SM90_BUILD_DIR=str(cache / "mla"), FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
                    TVM_FFI_CACHE_DIR=str(cache / "tvm"), TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
                    FLASHINFER_CUBIN_DIR=str(args.overlay.resolve() / "cubins"), MAX_JOBS="8"),
                engine_kwargs=dict(dtype="bfloat16", attention_backend="cuda_mla_sm90" if mla else "flashinfer",
                    vortex=dict(impl_backend="triton", use_tensor_core=True, attention_backend="trtllm",
                        layers_skip=[0, 1], block_reserved_bos=4, block_reserved_eos=32, topk_val=92, topk_ratio=0,
                        block_size=16, workload_chunk_size=64, module_name="quest_mla" if mla else "gqa_quest_sparse_attention",
                        max_topk_val=128, dtype="bfloat16", compilation_cache_dir=str(cache / "vortex"))))
            config["comparison_notes"] = [
                "Vortex QuEST: 4 sink + 32 recent + 92 selected pages, page16; total2048; layers0/1 dense.",
                "Page/head scoring differs from native token selection; matching budgets does not establish equal quality.",
                "Weight quantization is checkpoint-derived; dtype=bfloat16 controls activations, not dequantized weights."]
            config_path = root / f"{name}.json"
            write(config_path, config)
            command = ["python3", str(scripts / "sweep_decode_capacity.py"), "--config", str(config_path),
                       "--repo", str(source), "--model", model, "--gpus", f"auto:{base['models'][model]['tp']}",
                       "--lanes", "vortex-quest", "--attempt", "vortex-v1", "--export-measurements-dir",
                       str(args.export_data_dir.resolve() / f"p{length}-o{base['output_len']}")]
            jobs.append(dict(name=name, command=command))
    validate_export_namespaces(jobs)
    write(root / "jobs.json", jobs)
    print(root, flush=True)


def recover_case(case, destination):
    from plot_decode_capacity import validate_measurement, validate_grid_identity
    from benchmark.efficiency.paper import without_source_fingerprints

    queue, lane = case.parent.parent, case.parent.name
    config = json.loads((queue / "campaign.json").read_text())
    model = queue.parent.name
    row = json.loads((case / "performance.jsonl").read_text())
    batch = row["batch_size"]
    point = validate_measurement(case / "performance.jsonl", batch, config)
    validate_grid_identity(point, model, config)
    output = destination / model / queue.name / lane / f"bs{batch}.json"
    observed = case.parent / "observed_capacity.json"
    if output.exists() or observed.exists():
        raise FileExistsError("Refusing to replace recovered evidence")
    write(output, dict(measurement_protocol=config["measurement_protocol"], smoke_only=False,
        model=config["models"][model], lane=lane, config=config, point=point,
        run_manifest=without_source_fingerprints(json.loads((case / "run_manifest.json").read_text())),
        source_identity_sha256=hashlib.sha256((case / "identity.json").read_bytes()).hexdigest()))
    write(observed, dict(status="partial", model=model, lane=lane,
        reason="Raw-validated case recovered after export collision; no maximum established.",
        attempts=[dict(concurrency=batch, artifact=str(case / "performance.jsonl"), status="success")]))
    print(json.dumps(dict(export=str(output), capacity=str(observed), throughput=point["decode_throughput_tps"])))


def prepare_plot(root, previous):
    """Keep all old raw sources and add only the explicit Vortex attempts."""
    data = json.loads(previous.read_text())
    panels = []
    for panel_id, old in data["config"]["panel_protocols"].items():
        model, length = old["source_model"], old["input_len"]
        supplement = json.loads((root / f"{model}-{length}.json").read_text())
        for key in ("measurement_protocol", "input_len", "output_len", "decode_window_steps",
                    "decode_warmup_steps", "num_iters", "num_warmups", "gpu_memory_utilization"):
            if old[key] != supplement[key]:
                raise ValueError(f"Supplement differs on {key}")
        if old["models"][model] != supplement["models"][model]:
            raise ValueError("Supplement model/topology mismatch")
        old["external_lanes"]["vortex-quest"] = supplement["external_lanes"]["vortex-quest"]
        path = root / f"plot-{panel_id}.json"
        if path.exists():
            raise FileExistsError(path)
        write(path, old)
        policies = {}
        for curve in data["curves"]:
            if curve["model"] != panel_id or curve.get("status") == "unsupported":
                continue
            policies[curve["lane"]] = dict(capacity_artifact=curve["capacity_artifact"], allow_partial=True,
                reason=curve.get("reason", "Reuse previously validated formal points; preserve unverified capacity."))
        capacity = Path(supplement["output_root"]) / model / "vortex-quest-vortex-v1/vortex-quest/capacity.json"
        policies["vortex-quest"] = dict(capacity_artifact=str(capacity), allow_partial=True,
            reason="Matched Vortex supplement; preserve failed attempts, omit absent formal points, never infer a measured maximum.")
        panels.append(dict(model=model, config=str(path), curves=policies))
    write(root / "grid-with-vortex.json", dict(shape=data["config"]["grid_shape"], panels=panels,
        reused_plot_data_sha256=hashlib.sha256(previous.read_bytes()).hexdigest()))


def run(root, names=None, predecessors=(), status_name="status"):
    failures = []
    jobs = json.loads((root / "jobs.json").read_text())
    validate_export_namespaces(jobs)
    if names:
        selected = names.split(",")
        if set(selected) - {job["name"] for job in jobs}:
            raise ValueError("Unknown job selection")
        jobs = [job for job in jobs if job["name"] in selected]
    with (root / f"{status_name}.tsv").open("x", buffering=1) as status:
        deadline = time.monotonic() + 172800
        for predecessor in predecessors:
            status.write(f"{datetime.now(timezone.utc).isoformat()}\tpredecessor\twaiting\t{predecessor}\n")
            while True:
                state = predecessor / "status.tsv"
                rows = [line.split("\t") for line in state.read_text().splitlines()] if state.exists() else []
                terminal = [row for row in rows if len(row) >= 3 and row[1] == "queue" and row[2] in ("completed", "probe_completed", "failed")]
                if terminal:
                    if terminal[-1][2] == "failed":
                        summary = predecessor / "queue_summary.json"
                        if not summary.exists() or not json.loads(summary.read_text()).get("failures"):
                            raise RuntimeError(f"Predecessor resource/orchestration failure: {predecessor}")
                        failures.append(str(predecessor))
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Waiting for {predecessor} exceeded48h")
                time.sleep(5)
        for job in jobs:
            name, command = job["name"], job["command"]
            status.write(f"{datetime.now(timezone.utc).isoformat()}\t{name}\trunning\n")
            with (root / f"{name}.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=172800, start_new_session=True)
            status.write(f"{datetime.now(timezone.utc).isoformat()}\t{name}\texit={result.returncode}\n")
            if result.returncode:
                def option(flag):
                    return command[command.index(flag) + 1]
                config = json.loads(Path(option("--config")).read_text())
                summary_path = (Path(config["output_root"]) / option("--model")
                                / f"{option('--lanes')}-{option('--attempt')}" / "queue_summary.json")
                if not summary_path.exists() or not json.loads(summary_path.read_text()).get("failures"):
                    raise RuntimeError(f"Resource/orchestration failure: {name}; remaining jobs stopped")
                failures.append(name)
        write(root / f"{status_name}.summary.json", dict(status="failed" if failures else "completed", failed_jobs=failures))
        status.write(f"{datetime.now(timezone.utc).isoformat()}\tqueue\t{'failed' if failures else 'completed'}\n")
    if failures:
        raise RuntimeError(f"Method failures retained: {failures}")


if __name__ == "__main__":
    main()
