"""Bundle replot data and raw evidence, preserving portable artifact IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


p = argparse.ArgumentParser()
p.add_argument("--export", type=Path, required=True)
p.add_argument("--campaign", type=Path, required=True)
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--include-extended-history", action="store_true", help="Only use after extended jobs have stopped writing")
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
raw = read(args.export / "figures/plot_data.json")
portable = read(args.export / "plot_data.json")
shutil.copy2(args.export / "plot_data.json", args.output_dir)
shutil.copy2(args.export / "provenance.json", args.output_dir)
shutil.copytree(args.export / "portable-replot", args.output_dir / "figures")
files = {}


def add(source, relative):
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-portable archive path: {relative}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if str(relative) in files and sha(files[str(relative)]) != sha(source):
        raise ValueError(f"Conflicting raw artifact: {relative}")
    files[str(relative)] = source


def add_tree(source, relative):
    for path in source.rglob("*"):
        parts = path.relative_to(source).parts
        if any(part in {"cache", "__pycache__", ".pytest_cache", ".git"} for part in parts):
            continue
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".log", ".tsv", ".xml", ".patch", ".md", ".py", ".sh"}:
            add(path, relative / path.relative_to(source))


for curve, exported in zip(raw["curves"], portable["curves"], strict=True):
    if (curve["model"], curve["lane"]) != (exported["model"], exported["lane"]):
        raise ValueError("Raw and portable curve ordering differ")
    for kind in ("attempts", "points"):
        for actual, item in zip(curve[kind], exported[kind], strict=True):
            if "artifact" in actual:
                source, relative = Path(actual["artifact"]), Path(item["artifact"])
                add_tree(source.parent, relative.parent)
    if "capacity_artifact" in curve:
        add(Path(curve["capacity_artifact"]), Path(exported["capacity_artifact"]))
for name in ("micro", "tuning/micro", "refined/micro", "external_provenance"):
    add_tree(args.campaign / name, Path("validation") / name)
history = ["external1", "external2", "external3", "external4", "external5"]
if args.include_extended_history:
    history += ["external6_extended", "external7_extended"]
for name in history:
    add_tree(args.campaign / name, Path("attempt_history") / name)
# Also retain the standard external artifact IDs for its independent replot.
add_tree(args.campaign / "external5", Path("external"))
for name in ("04_glm_common8", "06_glm_wave64"):
    add_tree(args.campaign / "model/queue" / name, Path("validation/model") / name)
for name in ("comparison_validated.json", "manifest.json", "comparison_cases.json"):
    add(args.campaign / "model" / name, Path("validation/model") / name)
for case in read(args.campaign / "model/comparison_validated.json")["cases"]:
    if case["engine"] == "Legacy Triton":
        add_tree(Path(case["source_path"]).parent, Path("validation/model_baseline") / f"bs{case['concurrency']}")
for name in ("manifest.json", "source.patch", "refined/manifest.json", "refined/source.patch", "refined/tests.xml", "final_cpu_tests.xml", "final_adapter_tests.xml", "final_extended_tests.xml", "adapters5/manifest.json", "adapters5/source.patch"):
    add(args.campaign / name, Path("validation") / name)
forks = read(args.campaign / "external_provenance/manifest.json")["forks"]
for fork, name in (("tangram", "config/compression.py"), ("hisparse", "srt/arg_groups/hisparse_hook.py")):
    path = Path(forks[fork]["imported_package"]) / name
    if sha(path) != forks[fork]["imported_files_sha256"][name]:
        raise ValueError("Unsupported-contract evidence changed after capture")
    add(path, Path("validation/unsupported") / fork / name)
for name in ("scripts/official_experiments/triton_mla_sm_schedule", "scripts/official_experiments/sparse_decode_efficiency"):
    # Recipes and portable data only; no local/private path configs or old plots.
    for path in (args.repo / name).rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json", ".csv"} and not any(x in path.parts for x in ("__pycache__", "plots")) and ".local." not in path.name:
            add(path, Path("recipe") / path.relative_to(args.repo))
for name in ("benchmark/microbench.py", "benchmark/vllm_microbench.py", "benchmark/hisparse_microbench.py", "benchmark/efficiency/metrics.py",
             "src/sparsevllm/kernels/triton/mla/decode_schedule.py", "src/sparsevllm/operators/mla_attention.py", "tests/test_hisparse_decode_stage.py"):
    add(args.repo / name, Path("final_source") / name)
for name in ("benchmark/microbench.py", "benchmark/vllm_microbench.py", "benchmark/hisparse_microbench.py", "benchmark/efficiency/metrics.py"):
    add(args.campaign / "adapters5/source" / name, Path("measured_adapter") / name)
archive = args.output_dir / "raw_evidence.tar.gz"
with tarfile.open(archive, "w:gz") as handle:
    for name, source in sorted(files.items()):
        handle.add(source, arcname=name, recursive=False)
manifest = {"archive": archive.name, "archive_sha256": sha(archive),
            "files": {name: {"sha256": sha(source), "source": str(source)} for name, source in sorted(files.items())}}
(args.output_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files)}))
