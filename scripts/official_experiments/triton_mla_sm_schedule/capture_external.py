"""Archive imported fork code and hash binaries without copying CUDA binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


p = argparse.ArgumentParser()
p.add_argument("--tangram-package", type=Path, required=True)
p.add_argument("--tangram-checkout", type=Path, required=True)
p.add_argument("--tangram-env", type=Path, required=True)
p.add_argument("--hisparse-package", type=Path, required=True)
p.add_argument("--hisparse-checkout", type=Path, required=True)
p.add_argument("--hisparse-env", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
manifest = {"forks": {}, "model": {"path": str(args.model.resolve()), "files": {}}}
for name, package, checkout, environment in (
    ("tangram", args.tangram_package, args.tangram_checkout, args.tangram_env),
    ("hisparse", args.hisparse_package, args.hisparse_checkout, args.hisparse_env),
):
    dest = args.output_dir / name
    dest.mkdir()
    git = lambda *a: subprocess.check_output(["git", *a], cwd=checkout)
    (dest / "checkout.patch").write_bytes(git("diff", "HEAD", "--binary"))
    packages = subprocess.check_output([
        str(environment / "bin/python"), "-c",
        "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'packages':sorted([(d.metadata['Name'],d.version) for d in importlib.metadata.distributions()])},indent=2))",
    ])
    (dest / "environment.json").write_bytes(packages)
    files = sorted(path for path in package.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and ".git" not in path.parts)
    hashes = {str(path.relative_to(package)): digest(path) for path in files}
    archive = dest / "imported-source.tar.gz"
    # Keep every source/config file, but hash rather than commit large binaries.
    with tarfile.open(archive, "w:gz") as handle:
        for path in files:
            if path.suffix in {".py", ".pyi", ".json", ".toml", ".yaml", ".yml", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".c", ".md", ".txt"}:
                handle.add(path, arcname=str(path.relative_to(package)))
    manifest["forks"][name] = {
        "imported_package": str(package.resolve()), "checkout": str(checkout.resolve()),
        "checkout_head": git("rev-parse", "HEAD").decode().strip(),
        "checkout_status": git("status", "--short").decode(),
        "environment": str(environment.resolve()), "imported_files_sha256": hashes,
        "source_archive_sha256": digest(archive),
        "binary_policy": "Imported binaries are hashed, not included in the source-only archive.",
    }
for path in sorted(args.model.iterdir()):
    if path.is_file():
        item = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        if path.suffix == ".json":
            item["sha256"] = digest(path)
        manifest["model"]["files"][path.name] = item
(args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"output": str(args.output_dir), "forks": list(manifest["forks"])}))
