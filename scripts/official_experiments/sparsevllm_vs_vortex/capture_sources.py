"""Freeze source artifacts and checkpoint identities without copying weights."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--native-repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--vortex-repo', type=Path, required=True)
parser.add_argument('--model-root', type=Path, required=True)
args = parser.parse_args()
root = args.output_dir
root.mkdir(exist_ok=False)
for name, repo in [('native', args.native_repo.resolve()), ('vortex', args.vortex_repo.resolve())]:
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    paths = [Path(x.decode()) for x in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0') if x]
    paths = sorted(set(p for p in paths if (repo / p).is_file() and p.suffix in {'.py', '.sh', '.cu', '.cuh', '.h', '.cpp', '.toml', '.json', '.md', '.yaml'} and '__pycache__' not in p.parts))
    hashes = {str(p): hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths}
    manifest = {'repo': str(repo), 'head': git('rev-parse', 'HEAD').decode().strip(), 'status': git('status', '--short', '--untracked-files=all').decode(), 'source_sha256': hashes}
    (root / f'{name}.json').write_text(json.dumps(manifest, indent=2))
    (root / f'{name}.patch').write_bytes(git('diff', 'HEAD', '--binary'))
    with tarfile.open(root / f'{name}.tar.gz', 'w:gz') as archive:
        for p in paths:
            archive.add(repo / p, arcname=str(p))
    if any(hashlib.sha256((repo / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError(f'Sources changed during snapshot: {repo}')
    print(name, len(paths), flush=True)
models = {}
for model in ['Qwen3-4B-Instruct-2507', 'GLM-4.7-Flash']:
    directory = args.model_root.resolve() / model
    models[model] = {'path': str(directory), 'files': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.suffix == '.json'}}
(root / 'models.json').write_text(json.dumps(models, indent=2))
(root / 'gpu.txt').write_bytes(subprocess.check_output(['nvidia-smi', '-q']))
