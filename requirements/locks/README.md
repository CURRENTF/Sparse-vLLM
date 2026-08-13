# Dependency locks

`canonical-cu129-py310.txt` is an optional, fully validated reproducibility
baseline for Python 3.10 and CUDA 12.9. It freezes the complete resolved Python
environment; `pyproject.toml` is the default installation contract and describes
intentionally broader direct-dependency compatibility ranges. Training,
benchmark, and test packages are part of the main install.

Create an isolated environment from the repository root:

```bash
uv venv --python 3.10
uv pip install -r requirements/locks/canonical-cu129-py310.txt
uv pip install --no-deps -e .
```

For reproducible runs, the lock is the compatibility contract. The default
unlocked install is the development path, not evidence that a newly resolved
dependency combination is fully validated. Replace the lock only after
dependency checks, CPU tests, focused GPU operator tests, and a real model-path
smoke all pass.
