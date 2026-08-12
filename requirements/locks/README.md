# Dependency locks

`canonical-cu129-py310.txt` is the supported runtime and test baseline for
Python 3.10 and CUDA 12.9. It freezes the complete resolved Python environment;
`pyproject.toml` describes intentionally broader direct-dependency compatibility
ranges. Training, benchmark, and test packages are part of the main install.
The lock remains the only fully validated package combination; versions inside
the wider ranges are candidates until they pass the same validation gates.

Create an isolated environment from the repository root:

```bash
uv venv --python 3.10
uv pip install -r requirements/locks/canonical-cu129-py310.txt
uv pip install --no-deps -e .
```

The lock is the compatibility contract. An unlocked install is a development
convenience, not evidence that a newly resolved dependency combination is
supported. Replace the lock only after dependency checks, CPU tests, focused
GPU operator tests, and a real model-path smoke all pass.
