# Dependency locks

`canonical-cu129-py310.txt` is the supported runtime and test baseline for
Python 3.10 and CUDA 12.9. It freezes the complete resolved Python environment;
`pyproject.toml` describes intentionally broader direct-dependency compatibility
ranges. Training, benchmark, and test packages are part of the main install.
The lock remains the only fully validated package combination; versions inside
the wider ranges are candidates until they pass the same validation gates.

Create the environment on shared machines under `/data2`:

```bash
CONDA_PKGS_DIRS=/data2/$USER/cache/conda_pkgs \
  conda create -p /data2/$USER/conda_envs/sparse-vllm-cu129-py310 \
  python=3.10 pip setuptools wheel -y

PYTHONNOUSERSITE=1 PIP_CACHE_DIR=/data2/$USER/cache/pip \
  /data2/$USER/conda_envs/sparse-vllm-cu129-py310/bin/python -s -m pip \
  install -r requirements/locks/canonical-cu129-py310.txt

PYTHONNOUSERSITE=1 \
  /data2/$USER/conda_envs/sparse-vllm-cu129-py310/bin/python -s -m pip \
  install --no-deps -e .
```

The lock is the compatibility contract. An unlocked install is a development
convenience, not evidence that a newly resolved dependency combination is
supported. Replace the lock only after dependency checks, CPU tests, focused
GPU operator tests, and a real model-path smoke all pass.
