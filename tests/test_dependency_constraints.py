from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_runtime_compatibility_bounds_cover_canonical_lock():
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())["project"]
    dependencies = set(project["dependencies"])

    assert "torch>=2.9,<3" in dependencies
    assert "triton>=3.5,<4" in dependencies
    assert "tilelang==0.1.9" in dependencies
    assert "apache-tvm-ffi==0.1.10" in dependencies
    assert "transformers>=5.13,<6" in dependencies
    assert "flashinfer-python>=0.6.15,<0.7" in dependencies
    assert "flashinfer-jit-cache>=0.6.15,<0.7" in dependencies
    assert "sgl-kernel>=0.3.21,<0.4" in dependencies
    assert {"fire", "pillow", "einops", "tqdm", "loguru"} <= dependencies
    assert not any(
        dependency.startswith("torchvision") for dependency in dependencies
    )
    assert not any(
        dependency.startswith("flashinfer-cubin")
        for dependency in dependencies
    )


def test_canonical_lock_pins_validated_tilelang_runtime():
    lock_path = (
        Path(__file__).parents[1]
        / "requirements"
        / "locks"
        / "canonical-cu129-py310.txt"
    )
    locked_requirements = {
        line.strip()
        for line in lock_path.read_text().splitlines()
        if line and not line.startswith(("#", "--"))
    }

    assert "tilelang==0.1.9" in locked_requirements
    assert "apache-tvm-ffi==0.1.10" in locked_requirements


def test_uv_routes_cuda_packages_to_explicit_indexes():
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text())
    uv_config = config["tool"]["uv"]

    assert uv_config["sources"] == {
        "torch": {"index": "pytorch-cu130"},
        "flashinfer-jit-cache": {"index": "flashinfer-cu130"},
    }
    assert uv_config["index"] == [
        {
            "name": "pytorch-cu130",
            "url": "https://download.pytorch.org/whl/cu130",
            "explicit": True,
        },
        {
            "name": "flashinfer-cu130",
            "url": "https://flashinfer.ai/whl/cu130",
            "explicit": True,
        },
    ]


def test_workflow_dependencies_are_part_of_main_install():
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())["project"]
    dependencies = set(project["dependencies"])

    assert "optional-dependencies" not in project
    assert {
        "accelerate",
        "datasets",
        "socksio>=1,<2",
        "wandb",
        "bitsandbytes",
        "datatrove",
        "matplotlib",
        "seaborn",
        "math-verify==0.9.0",
        "fuzzywuzzy",
        "jieba",
        "pytest",
        "rouge",
        "tomli; python_version < '3.11'",
    } <= dependencies
