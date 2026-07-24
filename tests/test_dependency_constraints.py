import tomllib
from pathlib import Path


def test_flashinfer_minimum_version_matches_moe_api():
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())["project"]
    dependencies = set(project["dependencies"])

    assert "flashinfer-python>=0.6.15" in dependencies
    assert "flashinfer-jit-cache>=0.6.15" in dependencies
    assert not any(
        dependency.startswith("flashinfer-cubin")
        for dependency in dependencies
    )
