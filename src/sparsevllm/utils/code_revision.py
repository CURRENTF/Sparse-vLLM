import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path


def _git_command(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_value(repo_root: Path, *args: str) -> str | None:
    result = _git_command(repo_root, *args)
    if result is None:
        return None
    value = result.stdout.strip()
    return value or None


@lru_cache(maxsize=1)
def code_revision_info() -> dict[str, str | bool | None]:
    repo_root = Path(__file__).resolve().parents[3]
    dirty_result = _git_command(repo_root, "status", "--porcelain")
    try:
        package_version = version("deltakv")
    except PackageNotFoundError:
        package_version = None
    return {
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
        "git_dirty": (
            bool(dirty_result.stdout.strip())
            if dirty_result is not None
            else None
        ),
        "package_version": package_version,
    }
