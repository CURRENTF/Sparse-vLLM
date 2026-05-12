from __future__ import annotations

from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def iter_video_files(video_dir: Path):
    if not video_dir.exists():
        return
    for path in video_dir.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        yield path
