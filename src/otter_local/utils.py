from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    if include_ms:
        return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"
    return f"{hour:02d}:{minute:02d}:{sec:02d}"


def srt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds, include_ms=True).replace(".", ",")
