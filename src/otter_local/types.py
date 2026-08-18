from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "Speaker 1"
    confidence: float | None = None


@dataclass(slots=True)
class TranscriptResult:
    source_file: Path
    language: str
    duration: float
    segments: list[TranscriptSegment] = field(default_factory=list)


@dataclass(slots=True)
class PipelineOptions:
    backend: str = "faster-whisper"
    model: str = "small"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    diarization_mode: str = "auto"  # auto|on|off
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    speaker_names: list[str] = field(default_factory=list)
    hotwords: str = ""
    num_threads: int = 4
    keep_temp: bool = False
