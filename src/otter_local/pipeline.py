from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from .backends import transcribe_audio
from .types import PipelineOptions, TranscriptResult, TranscriptSegment


SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
}


class TranscriptionCancelled(RuntimeError):
    """Raised when a caller requests cancellation during transcription."""


ProgressCallback = Callable[[float, str], bool | None]
_DIARIZATION_LOCAL = threading.local()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _notify_progress(
    progress_callback: ProgressCallback | None,
    progress: float,
    stage: str,
) -> None:
    if progress_callback is None:
        return

    clamped = max(0.0, min(1.0, float(progress)))
    try:
        keep_going = progress_callback(clamped, stage)
    except TranscriptionCancelled:
        raise
    except Exception:
        # Progress updates should not fail transcription.
        return

    if keep_going is False:
        raise TranscriptionCancelled("Canceled by callback")


def _resolve_device_compute(options: PipelineOptions) -> tuple[str, str]:
    if options.device != "auto":
        device = options.device
    else:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if options.compute_type != "auto":
        compute_type = options.compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


@lru_cache(maxsize=1)
def _resolve_ffmpeg_binary() -> str:
    if os.name == "nt":
        portable_root = _PROJECT_ROOT / ".tools" / "ffmpeg"
        if portable_root.exists():
            portable = next(portable_root.glob("**/ffmpeg.exe"), None)
            if portable is not None:
                return str(portable)

    names = ["ffmpeg.exe", "ffmpeg"] if os.name == "nt" else ["ffmpeg"]
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            winget_links = Path(local_appdata) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
            if winget_links.exists():
                return str(winget_links)

            packages_root = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
            if packages_root.exists():
                for package_dir in packages_root.glob("Gyan.FFmpeg*"):
                    for candidate in package_dir.glob("**/bin/ffmpeg.exe"):
                        if candidate.exists():
                            return str(candidate)

        choco_ffmpeg = Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe")
        if choco_ffmpeg.exists():
            return str(choco_ffmpeg)

    raise RuntimeError(
        "ffmpeg is required but was not found in PATH. Install ffmpeg first."
    )


def _normalize_audio(input_file: Path, temp_dir: Path, sample_rate: int = 16000) -> Path:
    ffmpeg_bin = _resolve_ffmpeg_binary()
    normalized = temp_dir / f"{input_file.stem}.wav"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(normalized),
    ]
    subprocess.run(cmd, check=True)
    return normalized


def _is_native_mono_pcm_wav(input_file: Path, sample_rate: int) -> bool:
    if input_file.suffix.lower() != ".wav":
        return False

    try:
        with wave.open(str(input_file), "rb") as wav_f:
            return (
                wav_f.getnchannels() == 1
                and wav_f.getframerate() == sample_rate
                and wav_f.getsampwidth() == 2
                and wav_f.getcomptype() == "NONE"
            )
    except Exception:
        return False


def _diarize_segments(
    audio_file: Path,
    base_segments: list[TranscriptSegment],
    options: PipelineOptions,
    progress_callback: ProgressCallback | None = None,
) -> list[TranscriptSegment]:
    if options.diarization_mode == "off":
        return base_segments

    token = options.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        if options.diarization_mode == "on":
            raise RuntimeError(
                "Diarization requested but no HF token found. Set HUGGINGFACE_HUB_TOKEN."
            )
        return base_segments

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        if options.diarization_mode == "on":
            raise RuntimeError(
                "Diarization requested but pyannote.audio is missing. "
                "Install with: pip install pyannote.audio"
            ) from exc
        return base_segments

    _notify_progress(progress_callback, 0.88, "diarizing")
    device, _ = _resolve_device_compute(options)
    cache_key = (token, device)
    cached_key = getattr(_DIARIZATION_LOCAL, "key", None)
    diarize_model = getattr(_DIARIZATION_LOCAL, "model", None)
    if diarize_model is None or cached_key != cache_key:
        try:
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load pyannote Community-1. Confirm that the Hugging Face "
                "token is valid and that you accepted the model access conditions. "
                f"Original error: {exc}"
            ) from exc
        if device == "cuda":
            diarize_model.to(torch.device("cuda"))
        _DIARIZATION_LOCAL.key = cache_key
        _DIARIZATION_LOCAL.model = diarize_model

    import numpy as np

    with wave.open(str(audio_file), "rb") as wav_f:
        sample_rate = wav_f.getframerate()
        channels = wav_f.getnchannels()
        sample_width = wav_f.getsampwidth()
        raw_audio = wav_f.readframes(wav_f.getnframes())
    if sample_width != 2:
        raise RuntimeError("Diarization requires normalized 16-bit PCM audio")
    waveform_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    waveform_array = waveform_array.reshape(-1, channels).T
    waveform = torch.from_numpy(waveform_array)

    diarized = diarize_model(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=options.min_speakers,
        max_speakers=options.max_speakers,
    )
    annotation = getattr(diarized, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(diarized, "speaker_diarization", diarized)

    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    if not turns:
        _notify_progress(progress_callback, 0.95, "diarization_complete")
        return base_segments

    provided_names = [name.strip() for name in options.speaker_names if str(name).strip()]
    speaker_map: dict[str, str] = {}
    discovered_count = 0

    def map_speaker(raw: str) -> str:
        nonlocal discovered_count
        if raw not in speaker_map:
            discovered_count += 1
            if discovered_count <= len(provided_names):
                speaker_map[raw] = provided_names[discovered_count - 1]
            else:
                speaker_map[raw] = f"Speaker {discovered_count}"
        return speaker_map[raw]

    rebuilt: list[TranscriptSegment] = []
    for segment in base_segments:
        overlaps: dict[str, float] = {}
        for turn_start, turn_end, raw_speaker in turns:
            overlap = max(0.0, min(segment.end, turn_end) - max(segment.start, turn_start))
            if overlap > 0:
                overlaps[raw_speaker] = overlaps.get(raw_speaker, 0.0) + overlap
        raw_speaker = max(overlaps, key=overlaps.get) if overlaps else segment.speaker
        rebuilt.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=map_speaker(raw_speaker),
                confidence=segment.confidence,
            )
        )

    _notify_progress(progress_callback, 0.95, "diarization_complete")
    return rebuilt


def transcribe_file(
    input_file: Path,
    options: PipelineOptions,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptResult:
    input_file = input_file.expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        # We still try with ffmpeg, but surface that this was not one of the tested formats.
        pass

    _notify_progress(progress_callback, 0.02, "preparing")
    with tempfile.TemporaryDirectory(prefix="otter-local-") as tmp:
        temp_dir = Path(tmp)
        target_sample_rate = 24000 if options.backend == "vibeasr" else 16000
        _notify_progress(progress_callback, 0.05, "normalizing")
        if _is_native_mono_pcm_wav(input_file, target_sample_rate):
            audio_for_pipeline = input_file
            _notify_progress(progress_callback, 0.10, "normalization_skipped")
        else:
            audio_for_pipeline = _normalize_audio(input_file, temp_dir, target_sample_rate)
            _notify_progress(progress_callback, 0.10, "normalized")
        segments, language, duration = transcribe_audio(
            audio_for_pipeline,
            options,
            progress_callback=progress_callback,
        )
        _notify_progress(progress_callback, 0.84, "transcription_complete")
        segments = _diarize_segments(
            audio_for_pipeline,
            segments,
            options,
            progress_callback=progress_callback,
        )
        _notify_progress(progress_callback, 0.98, "finalizing")

        result = TranscriptResult(
            source_file=input_file,
            language=language,
            duration=duration,
            segments=segments,
        )

    _notify_progress(progress_callback, 1.0, "done")
    return result
