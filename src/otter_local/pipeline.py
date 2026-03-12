from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Callable
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

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
_MODEL_CACHE_LOCAL = threading.local()


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


def _normalize_audio(input_file: Path, temp_dir: Path) -> Path:
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
        "16000",
        str(normalized),
    ]
    subprocess.run(cmd, check=True)
    return normalized


def _is_native_16k_mono_pcm_wav(input_file: Path) -> bool:
    if input_file.suffix.lower() != ".wav":
        return False

    try:
        with wave.open(str(input_file), "rb") as wav_f:
            return (
                wav_f.getnchannels() == 1
                and wav_f.getframerate() == 16000
                and wav_f.getcomptype() == "NONE"
            )
    except Exception:
        return False


def _get_model_cache() -> dict[tuple[str, str, str], Any]:
    cache = getattr(_MODEL_CACHE_LOCAL, "models", None)
    if cache is None:
        cache = OrderedDict()
        _MODEL_CACHE_LOCAL.models = cache
    return cache


def _get_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install with: pip install faster-whisper"
        ) from exc

    cache = _get_model_cache()
    key = (model_name, device, compute_type)
    model = cache.get(key)
    if model is not None:
        cache.move_to_end(key)
        return model

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    cache[key] = model
    if len(cache) > 2:
        cache.popitem(last=False)
    return model


def _transcribe_with_whisper(
    audio_file: Path,
    options: PipelineOptions,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[TranscriptSegment], str, float]:
    device, compute_type = _resolve_device_compute(options)
    model = _get_whisper_model(options.model, device=device, compute_type=compute_type)

    seg_iter, info = model.transcribe(
        str(audio_file),
        language=options.language,
        beam_size=options.beam_size,
        vad_filter=options.vad_filter,
    )
    duration_hint = float(getattr(info, "duration", 0.0) or 0.0)
    _notify_progress(progress_callback, 0.12, "transcribing")

    segments: list[TranscriptSegment] = []
    segment_count = 0
    for seg in seg_iter:
        segment_count += 1
        confidence = None
        if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None:
            confidence = max(0.0, min(1.0, 1.0 - float(seg.no_speech_prob)))

        segments.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=str(seg.text).strip(),
                speaker="Speaker 1",
                confidence=confidence,
            )
        )
        if duration_hint > 0:
            ratio = max(0.0, min(1.0, float(seg.end) / duration_hint))
            _notify_progress(progress_callback, 0.12 + (ratio * 0.72), "transcribing")
        else:
            fallback_ratio = min(1.0, segment_count / 120.0)
            _notify_progress(progress_callback, 0.12 + (fallback_ratio * 0.72), "transcribing")

    language = info.language or (options.language or "unknown")
    duration = duration_hint
    _notify_progress(progress_callback, 0.84, "transcription_complete")
    return segments, language, duration


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
        import whisperx
    except ImportError as exc:
        if options.diarization_mode == "on":
            raise RuntimeError(
                "Diarization requested but whisperx is missing. Install with: pip install whisperx"
            ) from exc
        return base_segments

    _notify_progress(progress_callback, 0.88, "diarizing")
    device, _ = _resolve_device_compute(options)
    diarization_device = "cuda" if device == "cuda" else "cpu"

    audio = whisperx.load_audio(str(audio_file))
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=token,
        device=diarization_device,
    )
    diarized = diarize_model(
        audio,
        min_speakers=options.min_speakers,
        max_speakers=options.max_speakers,
    )

    aligned_input = {
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in base_segments
        ]
    }

    assigned = whisperx.assign_word_speakers(diarized, aligned_input)
    assigned_segments = assigned.get("segments", [])
    if not assigned_segments:
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
    for idx, seg_data in enumerate(assigned_segments):
        fallback = base_segments[min(idx, len(base_segments) - 1)]
        raw_speaker = str(seg_data.get("speaker") or fallback.speaker).strip() or "unknown"
        speaker = map_speaker(raw_speaker)
        rebuilt.append(
            TranscriptSegment(
                start=float(seg_data.get("start", fallback.start)),
                end=float(seg_data.get("end", fallback.end)),
                text=str(seg_data.get("text", fallback.text)).strip(),
                speaker=speaker,
                confidence=fallback.confidence,
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
        _notify_progress(progress_callback, 0.05, "normalizing")
        if _is_native_16k_mono_pcm_wav(input_file):
            audio_for_pipeline = input_file
            _notify_progress(progress_callback, 0.10, "normalization_skipped")
        else:
            audio_for_pipeline = _normalize_audio(input_file, temp_dir)
            _notify_progress(progress_callback, 0.10, "normalized")
        segments, language, duration = _transcribe_with_whisper(
            audio_for_pipeline,
            options,
            progress_callback=progress_callback,
        )
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
