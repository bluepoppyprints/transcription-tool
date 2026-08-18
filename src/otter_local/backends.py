from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import wave
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import PipelineOptions, TranscriptSegment


ProgressCallback = Callable[[float, str], bool | None]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(os.getenv("OTTER_LOCAL_MODELS_DIR", PROJECT_ROOT / ".models"))
TOOLS_ROOT = PROJECT_ROOT / ".tools"
_MODEL_CACHE_LOCAL = threading.local()


@dataclass(frozen=True, slots=True)
class BackendSpec:
    backend: str
    label: str
    models: tuple[str, ...]
    default_model: str
    description: str
    model_dir_name: str | None = None


BACKEND_SPECS: dict[str, BackendSpec] = {
    "faster-whisper": BackendSpec(
        backend="faster-whisper",
        label="Faster-Whisper",
        models=("tiny", "base", "small", "medium", "large-v3", "turbo", "distil-large-v3"),
        default_model="small",
        description="Reliable Whisper backend; multilingual except distil-large-v3.",
    ),
    "sherpa-qwen3": BackendSpec(
        backend="sherpa-qwen3",
        label="Qwen3-ASR 0.6B (sherpa-onnx)",
        models=("qwen3-0.6b-int8",),
        default_model="qwen3-0.6b-int8",
        description="Multilingual local Qwen3-ASR using CPU-friendly INT8 ONNX models.",
        model_dir_name="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    ),
    "sherpa-parakeet": BackendSpec(
        backend="sherpa-parakeet",
        label="Parakeet TDT 0.6B v3 (sherpa-onnx)",
        models=("parakeet-tdt-0.6b-v3-int8",),
        default_model="parakeet-tdt-0.6b-v3-int8",
        description="Fast local transcription for 25 European languages, including Polish.",
        model_dir_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    ),
    "vibeasr": BackendSpec(
        backend="vibeasr",
        label="VibeVoice ASR BitNet",
        models=("vibevoice-asr-bitnet",),
        default_model="vibevoice-asr-bitnet",
        description="Experimental Microsoft CPU backend invoked through VibeASR.cpp.",
        model_dir_name="vibeasr",
    ),
}


def _notify(callback: ProgressCallback | None, progress: float, stage: str) -> None:
    if callback is not None and callback(max(0.0, min(1.0, progress)), stage) is False:
        from .pipeline import TranscriptionCancelled

        raise TranscriptionCancelled("Canceled by callback")


def _cache() -> OrderedDict[tuple[Any, ...], Any]:
    cache = getattr(_MODEL_CACHE_LOCAL, "models", None)
    if cache is None:
        cache = OrderedDict()
        _MODEL_CACHE_LOCAL.models = cache
    return cache


def _remember(key: tuple[Any, ...], factory: Callable[[], Any]) -> Any:
    cache = _cache()
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
        return value
    value = factory()
    cache[key] = value
    while len(cache) > 2:
        cache.popitem(last=False)
    return value


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
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return device, compute_type


def _audio_duration(audio_file: Path) -> float:
    with wave.open(str(audio_file), "rb") as wav_f:
        rate = wav_f.getframerate()
        return wav_f.getnframes() / float(rate) if rate else 0.0


def _read_wav_mono(audio_file: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(audio_file), "rb") as wav_f:
        sample_rate = wav_f.getframerate()
        channels = wav_f.getnchannels()
        sample_width = wav_f.getsampwidth()
        raw = wav_f.readframes(wav_f.getnframes())
    if sample_width != 2:
        raise RuntimeError("Local ONNX backends require 16-bit PCM WAV input")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def _chunks(samples: Any, sample_rate: int, seconds: float = 30.0):
    chunk_size = max(1, int(sample_rate * seconds))
    for start in range(0, len(samples), chunk_size):
        end = min(len(samples), start + chunk_size)
        yield start, end, samples[start:end]


def _find_file(model_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        exact = model_dir / name
        if exact.exists():
            return exact
    for name in names:
        matches = sorted(model_dir.glob(f"**/{name}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing model file in {model_dir}: one of {', '.join(names)}")


def _require_model_dir(backend: str) -> Path:
    spec = BACKEND_SPECS[backend]
    assert spec.model_dir_name
    model_dir = MODELS_ROOT / spec.model_dir_name
    if not model_dir.exists():
        raise RuntimeError(
            f"{spec.label} is not installed. Run: "
            f".\\.venv\\Scripts\\python.exe scripts\\install_local_backends.py --backend {backend}"
        )
    return model_dir


def _get_whisper_model(options: PipelineOptions) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is missing; reinstall the base package") from exc

    device, compute_type = _resolve_device_compute(options)
    key = ("faster-whisper", options.model, device, compute_type)

    def load() -> Any:
        try:
            return WhisperModel(options.model, device=device, compute_type=compute_type)
        except Exception as exc:
            message = str(exc)
            if "cannot find the appropriate snapshot" in message or "ConnectError" in message:
                raise RuntimeError(
                    f"Model '{options.model}' is not cached and its download was blocked. "
                    "Start the app normally with run_app.bat, then retry, or run "
                    f"scripts\\install_local_backends.py --whisper-model {options.model}. "
                    f"Original error: {message}"
                ) from exc
            raise

    return _remember(key, load)


def _transcribe_faster_whisper(
    audio_file: Path,
    options: PipelineOptions,
    callback: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], str, float]:
    model = _get_whisper_model(options)
    _notify(callback, 0.12, "transcribing")
    segments_iter, info = model.transcribe(
        str(audio_file),
        language=options.language,
        beam_size=options.beam_size,
        vad_filter=options.vad_filter,
        hotwords=options.hotwords or None,
    )
    duration = float(getattr(info, "duration", 0.0) or _audio_duration(audio_file))
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(segments_iter, start=1):
        text = str(segment.text).strip()
        if text:
            no_speech = getattr(segment, "no_speech_prob", None)
            confidence = None if no_speech is None else max(0.0, min(1.0, 1.0 - float(no_speech)))
            segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    confidence=confidence,
                )
            )
        ratio = float(segment.end) / duration if duration else min(1.0, index / 120.0)
        _notify(callback, 0.12 + min(1.0, ratio) * 0.72, "transcribing")
    return segments, str(info.language or options.language or "unknown"), duration


def _create_sherpa_recognizer(options: PipelineOptions) -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError("sherpa-onnx is missing. Install the local-backends extra.") from exc

    backend = options.backend
    model_dir = _require_model_dir(backend)
    provider = "cuda" if options.device == "cuda" else "cpu"
    threads = max(1, int(options.num_threads))

    if backend == "sherpa-qwen3":
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(_find_file(model_dir, ("conv_frontend.onnx",))),
            encoder=str(_find_file(model_dir, ("encoder.int8.onnx", "encoder.onnx"))),
            decoder=str(_find_file(model_dir, ("decoder.int8.onnx", "decoder.onnx"))),
            tokenizer=str(_find_file(model_dir, ("tokenizer",))),
            num_threads=threads,
            provider=provider,
            max_total_len=1024,
            max_new_tokens=512,
            hotwords=options.hotwords,
        )

    if backend == "sherpa-parakeet":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(_find_file(model_dir, ("encoder.int8.onnx", "encoder.onnx"))),
            decoder=str(_find_file(model_dir, ("decoder.int8.onnx", "decoder.onnx"))),
            joiner=str(_find_file(model_dir, ("joiner.int8.onnx", "joiner.onnx"))),
            tokens=str(_find_file(model_dir, ("tokens.txt",))),
            num_threads=threads,
            provider=provider,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

    raise ValueError(f"Unsupported sherpa backend: {backend}")


def _transcribe_sherpa(
    audio_file: Path,
    options: PipelineOptions,
    callback: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], str, float]:
    model_dir = _require_model_dir(options.backend)
    key = (
        options.backend,
        str(model_dir),
        options.device,
        max(1, int(options.num_threads)),
        options.hotwords,
    )
    recognizer = _remember(key, lambda: _create_sherpa_recognizer(options))
    samples, sample_rate = _read_wav_mono(audio_file)
    duration = len(samples) / float(sample_rate)
    segments: list[TranscriptSegment] = []
    language = options.language or "unknown"
    _notify(callback, 0.12, "transcribing")

    for start_sample, end_sample, audio_chunk in _chunks(samples, sample_rate, seconds=30.0):
        if options.backend == "sherpa-qwen3":
            stream = recognizer.create_stream(hotwords=options.hotwords or None)
        else:
            stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_chunk)
        recognizer.decode_stream(stream)
        result = stream.result
        text = str(getattr(result, "text", "") or "").strip()
        detected = str(getattr(result, "lang", "") or "").strip()
        if detected:
            language = detected
        if text:
            segments.append(
                TranscriptSegment(
                    start=start_sample / float(sample_rate),
                    end=end_sample / float(sample_rate),
                    text=text,
                )
            )
        ratio = end_sample / max(1, len(samples))
        _notify(callback, 0.12 + ratio * 0.72, "transcribing")
    return segments, language, duration


def _vibe_executable() -> Path:
    configured = os.getenv("VIBEASR_EXECUTABLE")
    candidates = [
        Path(configured) if configured else None,
        TOOLS_ROOT / "vibeasr" / "build-otter" / "bin" / "asr_infer.exe",
        TOOLS_ROOT / "vibeasr" / "build" / "bin" / "asr_infer.exe",
        TOOLS_ROOT / "vibeasr" / "build" / "bin" / "asr_infer",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise RuntimeError(
        "VibeASR.cpp is not built. Run scripts\\install_vibeasr.ps1 -Bootstrap "
        "or set VIBEASR_EXECUTABLE to asr_infer.exe."
    )


def _transcribe_vibeasr(
    audio_file: Path,
    options: PipelineOptions,
    callback: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], str, float]:
    model_dir = _require_model_dir("vibeasr")
    vae = _find_file(model_dir, ("vibeasr-vae-encoder-i8_s.gguf",))
    lm = _find_file(model_dir, ("vibeasr-lm-i2_s-embed-q6_k.gguf",))
    duration = _audio_duration(audio_file)
    _notify(callback, 0.15, "transcribing")
    command = [
            str(_vibe_executable()),
            "--vae-model",
            str(vae),
            "--lm-model",
            str(lm),
            "--audio",
            str(audio_file),
            "-t",
            str(max(1, int(options.num_threads))),
        ]
    if options.hotwords:
        command.extend(["--context", options.hotwords])
    process_env = os.environ.copy()
    mingw_candidates = [TOOLS_ROOT / "winlibs" / "mingw64" / "bin"]
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        mingw_candidates.insert(0, Path(local_appdata) / "OtterLocalTools" / "winlibs" / "mingw64" / "bin")
    portable_mingw = None
    for item in mingw_candidates:
        try:
            if item.is_dir():
                portable_mingw = item
                break
        except OSError:
            continue
    if portable_mingw is not None:
        process_env["PATH"] = f"{portable_mingw}{os.pathsep}{process_env.get('PATH', '')}"
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=process_env,
    )
    output = completed.stdout.strip()
    text = output
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            text = str(parsed.get("text") or parsed.get("transcript") or output)
    except json.JSONDecodeError:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        transcript_lines = [
            re.sub(r"^(transcript|transcription)\s*:\s*", "", line, flags=re.I)
            for line in lines
            if not line.startswith("[") and "loading" not in line.lower()
        ]
        text = " ".join(transcript_lines).strip() or output
    _notify(callback, 0.84, "transcription_complete")
    segments: list[TranscriptSegment] = []
    timed_line = re.compile(
        r"^\[\s*([0-9.]+)\s*-\s*([0-9.]+)\s*\]\s*"
        r"(?:Speaker\s+([^:]+):\s*)?(.*)$",
        flags=re.I,
    )
    for line in output.splitlines():
        match = timed_line.match(line.strip())
        if match and match.group(4).strip():
            raw_speaker = (match.group(3) or "1").strip()
            speaker = raw_speaker if raw_speaker.lower().startswith("speaker") else f"Speaker {raw_speaker}"
            segments.append(
                TranscriptSegment(
                    start=float(match.group(1)),
                    end=float(match.group(2)),
                    text=match.group(4).strip(),
                    speaker=speaker,
                )
            )
    if not segments and text:
        segments = [TranscriptSegment(start=0.0, end=duration, text=text)]
    return segments, options.language or "unknown", duration


def transcribe_audio(
    audio_file: Path,
    options: PipelineOptions,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[TranscriptSegment], str, float]:
    if options.backend == "faster-whisper":
        return _transcribe_faster_whisper(audio_file, options, progress_callback)
    if options.backend in {"sherpa-qwen3", "sherpa-parakeet"}:
        return _transcribe_sherpa(audio_file, options, progress_callback)
    if options.backend == "vibeasr":
        return _transcribe_vibeasr(audio_file, options, progress_callback)
    raise ValueError(f"Unsupported ASR backend: {options.backend}")


def backend_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in BACKEND_SPECS.values():
        installed = True
        detail = "Ready"
        if spec.backend == "faster-whisper":
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                installed = False
                detail = "Python package is missing"
        elif spec.backend in {"sherpa-qwen3", "sherpa-parakeet"}:
            try:
                import sherpa_onnx  # noqa: F401
            except ImportError:
                installed = False
                detail = "sherpa-onnx package is missing"
            model_dir = MODELS_ROOT / str(spec.model_dir_name)
            if not model_dir.exists():
                installed = False
                detail = "Model files are not installed"
        else:
            model_dir = MODELS_ROOT / str(spec.model_dir_name)
            try:
                _vibe_executable()
            except RuntimeError as exc:
                installed = False
                detail = str(exc)
            if not model_dir.exists():
                installed = False
                detail = "VibeASR model files are not installed"
        row = asdict(spec)
        row.update({"installed": installed, "status": detail})
        rows.append(row)
    return rows
