from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.getenv("OTTER_LOCAL_MODELS_DIR", PROJECT_ROOT / ".models"))

ARCHIVES = {
    "sherpa-qwen3": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    ),
    "sherpa-parakeet": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    ),
}


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    percent = min(100, int((blocks * block_size * 100) / total))
    if percent != getattr(_progress, "last_percent", None):
        _progress.last_percent = percent
        print(f"\rDownloading: {percent:3d}%", end="", flush=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:bz2") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination, filter="data")


def install_archive_backend(backend: str) -> Path:
    url, directory_name = ARCHIVES[backend]
    target = MODELS_ROOT / directory_name
    if target.exists():
        print(f"{backend}: already installed at {target}")
        return target

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="otter-backend-", dir=MODELS_ROOT) as temp_raw:
        temp_dir = Path(temp_raw)
        archive = temp_dir / Path(url).name
        print(f"Downloading {backend} from the official sherpa-onnx release...")
        _progress.last_percent = None
        urllib.request.urlretrieve(url, archive, reporthook=_progress)
        print()
        print("Extracting model files...")
        _safe_extract(archive, temp_dir)
        extracted = temp_dir / directory_name
        if not extracted.exists():
            candidates = [item for item in temp_dir.iterdir() if item.is_dir()]
            if len(candidates) != 1:
                raise RuntimeError(f"Could not locate extracted model folder for {backend}")
            extracted = candidates[0]
        shutil.move(str(extracted), str(target))

    print(f"{backend}: installed at {target}")
    return target


def install_whisper_model(model: str) -> Path:
    try:
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required") from exc

    print(f"Caching Faster-Whisper model {model}...")
    path = Path(download_model(model))
    print(f"Whisper model cached at {path}")
    return path


def install_vibe_model() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required") from exc

    target = MODELS_ROOT / "vibeasr"
    target.mkdir(parents=True, exist_ok=True)
    print("Downloading Microsoft VibeVoice-ASR-BitNet model files...")
    snapshot_download(
        "microsoft/VibeVoice-ASR-BitNet",
        local_dir=target,
        allow_patterns=[
            "vibeasr-vae-encoder-i8_s.gguf",
            "vibeasr-lm-i2_s-embed-q6_k.gguf",
        ],
    )
    print(f"VibeASR model installed at {target}")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install optional fully local ASR model bundles.")
    parser.add_argument(
        "--backend",
        action="append",
        choices=["sherpa-qwen3", "sherpa-parakeet", "vibeasr", "all"],
        default=[],
        help="Backend model bundle to install; may be repeated.",
    )
    parser.add_argument(
        "--whisper-model",
        action="append",
        default=[],
        help="Faster-Whisper model to preload, e.g. turbo or large-v3.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested = set(args.backend)
    if "all" in requested:
        requested = {"sherpa-qwen3", "sherpa-parakeet", "vibeasr"}
    if not requested and not args.whisper_model:
        build_parser().print_help()
        return 2

    try:
        for backend in sorted(requested):
            if backend in ARCHIVES:
                install_archive_backend(backend)
            elif backend == "vibeasr":
                install_vibe_model()
        for model in args.whisper_model:
            install_whisper_model(model)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
