from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .exporters import export_docx, export_json, export_srt, export_txt
from .pipeline import transcribe_file
from .types import PipelineOptions
from .utils import ensure_dir, now_job_id


FORMAT_EXPORTERS = {
    "txt": (export_txt, ".txt"),
    "srt": (export_srt, ".srt"),
    "json": (export_json, ".json"),
    "docx": (export_docx, ".docx"),
}


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "audio"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="otter-local",
        description="Transcribe local audio/video files with Whisper and optional diarization.",
    )
    parser.add_argument("inputs", nargs="+", help="Path(s) to input audio/video files")
    parser.add_argument("--out-dir", default="", help="Output root directory")
    parser.add_argument("--job-id", default="", help="Optional custom job ID")

    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--language", default=None, help="Optional language code (e.g. en)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--compute-type",
        choices=["auto", "int8", "int8_float16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filtering")

    parser.add_argument(
        "--diarization",
        choices=["auto", "on", "off"],
        default="auto",
        help="Speaker diarization mode",
    )
    parser.add_argument("--hf-token", default="", help="HF token override")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--speaker-names",
        default="",
        help="Optional comma-separated speaker names (e.g. Alice,Bob)",
    )

    parser.add_argument(
        "--formats",
        default="txt,srt,json,docx",
        help="Comma-separated output formats: txt,srt,json,docx",
    )
    return parser


def _resolve_formats(raw: str) -> list[str]:
    formats = []
    for token in (p.strip().lower() for p in raw.split(",")):
        if not token:
            continue
        if token not in FORMAT_EXPORTERS:
            raise ValueError(f"Unsupported output format: {token}")
        formats.append(token)

    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def _build_options(args: argparse.Namespace) -> PipelineOptions:
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    speaker_names = [name.strip() for name in args.speaker_names.split(",") if name.strip()]
    return PipelineOptions(
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        diarization_mode=args.diarization,
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_names=speaker_names,
    )


def _export_result(result, output_dir: Path, formats: list[str]) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        exporter, ext = FORMAT_EXPORTERS[fmt]
        out_file = output_dir / f"transcript{ext}"
        exporter(result, out_file)
        written.append(out_file)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        formats = _resolve_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))

    job_id = args.job_id or now_job_id()
    out_root = Path(args.out_dir) if args.out_dir else Path("output") / "transcribe" / job_id
    ensure_dir(out_root)

    options = _build_options(args)
    failures = 0

    print(f"Output root: {out_root.resolve()}")
    for input_raw in args.inputs:
        input_file = Path(input_raw).expanduser()
        input_name = _safe_name(input_file.stem)
        item_dir = ensure_dir(out_root / input_name)

        print(f"\nProcessing: {input_file}")
        try:
            result = transcribe_file(input_file, options)
            written = _export_result(result, item_dir, formats)
            print(f"Language: {result.language} | Segments: {len(result.segments)}")
            for f in written:
                print(f"  wrote {f}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {exc}", file=sys.stderr)

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
