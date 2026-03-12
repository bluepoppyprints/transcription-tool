from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from otter_local.live_text_cleaner import clean_transcript_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clean_live_transcript",
        description="Remove repeated live-ASR fragments and keep only full sentences.",
    )
    parser.add_argument("input", type=Path, help="Input transcript file (txt)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Defaults to <input>.cleaned.txt",
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Keep cleaned partial text when no full sentence is detected.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path: Path
    if args.out is None:
        output_path = input_path.with_suffix(".cleaned.txt")
    else:
        output_path = args.out.expanduser().resolve()

    stats = clean_transcript_file(
        input_path=input_path,
        output_path=output_path,
        keep_partial=bool(args.keep_partial),
    )

    print(f"Input lines: {stats['input_lines']}")
    print(f"Kept lines: {stats['kept_lines']}")
    print(f"Dropped lines: {stats['dropped_lines']}")
    print(f"Output: {stats['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
