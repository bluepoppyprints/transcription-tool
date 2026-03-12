from __future__ import annotations

import json
import re
from pathlib import Path

from .types import TranscriptResult, TranscriptSegment
from .utils import format_timestamp, srt_timestamp


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


_WORD_EDGE_RE = re.compile(r"^\W+|\W+$")


def _normalize_words(text: str) -> tuple[list[str], list[str]]:
    words = _clean_text(text).split()
    normalized = [_WORD_EDGE_RE.sub("", word).lower() for word in words]
    paired = [(word, norm) for word, norm in zip(words, normalized) if norm]
    if not paired:
        return [], []
    filtered_words = [item[0] for item in paired]
    filtered_norm = [item[1] for item in paired]
    return filtered_words, filtered_norm


def _is_subsequence_in_order(longer: list[str], shorter: list[str]) -> bool:
    if not shorter:
        return True
    if len(shorter) > len(longer):
        return False
    span = len(shorter)
    for idx in range(len(longer) - span + 1):
        if longer[idx : idx + span] == shorter:
            return True
    return False


def _shared_prefix_words_count(a_words: list[str], b_words: list[str]) -> int:
    count = 0
    for left, right in zip(a_words, b_words):
        if left != right:
            break
        count += 1
    return count


def _looks_like_hypothesis_update(base_text: str, next_text: str, min_words: int = 4) -> bool:
    _, base_norm = _normalize_words(base_text)
    _, next_norm = _normalize_words(next_text)
    if not base_norm or not next_norm:
        return False

    prefix_words = _shared_prefix_words_count(base_norm, next_norm)
    if prefix_words >= min_words:
        return True

    shorter, longer = (base_norm, next_norm) if len(base_norm) <= len(next_norm) else (next_norm, base_norm)
    if len(shorter) < min_words:
        return False
    return _is_subsequence_in_order(longer, shorter)


def _merge_with_overlap(base_text: str, next_text: str, min_overlap_words: int = 3) -> str:
    base_clean = _clean_text(base_text)
    next_clean = _clean_text(next_text)
    if not base_clean:
        return next_clean
    if not next_clean:
        return base_clean

    base_words, base_norm = _normalize_words(base_clean)
    next_words, next_norm = _normalize_words(next_clean)
    if not base_words:
        return next_clean
    if not next_words:
        return base_clean

    if base_norm == next_norm:
        return base_clean

    # If the next segment is already present inside the base text, skip it.
    if len(next_norm) <= len(base_norm) and _is_subsequence_in_order(base_norm, next_norm):
        return base_clean

    max_overlap = min(len(base_norm), len(next_norm), 32)
    overlap = 0
    for size in range(max_overlap, min_overlap_words - 1, -1):
        if base_norm[-size:] == next_norm[:size]:
            overlap = size
            break

    if overlap > 0:
        merged_words = base_words + next_words[overlap:]
        return _clean_text(" ".join(merged_words))

    return _clean_text(f"{base_clean} {next_clean}")


def collapse_segments(
    segments: list[TranscriptSegment],
    merge_gap_seconds: float = 1.0,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    collapsed: list[TranscriptSegment] = []
    current = TranscriptSegment(
        start=ordered[0].start,
        end=ordered[0].end,
        text=_clean_text(ordered[0].text),
        speaker=ordered[0].speaker,
        confidence=ordered[0].confidence,
    )

    for seg in ordered[1:]:
        gap = max(0.0, seg.start - current.end)
        same_speaker = seg.speaker == current.speaker
        hypothesis_like = same_speaker and gap <= 4.0 and _looks_like_hypothesis_update(current.text, seg.text)
        if same_speaker and (gap <= merge_gap_seconds or hypothesis_like):
            current.end = max(current.end, seg.end)
            current.text = _merge_with_overlap(current.text, seg.text)
            if current.confidence is not None and seg.confidence is not None:
                current.confidence = (current.confidence + seg.confidence) / 2.0
            elif seg.confidence is not None:
                current.confidence = seg.confidence
            continue

        collapsed.append(current)
        current = TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=_clean_text(seg.text),
            speaker=seg.speaker,
            confidence=seg.confidence,
        )

    collapsed.append(current)
    deduped: list[TranscriptSegment] = []
    for seg in collapsed:
        if not deduped:
            deduped.append(seg)
            continue

        prev = deduped[-1]
        gap = max(0.0, seg.start - prev.end)
        if seg.speaker == prev.speaker and gap <= 6.0 and _looks_like_hypothesis_update(prev.text, seg.text):
            prev.end = max(prev.end, seg.end)
            prev.text = _merge_with_overlap(prev.text, seg.text, min_overlap_words=2)
            if prev.confidence is not None and seg.confidence is not None:
                prev.confidence = (prev.confidence + seg.confidence) / 2.0
            elif seg.confidence is not None:
                prev.confidence = seg.confidence
            continue

        deduped.append(seg)

    return deduped


def export_txt(result: TranscriptResult, out_file: Path) -> None:
    utterances = collapse_segments(result.segments)
    lines = []
    for seg in utterances:
        stamp = format_timestamp(seg.start)
        lines.append(f"[{stamp}] {seg.speaker}: {seg.text}")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_srt(result: TranscriptResult, out_file: Path) -> None:
    lines: list[str] = []
    utterances = collapse_segments(result.segments, merge_gap_seconds=0.3)
    for idx, seg in enumerate(utterances, start=1):
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}")
        lines.append(f"{seg.speaker}: {seg.text}")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def export_json(result: TranscriptResult, out_file: Path) -> None:
    payload = {
        "source_file": str(result.source_file),
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in result.segments
        ],
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_docx(result: TranscriptResult, out_file: Path) -> None:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "DOCX export requires python-docx. Install with: pip install python-docx"
        ) from exc

    doc = Document()
    doc.add_heading("Transcript", level=1)
    meta = doc.add_paragraph()
    meta.add_run("Source: ").bold = True
    meta.add_run(str(result.source_file))
    meta.add_run("\nLanguage: ").bold = True
    meta.add_run(result.language)
    meta.add_run("\nDuration: ").bold = True
    meta.add_run(format_timestamp(result.duration))

    doc.add_paragraph("")
    for seg in collapse_segments(result.segments):
        p = doc.add_paragraph()
        p.add_run(f"[{format_timestamp(seg.start)}] {seg.speaker}: ").bold = True
        p.add_run(seg.text)

    doc.save(out_file)
