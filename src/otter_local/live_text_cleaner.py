from __future__ import annotations

import re
from pathlib import Path


LINE_PATTERN = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$")
WORD_EDGE_RE = re.compile(r"^\W+|\W+$")


def _clean_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_token(token: str) -> str:
    return WORD_EDGE_RE.sub("", token).lower()


def _remove_adjacent_repeated_blocks(tokens: list[str], min_block: int = 3, max_block: int = 24) -> list[str]:
    if len(tokens) < min_block * 2:
        return tokens

    normalized = [_normalize_token(token) for token in tokens]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(tokens):
            remaining = len(tokens) - i
            max_size = min(max_block, remaining // 2)
            found = False
            for block in range(max_size, min_block - 1, -1):
                left = normalized[i : i + block]
                right = normalized[i + block : i + 2 * block]
                if not left or not right:
                    continue
                if left == right and any(left):
                    del tokens[i + block : i + 2 * block]
                    del normalized[i + block : i + 2 * block]
                    changed = True
                    found = True
                    break
            if not found:
                i += 1
    return tokens


def _dedupe_prefix_fragments(text: str) -> str:
    fragments = [part.strip() for part in re.split(r"\.\.\.+", text) if part.strip()]
    if len(fragments) <= 1:
        return _clean_spaces(text)

    normalized_fragments = [_clean_spaces(part) for part in fragments]
    keep = [True] * len(normalized_fragments)
    for i, item in enumerate(normalized_fragments):
        item_lower = item.lower()
        for j in range(i + 1, len(normalized_fragments)):
            candidate_lower = normalized_fragments[j].lower()
            if candidate_lower.startswith(item_lower) and len(candidate_lower) > len(item_lower):
                keep[i] = False
                break

    kept = [frag for idx, frag in enumerate(normalized_fragments) if keep[idx]]
    if not kept:
        kept = [normalized_fragments[-1]]
    return _clean_spaces(" ".join(kept))


def _extract_full_sentences(text: str) -> list[str]:
    content = _clean_spaces(text)
    if not content:
        return []

    sentences: list[str] = []
    buffer: list[str] = []
    for idx, char in enumerate(content):
        buffer.append(char)
        if char not in ".!?":
            continue

        prev = content[idx - 1] if idx > 0 else ""
        nxt = content[idx + 1] if idx + 1 < len(content) else ""
        if char == "." and (prev == "." or nxt == "."):
            continue

        sentence = _clean_spaces("".join(buffer))
        if sentence:
            sentences.append(sentence)
        buffer = []

    filtered: list[str] = []
    for sentence in sentences:
        if len([w for w in sentence.split() if _normalize_token(w)]) >= 3:
            filtered.append(sentence)
    return filtered


def clean_transcript_text(raw_text: str) -> tuple[str, list[str]]:
    normalized = _dedupe_prefix_fragments(raw_text)
    tokens = normalized.split()
    tokens = _remove_adjacent_repeated_blocks(tokens)
    merged = _clean_spaces(" ".join(tokens))
    merged = re.sub(r"\s+([,;:!?])", r"\1", merged)
    merged = re.sub(r"\s+\.", ".", merged)
    full_sentences = _extract_full_sentences(merged)
    return merged, full_sentences


def clean_line(line: str, keep_partial: bool) -> str | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None

    match = LINE_PATTERN.match(line)
    if match:
        timestamp = match.group("ts")
        speaker = match.group("speaker").strip()
        text = match.group("text")
        cleaned_text, full_sentences = clean_transcript_text(text)
        if full_sentences:
            return f"[{timestamp}] {speaker}: {' '.join(full_sentences)}"
        if keep_partial and cleaned_text:
            return f"[{timestamp}] {speaker}: {cleaned_text}"
        return None

    cleaned_text, full_sentences = clean_transcript_text(line)
    if full_sentences:
        return " ".join(full_sentences)
    if keep_partial and cleaned_text:
        return cleaned_text
    return None


def clean_transcript_lines(lines: list[str], keep_partial: bool = False) -> tuple[list[str], int]:
    cleaned_lines: list[str] = []
    dropped = 0
    for line in lines:
        cleaned = clean_line(line, keep_partial=keep_partial)
        if cleaned is None:
            dropped += 1
            continue
        cleaned_lines.append(cleaned)
    return cleaned_lines, dropped


def clean_transcript_file(
    input_path: Path,
    output_path: Path | None = None,
    keep_partial: bool = False,
) -> dict:
    src = input_path.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    out = src if output_path is None else output_path.expanduser().resolve()
    lines = src.read_text(encoding="utf-8").splitlines()
    cleaned_lines, dropped = clean_transcript_lines(lines, keep_partial=keep_partial)

    payload = "\n".join(cleaned_lines)
    if payload:
        payload += "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")

    return {
        "input_lines": len(lines),
        "kept_lines": len(cleaned_lines),
        "dropped_lines": dropped,
        "output_path": out,
        "keep_partial": bool(keep_partial),
    }
