"""
Semantic chunker.

Strategy (in order):
  1. Split on Markdown-ish headings (# / ## / ###).
  2. Split on blank-line paragraph boundaries.
  3. Merge adjacent paragraphs until ~`target_chars` characters.
  4. Respect a `max_chars` ceiling and a `min_chars` floor.

No embedding model required. The output is a plain list[str].
"""
from __future__ import annotations

import re
from typing import Iterable


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_text(
    text: str,
    *,
    target_chars: int = 1200,
    max_chars: int = 1800,
    min_chars: int = 200,
) -> list[str]:
    """Return a list of chunks suitable for Q/A synthesis."""
    if not text.strip():
        return []

    # 1. Section split on Markdown headings (keeps each section intact)
    sections = _split_on_headings(text)

    # 2. Paragraph packing within each section
    chunks: list[str] = []
    for sec in sections:
        paras = [p.strip() for p in re.split(r"\n{2,}", sec) if p.strip()]
        buf: list[str] = []
        buf_len = 0
        for p in paras:
            # If a single paragraph is way too long, split by sentences
            if len(p) > max_chars:
                if buf:
                    chunks.append("\n\n".join(buf))
                    buf, buf_len = [], 0
                chunks.extend(_split_by_sentences(p, max_chars))
                continue

            if buf_len + len(p) > target_chars and buf_len >= min_chars:
                chunks.append("\n\n".join(buf))
                buf, buf_len = [p], len(p)
            else:
                buf.append(p)
                buf_len += len(p)
        if buf:
            chunks.append("\n\n".join(buf))

    # Drop empty / tiny fragments
    return [c.strip() for c in chunks if len(c.strip()) >= 40]


def chunk_records(records: Iterable, **kw) -> list[dict]:
    """
    Convenience wrapper: take IngestedRecord(s) and return a flat list of
    {"source": ..., "chunk": ..., "chunk_index": ...} dicts.
    """
    out: list[dict] = []
    for rec in records:
        chunks = chunk_text(rec.text, **kw)
        for i, c in enumerate(chunks):
            out.append({
                "source": rec.source,
                "source_type": rec.source_type,
                "chunk": c,
                "chunk_index": i,
            })
    return out


# ──────────────────────────────────────────────────────────────
def _split_on_headings(text: str) -> list[str]:
    positions = [m.start() for m in _HEADING_RE.finditer(text)]
    if not positions:
        return [text]
    positions = [0] + positions + [len(text)]
    return [text[positions[i]:positions[i + 1]] for i in range(len(positions) - 1)]


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) < max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out
