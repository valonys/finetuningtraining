"""Chunking utilities for normalized multimodal text."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .schemas import ContentChunk, ContentRecord

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_records(
    records: Iterable[ContentRecord],
    *,
    target_chars: int = 1200,
    overlap_chars: int = 160,
) -> list[ContentChunk]:
    """Split records into retrieval-sized chunks.

    The chunker prefers sentence boundaries, but it can still handle long
    ASR/VLM/OCR runs without punctuation by falling back to character windows.
    """
    if target_chars < 200:
        raise ValueError("target_chars must be >= 200")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than target_chars")

    chunks: list[ContentChunk] = []
    for record in records:
        text = _normalize_text(record.text)
        if not text:
            continue
        parts = _split_text(text, target_chars=target_chars, overlap_chars=overlap_chars)
        for idx, part in enumerate(parts):
            chunk_id = _stable_chunk_id(record.record_id, idx, part)
            metadata = dict(record.metadata)
            metadata["char_count"] = len(part)
            chunks.append(
                ContentChunk(
                    chunk_id=chunk_id,
                    record_id=record.record_id,
                    text=part,
                    source=record.source,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )
    return chunks


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _split_text(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]

    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    if len(sentences) == 1:
        return _window_text(text, target_chars=target_chars, overlap_chars=overlap_chars)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = _tail(current, overlap_chars)
            current = f"{current} {sentence}".strip() if current else sentence
        while len(current) > target_chars:
            chunks.append(current[:target_chars].strip())
            current = current[target_chars - overlap_chars :].strip()
    if current:
        chunks.append(current)
    return chunks


def _window_text(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    out: list[str] = []
    step = target_chars - overlap_chars
    for start in range(0, len(text), step):
        chunk = text[start : start + target_chars].strip()
        if chunk:
            out.append(chunk)
        if start + target_chars >= len(text):
            break
    return out


def _tail(text: str, chars: int) -> str:
    if chars <= 0:
        return ""
    return text[-chars:].strip()


def _stable_chunk_id(record_id: str, idx: int, text: str) -> str:
    digest = hashlib.sha256(f"{record_id}:{idx}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{record_id}:{idx}:{digest}"
