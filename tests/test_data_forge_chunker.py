"""Chunker sanity tests — no file I/O, no downloads."""
from __future__ import annotations

from app.data_forge.chunker import chunk_text


def test_empty_input():
    assert chunk_text("") == []


def test_small_input_single_chunk():
    out = chunk_text("This is a short paragraph that should fit in one chunk. " * 3)
    assert len(out) == 1


def test_heading_split():
    text = "\n".join([
        "# First section",
        "Paragraph one." * 30,
        "",
        "## Second section",
        "Paragraph two." * 30,
    ])
    chunks = chunk_text(text, target_chars=200, max_chars=400)
    # Expect at least 2 chunks (one per heading)
    assert len(chunks) >= 2


def test_drops_tiny_fragments():
    out = chunk_text("tiny")
    assert out == []
