"""Plain text, Markdown, RST, and JSONL passthrough parsers."""
from __future__ import annotations

import json
from typing import Iterator

from ..ingest import IngestedRecord


def parse_txt(path: str, **_kw) -> IngestedRecord:
    """Read a text file with encoding sniffing."""
    try:
        import chardet
    except ImportError:
        chardet = None

    with open(path, "rb") as f:
        raw = f.read()

    enc = "utf-8"
    if chardet is not None:
        detected = chardet.detect(raw)
        enc = detected.get("encoding") or "utf-8"

    try:
        text = raw.decode(enc, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    return IngestedRecord(
        source=path,
        source_type="text",
        text=text,
        metadata={"encoding": enc, "bytes": len(raw)},
    )


def parse_json_passthrough(path: str, **_kw) -> Iterator[IngestedRecord]:
    """
    Passthrough for JSON / JSONL files that already hold training data.

    Each top-level JSON array element or JSONL line becomes an IngestedRecord
    whose `metadata` contains the raw object and whose `text` holds a canonical
    string rendering for quick previews.
    """
    ext = path.lower().split(".")[-1]
    with open(path, "r", encoding="utf-8") as f:
        if ext == "jsonl":
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield IngestedRecord(
                    source=path,
                    source_type="jsonl",
                    text=_stringify(obj),
                    metadata={"index": i, "raw": obj},
                )
        else:
            data = json.load(f)
            if isinstance(data, list):
                for i, obj in enumerate(data):
                    yield IngestedRecord(
                        source=path,
                        source_type="json",
                        text=_stringify(obj),
                        metadata={"index": i, "raw": obj},
                    )
            else:
                yield IngestedRecord(
                    source=path,
                    source_type="json",
                    text=_stringify(data),
                    metadata={"raw": data},
                )


def _stringify(obj) -> str:
    if isinstance(obj, dict):
        # Try common instruction-response keys
        for k_in in ("input", "question", "instruction", "prompt"):
            for k_out in ("output", "answer", "response", "completion"):
                if k_in in obj and k_out in obj:
                    return f"{obj[k_in]}\n\n→ {obj[k_out]}"
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)
