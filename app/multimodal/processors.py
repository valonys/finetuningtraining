"""Provider interfaces for modality-specific extraction.

Production deployments should implement these protocols with their chosen
providers: Snowflake Cortex, Whisper, AWS Transcribe, Azure Document
Intelligence, Tesseract, Qwen-VL, Gemini, Bedrock, or internal services.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .schemas import ContentRecord, Modality, SourceRef


class AudioTranscriber(Protocol):
    def transcribe(self, path: str, *, source: SourceRef) -> list[ContentRecord]: ...


class ImageTextExtractor(Protocol):
    def extract_text(self, path: str, *, source: SourceRef) -> list[ContentRecord]: ...


class VideoAnalyzer(Protocol):
    def analyze(self, path: str, *, source: SourceRef) -> list[ContentRecord]: ...


class TextFileProcessor:
    """Default processor for already-textual artifacts and transcripts."""

    def process(
        self,
        path: str,
        *,
        tenant_id: str = "public",
        collection: str = "default",
        source_type: Modality = Modality.TEXT,
        title: str | None = None,
        encoding: str = "utf-8",
    ) -> list[ContentRecord]:
        p = Path(path)
        text = p.read_text(encoding=encoding, errors="replace")
        source = SourceRef(
            source_uri=str(p),
            source_type=source_type,
            tenant_id=tenant_id,
            collection=collection,
            title=title or p.name,
        )
        return [
            ContentRecord(
                record_id=stable_record_id(str(p), source_type.value),
                text=text,
                source=source,
                metadata={"filename": p.name, "bytes": p.stat().st_size if p.exists() else None},
            )
        ]


def stable_record_id(source_uri: str, namespace: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{source_uri}".encode("utf-8")).hexdigest()[:20]
    return f"{namespace}:{digest}"


def records_from_data_forge(
    records,
    *,
    tenant_id: str = "public",
    collection: str = "default",
    source_type_override: Modality | None = None,
) -> list[ContentRecord]:
    """Convert existing Data Forge parser output into multimodal records."""
    out: list[ContentRecord] = []
    for idx, rec in enumerate(records):
        source_type = source_type_override or _modality_from_data_forge(rec.source_type)
        source = SourceRef(
            source_uri=rec.source,
            source_type=source_type,
            tenant_id=tenant_id,
            collection=collection,
            title=Path(rec.source).name,
            page=_first_page(rec.metadata),
            extra={
                "data_forge_source_type": rec.source_type,
                "metadata": rec.metadata,
                "tables": len(getattr(rec, "tables", []) or []),
                "pages": len(getattr(rec, "pages", []) or []),
                "images": len(getattr(rec, "images", []) or []),
            },
        )
        namespace = f"{source_type.value}:{idx}"
        out.append(
            ContentRecord(
                record_id=stable_record_id(rec.source, namespace),
                text=rec.text,
                source=source,
                metadata={
                    "source_type": rec.source_type,
                    "source": rec.source,
                    "tables": len(getattr(rec, "tables", []) or []),
                    "pages": len(getattr(rec, "pages", []) or []),
                },
            )
        )
    return out


def _modality_from_data_forge(source_type: str) -> Modality:
    lowered = (source_type or "").lower()
    if lowered in {"image", "png", "jpg", "jpeg", "webp", "tiff", "bmp"}:
        return Modality.IMAGE
    if lowered in {"pptx", "slide", "slides"}:
        return Modality.SLIDE
    if lowered in {"pdf", "docx", "xlsx", "html"}:
        return Modality.DOCUMENT
    if lowered in {"py", "ipynb", "code"}:
        return Modality.CODE
    return Modality.TEXT


def _first_page(metadata: dict | None) -> int | None:
    if not metadata:
        return None
    for key in ("page", "page_number", "page_index"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
    return None
