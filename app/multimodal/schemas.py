"""Core schemas for enterprise multimodal pipelines.

The course notebooks normalize audio transcripts, slide OCR, and VLM video
descriptions into one table. These schemas make that normalization explicit
and backend-neutral so Snowflake, local Python, or a managed vector DB can all
use the same contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    SLIDE = "slide"
    VIDEO = "video"
    DOCUMENT = "document"
    CODE = "code"


@dataclass(frozen=True)
class SourceRef:
    """Stable provenance for a derived piece of multimodal content."""

    source_uri: str
    source_type: Modality
    tenant_id: str = "public"
    collection: str = "default"
    title: str | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    page: int | None = None
    frame: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentRecord:
    """A normalized text-bearing unit before chunking.

    Examples:
      * ASR output from an audio file.
      * OCR text from a slide/image.
      * VLM description for a video segment.
      * Existing document text from Data Forge.
    """

    record_id: str
    text: str
    source: SourceRef
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentChunk:
    """A chunk ready for embedding and retrieval."""

    chunk_id: str
    record_id: str
    text: str
    source: SourceRef
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with a cosine-similarity score."""

    chunk: ContentChunk
    score: float


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime knobs that should stay stable across stack integrations."""

    tenant_id: str = "public"
    collection: str = "default"
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 160
    embedding_dim: int = 384
    default_top_k: int = 8
    max_context_chars: int = 12000

    def validate(self) -> None:
        if self.chunk_target_chars < 200:
            raise ValueError("chunk_target_chars must be >= 200")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be >= 0")
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_target_chars")
        if self.embedding_dim < 16:
            raise ValueError("embedding_dim must be >= 16")
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")
        if self.max_context_chars < 1000:
            raise ValueError("max_context_chars must be >= 1000")
