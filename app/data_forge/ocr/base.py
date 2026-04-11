"""
OCR engine interface. Every concrete backend implements `run(image)` and
returns an `OCRResult`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRResult:
    """A normalised OCR output, identical across engines."""
    text: str
    engine: str
    confidence: float = 0.0          # mean word confidence, 0..1
    tables: list[list[list[str]]] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    # blocks: [{"text": str, "bbox": [x1,y1,x2,y2], "conf": float, ...}, ...]


class OCREngine(ABC):
    name: str = "base"

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Return True iff the backend's dependency is importable at runtime."""

    @abstractmethod
    def run(self, image) -> OCRResult:
        """Run OCR on a PIL Image. Returns an OCRResult."""
