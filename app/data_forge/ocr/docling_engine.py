"""
Docling backend — IBM's layout-aware document conversion. Strong for
multi-column PDFs, tables, and technical reports. Datacenter-quality.
"""
from __future__ import annotations

import tempfile

from .base import OCREngine, OCRResult


class DoclingEngine(OCREngine):
    name = "docling"

    def __init__(self):
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()

    @classmethod
    def available(cls) -> bool:
        try:
            import docling  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, image) -> OCRResult:
        # Docling operates on file paths. Persist the PIL image first.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            result = self._converter.convert(f.name)

        doc = result.document
        markdown = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""

        # Best-effort table extraction — Docling keeps structured tables
        tables: list[list[list[str]]] = []
        for tbl in getattr(doc, "tables", []) or []:
            try:
                rows = [[cell.text or "" for cell in row.cells] for row in tbl.rows]
                tables.append(rows)
            except Exception:
                continue

        return OCRResult(
            text=markdown,
            engine=self.name,
            confidence=1.0,      # Docling does not report word confidences
            tables=tables,
        )
