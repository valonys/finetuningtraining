"""
app/data_forge/ingest.py
────────────────────────
Entry point for the Data Forge.

Routes each file to the right parser, returns a list of `IngestedRecord`s,
and — optionally — builds a training dataset (SFT / DPO / GRPO) using the
correct chat template for the chosen base model.

Supported extensions:
    .pdf .txt .md .rst
    .docx
    .pptx
    .xlsx .xls .csv .tsv
    .html .htm .xhtml
    .png .jpg .jpeg .webp .tiff .bmp       (→ OCR)
    .json .jsonl                             (passthrough)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

logger = logging.getLogger(__name__)

# ── Parser imports are lazy to avoid heavy deps when not needed ──
_PARSER_ROUTES: dict[str, str] = {
    # ext → module path
    ".pdf":  "app.data_forge.parsers.pdf:parse_pdf",
    ".txt":  "app.data_forge.parsers.txt:parse_txt",
    ".md":   "app.data_forge.parsers.txt:parse_txt",
    ".rst":  "app.data_forge.parsers.txt:parse_txt",
    ".docx": "app.data_forge.parsers.docx:parse_docx",
    ".pptx": "app.data_forge.parsers.pptx:parse_pptx",
    ".xlsx": "app.data_forge.parsers.xlsx:parse_xlsx",
    ".xls":  "app.data_forge.parsers.xlsx:parse_xlsx",
    ".csv":  "app.data_forge.parsers.xlsx:parse_csv",
    ".tsv":  "app.data_forge.parsers.xlsx:parse_csv",
    ".html": "app.data_forge.parsers.html:parse_html",
    ".htm":  "app.data_forge.parsers.html:parse_html",
    ".xhtml":"app.data_forge.parsers.html:parse_html",
    ".png":  "app.data_forge.parsers.image:parse_image",
    ".jpg":  "app.data_forge.parsers.image:parse_image",
    ".jpeg": "app.data_forge.parsers.image:parse_image",
    ".webp": "app.data_forge.parsers.image:parse_image",
    ".tiff": "app.data_forge.parsers.image:parse_image",
    ".tif":  "app.data_forge.parsers.image:parse_image",
    ".bmp":  "app.data_forge.parsers.image:parse_image",
    ".json": "app.data_forge.parsers.txt:parse_json_passthrough",
    ".jsonl":"app.data_forge.parsers.txt:parse_json_passthrough",
}


@dataclass
class IngestedRecord:
    """A normalised document record.

    Every parser in `parsers/` returns a list of these.
    """
    source: str                                 # absolute path
    source_type: str                            # "pdf" | "docx" | "image" | ...
    text: str                                   # plain text (possibly after OCR)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional structured data — tables, layouts, pages
    tables: list[list[list[str]]] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────
class DataForge:

    def __init__(self, ocr_engine: Optional[str] = None):
        """
        Args:
            ocr_engine: override OCR engine. One of:
                "rapidocr" | "paddleocr" | "docling" | "tesseract" | "trocr"
                If None, the default is picked from the hardware profile.
        """
        self.ocr_engine_override = ocr_engine

    # ── Public API ────────────────────────────────────────────
    def ingest(
        self,
        source: Union[str, Path, Iterable[Union[str, Path]]],
    ) -> list[IngestedRecord]:
        """Ingest one path (file or dir) or an iterable of paths."""
        if isinstance(source, (str, Path)):
            return self._ingest_one(Path(source))
        out: list[IngestedRecord] = []
        for s in source:
            out.extend(self._ingest_one(Path(s)))
        return out

    def build_dataset(self, records: list[IngestedRecord], **kwargs):
        """Delegates to DatasetBuilder — see dataset_builder.py."""
        from .dataset_builder import DatasetBuilder
        return DatasetBuilder(**kwargs).build(records)

    # ── Private ───────────────────────────────────────────────
    def _ingest_one(self, path: Path) -> list[IngestedRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Data Forge: {path} not found")

        if path.is_dir():
            records: list[IngestedRecord] = []
            for sub in sorted(path.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in _PARSER_ROUTES:
                    try:
                        records.extend(self._ingest_one(sub))
                    except Exception as e:
                        logger.warning(f"⚠️  Skipping {sub}: {e}")
            return records

        ext = path.suffix.lower()
        route = _PARSER_ROUTES.get(ext)
        if route is None:
            raise ValueError(f"Data Forge: unsupported file type '{ext}' ({path})")

        func = _load_func(route)
        logger.info(f"📥 Ingesting {path.name} via {route}")
        kwargs: dict[str, Any] = {}
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}:
            kwargs["ocr_engine"] = self.ocr_engine_override
        if ext == ".pdf":
            kwargs["ocr_engine"] = self.ocr_engine_override   # PDFs may fall back to OCR

        result = func(str(path), **kwargs)
        if isinstance(result, IngestedRecord):
            return [result]
        return list(result)


# ──────────────────────────────────────────────────────────────
def _load_func(route: str):
    module_path, func_name = route.split(":")
    mod = __import__(module_path, fromlist=[func_name])
    return getattr(mod, func_name)
