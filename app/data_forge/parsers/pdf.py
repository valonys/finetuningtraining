"""
PDF parser.

Strategy:
  1. PyMuPDF (fitz) for fast text extraction + layout.
  2. pdfplumber for table extraction (better structure).
  3. If a page has *no* extractable text, fall back to image OCR
     (render page → run OCR engine → merge the OCR text).

We return **one IngestedRecord per file** whose `pages` field holds per-page
details (text, tables, ocr_used).
"""
from __future__ import annotations

import io
import logging
from typing import Any

from ..ingest import IngestedRecord

logger = logging.getLogger(__name__)


def parse_pdf(path: str, ocr_engine: str | None = None, **_kw) -> IngestedRecord:
    pages: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    all_tables: list[list[list[str]]] = []

    # ── PyMuPDF primary pass ────────────────────────────────
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("pymupdf not installed — `pip install pymupdf`") from e

    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None

    doc = fitz.open(path)
    try:
        for pg_idx, page in enumerate(doc):
            txt = page.get_text("text") or ""
            ocr_used = False
            page_tables: list[list[list[str]]] = []

            # Fallback: image-OCR the rendered page if text layer is empty
            if not txt.strip():
                ocr_text = _ocr_rendered_page(page, ocr_engine)
                if ocr_text:
                    txt = ocr_text
                    ocr_used = True

            # Table extraction via pdfplumber (best-effort)
            if pdfplumber is not None:
                page_tables = _extract_tables_pdfplumber(path, pg_idx)

            pages.append({
                "page": pg_idx + 1,
                "text": txt,
                "tables": page_tables,
                "ocr_used": ocr_used,
            })
            full_text_parts.append(txt)
            all_tables.extend(page_tables)
    finally:
        doc.close()

    return IngestedRecord(
        source=path,
        source_type="pdf",
        text="\n\n".join(full_text_parts).strip(),
        tables=all_tables,
        pages=pages,
        metadata={"num_pages": len(pages)},
    )


# ──────────────────────────────────────────────────────────────
def _ocr_rendered_page(page, ocr_engine: str | None) -> str:
    """Render a PyMuPDF page to a PIL image and run OCR."""
    try:
        from PIL import Image
    except ImportError:
        return ""
    pix = page.get_pixmap(dpi=200)            # 200 DPI is good for text OCR
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    from ..ocr.pipeline import run_ocr
    try:
        return run_ocr(img, engine=ocr_engine).text
    except Exception as e:
        logger.warning(f"⚠️  OCR fallback failed: {e}")
        return ""


def _extract_tables_pdfplumber(path: str, page_index: int) -> list[list[list[str]]]:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pg = pdf.pages[page_index]
            tables = pg.extract_tables() or []
        # pdfplumber returns list[list[list[str|None]]]; normalise Nones
        return [[[cell or "" for cell in row] for row in tbl] for tbl in tables]
    except Exception:
        return []
