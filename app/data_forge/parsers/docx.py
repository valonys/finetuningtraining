"""DOCX parser (python-docx). Extracts paragraphs, tables, and headings."""
from __future__ import annotations

from ..ingest import IngestedRecord


def parse_docx(path: str, **_kw) -> IngestedRecord:
    try:
        from docx import Document
    except ImportError as e:
        raise RuntimeError("python-docx not installed — `pip install python-docx`") from e

    doc = Document(path)
    parts: list[str] = []
    tables: list[list[list[str]]] = []

    for block in _iter_block_items(doc):
        if block.__class__.__name__ == "Paragraph":
            style = getattr(block.style, "name", "") or ""
            txt = (block.text or "").strip()
            if not txt:
                continue
            if style.startswith("Heading"):
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 1
                parts.append("#" * max(1, min(6, level)) + " " + txt)
            else:
                parts.append(txt)
        elif block.__class__.__name__ == "Table":
            rows: list[list[str]] = []
            for row in block.rows:
                rows.append([cell.text.strip() for cell in row.cells])
            tables.append(rows)
            # Inline markdown-ish preview
            parts.append("\n".join([" | ".join(r) for r in rows]))

    return IngestedRecord(
        source=path,
        source_type="docx",
        text="\n\n".join(parts),
        tables=tables,
        metadata={"num_tables": len(tables)},
    )


def _iter_block_items(parent):
    """Yield paragraphs and tables in document order."""
    from docx.document import Document as _Doc
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import _Cell, Table
    from docx.text.paragraph import Paragraph

    if isinstance(parent, _Doc):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Unsupported parent")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)
