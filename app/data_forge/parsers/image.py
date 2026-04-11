"""Image parser — delegates everything to the OCR pipeline."""
from __future__ import annotations

from ..ingest import IngestedRecord


def parse_image(path: str, ocr_engine: str | None = None, **_kw) -> IngestedRecord:
    from ..ocr.pipeline import run_ocr
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow not installed") from e

    img = Image.open(path)
    result = run_ocr(img, engine=ocr_engine)

    return IngestedRecord(
        source=path,
        source_type="image",
        text=result.text,
        tables=result.tables,
        metadata={
            "ocr_engine": result.engine,
            "width": img.width,
            "height": img.height,
            "confidence": result.confidence,
        },
        images=[{"path": path, "width": img.width, "height": img.height}],
    )
