"""
OCR pipeline — selects an engine per the hardware profile and exposes
a single `run_ocr(image, engine=None)` function.

Preference order (overridden by explicit `engine=` or hardware profile):
    docling > paddleocr > rapidocr > tesseract > trocr

The chosen engine is cached per-name so we don't re-load weights on every call.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

# ── Registry (name → class) ────────────────────────────────────
_ENGINE_CLASSES: dict[str, type[OCREngine]] = {}


def _register(cls: type[OCREngine]):
    _ENGINE_CLASSES[cls.name] = cls
    return cls


# Lazy import + register. EVERY engine import is guarded — a missing
# top-level dep (e.g. numpy for rapidocr, pytesseract for tesseract)
# must NOT break the whole OCR subsystem or anything that calls
# `list_available_engines()` (including /healthz).
def _bootstrap_registry() -> None:
    if _ENGINE_CLASSES:
        return
    for loader in (
        _load_rapidocr, _load_tesseract,
        _load_paddleocr, _load_docling, _load_trocr,
    ):
        try:
            loader()
        except Exception as e:
            logger.debug(f"OCR engine unavailable: {loader.__name__} ({e})")


def _load_rapidocr() -> None:
    from .rapidocr_engine import RapidOCREngine
    _register(RapidOCREngine)

def _load_tesseract() -> None:
    from .tesseract_engine import TesseractEngine
    _register(TesseractEngine)

def _load_paddleocr() -> None:
    from .paddleocr_engine import PaddleOCREngine
    _register(PaddleOCREngine)

def _load_docling() -> None:
    from .docling_engine import DoclingEngine
    _register(DoclingEngine)

def _load_trocr() -> None:
    from .trocr_engine import TrOCREngine
    _register(TrOCREngine)


_instances: dict[str, OCREngine] = {}


def list_available_engines() -> list[str]:
    _bootstrap_registry()
    return [name for name, cls in _ENGINE_CLASSES.items() if cls.available()]


def run_ocr(image, engine: Optional[str] = None) -> OCRResult:
    """
    Run OCR on a PIL image. Resolves the engine in this order:
      1. Explicit `engine=` kwarg
      2. Hardware-profile default (rapidocr / paddleocr / docling / tesseract)
      3. First available engine in preference order
    """
    _bootstrap_registry()

    name = engine or _default_engine_name()
    if name not in _ENGINE_CLASSES or not _ENGINE_CLASSES[name].available():
        # Fall through to the first available engine
        for candidate in ["docling", "paddleocr", "rapidocr", "tesseract", "trocr"]:
            if candidate in _ENGINE_CLASSES and _ENGINE_CLASSES[candidate].available():
                name = candidate
                break
        else:
            raise RuntimeError(
                "No OCR engine available. Install one of: "
                "rapidocr-onnxruntime, pytesseract, paddleocr, docling, transformers (trocr)."
            )

    if name not in _instances:
        logger.info(f"🔤 Loading OCR engine: {name}")
        _instances[name] = _ENGINE_CLASSES[name]()

    return _instances[name].run(image)


def _default_engine_name() -> str:
    """Ask the hardware profile what the appropriate default is."""
    try:
        from app.hardware import resolve_profile
        return resolve_profile().default_ocr
    except Exception:
        return "rapidocr"
