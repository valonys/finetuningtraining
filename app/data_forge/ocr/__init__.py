"""OCR subsystem — engine-agnostic with graceful fallbacks."""
from .base import OCREngine, OCRResult
from .pipeline import run_ocr, list_available_engines

__all__ = ["OCREngine", "OCRResult", "run_ocr", "list_available_engines"]
