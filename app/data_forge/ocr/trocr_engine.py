"""
TrOCR backend — transformer OCR, excellent for handwriting.

Uses microsoft/trocr-base-handwritten by default. Heavy — only loads when
explicitly selected.
"""
from __future__ import annotations

from .base import OCREngine, OCRResult


class TrOCREngine(OCREngine):
    name = "trocr"

    def __init__(self, model_id: str = "microsoft/trocr-base-handwritten"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)

        try:
            import torch
            self._device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
            self.model.to(self._device)
        except Exception:
            self._device = "cpu"

    @classmethod
    def available(cls) -> bool:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, image) -> OCRResult:
        import torch
        pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=512)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return OCRResult(text=text.strip(), engine=self.name, confidence=0.0)
