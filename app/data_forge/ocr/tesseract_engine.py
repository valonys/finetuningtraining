"""Tesseract backend — always available fallback (CPU, no deep-learning deps)."""
from __future__ import annotations

from .base import OCREngine, OCRResult


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    @classmethod
    def available(cls) -> bool:
        try:
            import pytesseract   # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, image) -> OCRResult:
        import pytesseract
        text = pytesseract.image_to_string(image, lang=self.lang)
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=pytesseract.Output.DICT
            )
        except Exception:
            data = None

        blocks: list[dict] = []
        confs: list[float] = []
        if data and "text" in data:
            for i, txt in enumerate(data["text"]):
                if not txt.strip():
                    continue
                conf = float(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.0
                blocks.append({
                    "text": txt,
                    "bbox": [
                        int(data["left"][i]),
                        int(data["top"][i]),
                        int(data["left"][i] + data["width"][i]),
                        int(data["top"][i] + data["height"][i]),
                    ],
                    "conf": conf,
                })
                if conf > 0:
                    confs.append(conf)

        return OCRResult(
            text=text.strip(),
            engine=self.name,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            blocks=blocks,
        )
