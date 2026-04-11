"""
RapidOCR backend — ONNX runtime, fully cross-platform (Mac/Windows/Linux/CPU/GPU).
Our preferred default when CUDA isn't guaranteed.
"""
from __future__ import annotations

import numpy as np

from .base import OCREngine, OCRResult


class RapidOCREngine(OCREngine):
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._rapid = RapidOCR()

    @classmethod
    def available(cls) -> bool:
        try:
            import rapidocr_onnxruntime   # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, image) -> OCRResult:
        arr = np.array(image.convert("RGB"))
        results, _elapsed = self._rapid(arr)
        blocks: list[dict] = []
        texts: list[str] = []
        confs: list[float] = []

        if results:
            for item in results:
                # rapidocr tuple shape: (box, text, confidence)
                bbox, txt, conf = item
                if not txt:
                    continue
                texts.append(txt)
                confs.append(float(conf))
                blocks.append({
                    "text": txt,
                    "bbox": _flatten_bbox(bbox),
                    "conf": float(conf),
                })

        return OCRResult(
            text="\n".join(texts),
            engine=self.name,
            confidence=float(np.mean(confs)) if confs else 0.0,
            blocks=blocks,
        )


def _flatten_bbox(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return [min(xs), min(ys), max(xs), max(ys)]
